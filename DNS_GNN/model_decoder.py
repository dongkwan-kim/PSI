from pprint import pprint
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.pool.topk_pool import topk

from model_utils import MultiLinear, PositionalEncoding, AttentionPool, BilinearWith1d
from utils import act, get_extra_repr


class ObsSummarizer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.is_obs_sequential:
            self.pe = PositionalEncoding(max_len=args.obs_max_len, num_channels=args.hidden_channels)
        else:
            self.pe = None
        self.pool = AttentionPool(args.hidden_channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.hidden_channels, nhead=8, dim_feedforward=args.hidden_channels,
        )
        self.tf_enc = nn.TransformerEncoder(encoder_layer, num_layers=args.num_decoder_body_layers)

    def forward(self, x):
        if self.pe is not None:
            x = self.pe(x)
        x = x.view(1, x.size(0), x.size(1))
        x = self.tf_enc(x)
        x = x.squeeze()
        x = act(x, self.args.activation)
        x = self.pool(x)
        return x

    def _get_repr_list(self):
        return [
            str(self.pe),
            "{}(L={}, I={}, H={}, O={})".format(
                self.tf_enc.__class__.__name__, self.tf_enc.num_layers,
                self.args.hidden_channels, self.args.hidden_channels, self.args.hidden_channels,
            ),
            str(self.pool),
        ]

    def __repr__(self):
        return "{}({}, {}, {}\n)".format(
            self.__class__.__name__,
            *[f"\n    {s}" for s in self._get_repr_list()],
        )


class DNSDecoder(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.num_body_layers = self.args.num_decoder_body_layers
        self.main_decoder_type = self.args.main_decoder_type

        self.obs_summarizer_k = ObsSummarizer(args=args)  # [N_obs, F] -> [1, F]
        self.body_fc_q = self.build_body_fc()  # [N, F] -> [N, F]
        self.body_fc_v = self.build_body_fc()  # [N, F] -> [N, F]

        if self.main_decoder_type == "node":
            assert self.args.use_node_decoder, "use_node_decoder is not True."
            num_dec_classes = 2
            self.graph_fc = nn.Linear(self.args.hidden_channels, self.args.num_classes)
        elif self.main_decoder_type == "edge":
            assert self.args.use_edge_decoder, "use_edge_decoder is not True."
            num_dec_classes = 3
            self.graph_fc = nn.Linear(2 * self.args.hidden_channels, self.args.num_classes)
        else:
            raise ValueError(f"Wrong decoder_type: {self.main_decoder_type}")

        if self.args.use_node_decoder:
            # [1, F], [N, F] -> [N, 2]
            # classes: in-subgraph, or not.
            self.node_dec = BilinearWith1d(
                in1_features=self.args.hidden_channels,
                in2_features=self.args.hidden_channels,
                out_features=2,
            )
        if self.args.use_edge_decoder:
            # [1, 2F], [\sum E, 2F] -> [\sum E, 3]
            # classes: in-subgraph, in-graph, or not.
            self.edge_dec = BilinearWith1d(
                in1_features=self.args.hidden_channels,
                in2_features=2 * self.args.hidden_channels,  # *2 for edges
                out_features=3,
            )

        self.pool_ratio = self.args.pool_ratio
        self.pool_min_score = 1 / num_dec_classes if self.args.use_pool_min_score else None

    def build_body_fc(self):
        return MultiLinear(
            num_layers=self.num_body_layers,
            num_input=self.args.hidden_channels,
            num_hidden=self.args.hidden_channels,
            num_out=self.args.hidden_channels,
            activation=self.args.activation,
            use_bn=True,
            dropout=self.args.dropout_channels,
            activate_last=True,  # important
        )

    def forward(
            self, x, obs_x_idx, edge_index_01, edge_index_2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        obs_x_k = self.obs_summarizer_k(x[obs_x_idx])  # [1, F]
        x_q = self.body_fc_q(x)  # [N, F]
        x_v = self.body_fc_v(x)  # [N, F]

        dec_x, pool_x, dec_e, pool_e = None, None, None, None
        if self.args.use_node_decoder:
            dec_x, pool_x = self.decode_and_pool(
                obs_x_k, x_q, x_v, edge_index_01,
                decoder_type="node", use_pool=self.main_decoder_type == "node",
            )
        if self.args.use_edge_decoder:
            if edge_index_2 is not None:
                edge_index_012 = torch.cat([edge_index_01, edge_index_2], dim=1)
            else:
                edge_index_012 = edge_index_01
            dec_e, pool_e = self.decode_and_pool(
                obs_x_k, x_q, x_v, edge_index_012,
                decoder_type="edge", use_pool=self.main_decoder_type == "edge",
                idx_to_pool=edge_index_01.size(1),
            )

        o = pool_x if self.main_decoder_type == "node" else pool_e
        logits_g = self.graph_fc(o).view(1, -1)
        return logits_g, dec_x, dec_e

    def decode_and_pool(self, obs_x_k, x_q, x_v, edge_index, decoder_type, use_pool, idx_to_pool=None):
        # x_q, x_v: [N, F]
        # obs_x: [1, F]
        # edge_index: [2, E] or [2, \sum E]
        if decoder_type == "node":
            o_q = x_q  # [N, F]
            o_v = x_v
            decoded = self.node_dec(obs_x_k, o_q)  # [N, 2]
        elif decoder_type == "edge":
            o_q = x_q[edge_index].view(-1, 2 * self.args.hidden_channels)  # [\sum E, 2F]
            o_v = x_v[edge_index].view(-1, 2 * self.args.hidden_channels)  # [\sum E, 2F]
            decoded = self.edge_dec(obs_x_k, o_q)  # [\sum E, 3]
        else:
            raise ValueError(f"Wrong decoder_type: {decoder_type}")

        if use_pool:
            score = decoded[:, 0] if idx_to_pool is None else decoded[:idx_to_pool, 0]
            batch = edge_index.new_zeros(score.size(0))
            perm = topk(score, self.pool_ratio, batch, self.pool_min_score)
            pooled = torch.einsum("nf,n->f", o_v[perm], torch.softmax(score[perm], dim=0))
            return decoded, pooled
        else:
            return decoded, None

    def extra_repr(self):
        return get_extra_repr(self, ["main_decoder_type",
                                     'pool_ratio' if self.pool_min_score is None else 'pool_min_score'])


if __name__ == '__main__':
    from arguments import get_args

    _args = get_args("DNS", "FNTN", "TEST+MEMO")
    _args.use_pool_min_score = False  # todo
    _args.main_decoder_type = "node"  # todo
    _args.use_edge_decoder = True
    _args.use_node_decoder = True
    dec = DNSDecoder(_args)
    print(dec)

    _x = torch.arange(7 * 64).view(7, 64) * 0.1
    _ei = torch.randint(0, 7, [2, 17])
    _ei2 = torch.randint(0, 7, [2, 13])
    _obs_x_idx = torch.arange(3).long()
    _g, _dx, _de = dec(_x, _obs_x_idx, _ei, _ei2)

    print(f"__g: {_g.size()}")
    if _dx is not None:
        print(f"_dx: {_dx.size()}")
    if _de is not None:
        print(f"_de: {_de.size()}")
