from pprint import pprint
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint
from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.utils import to_dense_batch, softmax
from torch_scatter import scatter_add

from model_utils import MLP, PositionalEncoding, BilinearWith1d, GlobalAttentionHalf, GlobalMeanPool
from utils import act, get_extra_repr, softmax_half


class ObservedSubgraphPooler(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.pe = None

        if args.use_soft_attention_pooling:
            self.pool = GlobalAttentionHalf(gate_nn=nn.Linear(args.hidden_channels, 1, bias=False))
        else:
            self.pool = GlobalMeanPool()

        if args.use_transformer:
            if args.is_obs_sequential:
                self.pe = PositionalEncoding(max_len=args.obs_max_len, num_channels=args.hidden_channels)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=args.hidden_channels, nhead=8, dim_feedforward=args.hidden_channels,
            )
            self.obs_enc = nn.TransformerEncoder(encoder_layer, num_layers=args.num_decoder_body_layers)
        else:
            self.obs_enc = MLP(
                args.num_decoder_body_layers, args.hidden_channels, args.hidden_channels, args.hidden_channels,
                activation="relu", dropout=0.1,
            )

    def forward(self, x, batch=None, is_batch_sorted=True):

        if batch is not None and not is_batch_sorted:
            # Generally, batch is sorted.
            sorted_batch, perm = torch.sort(batch)
            x = x[perm, :]
            if torch.all(sorted_batch == batch):
                cprint("It is possible to turn off the batch-sort", "red")
            batch = sorted_batch
        x, mask = to_dense_batch(x, batch)  # [B, N_max, F]
        B, is_multi_batch = x.size(0), (x.size(0) > 1)

        if self.pe is not None:
            x = self.pe(x)
            if is_multi_batch:
                x[~mask] = 0.
        if self.args.use_transformer:
            x = self.obs_enc(x)
        else:  # MLP
            H = self.args.hidden_channels
            x = self.obs_enc(x.view(-1, H)).view(B, -1, H)
        x = act(x, self.args.activation)

        x = x[mask] if is_multi_batch else x.squeeze()  # [B, N_max, F] -> [\sum N, F]
        batch = (~mask).squeeze().long() if batch is None else batch
        x = self.pool(x, batch, size=B)
        return x

    def _get_repr_list(self):
        return [
            str(self.pe),
            "{}(L={}, I={}, H={}, O={})".format(
                self.obs_enc.__class__.__name__, self.obs_enc.num_layers,
                self.args.hidden_channels, self.args.hidden_channels, self.args.hidden_channels,
            ),
            str(self.pool),
        ]

    def __repr__(self):
        return "{}({}, {}, {}\n)".format(
            self.__class__.__name__,
            *[f"\n    {s}" for s in self._get_repr_list()],
        )


class SGIDecoder(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.num_body_layers = self.args.num_decoder_body_layers
        self.main_decoder_type = self.args.main_decoder_type
        self.include_obs_x_in_pooling = self.args.include_obs_x_in_pooling

        self.obs_sg_pooler_k = ObservedSubgraphPooler(args=args)  # [N_obs, F] -> [1, F]
        self.body_fc_q = self.build_body_fc()  # [N, F] -> [N, F]
        self.body_fc_v = self.build_body_fc()  # [N, F] -> [N, F]

        self.use_pergraph_attr = self.args.use_pergraph_attr
        if self.args.use_pergraph_attr:
            self.pergraph_fc = self.build_body_fc(
                num_layers=1, num_input=args.pergraph_channels, num_out=args.pergraph_hidden_channels,
            )
            pergraph_channels = self.args.pergraph_hidden_channels
        else:
            pergraph_channels = 0

        if self.main_decoder_type == "node":
            assert self.args.use_node_decoder, "use_node_decoder is not True."
            num_dec_classes = 2
            self.graph_fc = nn.Linear(self.args.hidden_channels + pergraph_channels, self.args.num_classes)
        elif self.main_decoder_type == "edge":
            assert self.args.use_edge_decoder, "use_edge_decoder is not True."
            num_dec_classes = 3
            self.graph_fc = nn.Linear(2 * self.args.hidden_channels + pergraph_channels, self.args.num_classes)
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

    def build_body_fc(self, **kwargs):
        kw = dict(
            num_layers=self.num_body_layers,
            num_input=self.args.hidden_channels,
            num_hidden=self.args.hidden_channels,
            num_out=self.args.hidden_channels,
            activation=self.args.activation,
            use_bn=self.args.use_bn,
            dropout=self.args.dropout_channels,
            activate_last=True,  # important
        )
        kw.update(**kwargs)
        return MLP(**kw)

    def forward(
            self, x, obs_x_index, edge_index_01, edge_index_2,
            pergraph_attr=None, batch=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        B = int(batch.max().item() + 1) if batch is not None else 1
        batch_obs_x = batch[obs_x_index] if batch is not None else None
        obs_g_k = self.obs_sg_pooler_k(x[obs_x_index], batch_obs_x)  # [B, F]
        x_q = self.body_fc_q(x)  # [N, F]
        x_v = self.body_fc_v(x)  # [N, F]

        dec_x, pool_x, dec_e, pool_e = None, None, None, None
        if self.args.use_node_decoder:
            dec_x, pool_x = self.decode_and_pool(
                obs_g_k, obs_x_index, x_q, x_v, edge_index_01,
                decoder_type="node",
                use_pool=self.main_decoder_type == "node",
                batch=batch, batch_size=B,
            )
        if self.args.use_edge_decoder:
            if edge_index_2 is not None:
                edge_index_012 = torch.cat([edge_index_01, edge_index_2], dim=1)
            else:
                edge_index_012 = edge_index_01
            dec_e, pool_e = self.decode_and_pool(
                obs_g_k, obs_x_index, x_q, x_v, edge_index_012,
                decoder_type="edge",
                use_pool=self.main_decoder_type == "edge",
                idx_to_pool=edge_index_01.size(1),
                batch=batch, batch_size=B,
            )

        z_g = pool_x if self.main_decoder_type == "node" else pool_e
        if self.args.use_pergraph_attr:
            p_g = self.pergraph_fc(pergraph_attr)  # [F] or [1, F]
            p_g = p_g.expand(B, -1) if batch is not None else p_g.view(B, -1)  # [B, F]
            z_with_p_g = torch.cat([z_g, p_g], dim=-1)  # [2F] or [B, 2F]
        else:
            z_with_p_g = z_g
        logits_g = self.graph_fc(z_with_p_g).view(B, -1)
        return z_g, logits_g, dec_x, dec_e

    def decode_and_pool(self, obs_g_k, obs_x_index, x_q, x_v, edge_index,
                        decoder_type, use_pool, idx_to_pool=None, batch=None, batch_size=1):
        # x_q, x_v: [N, F]
        # obs_x_k: [B, F]
        # edge_index: [2, E] or [2, \sum E]

        # [B, F] * [B, N_max, F] -> [B, N_max, 2] -> [N, 2]
        if decoder_type == "node":
            o_q, o_v = x_q, x_v  # [N, F]
            o_q, mask = to_dense_batch(o_q, batch)  # [B, N_max, F]
            decoded = self.node_dec(obs_g_k, o_q)  # [N, 2] or [B, N_max, 2]

        # [B, F] * [B, E_max, 2F] -> [B, E_max, 3] -> [\sum E, 3]
        elif decoder_type == "edge":
            o_q = x_q[edge_index].view(-1, 2 * self.args.hidden_channels)  # [\sum E, 2F]
            o_v = x_v[edge_index].view(-1, 2 * self.args.hidden_channels)  # [\sum E, 2F]
            o_q, mask = to_dense_batch(o_q, batch)   # [B, E_max, F]
            decoded = self.edge_dec(obs_g_k, o_q)  # [\sum E, 3] or [B, E_max, 3]

        else:
            raise ValueError(f"Wrong decoder_type: {decoder_type}")

        decoded = decoded.squeeze() if batch_size == 1 else decoded[mask]  # [N, 2] or [\sum E, 3]
        batch = (~mask).squeeze().long() if batch is None else batch

        if use_pool:
            score = decoded[:, 0] if idx_to_pool is None else decoded[:idx_to_pool, 0]  # [N,] or [\sum E,]
            perm = topk(score, self.pool_ratio, batch, self.pool_min_score)
            if self.include_obs_x_in_pooling:
                perm = torch.unique(torch.cat([obs_x_index, perm], dim=0))
            batch_topk, o_v_topk = batch[perm], o_v[perm]
            score_topk = softmax_half(score[perm].view(-1, 1), batch_topk, num_nodes=batch_size)
            pooled = scatter_add(score_topk * o_v_topk, batch_topk, dim=0, dim_size=batch_size)  # [B, F]
            return decoded, pooled
        else:
            return decoded, None

    def extra_repr(self):
        return get_extra_repr(self, ["main_decoder_type",
                                     'pool_ratio' if self.pool_min_score is None else 'pool_min_score',
                                     "use_pergraph_attr"])


if __name__ == '__main__':
    from arguments import get_args
    from pytorch_lightning import seed_everything

    def _print(_z, _logits, _dx, _de):
        print(f"_z: {_z.size(), _z.mean().item(), _z.std().item()}")
        print(f"_logits: {_logits.size(), _logits.mean().item(), _logits.std().item()}")
        if _dx is not None:
            print(f"_dx: {_dx.size(), _dx.mean().item(), _dx.std().item()}")
        if _de is not None:
            print(f"_de: {_de.size(), _de.mean().item(), _de.std().item()}")

    seed_everything(42)
    _args = get_args("SGI", "FNTN", "TEST+MEMO")
    _args.use_pool_min_score = False  # todo
    _args.pool_ratio = 0.3
    _args.main_decoder_type = "node"  # todo
    _args.use_edge_decoder = False  # todo
    _args.use_node_decoder = True
    _args.use_pergraph_attr = True
    _args.use_transformer = False  # todo
    _args.use_soft_attention_pooling = False  # todo
    dec = SGIDecoder(_args)
    print(dec)

    _N = 9
    _x = torch.arange(_N * 64).view(_N, 64) * 0.1
    _ei = torch.randint(0, _N, [2, 17])
    _ei2 = torch.randint(0, _N, [2, 13])
    _obs_x_index = torch.arange(5).long()
    _pga = torch.arange(_args.pergraph_channels) * 0.1
    _batch = torch.zeros(_N).long()
    _batch[3:] = 1

    seed_everything(42)
    _z, _logits, _dx, _de = dec(_x, _obs_x_index, _ei, _ei2, _pga, _batch)
    cprint("w/ pga, w/ batch", "red")
    _print(_z, _logits, _dx, _de)

    seed_everything(42)
    _batch[:] = 0
    _z, _logits, _dx, _de = dec(_x, _obs_x_index, _ei, _ei2, _pga, _batch)
    cprint("w/ pga, w/ batch but single", "red")
    _print(_z, _logits, _dx, _de)

    seed_everything(42)
    _z, _logits, _dx, _de = dec(_x, _obs_x_index, _ei, _ei2, _pga)
    cprint("w/ pga", "red")
    _print(_z, _logits, _dx, _de)
