from pprint import pprint
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint
from torch_geometric.nn.glob import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import to_dense_batch

from model_utils import MLP, PositionalEncoding, GlobalAttentionHalf


class Readout(nn.Module):

    def __init__(self, args, use_out_linear=True):
        super().__init__()
        self.args = args
        assert args.readout_name is not None
        self.name = args.readout_name
        self.num_body_layers = self.args.num_decoder_body_layers

        self.pe = None
        # [N, F] -> [N, F]
        if self.args.use_transformer:
            self.enc_in = self.build_body_transformer(args)
        else:
            self.enc_in = self.build_body_mlp()

        if self.args.use_pergraph_attr:
            self.pergraph_fc = MLP(
                num_layers=1,
                num_input=args.pergraph_channels,
                num_hidden=None,
                num_out=args.pergraph_hidden_channels,
                activation=self.args.activation,
                use_bn=self.args.use_bn,
                dropout=self.args.dropout_channels,
                activate_last=True,
            )
            pergraph_channels = self.args.pergraph_hidden_channels
        else:
            pergraph_channels = 0

        if "att" in self.name:
            self.att = GlobalAttentionHalf(
                gate_nn=nn.Linear(args.hidden_channels, 1, bias=False))
        else:
            self.att = None

        self.use_out_linear = use_out_linear
        if use_out_linear:
            num_readout_types = len(self.name.split("-"))
            self.fc_out = nn.Linear(
                pergraph_channels + num_readout_types * args.hidden_channels,
                args.num_classes,
            )

    def build_body_mlp(self, **kwargs):
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

    def build_body_transformer(self, args):
        if args.is_obs_sequential:
            self.pe = PositionalEncoding(
                max_len=args.obs_max_len,
                num_channels=args.hidden_channels,
            )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.hidden_channels,
            nhead=8,
            dim_feedforward=args.hidden_channels,
            dropout=args.dropout_channels,
            activation=args.activation,
        )
        layer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_body_layers)
        layer.layer_repr = f"Transformer(L={self.num_body_layers}, F={args.hidden_channels})"
        return layer

    def forward(self, x, pergraph_attr=None, batch=None):

        B = int(batch.max().item() + 1) if batch is not None else 1
        if self.args.use_transformer:
            x, mask = to_dense_batch(x, batch)  # [B, N_max, F]
            B, is_multi_batch = x.size(0), (x.size(0) > 1)
            if self.pe is not None:
                x = self.pe(x)
                if is_multi_batch:
                    x[~mask] = 0.
            x = self.enc_in(x)
            x = x[mask] if is_multi_batch else x.squeeze()  # [B, N_max, F] -> [\sum N, F]
        else:
            mask = None
            x = self.enc_in(x)

        o_list = []
        if "mean" in self.name:
            o_list.append(torch.mean(x, dim=0) if batch is None else
                          global_mean_pool(x, batch, B))
        if "max" in self.name:
            is_half = x.dtype == torch.half
            x = x.float() if is_half else x
            o_list.append(torch.max(x, dim=0).values if batch is None else
                          global_max_pool(x, batch, B).half() if is_half else
                          global_max_pool(x, batch, B))
        if "sum" in self.name:
            o_list.append(torch.sum(x, dim=0) if batch is None else
                          global_add_pool(x, batch, B))
        if "att" in self.name:
            if B == 1:
                o_list.append(self.att(x, (~mask).squeeze().long(), size=B).squeeze())
            else:
                o_list.append(self.att(x, batch, size=B).squeeze())
        z_g = torch.cat(o_list, dim=-1)  # [F * #type] or  [B, F * #type]
        if self.use_out_linear:
            if self.args.use_pergraph_attr:
                p_g = self.pergraph_fc(pergraph_attr)  # [F] or [1, F]
                if batch is not None:
                    p_g = p_g.expand(B, -1)  # [B, F]
                else:
                    p_g = p_g.squeeze()  # [F]
                z_with_p_g = torch.cat([z_g, p_g], dim=-1)  # [F * #type + F] or [B, F * #type + F]
            else:
                z_with_p_g = z_g
            return z_g, self.fc_out(z_with_p_g).view(B, -1)
        else:
            return z_g.view(B, -1)

    def __repr__(self):
        try:
            enc_in_layer_repr = self.enc_in.layer_repr()
        except TypeError:
            enc_in_layer_repr = self.enc_in.layer_repr
        return "{}(name={}, in_linear={}, out_linear={}, use_pergraph_attr={})".format(
            self.__class__.__name__, self.name,
            enc_in_layer_repr,
            None if not self.use_out_linear else "{}->{}".format(self.fc_out.in_features, self.fc_out.out_features),
            self.args.use_pergraph_attr,
        )


if __name__ == '__main__':
    from arguments import get_args

    _args = get_args("SGI", "FNTN", "TEST+MEMO")
    _args.readout_name = "mean-att"
    _args.use_pergraph_attr = True
    _args.num_decoder_body_layers = 2
    _args.use_transformer = True

    # Readout(name=mean-max, in_linear=64->64->64, out_linear=192->4, use_pergraph_attr=True)
    _ro = Readout(_args)

    _x = torch.ones(10 * 64).view(10, 64)
    _pga = torch.arange(_args.pergraph_channels) * 0.1
    _batch = torch.zeros(10).long()
    _batch[4:] = 1
    cprint("-- w/ batch w/ pga", "red")
    _z, _logits = _ro(_x, _pga, _batch)
    print(_ro)
    print("_z", _z.size())  # [2, 128]
    print("_logits", _logits.size())  # [2, 4]

    cprint("-- w/ pga", "red")
    _z, _logits = _ro(_x, _pga)
    print(_ro)
    print("_z", _z.size())  # [128]
    print("_logits", _logits.size())  # [1, 4]

    _args.use_pergraph_attr = False
    # Readout(name=mean-max, in_linear=64->64->64, out_linear=192->4, use_pergraph_attr=False)
    _ro = Readout(_args)
    cprint("-- wo/ all", "red")
    print(_ro)
    _z, _logits = _ro(_x)
    print("_z", _z.size())  # [128]
    print("_logits", _logits.size())  # [1, 4]
