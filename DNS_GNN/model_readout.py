from pprint import pprint
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint
from torch_geometric.nn.glob import global_mean_pool, global_max_pool, global_add_pool

from model_utils import MultiLinear


class Readout(nn.Module):

    def __init__(self, args, use_out_linear=True):
        super().__init__()
        self.args = args
        assert args.readout_name is not None
        self.name = args.readout_name
        self.num_body_layers = self.args.num_decoder_body_layers

        self.fc_in = self.build_body_fc()  # [N, F] -> [N, F]

        if self.args.use_pergraph_attr:
            self.pergraph_fc = MultiLinear(
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

        self.use_out_linear = use_out_linear
        if use_out_linear:
            num_readout_types = len(self.name.split("-"))
            self.fc_out = nn.Linear(
                pergraph_channels + num_readout_types * args.hidden_channels,
                args.num_classes,
            )

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
        return MultiLinear(**kw)

    def forward(self, x, pergraph_attr=None, batch=None):

        B = int(batch.max().item() + 1) if batch is not None else 1
        x = self.fc_in(x)

        o_list = []
        if "mean" in self.name:
            o_list.append(torch.mean(x, dim=0) if batch is None else
                          global_mean_pool(x, batch, B))
        if "max" in self.name:
            o_list.append(torch.max(x, dim=0).values if batch is None else
                          global_max_pool(x, batch, B))
        if "sum" in self.name:
            o_list.append(torch.sum(x, dim=0) if batch is None else
                          global_add_pool(x, batch, B))
        z_g = torch.cat(o_list, dim=-1)  # [F * #type] or  [B, F * #type]
        if self.use_out_linear:
            if self.args.use_pergraph_attr:
                p_g = self.pergraph_fc(pergraph_attr)  # [F] or [1, F]
                if batch is not None:
                    p_g = p_g.expand(B, -1)  # [B, F]
                z_with_p_g = torch.cat([z_g, p_g], dim=-1)  # [F * #type + F] or [B, F * #type + F]
            else:
                z_with_p_g = z_g
            return z_g, self.fc_out(z_with_p_g).view(B, -1)
        else:
            return z_g.view(B, -1)

    def __repr__(self):
        return "{}(name={}, in_linear={}, out_linear={}, use_pergraph_attr={})".format(
            self.__class__.__name__, self.name,
            self.fc_in.layer_repr(),
            None if not self.use_out_linear else "{}->{}".format(self.fc_out.in_features, self.fc_out.out_features),
            self.args.use_pergraph_attr,
        )


if __name__ == '__main__':
    from arguments import get_args

    _args = get_args("DNS", "FNTN", "TEST+MEMO")
    _args.readout_name = "mean-max"
    _args.use_pergraph_attr = True
    _args.num_decoder_body_layers = 2

    # Readout(name=mean-max, in_linear=64->64->64, out_linear=192->4, use_pergraph_attr=True)
    _ro = Readout(_args)

    _x = torch.ones(10 * 64).view(10, 64)
    _pga = torch.arange(_args.pergraph_channels) * 0.1
    _batch = torch.zeros(10).long()
    _batch[:4] = 1
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
