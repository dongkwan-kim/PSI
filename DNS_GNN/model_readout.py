from pprint import pprint
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, x, pergraph_attr=None):

        x = self.fc_in(x)

        o_list = []
        if "mean" in self.name:
            o_list.append(torch.mean(x, dim=0))
        if "max" in self.name:
            o_list.append(torch.mean(x, dim=0))
        if "sum" in self.name:
            o_list.append(torch.sum(x, dim=0))
        z_g = torch.cat(o_list, dim=0)
        if self.use_out_linear:
            if self.args.use_pergraph_attr:
                z_g = torch.cat([z_g, self.pergraph_fc(pergraph_attr)], dim=0)
            return z_g, self.fc_out(z_g).view(1, -1)
        else:
            return z_g.view(1, -1)

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
    print(_ro)

    _x = torch.ones(10 * 64).view(10, 64)
    _pga = torch.arange(_args.pergraph_channels) * 0.1
    _z, _logits = _ro(_x, _pga)
    print(_z.size())  # [192]
    print(_logits.size())  # [1, 4]
