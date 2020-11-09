from pprint import pprint
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import MultiLinear


class Readout(nn.Module):

    def __init__(self, args, with_linear=True):
        super().__init__()
        self.args = args
        assert args.readout_name is not None
        self.name = args.readout_name

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

        self.with_linear = with_linear
        if with_linear:
            num_readout_types = len(self.name.split("-"))
            self.fc = nn.Linear(
                pergraph_channels + num_readout_types * args.hidden_channels,
                args.num_classes,
            )

    def forward(self, x, pergraph_attr=None):
        o_list = []
        if "mean" in self.name:
            o_list.append(torch.mean(x, dim=0))
        if "max" in self.name:
            o_list.append(torch.mean(x, dim=0))
        if "sum" in self.name:
            o_list.append(torch.sum(x, dim=0))
        o = torch.cat(o_list, dim=0)
        if self.with_linear:
            if self.args.use_pergraph_attr:
                o = torch.cat([o, self.pergraph_fc(pergraph_attr)], dim=0)
            return self.fc(o).view(1, -1)
        else:
            return o.view(1, -1)

    def __repr__(self):
        return "{}(name={}, with_linear={}, use_pergraph_attr={})".format(
            self.__class__.__name__, self.name,
            False if not self.with_linear else "{}->{}".format(self.fc.in_features, self.fc.out_features),
            self.args.use_pergraph_attr,
        )


if __name__ == '__main__':
    from arguments import get_args

    _args = get_args("DNS", "FNTN", "TEST+MEMO")
    _args.readout_name = "mean-max"
    _args.use_pergraph_attr = True
    _ro = Readout(_args)
    print(_ro)

    _x = torch.ones(10 * 64).view(10, 64)
    _pga = torch.arange(_args.pergraph_channels) * 0.1
    print(_ro(_x, _pga).size())
