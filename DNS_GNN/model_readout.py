from pprint import pprint
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Readout(nn.Module):

    def __init__(self, args, with_linear=True):
        super().__init__()
        self.args = args
        assert args.readout_name is not None
        self.name = args.readout_name

        self.with_linear = with_linear
        if with_linear:
            num_readout_types = len(self.name.split("-"))
            self.fc = nn.Linear(num_readout_types * args.hidden_channels, args.num_classes)

    def forward(self, x):
        o_list = []
        if "mean" in self.name:
            o_list.append(torch.mean(x, dim=0))
        if "max" in self.name:
            o_list.append(torch.mean(x, dim=0))
        if "sum" in self.name:
            o_list.append(torch.sum(x, dim=0))
        o = torch.cat(o_list, dim=0).view(1, -1)
        if self.with_linear:
            return self.fc(o)
        else:
            return o

    def __repr__(self):
        return "{}(name={}, with_linear={})".format(
            self.__class__.__name__, self.name,
            False if not self.with_linear else "{}->{}".format(self.fc.in_features, self.fc.out_features)
        )


if __name__ == '__main__':
    from arguments import get_args
    _args = get_args("DNS", "FNTN", "TEST+MEMO")
    _args.readout_name = "mean-max"
    _ro = Readout(_args)
    print(_ro)

    _x = torch.ones(10 * 64).view(10, 64)
    print(_ro(_x).size())
