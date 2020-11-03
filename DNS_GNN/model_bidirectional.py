from copy import deepcopy
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiConv(nn.Module):

    def __init__(self, base_conv, reset_at_init=True):
        super().__init__()
        self.conv = deepcopy(base_conv)
        self.rev_conv = base_conv
        if reset_at_init:
            self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.rev_conv.reset_parameters()

    def forward(self, x, edge_index, *args, **kwargs):
        rev_edge_index = edge_index[[1, 0]]
        fwd_x = self.conv(x, edge_index, *args, **kwargs)
        rev_x = self.rev_conv(x, rev_edge_index, *args, **kwargs)
        return torch.cat([fwd_x, rev_x], dim=1)

    def __repr__(self):
        return "Bi{}".format(self.conv.__repr__())


if __name__ == '__main__':
    from model_encoder import DNSEncoder
    from arguments import get_args
    _args = get_args("DNS", "FNTN", "TEST+MEMO")
    enc = BiConv(DNSEncoder(_args))
    print(enc)

    _x = torch.ones(10 * 64).view(10, 64)
    _ei = torch.randint(0, 10, [2, 10])
    print(enc(_x, _ei).size())
