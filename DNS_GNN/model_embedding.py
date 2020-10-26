from pprint import pprint
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class DNSEmbedding(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        if self.args.global_channel_type == "Embedding":
            self.embedding = nn.Embedding(self.args.num_nodes_global, self.args.global_channels)
        elif self.args.global_channel_type == "Random":
            self.embedding = nn.Embedding(self.args.num_nodes_global, self.args.global_channels)
            self.embedding.weight.requires_grad = False
        elif self.args.global_channel_type == "Feature":
            self.embedding = None
        else:
            raise ValueError(f"Wrong global_channel_type: {self.args.global_channel_type}")

    def forward(self, x_indices):
        if self.embedding is not None:
            x_indices = x_indices.squeeze()
            return self.embedding(x_indices)
        else:
            return x_indices

    def __repr__(self):
        return '{}({}, {}, type={})'.format(
            self.__class__.__name__,
            self.args.num_nodes_global,
            self.args.global_channels,
            self.args.global_channel_type,
        )


if __name__ == '__main__':
    from arguments import get_args
    _args = get_args("DNS", "FNTN", "TEST+MEMO")
    _args.num_nodes_global = 11
    _args.global_channel_type = "Random"
    de = DNSEmbedding(_args)
    print(de)

    _x = torch.arange(11)
    print(de(_x).size())
