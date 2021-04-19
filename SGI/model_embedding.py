from pprint import pprint
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class VersatileEmbedding(nn.Module):

    def __init__(self, args, pretrained_embedding=None):
        super().__init__()
        self.args = args

        if self.args.global_channel_type == "Embedding":
            self.embedding = nn.Embedding(self.args.num_nodes_global, self.args.global_channels)
        elif self.args.global_channel_type == "Random":
            self.embedding = nn.Embedding(self.args.num_nodes_global, self.args.global_channels)
            self.embedding.weight.requires_grad = False
        elif self.args.global_channel_type == "Feature":
            self.embedding = None
        elif self.args.global_channel_type == "Pretrained":
            assert pretrained_embedding is not None
            N, C = pretrained_embedding.size()
            assert self.args.num_nodes_global == N
            assert self.args.global_channels == C
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=False)
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
    _args = get_args("SGI", "FNTN", "TEST+MEMO")
    _args.num_nodes_global = 11
    _args.global_channels = 32

    _args.global_channel_type = "Pretrained"

    if _args.global_channel_type == "Pretrained":
        _pte = torch.arange(11 * 32).view(11, 32).float()
    else:
        _pte = None

    de = VersatileEmbedding(_args, _pte)
    print(de)
    print("Embedding: {} +- {}".format(
        de.embedding.weight.mean().item(),
        de.embedding.weight.std().item(),
    ))

    _x = torch.arange(11)
    print(de(_x).size())
