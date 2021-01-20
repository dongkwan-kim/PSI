from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import DeepGraphInfomax
from torch_geometric.nn.inits import glorot

from model_utils import MultiLinear
from utils import EPSILON

EPS = EPSILON()


class InterSubgraphInfoMaxLoss(DeepGraphInfomax):

    def __init__(self, args, encoder=None):
        self.args = args
        if args.main_decoder_type == "node":
            self.summary_channels = args.hidden_channels
        elif args.main_decoder_type == "edge":
            self.summary_channels = 2 * args.hidden_channels
        else:
            raise ValueError

        if encoder is None:
            encoder = MultiLinear(
                num_layers=args.num_decoder_body_layers,
                num_input=args.hidden_channels,
                num_hidden=args.hidden_channels,
                num_out=args.hidden_channels,
                activation=args.activation,
                use_bn=args.use_bn,
                dropout=args.dropout_channels,
                activate_last=True,
            )

        super().__init__(args.hidden_channels, encoder=encoder, summary=None, corruption=None)

        # ref: In discriminate, torch.matmul(z, torch.matmul(self.weight, summary))
        del self.weight
        self.weight = Parameter(torch.Tensor(self.hidden_channels, self.summary_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)

    def forward(self, summarized, x_pos, x_neg):
        if self.encoder is not None:
            z_pos = self.encoder(x_pos)
            z_neg = self.encoder(x_neg)
        else:
            z_pos, z_neg = x_pos, x_neg
        loss = self.loss(pos_z=z_pos, neg_z=z_neg, summary=summarized)
        return loss

    def loss(self, pos_z, neg_z, summary):
        r"""Computes the mutual information maximization objective."""
        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(
            1 - self.discriminate(neg_z, summary, sigmoid=True) + EPS).mean()
        return pos_loss + neg_loss

    def __repr__(self):
        return '{}({}, {}, encoder={})'.format(
            self.__class__.__name__, self.hidden_channels, self.summary_channels,
            self.encoder,
        )


if __name__ == '__main__':
    from arguments import get_args
    _args = get_args("DNS", "FNTN", "TEST+MEMO")
    _isi = InterSubgraphInfoMaxLoss(_args)
    print(_isi)
    print("----")
    for m in _isi.modules():
        print(m)
    print("----")
    for p in _isi.named_parameters():
        print(p[0])
