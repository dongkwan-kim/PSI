from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import DeepGraphInfomax
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import to_dense_batch

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

    def forward(self, summarized, x_pos_and_neg, batch_pos_and_neg=None, ptr_pos_and_neg=None):
        """
        :param summarized: [B, F_s]
        :param x_pos_and_neg: [N_pos + N_neg, F_h]
        :param batch_pos_and_neg: [N_pos + N_neg,]
        :param ptr_pos_and_neg: [1] (special case of batch_size=1)
        :return:
        """
        z_pos_and_neg = self.encoder(x_pos_and_neg) if self.encoder is not None else x_pos_and_neg
        if batch_pos_and_neg is not None:
            dense_z_pos_and_neg, mask = to_dense_batch(z_pos_and_neg, batch_pos_and_neg)  # [2B, N_max, F]
            B = dense_z_pos_and_neg.size(0)
            z_pos, z_neg = dense_z_pos_and_neg[:B // 2], dense_z_pos_and_neg[B // 2:]
            loss = self.loss(pos_z=z_pos, neg_z=z_neg, summary=summarized,
                             is_batched=True, batch_mask=mask)
        elif ptr_pos_and_neg is not None:
            z_pos, z_neg = z_pos_and_neg[:ptr_pos_and_neg, :], z_pos_and_neg[ptr_pos_and_neg:, :]
            loss = self.loss(pos_z=z_pos, neg_z=z_neg, summary=summarized, is_batched=False)
        else:
            raise ValueError
        return loss

    def loss(self, pos_z, neg_z, summary, is_batched=False, batch_mask=None):
        r"""Computes the mutual information maximization objective.

        :param pos_z: [N, F_h] or [B, N_max, F_h]
        :param neg_z: [N, F_h] or [B, N_max, F_h]
        :param summary: [F_s] or [B, F_s]
        :param is_batched: bool
        :param batch_mask: [B, N_max]
        """
        if not is_batched:
            pos_loss = -torch.log(self.discriminate(
                pos_z, summary, sigmoid=True) + EPS).mean()
            neg_loss = -torch.log(1 - self.discriminate(
                neg_z, summary, sigmoid=True) + EPS).mean()
        else:
            pos_loss = -torch.log(self.batched_discriminate(
                pos_z, summary, batch_mask, sigmoid=True) + EPS).mean()
            neg_loss = -torch.log(1 - self.batched_discriminate(
                neg_z, summary, batch_mask, sigmoid=True) + EPS).mean()
        return pos_loss + neg_loss

    def batched_discriminate(self, z, summary, batch_mask, sigmoid=True):
        """
        :param z: [B, N_max, F_h]
        :param summary: [B, F_s]
        :param sigmoid: bool
        :param batch_mask: [B, N_max]
        """
        # value = torch.matmul(z, torch.matmul(self.weight, summary))
        value = torch.einsum("bnh,hs,bs->bn", z, self.weight, summary)
        value = value[batch_mask]
        return torch.sigmoid(value) if sigmoid else value

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
