from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import DeepGraphInfomax
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter

from model_utils import MLP
from utils import EPSILON

EPS = EPSILON()


class InfoNCELoss(nn.Module):
    r"""InfoNCELoss: Mostly adopted from codes below
        - https://github.com/GraphCL/PyGCL/blob/main/GCL/losses/infonce.py
        - https://github.com/GraphCL/PyGCL/blob/main/GCL/models/samplers.py#L64-L65
    InfoNCELoss_* = - log [ exp(sim(g, n_*)/t) ] / [ \sum_i exp(sim(g, n_i)/t) ]
                  = - exp(sim(g, n_*)/t) + log [ \sum_i exp(sim(g, n_i)/t) ]
    """

    def __init__(self, temperature):
        """
        :param temperature: The MoCo paper uses 0.07, while SimCLR uses 0.5.
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, g, x, batch):
        return self.get_loss(g, x, batch)

    def get_loss(self, anchor_g, samples_n, batch) -> torch.FloatTensor:
        sim = self._similarity(anchor_g, samples_n)  # [B, F], [N, F] --> [B, N]
        sim = sim / self.temperature
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))

        if batch is None:
            assert anchor_g.size(0) == samples_n.size(0), \
                f"anchor_g's size {anchor_g.size(0)} is not samples_n's size {samples_n.size(0)}"
            batch = torch.arange(anchor_g.size(0), device=log_prob.device)

        pos_idx, counts_per_batch = self.pos_index(batch, return_counts_per_batch=True)

        # same as (log_prob * pos_mask).sum(dim=1) / torch.sum(pos_mask, dim=1)
        pos_log_prob = scatter(torch.take(log_prob, pos_idx), batch, dim=0, reduce="sum")
        loss = pos_log_prob / counts_per_batch
        return -loss.mean()

    def __repr__(self):
        return f"{self.__class__.__name__}(t={self.temperature})"

    @staticmethod
    def pos_index(batch: torch.Tensor, return_counts_per_batch=False, device=None):
        """
        :param batch: e.g., [0, 0, 1, 1, 1, 2, 2]
        :param return_counts_per_batch:
        :param device: Use batch.device if not given.
        :return: the 1d index of pos_mask.
            e.g., [0*7+0, 0*7+1, 1*7+2, 1*7+3, 1*7+4, 2*7+5, 2*7+6],
            that is, 1d index of
                 [[ True,  True, False, False, False, False, False],
                  [False, False,  True,  True,  True, False, False],
                  [False, False, False, False, False,  True,  True]])
        """
        b_index, b_counts = torch.unique_consecutive(batch, return_counts=True)

        b_cum_counts = torch.cumsum(b_counts, dim=0)
        b_cum_counts = torch.roll(b_cum_counts, 1)  # [2, 5, 7] -> [7, 2, 5]
        b_cum_counts[0] = 0

        num_nodes = batch.size(0)
        start_at_rows = b_index * num_nodes + b_cum_counts

        sparse_mask_row_list = []
        for _start, _count in zip(start_at_rows.tolist(), b_counts.tolist()):
            sparse_mask_row_list.append(torch.arange(_count) + _start)
        sparse_pos_mask = torch.cat(sparse_mask_row_list)

        device = device or batch.device
        if not return_counts_per_batch:
            return sparse_pos_mask.to(device)
        else:
            return sparse_pos_mask.to(device), b_counts.to(device)

    @staticmethod
    def pos_mask(batch: torch.Tensor):
        """Practically not usable because of memory limits."""
        bools_one_hot = torch.eye(batch.size(0), dtype=torch.bool, device=batch.device)  # [N]
        pos_mask = scatter(bools_one_hot, batch, dim=0, reduce="sum")  # [B, N]
        return pos_mask

    @staticmethod
    def _similarity(h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1)  # [B, F]
        h2 = F.normalize(h2)  # [N, F]
        return h1 @ h2.t()  # [B, N]


class G2GContrastiveLoss(nn.Module):

    def __init__(self, args, projector=None):
        super().__init__()
        self.args = args

        if projector is None:
            projector = MLP(
                num_layers=args.num_decoder_body_layers,
                num_input=args.hidden_channels,
                num_hidden=args.hidden_channels,
                num_out=args.hidden_channels,
                activation=args.activation,
                use_bn=args.use_bn,
                dropout=args.dropout_channels,
                activate_last=True,
            )
        self.projector = projector
        self.infonce_loss = InfoNCELoss(args.infonce_temperature)

    @property
    def encoder(self):
        return self.projector

    def forward(self, summarized, summarized_2):
        """
        :param summarized: [B, F_s]
        :param summarized_2: [B, F_s]
        :return:
        """
        summarized_2 = self.encoder(summarized_2) if self.encoder is not None else summarized_2
        loss = self.infonce_loss(summarized, summarized_2, batch=None)
        return loss


class G2LContrastiveLoss(DeepGraphInfomax):

    def __init__(self, args, projector=None):
        self.args = args
        if args.main_decoder_type == "node":
            self.summary_channels = args.hidden_channels
        elif args.main_decoder_type == "edge":
            self.summary_channels = 2 * args.hidden_channels
        else:
            raise ValueError

        if projector is None:
            projector = MLP(
                num_layers=args.num_decoder_body_layers,
                num_input=args.hidden_channels,
                num_hidden=args.hidden_channels,
                num_out=args.hidden_channels,
                activation=args.activation,
                use_bn=args.use_bn,
                dropout=args.dropout_channels,
                activate_last=True,
            )

        super().__init__(args.hidden_channels, encoder=projector, summary=None, corruption=None)

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
            mask_pos, mask_neg = mask[:B // 2], mask[B // 2:]
            loss = self.loss(pos_z=z_pos, neg_z=z_neg, summary=summarized,
                             is_batched=True, pos_mask=mask_pos, neg_mask=mask_neg)
        elif ptr_pos_and_neg is not None:
            summarized = summarized.squeeze()
            z_pos, z_neg = z_pos_and_neg[:ptr_pos_and_neg, :], z_pos_and_neg[ptr_pos_and_neg:, :]
            loss = self.loss(pos_z=z_pos, neg_z=z_neg, summary=summarized, is_batched=False)
        else:
            raise ValueError
        return loss

    def loss(self, pos_z, neg_z, summary,
             is_batched=False, pos_mask=None, neg_mask=None):
        r"""Computes the mutual information maximization objective.

        :param pos_z: [N, F_h] or [B, N_max, F_h]
        :param neg_z: [N, F_h] or [B, N_max, F_h]
        :param summary: [F_s] or [B, F_s]
        :param is_batched: bool
        :param pos_mask: [B, N_max]
        :param neg_mask: [B, N_max]
        """
        if not is_batched:
            pos_loss = -torch.log(self.discriminate(
                pos_z, summary, sigmoid=True) + EPS).mean()
            neg_loss = -torch.log(1 - self.discriminate(
                neg_z, summary, sigmoid=True) + EPS).mean()
        else:
            pos_loss = -torch.log(self.batched_discriminate(
                pos_z, summary, pos_mask, sigmoid=True) + EPS).mean()
            neg_loss = -torch.log(1 - self.batched_discriminate(
                neg_z, summary, neg_mask, sigmoid=True) + EPS).mean()
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
        return '{}({}, {}, projector={})'.format(
            self.__class__.__name__, self.hidden_channels, self.summary_channels,
            self.encoder,
        )


if __name__ == '__main__':

    MODE = "G2GContrastiveLoss"

    from arguments import get_args
    from pytorch_lightning import seed_everything

    seed_everything(32)

    if MODE == "G2LContrastiveLoss":

        _args = get_args("SGI", "FNTN", "TEST+MEMO")
        _isi = G2LContrastiveLoss(_args)
        print(_isi)
        print("----")
        for m in _isi.modules():
            print(m)
        print("----")
        for p in _isi.named_parameters():
            print(p[0])

    elif MODE == "G2GContrastiveLoss":

        _args = get_args("SGI", "FNTN", "TEST+MEMO")
        _isi = G2GContrastiveLoss(_args)
        print(_isi)

    elif MODE == "InfoNCELoss":

        _d = "cpu"
        _infonce = InfoNCELoss(0.2)
        _ag = torch.randn([3, 7]).float().to(_d)
        _sn = torch.randn([3, 7]).float().to(_d)
        print(_infonce(_ag, _sn, None))
