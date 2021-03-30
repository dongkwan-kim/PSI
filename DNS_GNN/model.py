from copy import deepcopy
from pprint import pprint
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_bidirectional import BiConv
from model_decoder import DNSDecoder
from model_embedding import DNSEmbedding
from model_encoder import DNSEncoder
from model_isi import InterSubgraphInfoMaxLoss
from model_readout import Readout


class DNSNet(nn.Module):

    def __init__(self, args, embedding=None):
        super().__init__()
        self.args = args
        self.emb = DNSEmbedding(args, embedding)

        if not self.args.is_bidirectional:
            self.enc = DNSEncoder(args, activate_last=True)
        else:
            assert args.hidden_channels % 2 == 0
            args_with_half = deepcopy(args)
            args_with_half.hidden_channels //= 2
            self.enc = BiConv(DNSEncoder(args_with_half, activate_last=True))

        if self.args.use_decoder:
            self.dec_or_readout = DNSDecoder(args)
        else:
            self.dec_or_readout = Readout(args)

        if self.args.use_inter_subgraph_infomax:
            self.isi_loss = InterSubgraphInfoMaxLoss(args)
        else:
            self.isi_loss = None

    def pprint(self):
        pprint(next(self.modules()))

    def forward(self, x_idx, obs_x_index, edge_index_01, edge_index_2=None, pergraph_attr=None,
                x_idx_isi=None, edge_index_isi=None, ptr_isi=None):
        x = self.emb(x_idx)
        x = self.enc(x, edge_index_01)

        dec_x, dec_e, loss_isi = None, None, None

        if self.args.use_decoder:
            z_g, logits_g, dec_x, dec_e = self.dec_or_readout(
                x, obs_x_index, edge_index_01, edge_index_2, pergraph_attr,
            )
        else:
            z_g, logits_g = self.dec_or_readout(x, pergraph_attr)

        if self.args.use_inter_subgraph_infomax and self.training:  # only for training
            assert x_idx_isi is not None
            x_isi = self.emb(x_idx_isi)
            x_isi = self.enc(x_isi, edge_index_isi)
            x_pos, x_neg = x_isi[:ptr_isi, :], x_isi[ptr_isi:, :]
            loss_isi = self.isi_loss(summarized=z_g, x_pos=x_pos, x_neg=x_neg)

        # Returns which are not None:
        #   use_decoder & use_inter_subgraph_infomax: (logits_g, dec_x, dec_e, loss_isi)
        #   use_inter_subgraph_infomax: (logits_g, loss_isi)
        #   use_decoder: (logits_g, dec_x, dec_e)
        #   else: logits_g
        return logits_g, dec_x, dec_e, loss_isi


if __name__ == '__main__':
    from arguments import get_args
    from utils import count_parameters

    _args = get_args("DNS", "FNTN", "TEST+MEMO")
    _args.num_nodes_global = 19
    _args.main_decoder_type = "edge"
    _args.is_bidirectional = True
    _args.use_decoder = True
    _args.use_inter_subgraph_infomax = True
    _args.use_pergraph_attr = False
    _args.readout_name = "mean-max"
    net = DNSNet(_args)
    print(net)
    print(f"#: {count_parameters(net)}")

    _xi = torch.arange(7)
    _ei = torch.randint(0, 7, [2, 17])
    _ei2 = torch.randint(0, 7, [2, 13])
    _obs_x_index = torch.arange(3).long()
    _pga = torch.arange(_args.pergraph_channels) * 0.1

    _xi_isi = torch.arange(19)
    _ei_isi = torch.randint(0, 19, [2, 37])
    _ptr_isi = torch.Tensor([10]).long()

    def _print():
        if isinstance(_rvs, tuple):
            for _i, _oi in enumerate(_rvs):
                if _oi is None:
                    print(_i, None)
                elif len(_oi.size()) == 0:
                    print(_i, "(loss)", _oi)
                else:
                    print(_i, _oi.size())

    print("On 1st iter")
    _rvs = net(_xi, _obs_x_index, _ei, _ei2, _pga, _xi_isi, _ei_isi, _ptr_isi)
    _print()

    print("On 2nd iter")
    _rvs = net(_xi, _obs_x_index, _ei, _ei2, _pga, _xi_isi, _ei_isi, _ptr_isi)
    _print()
