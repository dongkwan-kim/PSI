from copy import deepcopy
from pprint import pprint
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from termcolor import cprint

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

    def forward(self, x_idx, obs_x_index, edge_index_01,
                edge_index_2=None, pergraph_attr=None, batch=None,
                x_idx_isi=None, edge_index_isi=None, batch_isi=None, ptr_isi=None):
        x = self.emb(x_idx)
        x = self.enc(x, edge_index_01)

        dec_x, dec_e, loss_isi = None, None, None

        if self.args.use_decoder:  # DNSDecoder
            z_g, logits_g, dec_x, dec_e = self.dec_or_readout(
                x, obs_x_index, edge_index_01, edge_index_2,
                pergraph_attr, batch,
            )
        else:  # Readout
            z_g, logits_g = self.dec_or_readout(x, pergraph_attr, batch)

        if self.args.use_inter_subgraph_infomax and self.training:  # only for training
            assert x_idx_isi is not None
            x_isi = self.emb(x_idx_isi)
            x_isi = self.enc(x_isi, edge_index_isi)
            loss_isi = self.isi_loss(
                summarized=z_g, x_pos_and_neg=x_isi,
                batch_pos_and_neg=batch_isi, ptr_pos_and_neg=ptr_isi,
            )

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
    _args.main_decoder_type = "node"
    _args.is_bidirectional = True
    _args.use_decoder = True
    _args.use_node_decoder = True
    _args.use_edge_decoder = False
    _args.use_inter_subgraph_infomax = True
    _args.use_pergraph_attr = True
    _args.readout_name = "mean-max"
    net = DNSNet(_args)
    print(net)
    print(f"#: {count_parameters(net)}")

    _N = _args.num_nodes_global
    _xi = torch.arange(_N)
    _ei = torch.randint(0, _N, [2, 17])
    _ei2 = torch.randint(0, _N, [2, 13])
    _obs_x_index = torch.Tensor([0, 1, 6, 7, 8]).long()
    _pga = torch.arange(_args.pergraph_channels) * 0.1
    _batch = torch.zeros(_N).long()
    _batch[5:] = 1


    def _print(_rvs, _names=["logits_g", "dec_x", "dec_e", "loss_isi"]):
        if isinstance(_rvs, tuple):
            for _name, _oi in zip(_names, _rvs):
                if _oi is None:
                    pass
                elif len(_oi.size()) == 0:
                    print("\t -", _name, "(loss)", _oi)
                else:
                    print("\t -", _name, _oi.size())


    """Data example
    
    Batch(batch=[9991], batch_pos_and_neg=[52], edge_index_01=[2, 7389503], edge_index_pos_and_neg=[2, 458],
          labels_x=[46], mask_x_index=[46], obs_x_index=[9], x=[9991], x_pos_and_neg=[52], y=[2, 10])

    DataPN(edge_index_01=[2, 3844995], edge_index_pos_and_neg=[2, 252], labels_x=[28], mask_x_index=[28],
           obs_x_index=[5], ptr_pos_and_neg=[1], x=[5391], x_pos_and_neg=[28], y=[1, 10])
    """

    _xi_isi = torch.arange(_N)
    _ei_isi = torch.randint(0, _N, [2, 37])
    _batch_isi = torch.zeros(_N).long()
    _batch_isi[4:] = 1
    _batch_isi[9:] = 2
    _batch_isi[16:] = 3

    cprint(f"w/ batch ({_batch.max().item() + 1})", "red")
    seed_everything(42)
    _rvs = net(_xi, _obs_x_index, _ei,
               edge_index_2=None, pergraph_attr=_pga, batch=_batch,
               x_idx_isi=_xi_isi, edge_index_isi=_ei_isi, batch_isi=_batch_isi, ptr_isi=None)
    _print(_rvs)

    _ptr_isi = torch.Tensor([10]).long()
    _batch_isi = torch.zeros(_N).long()
    _batch_isi[_ptr_isi:] = 1

    cprint("w/ ptr_isi", "red")
    seed_everything(42)
    _rvs = net(_xi, _obs_x_index, _ei,
               edge_index_2=None, pergraph_attr=_pga, batch=None,
               x_idx_isi=_xi_isi, edge_index_isi=_ei_isi, batch_isi=None, ptr_isi=_ptr_isi)
    _print(_rvs)

    _batch[:] = 0
    cprint(f"w/ batch ({_batch.max().item() + 1})", "red")
    seed_everything(42)
    _rvs = net(_xi, _obs_x_index, _ei,
               edge_index_2=None, pergraph_attr=_pga, batch=_batch,
               x_idx_isi=_xi_isi, edge_index_isi=_ei_isi, batch_isi=_batch_isi, ptr_isi=None)
    _print(_rvs)

