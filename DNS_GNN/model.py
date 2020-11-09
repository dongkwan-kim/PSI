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
from model_readout import Readout


class DNSNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.emb = DNSEmbedding(args)

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

    def pprint(self):
        pprint(next(self.modules()))

    def forward(self, x_idx, obs_x_idx, edge_index_01, edge_index_2=None, pergraph_attr=None):
        x = self.emb(x_idx)
        x = self.enc(x, edge_index_01)
        if self.args.use_decoder:
            logits_g, dec_x, dec_e = self.dec_or_readout(x, obs_x_idx, edge_index_01, edge_index_2, pergraph_attr)
            return logits_g, dec_x, dec_e
        else:
            return self.dec_or_readout(x, pergraph_attr)


if __name__ == '__main__':
    from arguments import get_args
    from utils import count_parameters

    _args = get_args("DNS", "FNTN", "TEST+MEMO")
    _args.num_nodes_global = 7
    _args.is_bidirectional = True
    _args.use_decoder = True
    _args.use_pergraph_attr = True
    _args.readout_name = "mean-max"
    net = DNSNet(_args)
    print(net)
    print(f"#: {count_parameters(net)}")

    _xi = torch.arange(7)
    _ei = torch.randint(0, 7, [2, 17])
    _ei2 = torch.randint(0, 7, [2, 13])
    _obs_x_idx = torch.arange(3).long()
    _pga = torch.arange(_args.pergraph_channels) * 0.1

    print(net(_xi, _obs_x_idx, _ei, _ei2, _pga))
