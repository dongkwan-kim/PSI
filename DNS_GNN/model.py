from pprint import pprint
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_decoder import DNSDecoder
from model_embedding import DNSEmbedding
from model_encoder import DNSEncoder
from model_utils import Act


class DNSNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.emb = DNSEmbedding(args)
        self.enc = DNSEncoder(args)
        self.after_enc = self.build_after_enc()
        self.dec = DNSDecoder(args)

    def build_after_enc(self):
        return nn.Sequential(
            nn.BatchNorm1d(self.args.hidden_channels),
            Act(self.args.activation),
            nn.Dropout(p=self.args.dropout_channels),
        )

    def pprint(self):
        pprint(next(self.modules()))

    def forward(self, x_idx, obs_x_idx, edge_index_01, edge_index_2=None):
        x = self.emb(x_idx)
        x = self.enc(x, edge_index_01)
        x = self.after_enc(x)
        logits_g, dec_x, dec_e = self.dec(x, obs_x_idx, edge_index_01, edge_index_2)
        return logits_g, dec_x, dec_e


if __name__ == '__main__':
    from arguments import get_args
    from utils import count_parameters

    _args = get_args("DNS", "FNTN", "TEST+MEMO")
    _args.num_nodes_global = 7
    net = DNSNet(_args)
    print(net)
    print(f"#: {count_parameters(net)}")

    _xi = torch.arange(7)
    _ei = torch.randint(0, 7, [2, 17])
    _ei2 = torch.randint(0, 7, [2, 13])
    _obs_x_idx = torch.arange(3).long()

    print(net(_xi, _obs_x_idx, _ei, _ei2, None, None))
