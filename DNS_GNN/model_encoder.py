from pprint import pprint
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import *

from model_utils import Act, MyLinear
from utils import act


def get_gnn_conv_and_kwargs(gnn_name, args):
    gkw = {}
    if gnn_name == "GCNConv":
        gnn_cls = GCNConv
    elif gnn_name == "SAGEConv":
        gnn_cls = SAGEConv
    elif gnn_name == "GATConv":
        gnn_cls = GATConv
    elif gnn_name == "Linear":
        gnn_cls = MyLinear
    else:
        raise ValueError(f"Wrong gnn conv name: {gnn_name}")
    return gnn_cls, gkw


class DNSEncoder(nn.Module):

    def __init__(self, args, activate_last=True):
        super().__init__()

        self.args = args
        self.activate_last = activate_last
        self.num_layers = self.args.num_encoder_layers

        self.convs = torch.nn.ModuleList()
        self.use_bn = args.use_bn
        self.bns = torch.nn.ModuleList() if self.use_bn else []
        self.build()

    def build(self):
        gnn, gkw = get_gnn_conv_and_kwargs(self.args.gnn_name, self.args)
        for conv_id in range(self.num_layers):
            if conv_id == 0:  # first
                in_channels = self.args.global_channels
            else:
                in_channels = self.args.hidden_channels
            self.convs.append(gnn(in_channels, self.args.hidden_channels, **gkw))
            if self.use_bn and (conv_id != self.num_layers - 1 or self.activate_last):
                self.bns.append(nn.BatchNorm1d(self.args.hidden_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, **kwargs):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, **kwargs)
            if i != self.num_layers - 1 or self.activate_last:
                if self.use_bn:
                    x = self.bns[i](x)
                x = act(x, self.args.activation)
                x = F.dropout(x, p=self.args.dropout_channels, training=self.training)
        return x

    def __repr__(self):
        return "{}(conv={}, L={}, I={}, H={}, O={}, act={}, act_last={}, bn={})".format(
            self.__class__.__name__, self.args.gnn_name, self.num_layers,
            self.args.global_channels, self.args.hidden_channels, self.args.hidden_channels,
            self.args.activation, self.activate_last, self.use_bn,
        )


if __name__ == '__main__':
    from arguments import get_args
    _args = get_args("DNS", "FNTN", "TEST+MEMO")
    enc = DNSEncoder(_args)
    print(enc)

    _x = torch.ones(10 * _args.global_channels).view(10, -1)
    _ei = torch.randint(0, 10, [2, 10])
    print(enc(_x, _ei).size())
