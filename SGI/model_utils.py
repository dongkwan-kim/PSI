import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from torch_geometric.nn import GlobalAttention, global_mean_pool
from torch_scatter import scatter_add

from utils import softmax_half


class MyLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)

    def forward(self, x, *args, **kwargs):
        return super(MyLinear, self).forward(x)


class PositionalEncoding(nn.Module):

    def __init__(self, max_len, num_channels, dropout=0.0):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.num_channels = num_channels
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, num_channels)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_channels, 2).float() * (-math.log(10000.0) / num_channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1).squeeze()
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [N, F] or [B, N_max, F]
        x = x + self.pe[:x.size(-2), :]
        return self.dropout(x)

    def __repr__(self):
        return "{}(max_len={}, num_channels={}, dropout={})".format(
            self.__class__.__name__, self.max_len, self.num_channels, self.dropout.p,
        )


class BilinearWith1d(nn.Bilinear):

    def __init__(self, in1_features, in2_features, out_features, bias=True):
        super().__init__(in1_features, in2_features, out_features, bias)
        # weight: [o, i1, i2], bias: [o,]

    def forward(self, x1, x2):

        x1_dim = x1.squeeze().dim()

        if x1_dim == 1:  # single-batch
            # x1 [F,] * weight [O, F, S] * x2 [N, S] -> [N, O]
            x1, x2 = x1.squeeze(), x2.squeeze()
            x = torch.einsum("f,ofs,ns->no", x1, self.weight, x2)

        elif x1_dim == 2:  # multi-batch
            # x1 [B, F] * weight [O, F, S] * x2 [B, N, S] -> [B, N, O]
            x = torch.einsum("bf,ofs,bns->bno", x1, self.weight, x2)

        else:
            raise ValueError("Wrong x1 shape: {}".format(x1.size()))

        if self.bias is not None:
            x += self.bias
        return x


class MLP(nn.Module):

    def __init__(self, num_layers, num_input, num_hidden, num_out, activation,
                 use_bn=False, dropout=0.0, activate_last=False):
        super().__init__()
        self.num_layers, self.num_input, self.num_hidden, self.num_out = num_layers, num_input, num_hidden, num_out
        self.activation, self.use_bn, self.dropout = activation, use_bn, dropout
        self.activate_last = activate_last
        layers = nn.ModuleList()

        for i in range(num_layers - 1):
            if i == 0:
                layers.append(nn.Linear(num_input, num_hidden))
            else:
                layers.append(nn.Linear(num_hidden, num_hidden))
            if use_bn:
                layers.append(nn.BatchNorm1d(num_hidden))
            layers.append(Act(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

        if num_layers != 1:
            layers.append(nn.Linear(num_hidden, num_out))
        else:  # single-layer
            layers.append(nn.Linear(num_input, num_out))

        if self.activate_last:
            if use_bn:
                layers.append(nn.BatchNorm1d(num_hidden))
            layers.append(Act(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

    def __repr__(self):
        if self.num_layers > 1:
            return "{}(L={}, I={}, H={}, O={}, act={}, bn={}, do={})".format(
                self.__class__.__name__, self.num_layers, self.num_input, self.num_hidden, self.num_out,
                self.activation, self.use_bn, self.dropout,
            )
        else:
            return "{}(L={}, I={}, O={}, act={}, bn={}, do={})".format(
                self.__class__.__name__, self.num_layers, self.num_input, self.num_out,
                self.activation, self.use_bn, self.dropout,
            )

    def layer_repr(self):
        """
        :return: e.g., '64->64'
        """
        hidden_layers = [self.num_hidden] * (self.num_layers - 1) if self.num_layers >= 2 else []
        layers = [self.num_input] + hidden_layers + [self.num_out]
        return "->".join(str(l) for l in layers)


class Act(nn.Module):

    def __init__(self, activation_name):
        super().__init__()
        if activation_name == "relu":
            self.a = nn.ReLU()
        elif activation_name == "elu":
            self.a = nn.ELU()
        elif activation_name == "leaky_relu":
            self.a = nn.LeakyReLU()
        elif activation_name == "sigmoid":
            self.a = nn.Sigmoid()
        elif activation_name == "tanh":
            self.a = nn.Tanh()
        else:
            raise ValueError(f"Wrong activation name: {activation_name}")

    def forward(self, tensor):
        return self.a(tensor)

    def __repr__(self):
        return self.a.__repr__()


class GlobalAttentionHalf(GlobalAttention):
    r"""GlobalAttention that supports torch.half tensors.
        See torch_geometric.nn.GlobalAttention for more details."""

    def __init__(self, gate_nn, nn=None):
        super(GlobalAttentionHalf, self).__init__(gate_nn, nn)

    def forward(self, x, batch, size=None):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax_half(gate, batch, num_nodes=size)  # A substitute for softmax
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        return out


class GlobalMeanPool(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch, size=None):
        return global_mean_pool(x, batch, size)


if __name__ == '__main__':

    MODE = "BILINEAR"
    if MODE == "BILINEAR":
        _bilinear = BilinearWith1d(in1_features=3, in2_features=6, out_features=7)
        _x1 = torch.randn((1, 3))
        _x2 = torch.randn((23, 6))
        print(_bilinear(_x1, _x2).size())  # [23, 7]
        _x1 = torch.randn((5, 3))
        _x2 = torch.randn((5, 23, 6))
        print(_bilinear(_x1, _x2).size())  # [5, 23, 7]
