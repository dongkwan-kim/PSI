import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

    def __repr__(self):
        return "{}(max_len={}, num_channels={}, dropout={})".format(
            self.__class__.__name__, self.max_len, self.num_channels, self.dropout.p,
        )


class AttentionPool(nn.Module):

    def __init__(self, num_channels):
        super(AttentionPool, self).__init__()
        self.att = nn.Linear(num_channels, 1, bias=False)
        self.num_channels = num_channels

    def forward(self, x):
        att_val = self.att(x)  # [N, F] -> [N, 1]
        w_sum = torch.einsum("nf,nu->f", x, torch.softmax(att_val, dim=0))  # [F] i.e., \sum a * x
        return w_sum.view(1, -1)

    def __repr__(self):
        return "{}(F={})".format(self.__class__.__name__, self.num_channels)


class BilinearWith1d(nn.Bilinear):

    def __init__(self, in1_features, in2_features, out_features, bias=True):
        super().__init__(in1_features, in2_features, out_features, bias)
        # weight = (o, i1, i2)

    def forward(self, x1, x2):
        # x1 [1, F] * weight [O, F, S] * x2 [N, S] -> [N, O]
        assert len(x1.squeeze().size()) == 1
        x1 = x1.view(1, -1)
        x = torch.einsum("uf,ofs->os", x1, self.weight)
        x = torch.einsum("ns,os->no", x2, x)
        if self.bias is not None:
            x += self.bias
        return x


class MultiLinear(nn.Module):

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


if __name__ == '__main__':

    MODE = "POOL"

    if MODE == "POOL":
        _att_pool = AttentionPool(3)
        _x = torch.randn((7, 3))
        print(_att_pool)
        print(_att_pool(_x).size())
    elif MODE == "BILINEAR":
        _bilinear = BilinearWith1d(3, 6, 7)
        _x1 = torch.randn((1, 3))
        _x2 = torch.randn((23, 6))
        print(_bilinear(_x1, _x2).size())
