import hashlib
from collections import Counter
import time
import random
from itertools import tee, islice
from typing import List, Tuple, Optional

import networkx as nx
import numpy as np
from termcolor import cprint
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric
from torch import Tensor
from torch_geometric.utils import to_dense_batch, softmax, subgraph


# PyTorch/PyTorch Geometric related


def dropout_nodes(x, edge_index, edge_attr=None, p=0.5, training=True):
    if p < 0. or p > 1.:
        raise ValueError('Dropout probability has to be between 0 and 1, '
                         'but got {}'.format(p))
    if not training or p == 0.0:
        return x, edge_index, edge_attr

    N = x.size(0)
    DN = int(N * (1.0 - p))
    idx = torch.randperm(N)[:DN]

    x = x[idx]
    edge_index, edge_attr = subgraph(idx, edge_index, edge_attr, relabel_nodes=True, num_nodes=N)

    return x, edge_index, edge_attr



def softmax_half(src: Tensor, index: Tensor, num_nodes: Optional[int] = None) -> Tensor:
    r"""softmax that supports torch.half tensors.
        See torch_geometric.utils.softmax for more details."""
    is_half = (src.dtype == torch.half)
    src = src.float() if is_half else src
    smx = softmax(src, index, num_nodes=num_nodes)
    return smx.half() if is_half else smx


def to_multiple_dense_batches(
        x_list: List[Tensor],
        batch=None, fill_value=0, max_num_nodes=None
) -> Tuple[List[Tensor], Tensor]:
    cat_x = torch.cat(x_list, dim=-1)
    cat_out, mask = to_dense_batch(cat_x, batch, fill_value, max_num_nodes)
    # [B, N, L*F] -> [B, N, F] * L
    return torch.chunk(cat_out, len(x_list), dim=-1), mask


def to_directed(edge_index, edge_attr=None):
    if edge_attr is not None:
        raise NotImplementedError
    N = edge_index.max().item() + 1
    row, col = torch.sort(edge_index.t()).values.t()
    sorted_idx = torch.unique(row * N + col)
    row, col = sorted_idx // N, sorted_idx % N
    return torch.stack([row, col], dim=0).long()


def convert_node_labels_to_integers_customized_ordering(
        G, first_label=0, ordering="default", label_attribute=None
):
    if ordering == "keep":
        mapping = dict(zip(G.nodes(), [int(v) for v in G.nodes()]))
        H = nx.relabel_nodes(G, mapping)
        if label_attribute is not None:
            nx.set_node_attributes(H, {v: k for k, v in mapping.items()}, label_attribute)
        return H
    else:
        return nx.convert_node_labels_to_integers(G, first_label, ordering, label_attribute)


def from_networkx_customized_ordering(G, ordering="default"):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """
    G = convert_node_labels_to_integers_customized_ordering(G, ordering=ordering)
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    data = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            data[key] = [value] if i == 0 else data[key] + [value]

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[key] = [value] if i == 0 else data[key] + [value]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()
    return data


def to_one_hot(labels_integer_tensor: torch.Tensor, n_classes: int) -> np.ndarray:
    labels = labels_integer_tensor.cpu().numpy()
    return np.eye(n_classes)[labels]


def act(tensor, activation_name, **kwargs):
    if activation_name == "relu":
        return F.relu(tensor, **kwargs)
    elif activation_name == "elu":
        return F.elu(tensor, **kwargs)
    elif activation_name == "leaky_relu":
        return F.leaky_relu(tensor, **kwargs)
    elif activation_name == "sigmoid":
        return torch.sigmoid(tensor)
    elif activation_name == "tanh":
        return torch.tanh(tensor)
    else:
        raise ValueError(f"Wrong activation name: {activation_name}")


def get_extra_repr(model, important_args):
    return "\n".join(["{}={},".format(a, getattr(model, a)) for a in important_args
                      if a in model.__dict__])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EPSILON(object):

    def __init__(self):
        super().__init__()

    def __add__(self, other):
        if other.dtype == torch.float16:
            return other + 1e-7
        else:
            return other + 1e-15

    def __radd__(self, other):
        return self.__add__(other)


# Others


def sample_index_with_replacement_and_exclusion(max_index, num_to_sample, set_to_exclude=None):
    set_to_exclude = set_to_exclude or set()
    populations = []
    num_candidates = num_to_sample + len(set_to_exclude)
    while num_candidates > 0:
        num_to_sample_at_this_iter = min(num_candidates, max_index)
        pops = list(set(random.sample(range(max_index), num_to_sample_at_this_iter))
                    - set_to_exclude)
        populations += pops
        num_candidates -= len(pops)
    return populations[:num_to_sample]


def n_wise(iterable, n=2):
    # https://stackoverflow.com/a/21303303
    iters = tee(iterable, n)
    for i, it in enumerate(iters):
        next(islice(it, i, i), None)
    return zip(*iters)


def del_attrs(o, keys: List[str]):
    for k in keys:
        delattr(o, k)


def create_hash(o: dict):
    def preprocess(v):
        if isinstance(v, torch.Tensor):
            return v.shape
        else:
            return v

    sorted_keys = sorted(o.keys())
    strings = "/ ".join(["{}: {}".format(k, preprocess(o[k])) for k in sorted_keys])
    return hashlib.md5(strings.encode()).hexdigest()


def debug_with_exit(func):  # Decorator
    def wrapped(*args, **kwargs):
        print()
        cprint("===== DEBUG ON {}=====".format(func.__name__), "red", "on_yellow")
        func(*args, **kwargs)
        cprint("=====   END  =====", "red", "on_yellow")
        exit()

    return wrapped


def print_time(method):
    """From https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d"""

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        cprint('%r  %2.2f s' % (method.__name__, (te - ts)), "red")
        return result

    return timed


def cprint_arg_conditionally(condition_func=lambda args: True,
                             filter_func=lambda arg: True,
                             out_func=lambda arg: arg,
                             color="red"):
    def decorator(func):
        def wrapped(*args):
            if condition_func(args):
                for arg in args:
                    if filter_func(arg):
                        cprint(out_func(arg), color)
            return func(*args)

        return wrapped

    return decorator


def cprint_multi_lines(prefix, color, is_sorted=True, **kwargs):
    kwargs_items = sorted(kwargs.items()) if is_sorted else kwargs.items()
    for k, v in kwargs_items:
        cprint("{}{}: {}".format(prefix, k, v), color)


def merge_or_update(old_dict: dict, new_dict: dict):
    for k, v_new in new_dict.items():
        if k in old_dict:
            v_old = old_dict[k]
            if type(v_new) == list and type(v_old) == list:
                old_dict[k] = v_old + v_new
            else:
                old_dict[k] = v_new
    return old_dict


if __name__ == '__main__':

    from pytorch_lightning import seed_everything
    seed_everything(42)

    MODE = "softmax_half"

    if MODE == "from_networkx_customized_ordering":
        nxg = nx.Graph()
        nxg.add_edges_from([(0, 1), (0, 2), (0, 5)])
        print(nxg.edges)
        pgg = from_networkx_customized_ordering(nxg, ordering="keep")
        print(pgg.edge_index)
        pgg = from_networkx_customized_ordering(nxg, ordering="default")
        print(pgg.edge_index)

    elif MODE == "softmax_half":
        _num_nodes = 3
        _src = torch.randn((6, 1))
        _index = torch.Tensor([0, 1, 1, 2, 2, 2]).long()
        print(softmax(_src, _index, num_nodes=_num_nodes).squeeze())
        print(softmax_half(_src.half(), _index, num_nodes=_num_nodes).squeeze())

    elif MODE == "to_multiple_dense_batches":
        _x_list = [torch.randn((13, 7)) for _ in range(3)]
        _batch = torch.zeros((13,)).long()
        _batch[5:] = 1

        _b_list, _mask = to_multiple_dense_batches(_x_list, _batch)
        for _i, _b in enumerate(_b_list):
            print(_i, _b.size())
        print("mask", _mask.size())

    elif MODE == "to_undirected":
        from torch_geometric.utils import to_undirected

        _ei = torch.randint(0, 7, [2, 5])
        print(_ei)
        _uei = to_undirected(_ei)
        print(_uei)
        print(to_directed(_uei))
