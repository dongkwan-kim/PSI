import hashlib
from collections import Counter

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric
from termcolor import cprint


# PyTorch/PyTorch Geometric related


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
    nxg = nx.Graph()
    nxg.add_edges_from([(0, 1), (0, 2), (0, 5)])
    print(nxg.edges)
    pgg = from_networkx_customized_ordering(nxg, ordering="keep")
    print(pgg.edge_index)
    pgg = from_networkx_customized_ordering(nxg, ordering="default")
    print(pgg.edge_index)
