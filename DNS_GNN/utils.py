import hashlib
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from termcolor import cprint


# Torch related

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
