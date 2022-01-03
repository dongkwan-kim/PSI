import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, NamedTuple, List, Union

from torch_geometric.data import Batch, Data
from torch_geometric.utils import dropout_adj

from augmentor_functional import add_edge, dropout_feature, drop_feature, drop_node, permute, random_walk_subgraph, \
    compute_ppr

"""
Most codes are adopted from
    - https://github.com/GraphCL/PyGCL/tree/main/GCL/augmentors
"""


class Graph(NamedTuple):
    x: torch.FloatTensor
    edge_index: torch.LongTensor
    edge_weights: Optional[torch.FloatTensor]

    def unfold(self) -> Tuple[torch.FloatTensor, torch.LongTensor, Optional[torch.FloatTensor]]:
        return self.x, self.edge_index, self.edge_weights


class Augmentor(ABC):
    """Base class for graph augmentors."""
    def __init__(self):
        pass

    @abstractmethod
    def augment(self, g: Graph) -> Graph:
        raise NotImplementedError(f"GraphAug.augment should be implemented.")

    def __call__(
            self, x: torch.FloatTensor,
            edge_index: torch.LongTensor, edge_weight: Optional[torch.FloatTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.augment(Graph(x, edge_index, edge_weight)).unfold()


class Compose(Augmentor):
    def __init__(self, augmentors: List[Augmentor]):
        super(Compose, self).__init__()
        self.augmentors = augmentors

    def augment(self, g: Graph) -> Graph:
        for aug in self.augmentors:
            g = aug.augment(g)
        return g


class RandomChoice(Augmentor):
    def __init__(self, augmentors: List[Augmentor], num_choices: int):
        super(RandomChoice, self).__init__()
        assert num_choices <= len(augmentors)
        self.augmentors = augmentors
        self.num_choices = num_choices

    def augment(self, g: Graph) -> Graph:
        num_augmentors = len(self.augmentors)
        perm = torch.randperm(num_augmentors)
        idx = perm[:self.num_choices]
        for i in idx:
            aug = self.augmentors[i]
            g = aug.augment(g)
        return g


class Identity(Augmentor):
    def __init__(self):
        super(Identity, self).__init__()

    def augment(self, g: Graph) -> Graph:
        return g


class EdgeAdding(Augmentor):
    def __init__(self, pe: float):
        super(EdgeAdding, self).__init__()
        self.pe = pe

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        edge_index = add_edge(edge_index, ratio=self.pe)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)


class EdgeRemoving(Augmentor):
    def __init__(self, pe: float):
        super(EdgeRemoving, self).__init__()
        self.pe = pe

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        edge_index, edge_weights = dropout_adj(edge_index, edge_attr=edge_weights, p=self.pe)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)


class FeatureDropout(Augmentor):
    def __init__(self, pf: float):
        super(FeatureDropout, self).__init__()
        self.pf = pf

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        x = dropout_feature(x, self.pf)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)


class FeatureMasking(Augmentor):
    def __init__(self, pf: float):
        super(FeatureMasking, self).__init__()
        self.pf = pf

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        x = drop_feature(x, self.pf)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)


class NodeDropping(Augmentor):
    def __init__(self, pn: float):
        super(NodeDropping, self).__init__()
        self.pn = pn

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        edge_index, edge_weights = drop_node(edge_index, edge_weights, keep_prob=1. - self.pn)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)


class NodeShuffling(Augmentor):
    def __init__(self):
        super(NodeShuffling, self).__init__()

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        x = permute(x)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)


class RWSampling(Augmentor):
    def __init__(self, num_seeds: int, walk_length: int):
        super(RWSampling, self).__init__()
        self.num_seeds = num_seeds
        self.walk_length = walk_length

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        edge_index, edge_weights = random_walk_subgraph(
            edge_index, edge_weights, batch_size=self.num_seeds, length=self.walk_length)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)


class PPRDiffusion(Augmentor):
    def __init__(self, alpha: float = 0.2, eps: float = 1e-4,
                 use_cache: bool = False, add_self_loop: bool = True):
        super(PPRDiffusion, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self._cache = None
        self.use_cache = use_cache
        self.add_self_loop = add_self_loop

    def augment(self, g: Graph) -> Graph:
        if self._cache is not None and self.use_cache:
            return self._cache
        x, edge_index, edge_weights = g.unfold()
        edge_index, edge_weights = compute_ppr(
            edge_index, edge_weights,
            alpha=self.alpha, eps=self.eps, ignore_edge_attr=False, add_self_loop=self.add_self_loop
        )
        res = Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
        self._cache = res
        return res


class PyGAugmentor:

    """PyG augmentor interfaces for PyGCL."""

    def __init__(self, *augmentors: Union[Augmentor, str]):
        if isinstance(augmentors[0], str):
            self.run_aug = Compose(list(eval(a) for a in augmentors))
        else:
            self.run_aug = Compose(list(augmentors))

    def __call__(self, data: Union[Data, Batch] = None,
                 x=None, edge_index=None, edge_weight=None, edge_attr=None) -> Union[Data, Batch, Tuple]:
        if data is not None:
            edge_feature, ef_name = None, None
            if hasattr(data, "edge_attr"):
                edge_feature, ef_name = data.edge_attr, "edge_attr"
            elif hasattr(data, "edge_weight"):
                edge_feature, ef_name = data.edge_weight, "edge_weight"
            augmented = self.run_aug(data.x, data.edge_index, edge_feature)
            if edge_feature is None:
                data.x, data.edge_index, _ = augmented
            else:
                data.x, data.edge_index, edge_feature = augmented
                setattr(data, ef_name, edge_feature)
            return data
        elif x is not None:
            use_ef = (edge_weight or edge_attr) is not None
            out = self.run_aug(x, edge_index, edge_weight or edge_attr)
            if use_ef:
                return out
            else:
                return out[0], out[1]

    def __repr__(self):
        repr_list = []
        for a in self.run_aug.augmentors:
            values = ", ".join(str(v) for v in a.__dict__.values())
            repr_list.append(f"{a.__class__.__name__}({values})")
        return "Aug({})".format(", ".join(repr_list))


if __name__ == '__main__':
    from pytorch_lightning import seed_everything

    seed_everything(41)

    _x = torch.randn((4, 5)).float()
    _edge_index = torch.Tensor(
        [[0, 0, 1, 1, 2, 3],
         [1, 2, 3, 0, 1, 2]]
    ).long()
    _d = Data(_x, _edge_index)

    print(_d)
    print("X", _x)
    print("E", _edge_index)

    print("- " * 7)
    pga = PyGAugmentor("EdgeAdding(0.5)", "PPRDiffusion(alpha=0.2)")
    print(pga)
    _o = pga(_d.clone())
    print(_o)
    print("X", _o.x)
    print("E", _o.edge_index)

    print("- " * 7)
    pga = PyGAugmentor(EdgeRemoving(0.5), FeatureDropout(0.25))
    print(pga)
    _o = pga(_d.clone())
    print(_o)
    print("X", _o.x)
    print("E", _o.edge_index)

    print("- " * 7)
    pga = PyGAugmentor("EdgeRemoving(0.0)", "FeatureDropout(0.25)")
    print(pga)
    _o = pga(x=_d.x, edge_index=_d.edge_index)
    print(_o)
    print("X", _o[0])
    print("E", _o[1])
    print(_d.x)
    print(_d.edge_index)


