from typing import Tuple

import torch
from torch_geometric.utils import k_hop_subgraph
from torch import Tensor

from data_sampler import get_observed_nodes_and_edges, create_khop_edge_attr


def decode_cke(khop_edge_idx, sparse_khop_edge_attr) -> Tuple[Tensor, Tensor]:
    """:return: khop_edge_index, khop_edge_attr"""


class CompressedKhopEdge(object):

    def __init__(self, num_hops, global_edge_index=None):
        self.num_hops = num_hops
        self.global_edge_index = global_edge_index
        self._N = None

    @property
    def N(self):
        if self._N is None:
            self._N = self.global_edge_index.max() + 1
        return self._N

    def __call__(self, data):
        """
        Add khop_edge_idx: torch.FloatTensor the size of which is [E_khp]
        Add sparse_khop_edge_attr: torch.sparse.FloatTensor

        :param data: Data class
            e.g., Data(edge_index=[2, 402], obs_x=[7], x=[23, 1], y=[1])
        :return: Data with khop_edge_idx, sparse_khop_edge_attr
        """
        assert hasattr(data, "edge_index")
        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)

        observed_nodes, observed_edge_index = get_observed_nodes_and_edges(
            data=data, obs_x_range=None,
        )

        khop_nodes, khop_edge_index, mapping, _ = k_hop_subgraph(
            node_idx=observed_nodes,
            num_hops=self.num_hops,
            edge_index=self.global_edge_index,  # self.G.edge_index
            relabel_nodes=False,  # Relabel later.
            num_nodes=self.N,
            flow="target_to_source",  # Important.
        )

        khop_edge_attr, is_khp_in_sub_edge = create_khop_edge_attr(
            khop_edge_index=khop_edge_index, edge_index=edge_index,
            edge_attr=edge_attr, N=self.N,
        )

        # Relabeling nodes
        _node_idx = observed_nodes.new_full((self.N,), -1)
        _node_idx[khop_nodes] = torch.arange(khop_nodes.size(0))
        khop_edge_index = _node_idx[khop_edge_index]

        # todo khop_edge_index -> khop_edge_idx
        # todo khop_edge_attr -> sparse_khop_edge_attr

    def __repr__(self):
        return '{}(k={})'.format(self.__class__.__name__, self.num_hops)


class CompleteSubgraph(object):

    def __init__(self, global_edge_index=None):
        self.global_edge_index = global_edge_index
        self._N = None

    @property
    def N(self):
        if self._N is None:
            self._N = self.global_edge_index.max() + 1
        return self._N

    def __call__(self, data):
        assert self.global_edge_index is not None
        x_cs, edge_index_cs, _, _ = k_hop_subgraph(
            node_idx=data.x.flatten(),
            num_hops=0,
            edge_index=self.global_edge_index,
            relabel_nodes=False,
            num_nodes=self.N,
            flow="target_to_source"
        )
        data.x_cs = x_cs.view(data.x.size(0), -1)
        data.edge_index_cs = edge_index_cs
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


if __name__ == '__main__':
    from torch_geometric.utils import add_self_loops
    from torch_geometric.data import Data
    from pytorch_lightning import  seed_everything

    seed_everything(23)

    _n = 7
    cke = CompressedKhopEdge(num_hops=1)
    _ge = torch.randint(_n, (2, 15))
    _ge, _ = add_self_loops(_ge)
    cke.global_edge_index = _ge

    # Data(edge_index=[2, 402], obs_x=[7], x=[23, 1], y=[1])
    _e_in_g = _ge[:, 5:10]
    _x = torch.unique(_e_in_g.flatten())
    _obs_x = torch.randperm(_x.size(0))[:3]

    _ni = torch.full((_n,), -1, dtype=torch.long)
    _ni[_x] = torch.arange(_x.size(0))
    _e = _ni[_e_in_g]
    _d = Data(edge_index=_e, obs_x=_obs_x, x=_x)
    print("---- Data ----")
    print(_d)
    print("x", _x)
    print("eg", _e_in_g)
    print("e", _e)
    print("obs_x", _obs_x)
    print("--------------")

    print(cke)
    cke(_d)
