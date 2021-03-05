import torch
from torch_geometric.transforms import Compose
from torch_geometric.utils import k_hop_subgraph

from data_sampler import get_observed_nodes_and_edges, create_khop_edge_attr


class TransformForDNS(object):

    def __init__(self, global_edge_index=None):
        self.global_edge_index = global_edge_index
        self._N = None

    @property
    def N(self):
        if self._N is None:
            self._N = self.global_edge_index.max() + 1
        return self._N

    @classmethod
    def has_instance(cls, transform):
        if isinstance(transform, cls):
            return True
        elif isinstance(transform, Compose):
            for t in transform.transforms:
                if isinstance(t, cls):
                    return True
        else:
            return False

    @classmethod
    def set_global_edge_index(cls, transform, global_edge_index):
        if isinstance(transform, cls):
            transform.global_edge_index = global_edge_index
        elif isinstance(transform, Compose):
            for t in transform.transforms:
                if isinstance(t, cls):
                    t.global_edge_index = global_edge_index
        else:
            raise AttributeError


class CompressedKhopEdge(TransformForDNS):

    def __init__(self, num_hops, global_edge_index=None):
        self.num_hops = num_hops
        super(CompressedKhopEdge, self).__init__(global_edge_index=global_edge_index)

    def __call__(self, data):
        """
        Add khop_edge_idx: torch.FloatTensor the size of which is [E_khp]
        Add sparse_khop_edge_attr: torch.sparse.FloatTensor

        :param data: Data class
            e.g., Data(edge_index=[2, 402], obs_x=[7], x=[23, 1], y=[1])
        :return: Data with khop_edge_idx,
                           sparse_khop_edge_attr_indices,
                           sparse_khop_edge_attr_values,
                           sparse_khop_edge_attr_size,
        """
        assert hasattr(data, "edge_index")
        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)

        observed_nodes, observed_edge_index = get_observed_nodes_and_edges(
            data=data, obs_x_range=None,
        )

        khop_nodes, khop_edge_index, obs_node_index, _ = k_hop_subgraph(
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
        _node_idx = torch.full((self.N,), fill_value=-1, dtype=torch.long)
        _node_idx[khop_nodes] = torch.arange(khop_nodes.size(0))
        khop_edge_index = _node_idx[khop_edge_index]

        # khop_edge_index -> khop_edge_idx (x2 compression)
        data.khop_edge_idx = khop_edge_index[0] * self.N + khop_edge_index[1]

        # khop_edge_attr -> sparse_khop_edge_attr
        #   Since pyg does not support sparse_tensor attribute,
        #   we instead save indices, values, size of sparse_tensor separately.
        sparse_khop_edge_attr = khop_edge_attr.flatten().to_sparse()
        data.sparse_khop_edge_attr_indices = sparse_khop_edge_attr.indices().squeeze()  # [1, *] -> [*]
        data.sparse_khop_edge_attr_values = sparse_khop_edge_attr.values()
        data.sparse_khop_edge_attr_size = sparse_khop_edge_attr.size()
        data.khop_nodes = khop_nodes
        data.obs_node_index = obs_node_index
        return data

    def __repr__(self):
        return '{}(k={})'.format(self.__class__.__name__, self.num_hops)


class CompleteSubgraph(TransformForDNS):

    def __init__(self, global_edge_index=None):
        super(CompleteSubgraph, self).__init__(global_edge_index=global_edge_index)

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
    from pytorch_lightning import seed_everything
    from pprint import pprint
    from data_sampler import decompress_khop_edges
    from termcolor import cprint

    seed_everything(23)

    _n = 7
    cke = CompressedKhopEdge(num_hops=1)
    _ge = torch.randint(_n, (2, 15))
    _ge_idx = torch.unique(_ge[0] * _n + _ge[1])
    _ge = torch.stack([_ge_idx // _n, _ge_idx % _n], dim=0)

    _ge, _ = add_self_loops(_ge)
    cke.global_edge_index = _ge
    print("---- GE ----")
    print(_ge)

    # Data(edge_index=[2, 402], obs_x=[7], x=[23, 1], y=[1])
    _e = _ge[:, 5:10]
    _x = torch.unique(_e.flatten())
    _obs_x = torch.randperm(_x.size(0))[:3]

    _d = Data(edge_index=_e, obs_x=_obs_x, x=_x)
    print("---- Data ----")
    print(_d)
    print("x", _x)
    print("e", _e)
    print("obs_x", _obs_x)
    print("--------------")

    print(cke)
    _new_d = cke(_d)
    print(_new_d)
    cprint("Compressed attributes", "red")
    print("khop_edge_idx", _new_d.khop_edge_idx)
    print("sparse_khop_edge_attr_indices", _new_d.sparse_khop_edge_attr_indices)
    print("sparse_khop_edge_attr_values", _new_d.sparse_khop_edge_attr_values)
    print("sparse_khop_edge_attr_size", _new_d.sparse_khop_edge_attr_size)

    cprint("Decompression test", "red")
    pprint(
        decompress_khop_edges(_new_d, _n)
    )

