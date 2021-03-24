import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch_cluster import random_walk


def random_walk_indices_from_data(data: Data, walk_length: int):
    conn_x = torch.unique(data.edge_index)
    a_x = conn_x[torch.randperm(conn_x.size(0))][:1]
    row, col = data.edge_index
    a_walk = random_walk(row, col, start=a_x, walk_length=walk_length - 1)
    a_walk = torch.unique(a_walk).squeeze()  # Remove duplicated nodes.

    # Node to idx
    max_x = max(conn_x.max().item(), data.x.max().item())
    _idx = torch.full((max_x + 1,), fill_value=-1.0, dtype=torch.long)
    _idx[data.x.squeeze()] = torch.arange(data.x.size(0))
    a_walk = _idx[a_walk]
    del _idx
    return a_walk


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
    cs = CompleteSubgraph()
    print(cs)
