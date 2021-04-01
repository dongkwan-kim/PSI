from typing import Tuple, Optional

import torch
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph
from torch_cluster import random_walk

from utils import del_attrs


class DataPN(Data):

    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None,
                 x_pos=None, edge_index_pos=None, x_neg=None, edge_index_neg=None,
                 x_pos_and_neg=None, edge_index_pos_and_neg=None, ptr_pos_and_neg=None,
                 **kwargs):
        super().__init__(x, edge_index, edge_attr, y, **kwargs)
        assert x_pos is None or x_pos_and_neg is None
        self.x_pos = x_pos
        self.edge_index_pos = edge_index_pos
        self.x_neg = x_neg
        self.edge_index_neg = edge_index_neg
        self.x_pos_and_neg = x_pos_and_neg
        self.edge_index_pos_and_neg = edge_index_pos_and_neg
        self.ptr_pos_and_neg = ptr_pos_and_neg

    def __inc__(self, key, value):
        if key == "edge_index_pos_and_neg":
            return self.x_pos_and_neg.size(0)
        elif key == "edge_index_pos":
            return self.x_pos.size(0)
        elif key == "edge_index_neg":
            return self.x_neg.size(0)
        else:
            return super().__inc__(key, value)

    @staticmethod
    def to_kwargs(pos_x_and_e: Optional[Tuple[Tensor, Tensor]],
                  neg_x_and_e: Optional[Tuple[Tensor, Tensor]],
                  concat: bool):

        if pos_x_and_e is None:
            return {}

        x_pos, edge_index_pos = pos_x_and_e
        x_neg, edge_index_neg = neg_x_and_e
        if not concat:
            return {
                "x_pos": x_pos, "edge_index_pos": edge_index_pos,
                "x_neg": x_neg, "edge_index_neg": edge_index_neg,
            }
        else:
            x_pos_and_neg = torch.cat([x_pos, x_neg], dim=0)  # [N_pos + N_neg]
            edge_index_pos_and_neg = torch.cat([edge_index_pos, edge_index_neg], dim=1)
            ptr_pos_and_neg = torch.Tensor([x_pos.size(0)]).long()
            return {
                "x_pos_and_neg": x_pos_and_neg, "edge_index_pos_and_neg": edge_index_pos_and_neg,
                "ptr_pos_and_neg": ptr_pos_and_neg,
            }

    @staticmethod
    def concat_pos_and_neg_in_batch_(batch: Batch, batch_size=None) -> Batch:
        """
        Change (x_pos, edge_index_pos, x_pos_batch,
                x_neg, edge_index_neg, x_neg_batch) in the batch to
               x_pos_and_neg, edge_index_pos_and_neg, batch_pos_and_neg

        :param batch:
        :param batch_size:
        :return:
        """
        x_pos, edge_index_pos, x_pos_batch = batch.x_pos, batch.edge_index_pos, batch.x_pos_batch
        x_neg, edge_index_neg, x_neg_batch = batch.x_neg, batch.edge_index_neg, batch.x_neg_batch

        batch.x_pos_and_neg = torch.cat([x_pos, x_neg], dim=-1)  # [N_pos + N_neg,]

        edge_index_neg += x_pos.size(0)  # + N_pos
        batch.edge_index_pos_and_neg = torch.cat([edge_index_pos, edge_index_neg], dim=-1)  # [2, E_pos+ E_neg]

        x_neg_batch += (batch_size or (x_pos_batch.max().item() + 1))  # + B
        batch.batch_pos_and_neg = torch.cat([x_pos_batch, x_neg_batch], dim=-1)  # [N_pos + N_neg,]

        del_attrs(batch, ["x_pos", "edge_index_pos", "x_pos_batch", "x_neg", "edge_index_neg", "x_neg_batch"])
        return batch


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
