import copy
from typing import List
import time

import torch
import torch.utils.data.dataloader
from termcolor import cprint
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, negative_sampling, dropout_adj

import numpy as np
import numpy_indexed as npi


def sort_by_edge_attr(edge_attr, edge_index, edge_labels=None):
    edge_attr_wo_zero = edge_attr.clone()
    edge_attr_wo_zero[edge_attr_wo_zero <= 0] = float("inf")
    perm = edge_attr_wo_zero.squeeze().argsort()

    ret = [edge_attr[perm, :], edge_index[:, perm]]
    if edge_labels is not None:
        ret.append(edge_labels[perm])

    return tuple(ret)


class KHopWithLabelsXESampler(torch.utils.data.DataLoader):

    def __init__(self, global_data, subdata_list, num_hops,
                 use_labels_x, use_labels_e,
                 neg_sample_ratio, dropout_edges,
                 use_obs_edge_only=False,
                 balanced_sampling=True, shuffle=False, verbose=0, **kwargs):

        self.G = global_data
        self.subdata_list: List[Data] = subdata_list
        self.num_hops = num_hops
        self.use_labels_x = use_labels_x
        self.use_labels_e = use_labels_e

        self.neg_sample_ratio = neg_sample_ratio * (1 - dropout_edges)
        self.dropout_edges = dropout_edges  # the bigger, the more sparse graph
        self.use_obs_edge_only = use_obs_edge_only
        self.balanced_sampling = balanced_sampling
        self.N = global_data.edge_index.max() + 1

        self.verbose = verbose
        if verbose > 0:
            cprint("Initialized: {}, len={}, with network={}".format(
                self.__class__.__name__, len(subdata_list), global_data), "blue")

        super(KHopWithLabelsXESampler, self).__init__(
            subdata_list, batch_size=1, collate_fn=self.__collate__,
            shuffle=shuffle, **kwargs,
        )

    def __collate__(self, uni_data):
        # Data(edge_attr=[219, 1], edge_index=[2, 219], global_attr=[5183], num_obs_x=[1], x=[220, 1], y=[1])
        d = uni_data[0]
        assert hasattr(d, "edge_index")
        assert hasattr(d, "edge_attr")

        edge_index, edge_attr = d.edge_index, d.edge_attr

        observed_edge_index = edge_index[:, :int(d.num_obs_x)]
        observed_nodes = observed_edge_index.flatten().unique()

        """
        It returns (1) the nodes involved in the subgraph, (2) the filtered
        :obj:`edge_index` connectivity, (3) the mapping from node indices in
        :obj:`node_idx` to their new location, and (4) the edge mask indicating
        which edges were preserved.
        """
        if not self.use_obs_edge_only:
            khop_nodes, khop_edge_index, mapping, _ = k_hop_subgraph(
                node_idx=observed_nodes,
                num_hops=self.num_hops,
                edge_index=self.G.edge_index,
                relabel_nodes=False,  # Relabel later.
                num_nodes=self.N,
                flow="target_to_source",  # Important.
            )
        else:
            khop_nodes, khop_edge_index, mapping = observed_nodes, observed_edge_index, None

        khp_edge_idx_npy = (khop_edge_index[0] * self.N + khop_edge_index[1]).numpy()
        sub_edge_idx_npy = (edge_index[0] * self.N + edge_index[1]).numpy()

        # Bottle neck for large k-hop.
        indices_of_sub_in_khp = npi.indices(sub_edge_idx_npy, khp_edge_idx_npy, missing=-1)  # [E_khp]
        is_khp_in_sub_edge = (indices_of_sub_in_khp >= 0)  # [E_khp]
        indices_of_sub_in_khp = indices_of_sub_in_khp[is_khp_in_sub_edge]  # [E_khp_and_in_sub_edge]

        # Construct edge_attr of khop_edges
        khop_edge_attr = np.zeros(khp_edge_idx_npy.shape[0])  # [E_khp]
        khop_edge_attr[is_khp_in_sub_edge] = edge_attr.numpy()[indices_of_sub_in_khp, :].squeeze()
        khop_edge_attr = torch.from_numpy(khop_edge_attr).float().unsqueeze(1)  # [E_khp, 1]

        # Relabeling nodes
        _node_idx = observed_nodes.new_full((self.N,), -1)
        _node_idx[khop_nodes] = torch.arange(khop_nodes.size(0))
        khop_edge_index = _node_idx[khop_edge_index]

        # Negative sampling of edges
        neg_edge_index = None
        num_neg_samples = int(self.neg_sample_ratio * torch.sum(khop_edge_attr > 0))
        if self.neg_sample_ratio > 0:
            neg_edge_index = negative_sampling(
                edge_index=khop_edge_index,
                num_neg_samples=num_neg_samples,
                method="dense",
            )

        # Drop Edges in graph and subgraph
        if self.dropout_edges > 0.0:
            khop_edge_index, khop_edge_attr = dropout_adj(
                khop_edge_index, khop_edge_attr, p=self.dropout_edges,
            )

        khop_edge_attr, khop_edge_index = sort_by_edge_attr(khop_edge_attr, khop_edge_index)

        # 0: in-subgraph, 1: in-graph, 2: not-in-graph
        labels_e, mask_e = None, None
        if self.use_labels_e:
            labels_e_2 = torch.full((neg_edge_index.size(1),), 2.).float()  # [E_khop_2]
            if self.balanced_sampling:
                full_labels_e_01 = torch.full((khop_edge_attr.size(0),), -1.).float()
                full_labels_e_01[khop_edge_attr.squeeze() > 0] = 0.
                idx_of_1 = torch.nonzero(full_labels_e_01, as_tuple=False)
                perm = torch.randperm(idx_of_1.size(0))
                idx_of_1 = idx_of_1[perm[:num_neg_samples]]
                full_labels_e_01[idx_of_1] = 1.
                full_labels_e = torch.cat([full_labels_e_01, labels_e_2])
                mask_e = full_labels_e >= 0.
                labels_e = full_labels_e[mask_e]  # [nearly 3 * E_khop_2]
            else:
                labels_e_01 = torch.zeros(khop_edge_attr.size(0)).float()  # [E_khop_01]
                labels_e_01[khop_edge_attr.squeeze() <= 0] = 1.
                labels_e = torch.cat([labels_e_01, labels_e_2])
            labels_e = labels_e.long()

        # 0: in-subgraph, 1: not-in-subgraph
        labels_x, mask_x = None, None
        if self.use_labels_x:
            x_0 = khop_edge_index[:, khop_edge_attr.squeeze() > 0].flatten().unique()
            if self.balanced_sampling:
                full_labels_x = torch.full((khop_nodes.size(0),), -1.0).float()
                full_labels_x[x_0] = 0.
                idx_of_1 = torch.nonzero(full_labels_x, as_tuple=False)
                perm = torch.randperm(idx_of_1.size(0))
                idx_of_1 = idx_of_1[perm[:x_0.size(0)]]
                full_labels_x[idx_of_1] = 1.
                mask_x = full_labels_x >= 0.
                labels_x = full_labels_x[mask_x]
            else:
                labels_x = torch.ones(khop_nodes.size(0)).float()
                labels_x[x_0] = 0.
            labels_x = labels_x.long()

        if mapping is not None and mapping.size(0) != khop_nodes.size(0):
            obs_x_idx = mapping
        else:
            obs_x_idx = None

        # noinspection PyUnresolvedReferences
        sampled_data = Data(
            x=khop_nodes,
            obs_x_idx=obs_x_idx,
            labels_x=labels_x,
            mask_x=mask_x,
            edge_index_01=khop_edge_index,
            edge_index_2=neg_edge_index,
            labels_e=labels_e,
            mask_e=mask_e,
            y=d.y,
        )
        return sampled_data


if __name__ == '__main__':
    from data_fntn import FNTN

    PATH = "/mnt/nas2/GNN-DATA"
    DEBUG = True

    fntn = FNTN(
        root=PATH,
        name="0.0",  # 0.0 0.001 0.002 0.003 0.004
        slice_type="num_edges",
        slice_range=(5, 10),
        num_slices=5,
        val_ratio=0.15,
        test_ratio=0.15,
        debug=DEBUG,
    )

    train_fntn, val_fntn, test_fntn = fntn.get_train_val_test()

    sampler = KHopWithLabelsXESampler(
        fntn.global_data, train_fntn,
        num_hops=1, use_labels_x=True, use_labels_e=True,
        neg_sample_ratio=1.0, dropout_edges=0.3, balanced_sampling=True,
        shuffle=True,
    )
    print("Train first")
    for i, b in enumerate(sampler):
        # Data(edge_index_01=[2, 97688], edge_index_2=[2, 147], labels_e=[440], labels_x=[298],
        #      mask_e=[97835], mask_x=[7261], obs_x_idx=[9], x=[7261], y=[1])
        print(i, b)
        if i == 2:
            break

    print("Train second")
    for b in sampler:  # shuffling test
        print(b)
        break

    sampler = KHopWithLabelsXESampler(
        fntn.global_data, val_fntn,
        num_hops=1, use_labels_x=False, use_labels_e=False,
        neg_sample_ratio=0.0, dropout_edges=0.0, balanced_sampling=True,
        shuffle=False,
    )
    print("Val")
    for b in sampler:
        # Data(edge_index_01=[2, 31539], obs_x_idx=[8], x=[3029], y=[1])
        print(b)
        break

    sampler = KHopWithLabelsXESampler(
        fntn.global_data, train_fntn,
        num_hops=0, use_labels_x=False, use_labels_e=False,
        neg_sample_ratio=0.0, dropout_edges=0.0,
        use_obs_edge_only=True,  # this.
        shuffle=True,
    )
    print("WO Sampler")
    for b in sampler:
        # Data(edge_index_01=[2, 15], x=[10], y=[1])
        print(b)
        break