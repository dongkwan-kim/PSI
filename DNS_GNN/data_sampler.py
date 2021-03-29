import copy
import random
from typing import List
import time

import torch
import torch.utils.data.dataloader
from termcolor import cprint
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph, negative_sampling, dropout_adj, subgraph, to_undirected
from torch_geometric.utils.num_nodes import maybe_num_nodes

import numpy as np
import numpy_indexed as npi

from data_utils import random_walk_indices_from_data
from utils import print_time


def sort_by_edge_attr(edge_attr, edge_index, edge_labels=None):
    edge_attr_wo_zero = edge_attr.clone()
    edge_attr_wo_zero[edge_attr_wo_zero <= 0] = float("inf")
    perm = edge_attr_wo_zero.squeeze().argsort()

    ret = [edge_attr[perm, :], edge_index[:, perm]]
    if edge_labels is not None:
        ret.append(edge_labels[perm])

    return tuple(ret)


def get_observed_nodes_and_edges(data, obs_x_range):
    # Training stage: obs_x_range is not None -> sampling
    # Evaluation stage: obs_x_range is None -> use it from Data
    if hasattr(data, "num_obs_x"):
        # Data(edge_attr=[219, 1], edge_index=[2, 219],
        #      global_attr=[5183], num_obs_x=[1], x=[220, 1], y=[1])
        if obs_x_range is not None:
            num_obs_x = int(torch.randint(obs_x_range[0], obs_x_range[1], (1,)))
        else:
            num_obs_x = int(data.num_obs_x)
        observed_edge_index = data.edge_index[:, :num_obs_x]
        observed_nodes = observed_edge_index.flatten().unique()
    elif hasattr(data, "obs_x"):
        # Data(edge_index=[2, 402], obs_x=[7], x=[23, 1], y=[1])
        if obs_x_range is not None:
            num_obs_x = int(torch.randint(obs_x_range[0], obs_x_range[1], (1,)))
            obs_x = torch.randperm(data.x.size(0))[:num_obs_x]
        else:
            obs_x = data.obs_x
        observed_nodes = data.x[obs_x].flatten().unique()
        observed_edge_index = None  # temporarily, will be assigned in the main loop.
    elif hasattr(data, "obs_rw_x"):
        if obs_x_range is not None:
            num_obs_x = int(torch.randint(obs_x_range[0], obs_x_range[1], (1,)))
            obs_x = random_walk_indices_from_data(data, num_obs_x)
        else:
            obs_x = data.obs_rw_x
        observed_nodes = data.x[obs_x].flatten().unique()
        observed_edge_index = None  # temporarily, will be assigned in the main loop.
    else:
        raise AttributeError
    return observed_nodes, observed_edge_index


def get_khop_attributes(
        global_edge_index,
        num_global_nodes,
        observed_nodes,
        observed_edge_index,
        edge_index,
        num_hops,
        use_obs_edge_only,
):
    if not use_obs_edge_only:
        # Get neighbor nodes from k-hop sampling.
        khop_nodes, khop_edge_index, obs_x_idx, _ = k_hop_subgraph(
            node_idx=observed_nodes,
            num_hops=num_hops,
            edge_index=global_edge_index,
            relabel_nodes=False,  # Relabel later.
            num_nodes=num_global_nodes,
            flow="target_to_source",  # Important.
        )
    else:
        # use_obs_edge_only == True, i.e., observed_nodes are all you need.
        khop_nodes, obs_x_idx = observed_nodes, None
        if observed_edge_index is not None:
            khop_edge_index = observed_edge_index
        else:
            # Case of hasattr(data, "obs_x"),
            #   that is, observed_edge_index is None,
            #   find khop_edge_index from edge_index with observed_nodes.
            if edge_index.size(1) > 0:
                num_nodes = max(maybe_num_nodes(edge_index),
                                observed_nodes.max().item() + 1)
            else:  # necessary, since there can be isolated nodes.
                num_nodes = observed_nodes.max().item() + 1
            khop_edge_index, _ = subgraph(
                subset=observed_nodes,
                edge_index=edge_index,
                edge_attr=None, relabel_nodes=False,
                num_nodes=num_nodes,
            )
    return khop_nodes, khop_edge_index, obs_x_idx


def create_khop_edge_attr(khop_edge_index, edge_index, edge_attr, N, method):
    if method == "edge":
        is_khp_in_subgraph, indices_of_sub_in_khp = _create_khop_edge_attr_edge_ver(
            khop_edge_index, edge_index, N)
    elif method == "node":
        assert edge_attr is None
        nodes = torch.unique(edge_index)
        is_khp_in_subgraph = _create_khop_edge_attr_node_ver(
            khop_edge_index, nodes,
        )
        indices_of_sub_in_khp = None
    else:
        raise ValueError("Wrong method: {}".format(method))

    # Construct edge_attr of khop_edges
    if edge_attr is not None:
        khop_edge_attr = torch.zeros(is_khp_in_subgraph.shape[0])  # [E_khp]
        khop_edge_attr[is_khp_in_subgraph] = edge_attr[indices_of_sub_in_khp, :].squeeze()
        khop_edge_attr[is_khp_in_subgraph] = 1.0
    else:
        khop_edge_attr = torch.Tensor(is_khp_in_subgraph).float()
    khop_edge_attr = khop_edge_attr.unsqueeze(1)  # [E_khp, 1]
    return khop_edge_attr, is_khp_in_subgraph


def _create_khop_edge_attr_edge_ver(khop_edge_index, edge_index, N):
    khp_edge_idx_npy = (khop_edge_index[0] * N + khop_edge_index[1]).numpy()
    sub_edge_idx_npy = (edge_index[0] * N + edge_index[1]).numpy()

    # Bottleneck for large k-hop.
    indices_of_sub_edge_in_khp = npi.indices(sub_edge_idx_npy, khp_edge_idx_npy, missing=-1)  # [E_khp]
    is_khp_in_sub_edge = (indices_of_sub_edge_in_khp >= 0)  # [E_khp]
    indices_of_sub_edge_in_khp = indices_of_sub_edge_in_khp[is_khp_in_sub_edge]  # [E_khp_and_in_sub_edge]
    return is_khp_in_sub_edge, indices_of_sub_edge_in_khp


def _create_khop_edge_attr_node_ver(khop_edge_index, nodes):
    nodes = nodes.numpy()
    khp_row, khp_col = khop_edge_index.numpy()
    is_khp_row_in_sub_nodes = npi.contains(nodes, khp_row)
    is_khp_col_in_sub_nodes = npi.contains(nodes, khp_col[is_khp_row_in_sub_nodes])

    subarray = is_khp_row_in_sub_nodes[is_khp_row_in_sub_nodes]
    subarray[~is_khp_col_in_sub_nodes] = False
    is_khp_row_in_sub_nodes[is_khp_row_in_sub_nodes] = subarray
    return is_khp_row_in_sub_nodes


class KHopWithLabelsXESampler(torch.utils.data.DataLoader):

    def __init__(self, global_data, subdata_list, num_hops,
                 use_labels_x, use_labels_e,
                 neg_sample_ratio, dropout_edges,
                 obs_x_range=None, use_obs_edge_only=False,
                 use_pergraph_attr=False, balanced_sampling=True,
                 use_inter_subgraph_infomax=False,
                 batch_size=1,
                 subdata_filter_func=None,
                 cache_hop_computation=False,
                 ke_method="node",
                 shuffle=False, verbose=0, **kwargs):

        self.G = global_data
        if subdata_filter_func is None:
            self.subdata_list: List[Data] = subdata_list
        else:
            self.subdata_list: List[Data] = [d for d in subdata_list
                                             if subdata_filter_func(d)]
        self.num_hops = num_hops
        self.use_labels_x = use_labels_x
        self.use_labels_e = use_labels_e
        self.cache_hop_computation = cache_hop_computation
        self.ke_method = ke_method
        assert self.ke_method in ["node", "edge"]

        self.neg_sample_ratio = neg_sample_ratio * (1 - dropout_edges)
        self.dropout_edges = dropout_edges  # the bigger, the more sparse graph
        self.obs_x_range = obs_x_range
        self.use_obs_edge_only = use_obs_edge_only
        self.use_pergraph_attr = use_pergraph_attr
        self.balanced_sampling = balanced_sampling
        self.use_inter_subgraph_infomax = use_inter_subgraph_infomax
        self.N = global_data.edge_index.max() + 1

        self.verbose = verbose
        if verbose > 0:
            cprint("Initialized: {}, len={}, with network={}".format(
                self.__class__.__name__, len(subdata_list), global_data), "blue")

        super(KHopWithLabelsXESampler, self).__init__(
            self.subdata_list, batch_size=batch_size, collate_fn=self.__collate__,
            shuffle=shuffle, **kwargs,
        )

    def __collate__(self, data_list):

        if len(data_list) == 1:
            return self.__collate_one__(data_list[0])

        collated = []
        for d in data_list:
            collated.append(self.__collate_one__(d))
        return Batch.from_data_list(collated)

    def __collate_one__(self, d: Data):
        assert hasattr(d, "edge_index")
        edge_index = d.edge_index
        edge_attr = getattr(d, "edge_attr", None)

        if self.cache_hop_computation and hasattr(d, "khop_edge_index"):
            # Use cached version.
            khop_nodes = d.khop_nodes
            khop_edge_index = d.khop_edge_index
            obs_x_idx = d.obs_x_idx
            khop_edge_attr = d.khop_edge_attr
        else:
            observed_nodes, observed_edge_index = get_observed_nodes_and_edges(
                data=d, obs_x_range=self.obs_x_range,
            )
            """
            It returns (1) the nodes involved in the subgraph, (2) the filtered
            :obj:`edge_index` connectivity, (3) the mapping (obs_x_idx) from node
            indices in :obj:`node_idx` to their new location.
            """
            khop_nodes, khop_edge_index, obs_x_idx = get_khop_attributes(
                global_edge_index=self.G.edge_index,
                num_global_nodes=self.N,
                observed_nodes=observed_nodes,
                observed_edge_index=observed_edge_index,
                edge_index=edge_index,
                num_hops=self.num_hops,
                use_obs_edge_only=self.use_obs_edge_only,
            )
            khop_edge_attr, is_khp_in_sub_edge = create_khop_edge_attr(
                khop_edge_index=khop_edge_index, edge_index=edge_index,
                edge_attr=edge_attr, N=self.N, method=self.ke_method,
            )

            # Relabeling nodes
            _node_idx = observed_nodes.new_full((self.N,), -1)
            _node_idx[khop_nodes] = torch.arange(khop_nodes.size(0))
            khop_edge_index = _node_idx[khop_edge_index]

            if self.cache_hop_computation:
                # Update the cache.
                d.khop_nodes = khop_nodes
                d.khop_edge_index = khop_edge_index
                d.obs_x_idx = obs_x_idx
                d.khop_edge_attr = khop_edge_attr
                d.cached = True

        # Negative sampling of edges
        neg_edge_index = None
        num_pos_samples = torch.sum(khop_edge_attr > 0).item()
        num_neg_samples = int(self.neg_sample_ratio * num_pos_samples)
        if self.use_labels_e and self.neg_sample_ratio > 0:
            neg_edge_index = negative_sampling(
                edge_index=khop_edge_index,
                num_neg_samples=num_neg_samples,
                method="dense",
            )

        # Drop Edges in graph and subgraph
        if self.dropout_edges > 0.0:
            khop_edge_index, khop_edge_attr = dropout_adj(
                khop_edge_index, khop_edge_attr,
                p=self.dropout_edges, num_nodes=self.N,
            )

        KE = khop_edge_index.size(1)
        if edge_attr is not None:
            khop_edge_attr, khop_edge_index = sort_by_edge_attr(khop_edge_attr, khop_edge_index)

        # 0: in-subgraph, 1: in-graph, 2: not-in-graph
        labels_e, mask_e = None, None
        if self.use_labels_e:
            labels_e_2 = torch.full((neg_edge_index.size(1),), 2.).float()  # [E_khop_2]
            if self.balanced_sampling:
                full_labels_e_01 = torch.full((KE,), -1.).float()
                full_labels_e_01[khop_edge_attr.squeeze() > 0] = 0.
                idx_of_1 = torch.nonzero(full_labels_e_01, as_tuple=False)
                perm = torch.randperm(idx_of_1.size(0))
                idx_of_1 = idx_of_1[perm[:num_neg_samples]]
                full_labels_e_01[idx_of_1] = 1.
                full_labels_e = torch.cat([full_labels_e_01, labels_e_2])
                mask_e = torch.nonzero(full_labels_e >= 0., as_tuple=False).squeeze()
                labels_e = full_labels_e[mask_e]  # [nearly 3 * E_khop_2]
            else:
                labels_e_01 = torch.zeros(KE).float()  # [E_khop_01]
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
                mask_x = torch.nonzero(full_labels_x >= 0., as_tuple=False).squeeze()
                labels_x = full_labels_x[mask_x]
            else:
                labels_x = torch.ones(khop_nodes.size(0)).float()
                labels_x[x_0] = 0.
            labels_x = labels_x.long()

        if obs_x_idx is None or obs_x_idx.size(0) == khop_nodes.size(0):
            obs_x_idx = None

        if self.use_pergraph_attr:
            pergraph_attr = d.pergraph_attr
        else:
            pergraph_attr = None

        if self.use_inter_subgraph_infomax:
            d_neg = self.sample_uni_data_neg(uni_data_pos=d)

            try:
                x_pos_isi, edge_index_pos_isi = d.x_cs, d.edge_index_cs
                x_neg_isi, edge_index_neg_isi = d_neg.x_cs, d_neg.edge_index_cs
            except AttributeError:
                x_pos_isi, edge_index_pos_isi = d.x, d.edge_index
                x_neg_isi, edge_index_neg_isi = d_neg.x, d_neg.edge_index

            x_isi = torch.cat([x_pos_isi, x_neg_isi], dim=0).squeeze()  # [N_pos + N_neg]
            edge_index_isi = torch.cat([edge_index_pos_isi, edge_index_neg_isi], dim=1)

            # Relabeling
            _node_idx = torch.full((self.N,), fill_value=-1, dtype=torch.long)
            _node_idx[x_isi] = torch.arange(x_isi.size(0))
            edge_index_isi = _node_idx[edge_index_isi]

            ptr_isi = torch.Tensor([x_pos_isi.size(0)]).long()
        else:
            x_isi, edge_index_isi, ptr_isi = None, None, None

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
            pergraph_attr=pergraph_attr,
            x_isi=x_isi,
            edge_index_isi=edge_index_isi,
            ptr_isi=ptr_isi,
        )
        return sampled_data

    def sample_uni_data_neg(self, uni_data_pos):
        # todo: support multiple neg samples.
        neg_idx_list = random.sample(range(len(self.subdata_list)), 2)
        data_neg_0 = self.subdata_list[neg_idx_list[0]]
        is_not_neg = (uni_data_pos.x.size(0) == data_neg_0.x.size(0) and
                      uni_data_pos.x.sum() == data_neg_0.x.sum())
        if is_not_neg:
            return self.subdata_list[neg_idx_list[1]]
        else:
            return data_neg_0


if __name__ == '__main__':
    from data_fntn import FNTN
    from data_sub import HPOMetab, HPONeuro
    from pytorch_lightning import seed_everything
    from data_utils import CompleteSubgraph

    PATH = "/mnt/nas2/GNN-DATA"
    DATASET = "HPONeuro"
    DEBUG = False

    if DATASET == "FNTN":
        SLICE_RANGE = (5, 10)
        dataset_instance = FNTN(
            root=PATH,
            name="0.0",  # 0.0 0.001 0.002 0.003 0.004
            slice_type="num_edges",
            slice_range=SLICE_RANGE,
            num_slices=1,
            val_ratio=0.15,
            test_ratio=0.15,
            pre_transform=CompleteSubgraph(),
            debug=DEBUG,
        )
    elif DATASET == "HPOMetab":
        SLICE_RANGE = (3, 8)
        dataset_instance = HPOMetab(
            root=PATH,
            name="HPOMetab",
            slice_type="random_walk",
            slice_range=SLICE_RANGE,
            num_slices=1,
            val_ratio=0.15,
            test_ratio=0.15,
            pre_transform=None,  # not CompleteSubgraph
            debug=DEBUG,
        )
    elif DATASET == "HPONeuro":
        SLICE_RANGE = (3, 8)
        dataset_instance = HPONeuro(
            root=PATH,
            name="HPONeuro",
            slice_type="random",
            slice_range=SLICE_RANGE,
            num_slices=1,
            val_ratio=0.15,
            test_ratio=0.15,
            pre_transform=None,  # not CompleteSubgraph
            debug=DEBUG,
        )
    else:
        raise ValueError

    train_fntn, val_fntn, test_fntn = dataset_instance.get_train_val_test()

    sampler = KHopWithLabelsXESampler(
        dataset_instance.global_data, train_fntn,
        num_hops=1, use_labels_x=True, use_labels_e=False,
        neg_sample_ratio=1.0, dropout_edges=0.3, balanced_sampling=True,
        obs_x_range=SLICE_RANGE,
        use_inter_subgraph_infomax=True,  # todo
        cache_hop_computation=False,
        ke_method="node",
        shuffle=True,
    )
    seed_everything(42)
    cprint("Train w/ ISI", "green")
    for i, b in enumerate(sampler):
        print(i, b)
        if i >= 4:
            break

    sampler = KHopWithLabelsXESampler(
        dataset_instance.global_data, train_fntn,
        num_hops=1, use_labels_x=True, use_labels_e=True,
        neg_sample_ratio=1.0, dropout_edges=0.3, balanced_sampling=True,
        obs_x_range=SLICE_RANGE,
        use_inter_subgraph_infomax=False,  # todo
        cache_hop_computation=True,
        shuffle=True,
    )
    cprint("Train first", "green")
    for i, b in enumerate(sampler):
        # Data(edge_index_01=[2, 97688], edge_index_2=[2, 147], labels_e=[440], labels_x=[298],
        #      mask_e=[97835], mask_x=[7261], obs_x_idx=[9], x=[7261], y=[1])
        print(i, b)
        if i >= 4:
            break

    cprint("Train second", "green")
    for i, b in enumerate(sampler):  # shuffling test
        print(b)
        if i >= 4:
            break

    sampler = KHopWithLabelsXESampler(
        dataset_instance.global_data, val_fntn,
        num_hops=1, use_labels_x=False, use_labels_e=False,
        neg_sample_ratio=0.0, dropout_edges=0.0, balanced_sampling=True,
        cache_hop_computation=False,
        shuffle=False,
    )
    cprint("Val", "green")
    for i, b in enumerate(sampler):
        # Data(edge_index_01=[2, 31539], obs_x_idx=[8], x=[3029], y=[1])
        print(b)
        if i >= 4:
            break

    sampler = KHopWithLabelsXESampler(
        dataset_instance.global_data, train_fntn,
        num_hops=0, use_labels_x=False, use_labels_e=False,
        neg_sample_ratio=0.0, dropout_edges=0.0,
        use_obs_edge_only=True,  # this.
        cache_hop_computation=False,
        shuffle=True,
    )
    cprint("WO Sampler", "green")
    for i, b in enumerate(sampler):
        # Data(edge_index_01=[2, 15], x=[10], y=[1])
        print(b)
        if i >= 4:
            break
