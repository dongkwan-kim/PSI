import copy
import random
from typing import List
import time

import torch
import torch.utils.data.dataloader
from termcolor import cprint
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Compose
from torch_geometric.utils import k_hop_subgraph, negative_sampling, dropout_adj, subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes

import numpy as np
import numpy_indexed as npi

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
        observed_edge_index = None  # temporarily
    else:
        raise AttributeError
    return observed_nodes, observed_edge_index


def create_khop_edge_attr(khop_edge_index, edge_index, edge_attr, N):
    khp_edge_idx_npy = (khop_edge_index[0] * N + khop_edge_index[1]).numpy()
    sub_edge_idx_npy = (edge_index[0] * N + edge_index[1]).numpy()

    # Bottleneck for large k-hop.
    indices_of_sub_in_khp = npi.indices(sub_edge_idx_npy, khp_edge_idx_npy, missing=-1)  # [E_khp]
    is_khp_in_sub_edge = (indices_of_sub_in_khp >= 0)  # [E_khp]
    indices_of_sub_in_khp = indices_of_sub_in_khp[is_khp_in_sub_edge]  # [E_khp_and_in_sub_edge]

    # Construct edge_attr of khop_edges
    khop_edge_attr = np.zeros(khp_edge_idx_npy.shape[0])  # [E_khp]
    if edge_attr is not None:
        khop_edge_attr[is_khp_in_sub_edge] = edge_attr.numpy()[indices_of_sub_in_khp, :].squeeze()
    else:
        khop_edge_attr[is_khp_in_sub_edge] = 1.0
    khop_edge_attr = torch.from_numpy(khop_edge_attr).float().unsqueeze(1)  # [E_khp, 1]
    return khop_edge_attr, is_khp_in_sub_edge


def decompress_khop_edges(data: Data, num_nodes: int, clean_up=True) -> Data:
    """:return: khop_edge_index, khop_edge_attr"""
    N = num_nodes

    khop_edge_idx = data.khop_edge_idx
    data.khop_edge_index = torch.stack([khop_edge_idx // N, khop_edge_idx % N], dim=0)

    skea_indices = data.sparse_khop_edge_attr_indices.unsqueeze(0)  # [*] -> [1, *]
    skea_values = data.sparse_khop_edge_attr_values
    skea_size = data.sparse_khop_edge_attr_size
    sparse_khop_edge_attr = torch.sparse.FloatTensor(
        skea_indices, skea_values, skea_size,
    )  # API from torch==1.6 https://pytorch.org/docs/1.6.0//sparse.html
    data.khop_edge_attr = sparse_khop_edge_attr.to_dense()

    if clean_up:
        del data.khop_edge_idx
        del data.sparse_khop_edge_attr_indices
        del data.sparse_khop_edge_attr_values
        del data.sparse_khop_edge_attr_size
    return data


class KHopWithLabelsXESampler(torch.utils.data.DataLoader):

    def __init__(self, global_data, subdata_list, num_hops,
                 use_labels_x, use_labels_e,
                 neg_sample_ratio, dropout_edges,
                 obs_x_range=None, use_obs_edge_only=False,
                 use_pergraph_attr=False, balanced_sampling=True,
                 use_inter_subgraph_infomax=False,
                 batch_size=1,
                 subdata_filter_func=None,
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

    def __collate_one__(self, d):
        assert hasattr(d, "edge_index")
        edge_index = d.edge_index
        edge_attr = getattr(d, "edge_attr", None)
        if hasattr(d, "khop_edge_idx"):
            d = decompress_khop_edges(d, self.N)
            khop_edge_index = d.khop_edge_index
            khop_edge_attr = d.khop_edge_attr.unsqueeze(1)
            khop_nodes = d.khop_nodes
            obs_node_index = d.obs_node_index
            num_khop_nodes = khop_nodes.size(0)
        else:
            observed_nodes, observed_edge_index = get_observed_nodes_and_edges(
                data=d, obs_x_range=self.obs_x_range,
            )

            if not self.use_obs_edge_only:
                """
                It returns (1) the nodes involved in the subgraph, (2) the filtered
                :obj:`edge_index` connectivity, (3) the mapping from node indices in
                :obj:`node_idx` to their new location, and (4) the edge mask indicating
                which edges were preserved.
                """
                khop_nodes, khop_edge_index, obs_node_index, _ = k_hop_subgraph(
                    node_idx=observed_nodes,
                    num_hops=self.num_hops,
                    edge_index=self.G.edge_index,
                    relabel_nodes=False,  # Relabel later.
                    num_nodes=self.N,
                    flow="target_to_source",  # Important.
                )
            else:
                khop_nodes, obs_node_index = observed_nodes, None
                if observed_edge_index is not None:
                    khop_edge_index = observed_edge_index
                else:
                    # The latter is necessary, since there can be isolated nodes.
                    if edge_index.size(1) > 0:
                        num_nodes = max(maybe_num_nodes(edge_index),
                                        observed_nodes.max().item() + 1)
                    else:
                        num_nodes = observed_nodes.max().item() + 1
                    khop_edge_index, _ = subgraph(
                        subset=observed_nodes,
                        edge_index=edge_index,
                        edge_attr=None, relabel_nodes=False,
                        num_nodes=num_nodes,
                    )

            khop_edge_attr, is_khp_in_sub_edge = create_khop_edge_attr(
                khop_edge_index=khop_edge_index, edge_index=edge_index,
                edge_attr=edge_attr, N=self.N,
            )

            # Relabeling nodes
            num_khop_nodes = khop_nodes.size(0)
            _node_idx = observed_nodes.new_full((self.N,), -1)
            _node_idx[khop_nodes] = torch.arange(num_khop_nodes)
            khop_edge_index = _node_idx[khop_edge_index]

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
                khop_edge_index, khop_edge_attr, p=self.dropout_edges,
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
                full_labels_x = torch.full((num_khop_nodes,), -1.0).float()
                full_labels_x[x_0] = 0.
                idx_of_1 = torch.nonzero(full_labels_x, as_tuple=False)
                perm = torch.randperm(idx_of_1.size(0))
                idx_of_1 = idx_of_1[perm[:x_0.size(0)]]
                full_labels_x[idx_of_1] = 1.
                mask_x = torch.nonzero(full_labels_x >= 0., as_tuple=False).squeeze()
                labels_x = full_labels_x[mask_x]
            else:
                labels_x = torch.ones(num_khop_nodes).float()
                labels_x[x_0] = 0.
            labels_x = labels_x.long()

        if obs_node_index is not None and obs_node_index.size(0) != num_khop_nodes:
            obs_x_idx = obs_node_index
        else:
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
    from data_transform import CompleteSubgraph, CompressedKhopEdge

    PATH = "/mnt/nas2/GNN-DATA"
    DATASET = "FNTN"
    DEBUG = True
    USE_KHOP_PREFETCH = True

    pre_transform = None
    if DATASET == "FNTN":
        if USE_KHOP_PREFETCH:
            pre_transform = Compose([CompressedKhopEdge(num_hops=1),
                                     CompleteSubgraph()])
        else:
            pre_transform = CompleteSubgraph()
        dataset_instance = FNTN(
            root=PATH,
            name="0.0",  # 0.0 0.001 0.002 0.003 0.004
            slice_type="num_edges",
            slice_range=(5, 10),
            num_slices=1,
            val_ratio=0.15,
            test_ratio=0.15,
            pre_transform=pre_transform,
            debug=DEBUG,
        )
    elif DATASET == "HPOMetab":
        if USE_KHOP_PREFETCH:
            # Do not use CompleteSubgraph, this dataset already has complete subgraphs.
            pre_transform = CompressedKhopEdge(num_hops=1)
        dataset_instance = HPOMetab(
            root=PATH,
            name="HPOMetab",
            slice_type="random",
            slice_range=(3, 8),
            num_slices=1,
            val_ratio=0.15,
            test_ratio=0.15,
            pre_transform=pre_transform,
            debug=DEBUG,
        )
    elif DATASET == "HPONeuro":
        if USE_KHOP_PREFETCH:
            # Do not use CompleteSubgraph, this dataset already has complete subgraphs.
            pre_transform = CompressedKhopEdge(num_hops=1)
        dataset_instance = HPONeuro(
            root=PATH,
            name="HPONeuro",
            slice_type="random",
            slice_range=(3, 8),
            num_slices=1,
            val_ratio=0.15,
            test_ratio=0.15,
            pre_transform=pre_transform,
            debug=DEBUG,
        )
    else:
        raise ValueError

    train_fntn, val_fntn, test_fntn = dataset_instance.get_train_val_test()

    sampler = KHopWithLabelsXESampler(
        dataset_instance.global_data, train_fntn,
        num_hops=1, use_labels_x=True, use_labels_e=False,
        neg_sample_ratio=1.0, dropout_edges=0.3, balanced_sampling=True,
        obs_x_range=(5, 10),
        use_inter_subgraph_infomax=True,  # todo
        shuffle=True,
    )
    seed_everything(42)
    cprint("Train w/ ISI", "green")
    for i, b in enumerate(sampler):
        print(i, b)
        if i == 2:
            break

    sampler = KHopWithLabelsXESampler(
        dataset_instance.global_data, train_fntn,
        num_hops=1, use_labels_x=True, use_labels_e=True,
        neg_sample_ratio=1.0, dropout_edges=0.3, balanced_sampling=True,
        obs_x_range=(5, 10),
        shuffle=True,
    )
    cprint("Train first", "green")
    for i, b in enumerate(sampler):
        # Data(edge_index_01=[2, 97688], edge_index_2=[2, 147], labels_e=[440], labels_x=[298],
        #      mask_e=[97835], mask_x=[7261], obs_x_idx=[9], x=[7261], y=[1])
        print(i, b)
        if i == 2:
            break

    cprint("Train second", "green")
    for b in sampler:  # shuffling test
        print(b)

    sampler = KHopWithLabelsXESampler(
        dataset_instance.global_data, val_fntn,
        num_hops=1, use_labels_x=False, use_labels_e=False,
        neg_sample_ratio=0.0, dropout_edges=0.0, balanced_sampling=True,
        shuffle=False,
    )
    cprint("Val", "green")
    for b in sampler:
        # Data(edge_index_01=[2, 31539], obs_x_idx=[8], x=[3029], y=[1])
        print(b)

    sampler = KHopWithLabelsXESampler(
        dataset_instance.global_data, train_fntn,
        num_hops=0, use_labels_x=False, use_labels_e=False,
        neg_sample_ratio=0.0, dropout_edges=0.0,
        use_obs_edge_only=True,  # this.
        shuffle=True,
    )
    cprint("WO Sampler", "green")
    for b in sampler:
        # Data(edge_index_01=[2, 15], x=[10], y=[1])
        print(b)
