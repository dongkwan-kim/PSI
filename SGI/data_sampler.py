import copy
import math
import random
from pprint import pprint
from typing import List, Tuple, Dict
import time
from types import MethodType

import torch
import torch.utils.data.dataloader
from termcolor import cprint
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Compose
from torch_geometric.utils import k_hop_subgraph, negative_sampling, dropout_adj, subgraph, to_undirected
from torch_geometric.utils.num_nodes import maybe_num_nodes

import numpy as np
import numpy_indexed as npi
from tqdm import tqdm

from data_base import get_int_range
from data_utils import random_walk_indices_from_data, DataPN, DigitizeY
from utils import print_time, n_wise, dropout_nodes, sample_index_with_replacement_and_exclusion


def sort_by_edge_attr(edge_attr, edge_index, edge_labels=None):
    edge_attr_wo_zero = edge_attr.clone()
    edge_attr_wo_zero[edge_attr_wo_zero <= 0] = float("inf")
    perm = edge_attr_wo_zero.squeeze().argsort()

    ret = [edge_attr[perm, :], edge_index[:, perm]]
    if edge_labels is not None:
        ret.append(edge_labels[perm])

    return tuple(ret)


def get_observed_nodes_and_edges(data, obs_x_range):
    if obs_x_range is not None:
        # (r1: int, r2: int) -> same (r1, r2)
        # (r1: float, window: int) -> (N * r1 - window, N * r1 + window)
        obs_x_range = get_int_range(*obs_x_range, data.x.size(0))

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
        khop_nodes, khop_edge_index, obs_x_index, _ = k_hop_subgraph(
            node_idx=observed_nodes,
            num_hops=num_hops,
            edge_index=global_edge_index,
            relabel_nodes=False,  # Relabel later.
            num_nodes=num_global_nodes,
            flow="target_to_source",  # Important.
        )
    else:
        # use_obs_edge_only == True, i.e., observed_nodes are all you need.
        khop_nodes, obs_x_index = observed_nodes, None
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
    return khop_nodes, khop_edge_index, obs_x_index


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
    return khop_edge_attr, torch.Tensor(is_khp_in_subgraph).bool()


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
                 use_pergraph_attr=False,
                 balanced_sampling=True,
                 subgraph_infomax_type=None,
                 negative_sample_type_in_isi="SGI",
                 neg_sample_ratio_in_isi=1.0,
                 no_drop_pos_edges=False,
                 batch_size=1,
                 subdata_filter_func=None,
                 cache_hop_computation=False,
                 ke_method=None,
                 shuffle=False,
                 num_workers=0,
                 verbose=0,
                 **kwargs):

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
        self.subgraph_infomax_type = subgraph_infomax_type
        self.negative_sample_type_in_isi = negative_sample_type_in_isi
        self.neg_sample_ratio_in_isi = neg_sample_ratio_in_isi
        self.no_drop_pos_edges = no_drop_pos_edges

        self.G = global_data
        self.N = global_data.edge_index.max() + 1

        if subdata_filter_func is None:
            self.subdata_list: List[Data] = subdata_list
        else:
            self.subdata_list: List[Data] = [d for d in subdata_list
                                             if subdata_filter_func(d)]

        if self.subgraph_infomax_type is not None:
            for idx, d in enumerate(self.subdata_list):
                d.idx = idx

        self.verbose = verbose
        if verbose > 0:
            cprint("Initialized: {}, len={}, with network={}".format(
                self.__class__.__name__, len(subdata_list), global_data), "blue")

        super(KHopWithLabelsXESampler, self).__init__(
            self.subdata_list, batch_size=batch_size, collate_fn=self.__collate__,
            shuffle=shuffle, num_workers=num_workers, **kwargs,
        )

    @property
    def B(self):
        return self.batch_size

    def __collate__(self, data_list):

        is_single_batch = len(data_list) == 1

        if self.subgraph_infomax_type is not None:
            pos_attr_list, neg_attr_list = self.get_isi_attr_from_pos_data_list(data_list)
        else:
            pos_attr_list, neg_attr_list = [None for _ in range(self.B)], [None for _ in range(self.B)]

        # DataPN.to_kwargs:
        # Construct (x_pos, edge_index_pos, x_neg, edge_index_neg) for multi-batch B > 1,
        #        or (x_pos_and_neg, edge_index_pos_and_neg, ptr_pos_and_neg) for single-batch B = 1.
        processed_data_list = [
            self.__process_to_data__(
                d,
                isi_kwargs=DataPN.to_kwargs(pos_x_and_e, neg_x_and_e, concat=is_single_batch))
            for d, pos_x_and_e, neg_x_and_e in zip(data_list, pos_attr_list, neg_attr_list)
        ]
        if is_single_batch:
            return processed_data_list[0]
        else:
            collated_batch = Batch.from_data_list(processed_data_list, follow_batch=["x_pos", "x_neg"])
            if self.subgraph_infomax_type is not None:
                collated_batch = DataPN.concat_pos_and_neg_in_batch_(collated_batch, batch_size=len(data_list))
            return collated_batch

    def __process_to_data__(self, d: Data, isi_kwargs: dict) -> DataPN:
        assert hasattr(d, "edge_index")
        edge_index = d.edge_index
        edge_attr = getattr(d, "edge_attr", None)

        if self.cache_hop_computation and hasattr(d, "khop_edge_index"):
            # Use cached version.
            khop_nodes = d.khop_nodes
            khop_edge_index = d.khop_edge_index
            obs_x_index = d.obs_x_index
            khop_edge_attr = d.khop_edge_attr
            is_khp_in_sub_edge = None
        else:
            observed_nodes, observed_edge_index = get_observed_nodes_and_edges(
                data=d, obs_x_range=self.obs_x_range,
            )
            """
            It returns (1) the nodes involved in the subgraph, (2) the filtered
            :obj:`edge_index` connectivity, (3) the mapping (obs_x_index) from node
            indices in :obj:`node_idx` to their new location.
            """
            khop_nodes, khop_edge_index, obs_x_index = get_khop_attributes(
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
                d.obs_x_index = obs_x_index
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
            if not self.no_drop_pos_edges:  # General cases
                khop_edge_index, khop_edge_attr = dropout_adj(
                    khop_edge_index, khop_edge_attr,
                    p=self.dropout_edges, num_nodes=self.N,
                )
            else:
                _ks_mask = is_khp_in_sub_edge
                khop_edge_index_pos, khop_edge_attr_pos = khop_edge_index[:, _ks_mask], khop_edge_attr[_ks_mask]
                khop_edge_index_neg, khop_edge_attr_neg = dropout_adj(
                    khop_edge_index[:, ~_ks_mask], khop_edge_attr[~_ks_mask],
                    p=self.dropout_edges, num_nodes=self.N,
                )
                khop_edge_index = torch.cat([khop_edge_index_pos, khop_edge_index_neg], dim=1)
                khop_edge_attr = torch.cat([khop_edge_attr_pos, khop_edge_attr_neg], dim=0)

        KE = khop_edge_index.size(1)
        if edge_attr is not None:
            khop_edge_attr, khop_edge_index = sort_by_edge_attr(khop_edge_attr, khop_edge_index)

        # 0: in-subgraph, 1: in-graph, 2: not-in-graph
        labels_e, mask_e_index = None, None
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
                mask_e_index = torch.nonzero(full_labels_e >= 0., as_tuple=False).squeeze()
                labels_e = full_labels_e[mask_e_index]  # [nearly 3 * E_khop_2]
            else:
                labels_e_01 = torch.zeros(KE).float()  # [E_khop_01]
                labels_e_01[khop_edge_attr.squeeze() <= 0] = 1.
                labels_e = torch.cat([labels_e_01, labels_e_2])
            labels_e = labels_e.long()

        # 0: in-subgraph, 1: not-in-subgraph
        labels_x, mask_x_index = None, None
        if self.use_labels_x:
            x_0 = khop_edge_index[:, khop_edge_attr.squeeze() > 0].flatten().unique()
            if self.balanced_sampling:
                full_labels_x = torch.full((khop_nodes.size(0),), -1.0).float()
                full_labels_x[x_0] = 0.
                idx_of_1 = torch.nonzero(full_labels_x, as_tuple=False)
                perm = torch.randperm(idx_of_1.size(0))
                idx_of_1 = idx_of_1[perm[:x_0.size(0)]]
                full_labels_x[idx_of_1] = 1.
                mask_x_index = torch.nonzero(full_labels_x >= 0., as_tuple=False).squeeze()
                labels_x = full_labels_x[mask_x_index]
            else:
                labels_x = torch.ones(khop_nodes.size(0)).float()
                labels_x[x_0] = 0.
            labels_x = labels_x.long()

        if obs_x_index is None or obs_x_index.size(0) == khop_nodes.size(0):
            obs_x_index = None

        if self.use_pergraph_attr:
            pergraph_attr = d.pergraph_attr.view(1, -1)
        else:
            pergraph_attr = None

        sampled_data = DataPN(
            x=khop_nodes,
            obs_x_index=obs_x_index,
            labels_x=labels_x,
            mask_x_index=mask_x_index,
            edge_index_01=khop_edge_index,
            edge_index_2=neg_edge_index,
            labels_e=labels_e,
            mask_e_index=mask_e_index,
            y=d.y,
            pergraph_attr=pergraph_attr,
            **isi_kwargs,
        )
        return sampled_data

    def get_isi_attr_from_pos_data_list(self, pos_data_list: List[Data]):
        num_subdata = len(self.subdata_list)
        neg_magnification = math.ceil(self.neg_sample_ratio_in_isi)
        neg_non_dropout = self.neg_sample_ratio_in_isi / neg_magnification
        num_samples = min(len(pos_data_list), num_subdata // 2) * neg_magnification
        pos_idx_list = [d.idx for d in pos_data_list]

        def corruption(_x):
            return _x[torch.randperm(_x.size(0))]

        if self.negative_sample_type_in_isi in ["SGI", "INFOGRAPH"]:
            neg_data_idx_list = sample_index_with_replacement_and_exclusion(
                num_subdata, num_to_sample=num_samples, set_to_exclude=set(pos_idx_list))
            neg_data_list = []
            for neg_data_indices in n_wise(neg_data_idx_list[:num_samples], n=neg_magnification):
                if neg_magnification == 1:
                    neg_data_list.append(self.subdata_list[neg_data_indices[0]])
                else:
                    neg_subdata_list = []
                    for neg_idx in neg_data_indices:
                        subdata = self.subdata_list[neg_idx]
                        subdata.__inc__ = MethodType(lambda s, k, v: 0, subdata)  # not accumulating edge_index
                        neg_subdata_list.append(subdata)
                    d = Batch.from_data_list(neg_subdata_list) if neg_magnification > 1 else neg_subdata_list[0]
                    neg_data_list.append(d)
        elif self.negative_sample_type_in_isi == "DGI":
            neg_data_list = []
            for pos_data in pos_data_list:
                corr_data = pos_data.clone()
                corr_data.x = corruption(corr_data.x)
                neg_data_list.append(corr_data)
        else:
            raise ValueError(f"Wrong negative_sample_type: {self.negative_sample_type_in_isi}")

        _node_idx = torch.full((self.N,), fill_value=-1, dtype=torch.long)

        def get_isi_attr(d: Data, isi_type: str = None) -> Tuple[Tensor, Tensor]:
            _x_isi = getattr(d, "x_cs", d.x).squeeze()
            _edge_index_isi = getattr(d, "edge_index_cs", d.edge_index)
            # Relabeling
            _node_idx[_x_isi] = torch.arange(_x_isi.size(0))
            _edge_index_isi = _node_idx[_edge_index_isi]
            if neg_non_dropout == 1.0 or isi_type == "pos":  # no neg_dropout, or positive
                return _x_isi, _edge_index_isi
            else:  # node drop
                _x_isi, _edge_index_isi, _ = dropout_nodes(_x_isi, _edge_index_isi, p=1.0 - neg_non_dropout)
                return _x_isi, _edge_index_isi

        return ([get_isi_attr(d, "pos") for d in pos_data_list],
                [get_isi_attr(d, "neg") for d in neg_data_list])


def print_obs_stats(data_iters):
    oxi_list = []
    for idx, data in enumerate(tqdm(data_iters)):
        if hasattr(data, "obs_x"):
            oxi_list.append(data.obs_x.size(0))
        elif hasattr(data, "num_obs_x"):
            oxi_list.append(data.num_obs_x.item())
        else:
            raise ValueError
    print({
        "mean": np.mean(oxi_list),
        "std": np.std(oxi_list),
        "min": np.min(oxi_list),
        "max": np.max(oxi_list),
        "median": np.median(oxi_list),
    })


if __name__ == '__main__':
    from data_fntn import FNTN
    from data_sub import HPOMetab, HPONeuro, EMUser
    from pytorch_lightning import seed_everything
    from data_utils import CompleteSubgraph

    PATH = "/mnt/nas2/GNN-DATA"
    DATASET = "EMUser"
    DEBUG = False

    if DATASET == "FNTN":
        SLICE_RANGE = (5, 10)
        KE_METHOD = "edge"
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
        SLICE_RANGE = (2, 7)
        KE_METHOD = "node"
        dataset_instance = HPOMetab(
            root=PATH,
            name="HPOMetab",
            slice_type="random",
            slice_range=SLICE_RANGE,
            num_slices=1,
            pre_transform=None,  # not CompleteSubgraph
            debug=DEBUG,
        )
    elif DATASET == "HPONeuro":
        SLICE_RANGE = (2, 7)
        KE_METHOD = "node"
        dataset_instance = HPONeuro(
            root=PATH,
            name="HPONeuro",
            slice_type="random",
            slice_range=SLICE_RANGE,
            num_slices=1,
            pre_transform=None,  # not CompleteSubgraph
            debug=DEBUG,
        )
    elif DATASET == "EMUser":
        SLICE_RANGE = (6, 11)
        KE_METHOD = "node"
        dataset_instance = EMUser(
            root=PATH,
            name="EMUser",
            slice_type="random",
            slice_range=SLICE_RANGE,
            num_slices=1,
            pre_transform=None,  # not CompleteSubgraph
            debug=DEBUG,
        )
    else:
        raise ValueError

    print_obs_stats(dataset_instance)
    train_data, val_data, test_data = dataset_instance.get_train_val_test()

    cprint("Sampler Test", "green")
    sampler = KHopWithLabelsXESampler(
        dataset_instance.global_data, train_data,
        num_hops=1, use_labels_x=True, use_labels_e=False,
        neg_sample_ratio=1.0, dropout_edges=0.0, balanced_sampling=True,
        obs_x_range=None,
        subgraph_infomax_type=None,  # todo
        no_drop_pos_edges=False,  # todo
        cache_hop_computation=False,
        ke_method=KE_METHOD,
        shuffle=False,
    )
    seed_everything(42)
    for i, b in enumerate(sampler):
        print(i, b)
        if i >= 4:
            break

    sampler = KHopWithLabelsXESampler(
        dataset_instance.global_data, train_data,
        num_hops=1, use_labels_x=True, use_labels_e=False,
        neg_sample_ratio=1.0, dropout_edges=0.9, balanced_sampling=True,
        obs_x_range=SLICE_RANGE,
        subgraph_infomax_type="single",  # todo
        no_drop_pos_edges=True,  # todo
        cache_hop_computation=False,
        ke_method=KE_METHOD,
        shuffle=True,
    )
    seed_everything(42)
    cprint("Train ISI-X-GB no_drop_pos_edges=True", "green")
    for i, b in enumerate(sampler):
        print(i, b)
        if i >= 4:
            break

    sampler = KHopWithLabelsXESampler(
        dataset_instance.global_data, train_data,
        num_hops=1, use_labels_x=True, use_labels_e=False,
        neg_sample_ratio=1.0, dropout_edges=0.3, balanced_sampling=True,
        obs_x_range=SLICE_RANGE,
        subgraph_infomax_type="single",  # todo
        cache_hop_computation=False,
        batch_size=2,  # todo
        ke_method=KE_METHOD,
        shuffle=True,
    )
    seed_everything(42)
    cprint("Train ISI-X-GB multi-batch", "green")
    for i, b in enumerate(sampler):
        print(i, b)
        if i >= 4:
            break

    sampler = KHopWithLabelsXESampler(
        dataset_instance.global_data, train_data,
        num_hops=1, use_labels_x=True, use_labels_e=False,
        neg_sample_ratio=1.0, dropout_edges=0.3, balanced_sampling=True,
        obs_x_range=SLICE_RANGE,
        subgraph_infomax_type="single",  # todo
        cache_hop_computation=False,
        ke_method=KE_METHOD,
        shuffle=True,
    )
    seed_everything(42)
    cprint("Train ISI-X-GB", "green")
    for i, b in enumerate(sampler):
        print(i, b)
        if i >= 4:
            break

    sampler = KHopWithLabelsXESampler(
        dataset_instance.global_data, train_data,
        num_hops=1, use_labels_x=True, use_labels_e=True,  # todo
        neg_sample_ratio=1.0, dropout_edges=0.3, balanced_sampling=True,
        obs_x_range=SLICE_RANGE,
        subgraph_infomax_type=None,  # todo
        cache_hop_computation=False,
        ke_method=KE_METHOD,
        shuffle=True,
    )
    cprint("Train XE first", "green")
    for i, b in enumerate(sampler):
        # Data(edge_index_01=[2, 97688], edge_index_2=[2, 147], labels_e=[440], labels_x=[298],
        #      mask_e_index=[97835], mask_x_index=[7261], obs_x_index=[9], x=[7261], y=[1])
        print(i, b)
        if i >= 4:
            break

    cprint("Train XE second (shuffling test)", "green")
    for i, b in enumerate(sampler):
        print(b)
        if i >= 4:
            break

    sampler = KHopWithLabelsXESampler(
        dataset_instance.global_data, val_data,
        num_hops=1, use_labels_x=False, use_labels_e=False,
        neg_sample_ratio=0.0, dropout_edges=0.0, balanced_sampling=True,
        cache_hop_computation=False,
        ke_method=KE_METHOD,
        shuffle=False,
    )
    cprint("Val", "green")
    for i, b in enumerate(sampler):
        # Data(edge_index_01=[2, 31539], obs_x_index=[8], x=[3029], y=[1])
        print(b)
        if i >= 4:
            break

    sampler = KHopWithLabelsXESampler(
        dataset_instance.global_data, train_data,
        num_hops=0, use_labels_x=False, use_labels_e=False,
        neg_sample_ratio=0.0, dropout_edges=0.0,
        use_obs_edge_only=True,  # this.
        cache_hop_computation=False,
        ke_method=KE_METHOD,
        shuffle=True,
    )
    cprint("WO Sampler", "green")
    for i, b in enumerate(sampler):
        # Data(edge_index_01=[2, 15], x=[10], y=[1])
        print(b)
        if i >= 4:
            break

    cprint("Done!", "green")
