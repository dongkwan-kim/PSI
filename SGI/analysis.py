import random
from multiprocessing import Pool
from pprint import pprint
from typing import List, Dict, Tuple, Callable, Any

import os
from termcolor import cprint
from torch_geometric.data import Batch
from torch_geometric.utils import degree, to_networkx

import networkx as nx
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from data_sampler import KHopWithLabelsXESampler
from data_utils import CompleteSubgraph, DataPN


# See https://networkx.org/documentation/stable/reference/algorithms/index.html


def print_degree_assortativity_coefficient(*args):
    o = nx.degree_assortativity_coefficient(*args)
    print(o)
    return o


def print_average_clustering(*args):
    o = nx.average_clustering(*args)
    print(o)
    return o


def print_table(name_to_values_list: Dict[str, List[Dict[str, float]]]):
    for i, (name, values_list) in enumerate(name_to_values_list.items()):
        print("\t".join([name, *[str(v) for v in values_list[0].values()]]))


def my_to_networkx(data, *args, **kwargs):
    _data = data.clone()
    if hasattr(_data, "edge_index_01"):
        ei = _data.edge_index_01
        _data.edge_index = ei
        del _data.edge_index_01
    return to_networkx(_data, *args, **kwargs)


def analyze_network(nxg_list: List[nx.Graph],
                    name_to_function_and_args: Dict[str, Tuple[Callable, List[Any]]],
                    summary_function_list: List[Callable] = None):
    # example of name_to_function_and_args
    # {"density": (get_density, [])}

    if summary_function_list is None:
        summary_function_list = [np.mean, np.std]

    print("Analyze networks ...")
    name_to_values_list = dict()
    for name, (_func, _args) in tqdm(name_to_function_and_args.items()):
        _nxg_values_1 = _func(nxg_list, *_args)
        name_to_values_list[name] = [
            {_sum_func.__name__: _sum_func(_nxg_values_1) for _sum_func in summary_function_list},
        ]
        pprint(name_to_values_list)

    return name_to_values_list


def iterate_for_nxgs(nxg_list: List[nx.Graph], function: Callable, *args):
    with Pool(min(len(nxg_list), os.cpu_count())) as p:
        ret = p.starmap(function, [[nxg, *args] for nxg in nxg_list if nxg.number_of_edges() > 0])
    return ret


if __name__ == '__main__':
    from data_fntn import FNTN
    from data_sub import HPOMetab, HPONeuro, EMUser
    from pytorch_lightning import seed_everything

    L = 50

    KHOP_OR_ALL = "KHOP"  # KHOP, ALL
    DATASET = "HPOMetab"  # FNTN, EMUser, HPOMetab

    PATH = "/mnt/nas2/GNN-DATA"
    DEBUG = False

    if DATASET == "FNTN":
        SLICE_RANGE = (5, 10)
        MAX_SLICE_RANGE = (3089, 3090)
        KE_METHOD = "edge"
        kws = dict(
            root=PATH,
            name="0.0",  # 0.0 0.001 0.002 0.003 0.004
            slice_type="num_edges",
            num_slices=1,
            val_ratio=0.15,
            test_ratio=0.15,
            pre_transform=CompleteSubgraph(),
            debug=DEBUG,
        )
        dataset_instance = FNTN(slice_range=SLICE_RANGE, **kws)
        max_dataset_instance = FNTN(slice_range=MAX_SLICE_RANGE, **kws)

    elif DATASET == "EMUser":
        SLICE_RANGE = (6, 11)
        MAX_SLICE_RANGE = (499, 500)
        KE_METHOD = "node"
        kws = dict(
            root=PATH,
            name="EMUser",
            slice_type="random",
            num_slices=1,
            pre_transform=None,  # not CompleteSubgraph
            debug=DEBUG,
        )
        dataset_instance = EMUser(slice_range=SLICE_RANGE, **kws)
        max_dataset_instance = EMUser(slice_range=MAX_SLICE_RANGE, **kws)

    elif DATASET == "HPOMetab":
        SLICE_RANGE = (2, 7)
        MAX_SLICE_RANGE = (50, 51)
        KE_METHOD = "node"
        kws = dict(
            root=PATH,
            name="HPOMetab",
            slice_type="random",
            num_slices=1,
            pre_transform=None,  # not CompleteSubgraph
            debug=DEBUG,
        )
        dataset_instance = HPOMetab(slice_range=SLICE_RANGE, **kws)
        max_dataset_instance = HPOMetab(slice_range=MAX_SLICE_RANGE, **kws)

    else:
        raise ValueError

    train_data, val_data, test_data = dataset_instance.get_train_val_test()
    max_train_data, max_val_data, max_test_data = max_dataset_instance.get_train_val_test()

    sampler_kws = dict(
        use_labels_x=False, use_labels_e=False,
        neg_sample_ratio=0.0, dropout_edges=0.0,
        cache_hop_computation=False, ke_method=KE_METHOD, shuffle=False,
        num_workers=40,
    )

    random.seed(L)

    if KHOP_OR_ALL == "KHOP":

        sub_data = [*train_data, *val_data, *test_data]
        random.shuffle(sub_data)
        sub_data = sub_data[:L]

        sampler = KHopWithLabelsXESampler(
            dataset_instance.global_data,
            sub_data,
            num_hops=1,  # this.
            **sampler_kws,
        )
        cprint("Building k-hop subgraphs ...", "green")
        # DataPN(edge_index_01=[2, 216164], obs_x_index=[10], x=[3905], y=[1])
        lst: List[nx.Graph] = [my_to_networkx(b) for b in tqdm(sampler)]
        results_1 = analyze_network(
            lst,
            name_to_function_and_args={
                "degree_assortativity_coefficient": (iterate_for_nxgs, [print_degree_assortativity_coefficient]),
                "average_clustering": (iterate_for_nxgs, [print_average_clustering]),
                "density": (iterate_for_nxgs, [nx.density]),
            },
            summary_function_list=[np.nanmean, np.nanstd],
        )
        print_table(results_1)

    else:

        max_sub_data = [*max_train_data, *max_val_data, *max_test_data]
        random.shuffle(max_sub_data)
        max_sub_data = max_sub_data[:L]

        sampler = KHopWithLabelsXESampler(
            dataset_instance.global_data,
            max_sub_data,
            num_hops=0,  # this.
            **sampler_kws,
        )
        cprint("Building entire subgraphs ...", "green")
        max_lst: List[nx.Graph] = [my_to_networkx(b) for b in tqdm(sampler)]

        results_2 = analyze_network(
            max_lst,
            name_to_function_and_args={
                "degree_assortativity_coefficient": (iterate_for_nxgs, [print_degree_assortativity_coefficient]),
                "average_clustering": (iterate_for_nxgs, [nx.average_clustering]),
                "density": (iterate_for_nxgs, [nx.density]),
            },
            summary_function_list=[np.nanmean, np.nanstd],
        )
        print_table(results_2)
