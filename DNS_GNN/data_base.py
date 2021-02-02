import pickle
from collections import OrderedDict
from itertools import chain
from pprint import pprint
from typing import List, Dict, Tuple

import torch
from termcolor import cprint
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx
import numpy as np
import networkx as nx
import os.path as osp

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from tqdm import tqdm


class DatasetBase(InMemoryDataset):
    """Dataset base class"""

    def __init__(self, root, name,
                 slice_type, slice_range: Tuple[int, int] or Tuple[float, float], num_slices,
                 val_ratio=0.15, test_ratio=0.15, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):

        self.name = name
        self.slice_type = slice_type
        self.slice_range = slice_range
        self.num_slices = num_slices
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.debug = debug
        self.seed = seed

        self.num_train = -1
        self.num_val = -1
        self.global_data = None
        self.vocab = None
        super(DatasetBase, self).__init__(root, transform, pre_transform)

        self.load()
        self.cprint()

    def load(self):
        raise NotImplementedError

    def cprint(self):
        cprint(
            "Initialized: {} (debug={}) \n"
            "/ num_nodes: {}, num_edges: {} \n"
            "/ num_train: {}, num_val: {}, num_test: {}".format(
                self.__class__.__name__, self.debug,
                self.global_data.edge_index.max() + 1, self.global_data.edge_index.size(),
                self.num_train, self.num_val, len(self) - self.num_train - self.num_val),
            "blue",
        )

    @property
    def num_nodes_global(self):
        return self.global_data.edge_index.max().item() + 1

    @property
    def vocab_size(self):
        assert self.vocab is not None
        return len(self.vocab)

    def _get_important_elements(self):
        ie = {
            "name": self.name,
            "slice_type": self.slice_type,
            "slice_criteria_range": self.slice_range,
            "num_slices": self.num_slices,
            "seed": self.seed,
            "debug": self.debug,
        }
        if self.pre_transform is not None:
            ie["pre_transform"] = str(self.pre_transform)
        return ie

    def _logging_args(self):
        with open(osp.join(self.processed_dir, "args.txt"), "w") as f:
            f.writelines(["{}: {}\n".format(k, v) for k, v in self._get_important_elements().items()])
        cprint("Args logged: ")
        pprint(self._get_important_elements())

    def _get_stats(self, stat_names=None, stat_functions=None):
        if stat_names is None:
            stat_names = ['x', 'edge_index']
        if stat_functions is None:
            stat_functions = [
                torch.mean, torch.std,
                torch.min, torch.max, torch.median,
            ]
        stat_dict = OrderedDict()
        for name in stat_names:
            if name in self.slices:
                s_vec = (self.slices[name][1:] - self.slices[name][:-1])
                s_vec = s_vec.float()
                for func in stat_functions:
                    printing_name = "{}/#{}".format(func.__name__, name)
                    printing_value = func(s_vec)
                    stat_dict[printing_name] = printing_value
        s = {
            "num_graphs": len(self),
            "num_train": self.num_train, "num_val": self.num_val,
            "num_test": len(self) - self.num_train - self.num_val,
            "num_classes": self.num_classes,
            "num_global_nodes": self.global_data.edge_index.max() + 1,
            "num_global_edges": self.global_data.edge_index.size(1),
            **stat_dict,
        }
        return s

    @property
    def raw_dir(self):
        return osp.join(self.root, self.__class__.__name__.upper(), 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.__class__.__name__.upper(),
                        'processed_{}'.format("_".join([str(e) for e in self._get_important_elements().values()])))

    @property
    def raw_file_names(self):
        raise NotImplementedError

    @property
    def processed_file_names(self):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError

    def process_with_slice_data_train_val_test(self, data_list_or_triplet: Tuple or List):

        if isinstance(data_list_or_triplet, tuple):
            data_train, data_val, data_test = data_list_or_triplet
        elif isinstance(data_list_or_triplet, list):
            data_train, data_val, data_test = self.train_val_test_split(data_list_or_triplet)
        else:
            raise TypeError("{} is not appropriate type".format(type(data_list_or_triplet)))

        if self.debug:
            data_train = data_train[:5]
            data_val = data_val[:3]
            data_test = data_test[:3]
            cprint("USING DEBUG MODE", "red")

        data_train = self.corrupt_data(data_train, is_eval=False)
        cprint("Sliced edge_attr for data_train by [{}, {}], counts: {}".format(
            self.slice_type, self.slice_range, len(data_train)), "green")

        data_val = self.corrupt_data(data_val, is_eval=True)
        cprint("Sliced edge_attr for data_val by [{}, {}], counts: {}".format(
            self.slice_type, self.slice_range, len(data_val)), "green")

        data_test = self.corrupt_data(data_test, is_eval=True)
        cprint("Sliced edge_attr for data_test by [{}, {}], counts: {}".format(
            self.slice_type, self.slice_range, len(data_test)), "green")

        return data_train, data_val, data_test

    def train_val_test_split(self, data_list):
        num_total = len(data_list)
        num_val = int(num_total * self.val_ratio)
        num_test = int(num_total * self.test_ratio)
        y = np.asarray([int(d.y) for d in data_list])
        data_train_and_val, data_test = train_test_split(
            data_list,
            test_size=num_test, random_state=self.seed, stratify=y,
        )
        y_train_and_val = np.asarray([int(d.y) for d in data_train_and_val])
        data_train, data_val = train_test_split(
            data_train_and_val,
            test_size=num_val, random_state=self.seed, stratify=y_train_and_val,
        )
        return data_train, data_val, data_test

    def corrupt_data(self, data_list: List[Data], is_eval=False):
        """Randomly slice nodes or edges by time, number of nodes, or just random sampling.
            There can be duplicates.
            e.g.,
                edge_attr = [0., 1., 2., 3., 4.],
                slice_type = "time",
                slice_range = (2., 3.),
                num_slices = 3,

        :param data_list: List of Data
            e.g., [..., Data(edge_index=[2, 250], x=[17, 1], y=[1]), ...]
        :param is_eval: default False
        :return: data_list with num_obs_x or obs_x
            e.g., Data(edge_attr=[E, 1], edge_index=[2, E], pergraph_attr=[D], num_obs_x=[1], x=[N, 1], y=[1])
        """
        new_data_list = []

        if self.slice_type == "time" or self.slice_type == "num_edges":

            for data in data_list:

                if self.slice_type == "time":
                    time_range_mask = (self.slice_range[0] <= data.edge_attr) & (data.edge_attr < self.slice_range[1])
                    targets = time_range_mask.squeeze().nonzero().squeeze().tolist()
                elif self.slice_type == "num_edges":
                    targets = range(self.slice_range[0], self.slice_range[1])

                if not is_eval:
                    random_slices = np.random.choice(targets, self.num_slices, replace=True)
                else:
                    random_slices = [targets[len(targets) // 2]]

                for one_slice in random_slices:  # int-iterators
                    new_data = data.clone()
                    new_data.num_obs_x = torch.Tensor([one_slice]).long()
                    new_data_list.append(new_data)

        elif self.slice_type == "random":
            for data in data_list:
                # e.g., Data(edge_index=[2, 250], x=[17, 1], y=[1])
                N = data.x.size(0)
                targets = range(self.slice_range[0], self.slice_range[1])

                if not is_eval:
                    random_slices = np.random.choice(targets, self.num_slices, replace=True)
                else:
                    random_slices = [targets[len(targets) // 2]]

                for one_slice in random_slices:  # int-iterators
                    new_data = data.clone()
                    new_data.obs_x = torch.randperm(N)[:one_slice]
                    new_data_list.append(new_data)

        else:
            raise ValueError("{} is not appropriate slice_type".format(self.slice_type))

        return new_data_list

    def tolist(self):
        return list(self)

    def get_train_val_test(self):
        data_list = self.tolist()
        num_train_and_val = self.num_train + self.num_val
        data_train = data_list[:self.num_train]
        data_val = data_list[self.num_train:num_train_and_val]
        data_test = data_list[num_train_and_val:]
        return data_train, data_val, data_test

    def print_summary(self):

        def out(v):
            return str(float(v)) if isinstance(v, torch.Tensor) else str(v)

        print("---------------------------------------------")
        for k, v in chain(self._get_important_elements().items(),
                          self._get_stats().items()):
            print("{:>20}{:>25}".format(k, out(v)))
        print("---------------------------------------------")

    def __repr__(self):
        return '{}(\n{}\n)'.format(
            self.__class__.__name__,
            "\n".join("\t{}={},".format(k, v) for k, v in self._get_important_elements().items()),
        )
