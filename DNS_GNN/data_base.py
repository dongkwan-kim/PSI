import pickle
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
        return {
            "name": self.name,
            "slice_type": self.slice_type,
            "slice_criteria_range": self.slice_range,
            "num_slices": self.num_slices,
            "seed": self.seed,
            "debug": self.debug,
        }

    def _logging_args(self):
        with open(osp.join(self.processed_dir, "args.txt"), "w") as f:
            f.writelines(["{}: {}\n".format(k, v) for k, v in self._get_important_elements().items()])
        cprint("Args logged: ")
        pprint(self._get_important_elements())

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

    def process_with_slice_data_train_val_test(self, data_list):

        data_train, data_val, data_test = self.train_val_test_split(data_list)

        if self.debug:
            data_train = data_train[:5]
            data_val = data_val[:3]
            data_test = data_test[:3]
            cprint("USING DEBUG MODE", "red")

        data_train = self.slice_edge_attr(data_train, is_eval=False)
        cprint("Sliced edge_attr for data_train by [{}, {}], counts: {}".format(
            self.slice_type, self.slice_range, len(data_train)), "green")

        data_val = self.slice_edge_attr(data_val, is_eval=True)
        cprint("Sliced edge_attr for data_val by [{}, {}], counts: {}".format(
            self.slice_type, self.slice_range, len(data_val)), "green")

        data_test = self.slice_edge_attr(data_test, is_eval=True)
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

    def slice_edge_attr(self, data_list: List[Data], is_eval=False):
        """Randomly slice edge_attr by time or number of nodes.
            There can be duplicates.
            e.g.,
                edge_attr = [0., 1., 2., 3., 4.],
                slice_type = "time",
                slice_range = (2., 3.),
                num_slices = 3,

                Then, one of the possible new_edge_attr is
                    [0., 1., 2.] and [0., 1., 2.] and [0., 1., 2., 3.]

        :param data_list: List of Data
        :param is_eval: default False
        :return: data_list with sliced edge_attr
                Data(edge_attr=[new_E, 1], edge_index=[2, E], pergraph_attr=[D], num_obs_x=[1], x=[N, 1], y=[1])
                where new_E is num_edges after slicing
        """
        new_data_list = []
        slice_list = []
        if self.slice_type == 'time':
            for data in tqdm(data_list):
                time_range_mask = (self.slice_range[0] <= data.edge_attr) & (data.edge_attr < self.slice_range[1])
                targets = time_range_mask.squeeze().nonzero().squeeze().tolist()
                if not is_eval:
                    random_slices = np.random.choice(targets, self.num_slices, replace=True)
                else:
                    random_slices = [targets[len(targets) // 2]]
                slice_list.append(random_slices)

        elif self.slice_type == "num_edges":
            for _ in range(len(data_list)):
                targets = range(self.slice_range[0], self.slice_range[1])
                if not is_eval:
                    random_slices = np.random.choice(targets, self.num_slices, replace=True)
                else:
                    random_slices = [targets[len(targets) // 2]]
                slice_list.append(random_slices)

        for data, random_slices in tqdm(zip(data_list, slice_list), total=len(data_list)):
            for idx in random_slices:  # int-iterators
                data = data.clone()
                data.num_obs_x = torch.Tensor([idx]).long()
                new_data_list.append(data)

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

    def __repr__(self):
        return '{}(\n{}\n)'.format(
            self.__class__.__name__,
            "\n".join("\t{}={},".format(k, v) for k, v in self._get_important_elements().items()),
        )
