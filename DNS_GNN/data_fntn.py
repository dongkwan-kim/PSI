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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from tqdm import tqdm


def preprocess_text(text_list: List[str], **kwargs) -> (torch.Tensor, Dict[int, str]):
    """
    :param text_list:
    :return: Tuple of LongTensor of [G, max_len] and vocabulary
    """
    encoded_texts = []

    vectorizer = CountVectorizer(stop_words="english")
    vectorizer.fit(text_list)
    vocab = vectorizer.vocabulary_
    preprocess_and_tokenize = vectorizer.build_analyzer()

    for text in text_list:
        tokens = preprocess_and_tokenize(text)
        indexed_tokens = [vocab[token] + 1 for token in tokens if token in vocab]
        encoded_texts.append(indexed_tokens)

    max_len = kwargs['max_len'] if 'max_len' in kwargs.keys() else max(len(et) for et in encoded_texts)

    pad_encoded_texts = torch.zeros([len(text_list), max_len], dtype=torch.int32)
    for idx, et in enumerate(encoded_texts):
        length = len(et) if len(et) <= max_len else max_len
        pad_encoded_texts[idx, :length] = torch.tensor(et[:length])

    return pad_encoded_texts, vocab


def load_propagation_graphs_and_preprocess_text(paths, **kwargs) -> (List[Data], Dict[int, str]):
    propagation = pickle.load(open(paths[1], "rb"))
    story = pickle.load(open(paths[2], "rb"))

    x_index_list = propagation["x_index_list"]
    edge_index_list = propagation["edge_index_list"]
    edge_attr_list = propagation["edge_attr_list"]
    text_list = story["text_list"]
    labels = story["labels"]

    text_tensor, vocab = preprocess_text(text_list, **kwargs)

    data_list = []
    for x_index, edge_index, edge_attr, global_attr, y in tqdm(zip(x_index_list,
                                                                   edge_index_list,
                                                                   edge_attr_list,
                                                                   text_tensor,
                                                                   labels),
                                                               total=len(labels)):
        x_index = torch.Tensor(x_index).long().view(-1, 1)  # [N, 1]
        edge_attr = torch.Tensor(edge_attr).float().view(-1, 1)  # [E, 1]

        sorted_edge_attr, sorted_indices = torch.sort(edge_attr.squeeze())
        sorted_edge_attr = sorted_edge_attr.unsqueeze(1)
        sorted_edge_index = torch.Tensor(edge_index).long()[:, sorted_indices]  # [2, E]

        y = torch.Tensor([y]).long()  # [1]
        data = Data(x=x_index,
                    edge_index=sorted_edge_index, edge_attr=sorted_edge_attr,
                    y=y, global_attr=global_attr)
        data_list.append(data)
    return data_list, vocab


class FNTN(InMemoryDataset):
    """Dataset of Fake News Twitter Network"""

    def __init__(self, root, name,
                 slice_type, slice_range: Tuple[int, int] or Tuple[float, float], num_slices,
                 val_ratio=0.15, test_ratio=0.15, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):

        self.name = name
        assert self.name in ["0.0", "0.001", "0.002", "0.003", "0.004"]

        self.slice_type = slice_type
        assert self.slice_type in ["time", "num_edges"]

        # (float, float) for time, (int, int) for num_edges
        self.slice_range = slice_range
        self.num_slices = num_slices
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.debug = debug
        self.seed = seed
        super(FNTN, self).__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.vocab = torch.load(self.processed_paths[1])

        meta = torch.load(self.processed_paths[2])
        self.num_train = int(meta[0])
        self.num_val = int(meta[1])

        self.global_data = torch.load(self.raw_paths[3])
        self.global_data.edge_index = self.global_data.edge_index[[1, 0]]

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

    @property
    def raw_dir(self):
        return osp.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.__class__.__name__,
                        'processed_{}'.format("_".join([str(e) for e in self._get_important_elements().values()])))

    @property
    def raw_file_names(self):
        return ["idx_network_{}.gpickle".format(self.name),
                "idx_propagation_{}.pkl".format(self.name),
                "story.pkl",
                "user_network_data_{}.pt".format(self.name)]

    @property
    def processed_file_names(self):
        return ["data.pt", "vocab.pt", "meta.pt"]

    def download(self):
        print("Please download: {} at {}".format(self.raw_file_names[:3], self.raw_dir))
        print("Now save_global_data is performed to save {}".format(self.raw_file_names[3]))
        self.save_global_data()

    def save_global_data(self):
        global_graph: nx.DiGraph = nx.read_gpickle(self.raw_paths[0])
        cprint("Loaded global_graph at {}".format(self.raw_paths[0]), "green")
        print("\t- num_nodes: {},\n\t- num_edges: {}".format(global_graph.number_of_nodes(),
                                                             global_graph.number_of_edges()))
        global_data = from_networkx(global_graph)
        cprint("Converted global_graph to PyG format", "green")
        torch.save(global_data, self.raw_paths[3])
        cprint("Saved global_data at {}".format(self.raw_paths[3]), "green")

    def process(self):
        global_data: Data = torch.load(self.raw_paths[3])
        cprint("Loaded global_data at {}".format(self.raw_paths[3]), "green")
        print("\t- {}".format(global_data))

        data_list, vocab = load_propagation_graphs_and_preprocess_text(self.raw_paths)
        cprint("Loaded data_list at {}".format(self.raw_paths[1:3]), "green")
        print("\t- num_graphs: {}".format(len(data_list)))
        print("\t- num_vocab: {}".format(len(vocab)))

        data_train, data_val, data_test = self.train_val_test_split(data_list)

        if self.debug:
            data_train = data_train[:5]
            data_val = data_val[:3]
            data_test = data_test[:3]
            cprint("USING DEBUG MODE", "red")

        data_train = self.slice_edge_attr_by_time_or_num_edges(data_train, is_not_train=False)
        self.num_train = len(data_train)
        cprint("Sliced edge_attr for data_train by [{}, {}], counts: {}".format(
            self.slice_type, self.slice_range, self.num_train), "green")

        data_val = self.slice_edge_attr_by_time_or_num_edges(data_val, is_not_train=True)
        self.num_val = len(data_val)
        cprint("Sliced edge_attr for data_val by [{}, {}], counts: {}".format(
            self.slice_type, self.slice_range, self.num_val), "green")

        data_test = self.slice_edge_attr_by_time_or_num_edges(data_test, is_not_train=True)
        cprint("Sliced edge_attr for data_test by [{}, {}], counts: {}".format(
            self.slice_type, self.slice_range, len(data_test)), "green")

        data_total = data_train + data_val + data_test
        torch.save(self.collate(data_total), self.processed_paths[0])
        torch.save(vocab, self.processed_paths[1])
        torch.save(torch.as_tensor([self.num_train, self.num_val]).long(), self.processed_paths[2])
        cprint("Saved data at {}".format(self.processed_paths), "green")

        self._logging_args()

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

    def slice_edge_attr_by_time_or_num_edges(self, data_list: List[Data], is_not_train=False):
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
        :param is_not_train: default False
        :return: data_list with sliced edge_attr
                Data(edge_attr=[new_E, 1], edge_index=[2, E], global_attr=[D], num_obs_x=[1], x=[N, 1], y=[1])
                where new_E is num_edges after slicing
        """
        new_data_list = []
        slice_list = []
        if self.slice_type == 'time':
            for data in tqdm(data_list):
                time_range_mask = (self.slice_range[0] <= data.edge_attr) & (data.edge_attr < self.slice_range[1])
                targets = time_range_mask.squeeze().nonzero().squeeze().tolist()
                if not is_not_train:
                    random_slices = np.random.choice(targets, self.num_slices, replace=True)
                else:
                    random_slices = [targets[len(targets) // 2]]
                slice_list.append(random_slices)

        elif self.slice_type == "num_edges":
            for _ in range(len(data_list)):
                targets = range(self.slice_range[0], self.slice_range[1])
                if not is_not_train:
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


if __name__ == '__main__':

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

    for b in train_fntn:
        print(b, b.num_obs_x)
