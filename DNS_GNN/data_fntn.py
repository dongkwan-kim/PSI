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

from data_base import DatasetBase
from data_transform import CompleteSubgraph


def preprocess_text(text_list: List[str], repr_type="tfidf", **kwargs) -> (torch.Tensor, Dict[int, str]):
    """
    :param text_list:
    :param repr_type:
    :return: Tuple of LongTensor of [G, max_len] and vocabulary
    """
    encoded_texts = []

    if repr_type == "tfidf":
        vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
        vectorizer.fit(text_list)
        vocab = vectorizer.vocabulary_
        texts = vectorizer.transform(text_list).toarray()
        texts = torch.Tensor(texts).float()
    else:
        vectorizer = CountVectorizer(stop_words="english")
        vectorizer.fit(text_list)
        vocab = vectorizer.vocabulary_
        preprocess_and_tokenize = vectorizer.build_analyzer()

        for text in text_list:
            tokens = preprocess_and_tokenize(text)
            indexed_tokens = [vocab[token] + 1 for token in tokens if token in vocab]
            encoded_texts.append(indexed_tokens)

        max_len = kwargs['max_len'] if 'max_len' in kwargs.keys() else max(len(et) for et in encoded_texts)

        texts = torch.zeros([len(text_list), max_len], dtype=torch.int32)
        for idx, et in enumerate(encoded_texts):
            length = len(et) if len(et) <= max_len else max_len
            texts[idx, :length] = torch.tensor(et[:length])
    return texts, vocab


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
    for x_index, edge_index, edge_attr, pergraph_attr, y in tqdm(zip(x_index_list,
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
                    y=y, pergraph_attr=pergraph_attr)
        data_list.append(data)
    return data_list, vocab


class FNTN(DatasetBase):
    """Dataset of Fake News Twitter Network"""

    def __init__(self, root, name,
                 slice_type, slice_range: Tuple[int, int] or Tuple[float, float], num_slices,
                 val_ratio=0.15, test_ratio=0.15, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):

        assert name in ["0.0", "0.001", "0.002", "0.003", "0.004"]
        assert slice_type in ["time", "num_edges"]
        super(FNTN, self).__init__(
            root, name, slice_type, slice_range, num_slices, val_ratio, test_ratio, debug, seed,
            transform, pre_transform, **kwargs,
        )

    def load(self):
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.vocab = torch.load(self.processed_paths[1])

        meta = torch.load(self.processed_paths[2])
        self.num_train = int(meta[0])
        self.num_val = int(meta[1])

        self.global_data = torch.load(self.raw_paths[3])
        self.global_data.edge_index = self.global_data.edge_index[[1, 0]]

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

        data_train, data_val, data_test = self.process_with_slice_data_train_val_test(data_list)

        data_total = data_train + data_val + data_test
        if self.pre_transform is not None:
            if isinstance(self.pre_transform, CompleteSubgraph):
                self.pre_transform.global_edge_index = global_data.edge_index[[1, 0]]
            data_total = [self.pre_transform(d) for d in tqdm(data_total)]
            cprint("Pre-transformed: {}".format(self.pre_transform), "green")

        torch.save(self.collate(data_total), self.processed_paths[0])

        self.num_train = len(data_train)
        self.num_val = len(data_val)
        torch.save(torch.as_tensor([self.num_train, self.num_val]).long(), self.processed_paths[2])

        torch.save(vocab, self.processed_paths[1])
        cprint("Saved data at {}".format(self.processed_paths), "green")

        self._logging_args()


if __name__ == '__main__':

    PATH = "/mnt/nas2/GNN-DATA"
    DEBUG = False

    fntn_kwargs = dict(
        root=PATH,
        name="0.0",  # 0.0 0.001 0.002 0.003 0.004
        slice_type="num_edges",
        slice_range=(5, 10),
        num_slices=1,
        val_ratio=0.15,
        test_ratio=0.15,
        debug=DEBUG,
        seed=42,
    )

    fntn = FNTN(**fntn_kwargs)
    fntn_cs = FNTN(pre_transform=CompleteSubgraph(), **fntn_kwargs)

    train_fntn, val_fntn, test_fntn = fntn.get_train_val_test()
    train_fntn_cs, val_fntn_cs, test_fntn_cs = fntn_cs.get_train_val_test()

    print("Train")
    for b, b_cs in zip(train_fntn, train_fntn_cs):
        print(b, b.num_obs_x)
        print(b_cs, b_cs.num_obs_x)
        print("------")

    print("Test")
    for b, b_cs in zip(test_fntn, test_fntn_cs):
        print(b, b.num_obs_x)
        print(b_cs, b_cs.num_obs_x)
        exit()
