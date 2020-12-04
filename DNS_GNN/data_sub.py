import pickle
from pprint import pprint
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer
from termcolor import cprint
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx, subgraph
import numpy as np
import networkx as nx
import os.path as osp

from tqdm import tqdm

from data_base import DatasetBase


def read_subgnn_data(edge_list_path, subgraph_path, embedding_path):
    """
    Read in the subgraphs & their associated labels
    Reference: https://github.com/mims-harvard/SubGNN/blob/main/SubGNN/SubGNN.py#L519
    """
    # read list of node ids for each subgraph & their labels
    train_nodes, _train_ys, val_nodes, _val_ys, test_nodes, _test_ys = read_subgraphs(subgraph_path)
    cprint("Loaded subgraphs at {}".format(subgraph_path), "green")

    # check if the dataset is multilabel (e.g. HPO-NEURO)
    if type(_train_ys) == list:
        all_labels = _train_ys + _val_ys + _test_ys
        mlb = MultiLabelBinarizer()
        mlb.fit(all_labels)
        train_sub_ys = torch.Tensor(mlb.transform(_train_ys)).long()
        val_sub_ys = torch.Tensor(mlb.transform(_val_ys)).long()
        test_sub_ys = torch.Tensor(mlb.transform(_test_ys)).long()
    else:
        train_sub_ys, val_sub_ys, test_sub_ys = _train_ys, _val_ys, _test_ys

    # Initialize pretrained node embeddings
    xs = torch.load(embedding_path)  # feature matrix should be initialized to the node embeddings
    # xs_with_zp = torch.cat([torch.zeros(1, xs.shape[1]), xs], 0)  # there's a zeros in the first index for padding
    cprint("Loaded embeddings at {}".format(embedding_path), "green")

    # read networkx graph from edge list
    global_nxg: nx.Graph = nx.read_edgelist(edge_list_path)
    cprint("Loaded global_graph at {}".format(edge_list_path), "green")
    global_data = from_networkx(global_nxg)
    cprint("Converted global_graph to PyG format", "green")
    global_data.x = xs

    train_data_list = get_data_list_from_subgraphs(global_data.edge_index, train_nodes, train_sub_ys)
    cprint("Converted train_subgraph to PyG format", "green")
    val_data_list = get_data_list_from_subgraphs(global_data.edge_index, val_nodes, val_sub_ys)
    cprint("Converted val_subgraph to PyG format", "green")
    test_data_list = get_data_list_from_subgraphs(global_data.edge_index, test_nodes, test_sub_ys)
    cprint("Converted test_subgraph to PyG format", "green")
    return global_data, train_data_list, val_data_list, test_data_list


def get_data_list_from_subgraphs(global_edge_index, sub_nodes: List[List[int]], sub_ys):
    data_list = []
    for x_index, y in zip(sub_nodes, tqdm(sub_ys)):
        x_index = torch.Tensor(x_index).long().view(-1, 1)
        if len(y.size()) == 0:
            y = torch.Tensor([y]).long()
        else:
            y = y.long()
        edge_index, _ = subgraph(x_index, global_edge_index, relabel_nodes=False)
        data = Data(x=x_index, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list


def read_subgraphs(subgraph_path):
    """
    Read subgraphs from file

    Args
       - sub_f (str): filename where subgraphs are stored

    Return for each train, val, test split:
       - sub_G (list): list of nodes belonging to each subgraph
       - sub_G_label (list): labels for each subgraph
    """

    # Enumerate/track labels
    label_idx = 0
    labels = {}

    # Train/Val/Test subgraphs
    train_sub_g, val_sub_g, test_sub_g = [], [], []

    # Train/Val/Test subgraph labels
    train_sub_y, val_sub_y, test_sub_y = [], [], []

    # Train/Val/Test masks
    train_mask, val_mask, test_mask = [], [], []

    multilabel = False

    # Parse data
    with open(subgraph_path) as fin:
        subgraph_idx = 0
        for line in fin:
            nodes = [int(n) for n in line.split("\t")[0].split("-") if n != ""]
            if len(nodes) != 0:
                if len(nodes) == 1:
                    print("G with one node: ", nodes)
                l = line.split("\t")[1].split("-")
                if len(l) > 1:
                    multilabel = True
                for lab in l:
                    if lab not in labels.keys():
                        labels[lab] = label_idx
                        label_idx += 1
                if line.split("\t")[2].strip() == "train":
                    train_sub_g.append(nodes)
                    train_sub_y.append([labels[lab] for lab in l])
                    train_mask.append(subgraph_idx)
                elif line.split("\t")[2].strip() == "val":
                    val_sub_g.append(nodes)
                    val_sub_y.append([labels[lab] for lab in l])
                    val_mask.append(subgraph_idx)
                elif line.split("\t")[2].strip() == "test":
                    test_sub_g.append(nodes)
                    test_sub_y.append([labels[lab] for lab in l])
                    test_mask.append(subgraph_idx)
                subgraph_idx += 1

    if not multilabel:
        train_sub_y = torch.tensor(train_sub_y).long().squeeze()
        val_sub_y = torch.tensor(val_sub_y).long().squeeze()
        test_sub_y = torch.tensor(test_sub_y).long().squeeze()

    if len(val_mask) < len(test_mask):
        return train_sub_g, train_sub_y, test_sub_g, test_sub_y, val_sub_g, val_sub_y

    return train_sub_g, train_sub_y, val_sub_g, val_sub_y, test_sub_g, test_sub_y


class DatasetSubGNN(DatasetBase):
    """Dataset of Human Phenotype Ontology Disease - NEURO/METAB"""

    def __init__(self, root, name,
                 slice_type, slice_range: Tuple[int, int] or Tuple[float, float], num_slices,
                 val_ratio=0.15, test_ratio=0.15, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):
        assert slice_type in ["random"]
        super(DatasetSubGNN, self).__init__(
            root, name, slice_type, slice_range, num_slices, val_ratio, test_ratio, debug, seed,
            transform, pre_transform, **kwargs,
        )

    def load(self):
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.global_data = torch.load(self.processed_paths[1])
        meta = torch.load(self.processed_paths[2])
        self.num_train = int(meta[0])
        self.num_val = int(meta[1])

    @property
    def raw_file_names(self):
        return ["edge_list.txt", "subgraphs.pth", "graphsaint_gcn_embeddings.pth"]

    @property
    def processed_file_names(self):
        return ["data.pt", "global.pt", "meta.pt"]

    def download(self):
        raise FileNotFoundError("Please download: {} at {} from {}".format(
            self.raw_file_names, self.raw_dir, "https://github.com/mims-harvard/SubGNN",
        ))

    def process(self):
        global_data, data_train, data_val, data_test = read_subgnn_data(*self.raw_paths)

        data_train, data_val, data_test = self.process_with_slice_data_train_val_test(
            (data_train, data_val, data_test),
        )

        data_total = data_train + data_val + data_test
        torch.save(self.collate(data_total), self.processed_paths[0])
        cprint("Saved data at {}".format(self.processed_paths[0]), "green")
        torch.save(global_data, self.processed_paths[1])
        cprint("Saved global_data at {}".format(self.processed_paths[1]), "green")

        self.num_train = len(data_train)
        self.num_val = len(data_val)
        torch.save(torch.as_tensor([self.num_train, self.num_val]).long(), self.processed_paths[2])

        self._logging_args()


class HPONeuro(DatasetSubGNN):

    def __init__(self, root, name,
                 slice_type, slice_range: Tuple[int, int] or Tuple[float, float], num_slices,
                 val_ratio=0.15, test_ratio=0.15, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):
        super(HPONeuro, self).__init__(
            root, name, slice_type, slice_range, num_slices, val_ratio, test_ratio, debug, seed,
            transform, pre_transform, **kwargs,
        )

    def download(self):
        super().download()

    def process(self):
        super().process()


class HPOMetab(DatasetSubGNN):

    def __init__(self, root, name,
                 slice_type, slice_range: Tuple[int, int] or Tuple[float, float], num_slices,
                 val_ratio=0.15, test_ratio=0.15, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):
        super(HPOMetab, self).__init__(
            root, name, slice_type, slice_range, num_slices, val_ratio, test_ratio, debug, seed,
            transform, pre_transform, **kwargs,
        )

    def download(self):
        super().download()

    def process(self):
        super().process()


if __name__ == '__main__':

    TYPE = "HPOMetab"

    PATH = "/mnt/nas2/GNN-DATA"
    DEBUG = True

    if TYPE == "HPONeuro":  # multi-label
        dts = HPONeuro(
            root=PATH,
            name="HPO",
            slice_type="random",
            slice_range=(5, 10),
            num_slices=5,
            val_ratio=0.15,
            test_ratio=0.15,
            debug=DEBUG,
        )
    elif TYPE == "HPOMetab":
        dts = HPOMetab(
            root=PATH,
            name="HPO",
            slice_type="random",
            slice_range=(5, 10),
            num_slices=5,
            val_ratio=0.15,
            test_ratio=0.15,
            debug=DEBUG,
        )
    else:
        raise ValueError

    train_dts, val_dts, test_dts = dts.get_train_val_test()

    for b in train_dts:
        print(b)
