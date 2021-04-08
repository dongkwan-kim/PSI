import pickle
from pprint import pprint
from typing import List, Dict, Tuple

import torch
from termcolor import cprint
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx
import networkx as nx
from tqdm import tqdm

from data_base import DatasetBase
from data_utils import CompleteSubgraph


def load_propagation_graphs_data_split(paths, **kwargs) -> (List[Data], Dict[int, str]):
    propagation = pickle.load(open(paths[1], "rb"))

    data_train = load_propagation_graphs(propagation, 'train')
    data_val = load_propagation_graphs(propagation, 'dev')
    data_test = load_propagation_graphs(propagation, 'test')

    return data_train, data_val, data_test


def load_propagation_graphs(file, file_type, **kwargs) -> (List[Data], Dict[int, str]):
    x_index_list = file[file_type]["x_index_list"]
    edge_index_list = file[file_type]["edge_index_list"]
    labels = file[file_type]["ys1_list"]  # changeable ['ys1_list', 'ys2_list']

    data_list = []
    for x_index, edge_index, y in tqdm(zip(x_index_list,
                                           edge_index_list,
                                           labels),
                                       total=len(labels)):
        x_index = torch.Tensor(x_index).long().view(-1, 1)  # [N, 1]
        sorted_edge_index = torch.Tensor(edge_index).long()  # [2, E]
        y = torch.Tensor([y]).long()  # [1]
        data = Data(x=x_index,
                    edge_index=sorted_edge_index,
                    y=y)
        data_list.append(data)
    return data_list


class ICTN(DatasetBase):
    """Dataset of Fake News Twitter Network"""

    def __init__(self, root, name,
                 slice_type, slice_range: Tuple[int, int] or Tuple[float, float], num_slices,
                 val_ratio=0.15, test_ratio=0.15, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):

        assert name in ["0.0", "0.001", "0.002", "0.003", "0.004"]
        assert slice_type in ["time", "num_edges"]
        super(ICTN, self).__init__(
            root, name, slice_type, slice_range, num_slices, val_ratio, test_ratio, debug, seed,
            transform, pre_transform, **kwargs,
        )

    def load(self):
        self.data, self.slices = torch.load(self.processed_paths[0])

        meta = torch.load(self.processed_paths[1])
        self.num_train = int(meta[0])
        self.num_val = int(meta[1])

        self.global_data = torch.load(self.raw_paths[2])
        self.global_data.edge_index = self.global_data.edge_index[[1, 0]]

    @property
    def raw_file_names(self):
        return ["idx_network_{}.gpickle".format(self.name),
                "idx_propagation_{}.pkl".format(self.name),
                "user_network_data_{}.pt".format(self.name)]

    @property
    def processed_file_names(self):
        return ["data.pt", "meta.pt"]

    def download(self):
        print("Please download: {} at {}".format(self.raw_file_names[:2], self.raw_dir))
        print("Now save_global_data is performed to save {}".format(self.raw_file_names[2]))
        self.save_global_data()

    def save_global_data(self):
        global_graph: nx.DiGraph = nx.read_gpickle(self.raw_paths[0])
        cprint("Loaded global_graph at {}".format(self.raw_paths[0]), "green")
        print("\t- num_nodes: {},\n\t- num_edges: {}".format(global_graph.number_of_nodes(),
                                                             global_graph.number_of_edges()))
        global_data = from_networkx(global_graph)
        cprint("Converted global_graph to PyG format", "green")
        torch.save(global_data, self.raw_paths[2])
        cprint("Saved global_data at {}".format(self.raw_paths[2]), "green")

    def process(self):
        global_data: Data = torch.load(self.raw_paths[2])
        cprint("Loaded global_data at {}".format(self.raw_paths[2]), "green")
        print("\t- {}".format(global_data))

        data_list = load_propagation_graphs_data_split(self.raw_paths)
        cprint("Loaded data_list at {}".format(self.raw_paths[1]), "green")
        print("\t- num_graphs: {}".format(len(data_list)))

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
        torch.save(torch.as_tensor([self.num_train, self.num_val]).long(), self.processed_paths[1])

        cprint("Saved data at {}".format(self.processed_paths), "green")

        self._logging_args()


if __name__ == '__main__':

    PATH = "/mnt/nas2/GNN-DATA"
    DEBUG = False

    ictn_kwargs = dict(
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

    ictn = ICTN(**ictn_kwargs)
    ictn.save_global_data()

    train_ictn, val_ictn, test_ictn = ictn.get_train_val_test()

    print("Train")
    for b in train_ictn:
        print(b, b.num_obs_x)
        print("------")

    print("Test")
    for b in test_ictn:
        print(b, b.num_obs_x)
        break

    ictn.print_summary()
