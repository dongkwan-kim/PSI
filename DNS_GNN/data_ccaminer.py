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


def load_cascades_train_val_test(paths, **kwargs) -> (List[Data], Dict[int, str]):
    cascades = pickle.load(open(paths[1], "rb"))
    data_train = load_cascades(cascades, 'train')
    data_val = load_cascades(cascades, 'dev')
    data_test = load_cascades(cascades, 'test')
    return data_train, data_val, data_test


def load_cascades(file, file_type, **kwargs) -> (List[Data], Dict[int, str]):
    x_index_list = file[file_type]["x_index_list"]
    edge_index_list = file[file_type]["edge_index_list"]
    labels_1 = file[file_type]["ys1_list"]
    labels_2 = file[file_type]["ys2_list"]

    data_list = []
    for x_index, edge_index, y1, y2 in tqdm(zip(x_index_list,
                                                edge_index_list,
                                                labels_1,
                                                labels_2),
                                            total=len(labels_1)):
        x_index = torch.Tensor(x_index).long().view(-1, 1)  # [N, 1]
        edge_index = torch.Tensor(edge_index).long()  # [2, E]
        y = torch.Tensor([y1, y2]).long().view(1, -1)  # [1, 2]
        data = Data(x=x_index, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list


class CCAMiner(DatasetBase):
    """Dataset of Citation Cascades from ArnetMiner published in
       DeepCas: an End-to-end Predictor of Information Cascades (WWW 2017)"""

    def __init__(self, root, name,
                 slice_type, slice_range: Tuple[int, int] or Tuple[float, float], num_slices,
                 val_ratio=0.15, test_ratio=0.15, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):

        assert name in ["y1", "y2"]
        assert slice_type in ["num_edges"]
        super(CCAMiner, self).__init__(
            root, name, slice_type, slice_range, num_slices, val_ratio, test_ratio, debug, seed,
            transform, pre_transform, **kwargs,
        )

    def load(self):
        self.data, self.slices = torch.load(self.processed_paths[0])

        y_idx = 0 if self.name == "y1" else 1
        self.data.y = self.data.y[:, y_idx]

        meta = torch.load(self.processed_paths[1])
        self.num_train = int(meta[0])
        self.num_val = int(meta[1])

        self.global_data = torch.load(self.raw_paths[2])

    @property
    def raw_file_names(self):
        return ["idx_network.gpickle", "idx_cascade.pkl", "network_data.pt"]

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

        data_train, data_val, data_test = load_cascades_train_val_test(self.raw_paths)
        cprint("Loaded data_list at {}".format(self.raw_paths[1]), "green")
        print("\t- num_graphs: {}".format(len(data_train + data_val + data_test)))

        data_train, data_val, data_test = self.process_with_slice_data_train_val_test(
            (data_train, data_val, data_test),
        )

        data_total = data_train + data_val + data_test
        if self.pre_transform is not None:
            if isinstance(self.pre_transform, CompleteSubgraph):
                self.pre_transform.global_edge_index = global_data.edge_index
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

    ccaminer_kwargs = dict(
        root=PATH,
        name="y1",  # y1, y2
        slice_type="num_edges",
        slice_range=(5, 10),
        num_slices=1,
        val_ratio=0.15,
        test_ratio=0.15,
        debug=DEBUG,
        seed=42,
    )

    ccaminer = CCAMiner(**ccaminer_kwargs)

    train_ccaminer, val_ccaminer, test_ccaminer = ccaminer.get_train_val_test()

    print("Train")
    for i, b in enumerate(train_ccaminer):
        print(b, b.num_obs_x)
        if i > 4:
            break

    print("Test")
    for i, b in enumerate(test_ccaminer):
        print(b, b.num_obs_x)
        if i > 4:
            break

    ccaminer.print_summary()
