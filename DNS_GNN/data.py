from typing import Type, Any
from pprint import pprint

import pytorch_lightning as pl
import torch
from termcolor import cprint
from torch_geometric.data import InMemoryDataset, Data

from data_fntn import FNTN
from data_sampler import KHopWithLabelsXESampler
from data_sub import HPONeuro, HPOMetab, EMUser
from data_utils import CompleteSubgraph


def _subdata_filter_func(data: Data):
    if data.edge_index.size(1) <= 0:
        cprint("Data filtered: {}".format(data), "red")
        return False
    return True


class DNSDataModule(pl.LightningDataModule):

    def __init__(self, hparams, prepare_data_and_setup):
        super().__init__()
        self.hparams = hparams
        self.dataset: Type[InMemoryDataset] or None = None
        self.train_data, self.val_data, self.test_data = None, None, None
        if prepare_data_and_setup:
            self.prepare_data_and_setup()
            print(f"{self.__class__.__name__}/{self.dataset_name}: prepared and set up!")

    @property
    def dataset_name(self):
        return self.hparams.dataset_name

    @property
    def sampler_name(self):
        return self.hparams.dataset_sampler_name

    @property
    def num_nodes_global(self):
        return self.dataset.num_nodes_global

    @property
    def num_classes(self):
        return self.dataset.num_classes

    @property
    def embedding(self):
        return self.dataset.global_data.x

    @property
    def data_kwargs(self):
        return dict(
            root=self.hparams.dataset_path,
            name=self.hparams.dataset_id,
            slice_type=self.hparams.dataset_slice_type,
            slice_range=(self.hparams.dataset_slice_range_1,
                         self.hparams.dataset_slice_range_2),
            num_slices=self.hparams.dataset_num_slices,
            val_ratio=self.hparams.dataset_val_ratio,
            test_ratio=self.hparams.dataset_test_ratio,
            debug=self.hparams.dataset_debug,
            seed=self.hparams.dataset_seed,
            transform=None,
        )

    @property
    def ke_method(self):
        if self.dataset_name == "FNTN":
            return "edge"
        elif self.dataset_name in ["HPONeuro", "HPOMetab"]:
            return "node"
        else:
            raise ValueError("Wrong dataset_name: {}".format(self.dataset_name))

    def prepare_data(self):
        pre_transform = None
        cprint("Dataset prepared", "yellow")
        for k, v in self.data_kwargs.items():
            print("\t- {}: {}".format(k, v))
        if self.dataset_name == "FNTN":
            if self.hparams.inter_subgraph_infomax_edge_type == "global":
                pre_transform = CompleteSubgraph()
            self.dataset = FNTN(**self.data_kwargs, pre_transform=pre_transform)
        elif self.dataset_name == "HPONeuro":
            self.dataset = HPONeuro(**self.data_kwargs, pre_transform=pre_transform)
        elif self.dataset_name == "HPOMetab":
            self.dataset = HPOMetab(**self.data_kwargs, pre_transform=pre_transform)
        elif self.dataset_name == "EMUser":
            self.dataset = EMUser(**self.data_kwargs, pre_transform=pre_transform)
        else:
            raise ValueError(f"Wrong dataset: {self.dataset_name}")

    def setup(self, stage=None):
        self.train_data, self.val_data, self.test_data = self.dataset.get_train_val_test()

    def train_dataloader(self):
        sampler = KHopWithLabelsXESampler(
            self.dataset.global_data, self.train_data,
            num_hops=self.hparams.data_sampler_num_hops,
            use_labels_x=self.hparams.use_node_decoder,
            use_labels_e=self.hparams.use_edge_decoder,
            neg_sample_ratio=self.hparams.data_sampler_neg_sample_ratio,
            dropout_edges=self.hparams.data_sampler_dropout_edges,
            obs_x_range=(self.hparams.dataset_slice_range_1, self.hparams.dataset_slice_range_2),
            use_obs_edge_only=self.hparams.data_use_obs_edge_only,
            use_pergraph_attr=self.hparams.use_pergraph_attr,
            balanced_sampling=self.hparams.data_sampler_balanced_sampling,
            use_inter_subgraph_infomax=self.hparams.use_inter_subgraph_infomax,
            no_drop_pos_edges=self.hparams.data_sampler_no_drop_pos_edges,
            batch_size=self.hparams.batch_size,
            subdata_filter_func=_subdata_filter_func,
            cache_hop_computation=self.hparams.data_sampler_cache_hop_computation,
            ke_method=self.ke_method,
            shuffle=self.hparams.data_sampler_shuffle,
            num_workers=self.hparams.data_sampler_num_workers,
        )
        return sampler

    def eval_loader(self, sub_data, **kwargs):
        kw = dict(
            num_hops=self.hparams.data_sampler_num_hops,
            use_labels_x=False, use_labels_e=False,
            neg_sample_ratio=0.0, dropout_edges=0.0, obs_x_range=None,
            use_obs_edge_only=self.hparams.data_use_obs_edge_only,
            use_pergraph_attr=self.hparams.use_pergraph_attr,
            balanced_sampling=False,
            use_inter_subgraph_infomax=False,
            no_drop_pos_edges=False,  # important
            batch_size=(self.hparams.eval_batch_size or self.hparams.batch_size),  # important
            subdata_filter_func=_subdata_filter_func,
            cache_hop_computation=self.hparams.data_sampler_cache_hop_computation,
            ke_method=self.ke_method,
            shuffle=False,
            num_workers=self.hparams.data_sampler_num_workers,
        )
        kw.update(kwargs)
        return KHopWithLabelsXESampler(
            self.dataset.global_data, sub_data, **kw
        )

    def val_dataloader(self):
        return self.eval_loader(self.val_data)

    def test_dataloader(self):
        return self.eval_loader(self.test_data)

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        return batch.to(device)

    def prepare_data_and_setup(self):
        self.prepare_data()
        self.setup()

    def __repr__(self):
        return "{}(dataset={})".format(
            self.__class__.__name__, self.dataset,
        )


def random_input_generator(n_p=49, n_n=50, F_f=10, e_p=11, e_n=13, batch_size=1, F_w=23):
    """
    :return x: FloatTensor of [\sum_{i \in B} (N^p_i + N^n_i), F^f]
    :return edge_index: LongTensor of [2, \sum_{i \in B} (E^p_i + E^n_i)]
    :return edge_attr: FloatTensor of [\sum_{i \in B} (E^p_i + E^n_i), 1]
    :return global_attr: LongTensor of [|B|, F^w]
    :return batch: LongTensor of [N,]
    """
    x = torch.rand(n_p + n_n, F_f)
    edge_index = torch.randint(n_p + n_n, (2, e_p + e_n))
    edge_attr = torch.rand(e_p + e_n, )
    global_attr = torch.rand(batch_size, F_w)
    num_obs_x = torch.randint(10, (1,))
    y = torch.randint(10, (1,))
    return Data(
        x=x, num_obs_x=num_obs_x,
        edge_index=edge_index, edge_attr=edge_attr,
        y=y, global_attr=global_attr,
    )


if __name__ == '__main__':
    from arguments import get_args

    _args = get_args("DNS", "HPOMetab", "BIE2D2F64-ISI-X-GB")
    pprint(_args)

    dm = DNSDataModule(_args, prepare_data_and_setup=True)
    print(dm)
    print("num_classes", dm.num_classes)
    print("num_nodes_global", dm.num_nodes_global)
    print("embedding", dm.embedding)
