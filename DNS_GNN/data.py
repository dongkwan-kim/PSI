from typing import Type, Any

import pytorch_lightning as pl
import torch
from torch_geometric.data import InMemoryDataset, Data

from data_fntn import FNTN
from data_sampler import KHopWithLabelsXESampler


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

    def prepare_data(self):
        if self.dataset_name == "FNTN":
            self.dataset = FNTN(
                root=self.hparams.dataset_path,
                name=self.hparams.dataset_id,
                slice_type=self.hparams.dataset_slice_type,
                slice_range=(self.hparams.dataset_slice_range_1, self.hparams.dataset_slice_range_2),
                num_slices=self.hparams.dataset_num_slices,
                val_ratio=self.hparams.dataset_val_ratio,
                test_ratio=self.hparams.dataset_test_ratio,
                debug=self.hparams.dataset_debug,
                seed=self.hparams.seed,
                transform=None,
                pre_transform=None,
            )
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
            balanced_sampling=self.hparams.data_sampler_balanced_sampling,
            shuffle=self.hparams.data_sampler_shuffle,
        )
        return sampler

    def eval_loader(self, sub_data, **kwargs):
        kw = dict(
            num_hops=self.hparams.data_sampler_num_hops,
            use_labels_x=False, use_labels_e=False,
            neg_sample_ratio=0.0, dropout_edges=0.0,
            balanced_sampling=False, shuffle=False,
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

    _args = get_args("DNS", "FNTN", "TEST+MEMO")

    dm = DNSDataModule(_args, prepare_data_and_setup=True)
    print(dm)
    print(dm.num_classes)
    print(dm.num_nodes_global)
