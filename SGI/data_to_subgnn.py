import os
import torch
import numpy as np
from arguments import get_args
from data import pprint, NoisySubgraphDataModule


def write_edgelist(file_path, edge_index):
    np.savetxt(file_path, edge_index.T.numpy(), fmt='%d')


def write_subgraphs_by_dataset(f, data, data_type):
    DATASET = data_type # "train" or "val" or "test"

    for i, b in enumerate(data):
        SUBGRAPH_IDS = "-".join(list(map(str, b.x.view(-1).tolist())))

        LABEL = b.y.item()
        # (Multi-labels)
        # multi_label = torch.nonzero(b.y.squeeze(), as_tuple=True)[0].tolist()
        # LABEL = '-'.join(list(map(str, multi_label)))

        f.write(f"{SUBGRAPH_IDS}\t{LABEL}\t{DATASET}\n")


if __name__ == '__main__':
    path = "/mnt/nas2/jiho/hpo_metab_max/"

    _args = get_args("SubGNN", "EMUser", "SHORT") # "SHORT" or "TRAIN-MAX-EVAL-SHORT"
    pprint(_args)

    dm = NoisySubgraphDataModule(_args, prepare_data_and_setup=True)
    print(dm)

    train_sampler = dm.train_dataloader()
    val_sampler = dm.val_dataloader()
    test_sampler = dm.test_dataloader()

    for i, b in enumerate(train_sampler):
        if i >= 3:
            break
        print("[train example]", b)

    for i, b in enumerate(test_sampler):
        if i >= 3:
            break
        print("[test example]", b)

    write_edgelist(os.path.join(path, "edge_list.txt"), dm.dataset.global_data.edge_index)

    with open(os.path.join(path, "subgraphs.pth"), 'w') as f:
        write_subgraphs_by_dataset(f, train_sampler, "train")
        write_subgraphs_by_dataset(f, val_sampler, "val")
        write_subgraphs_by_dataset(f, test_sampler, "test")
