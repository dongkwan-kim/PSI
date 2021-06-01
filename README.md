# SGI

Official Implementation of 'Intra- and Inter-Subgraph InfoMax' from 'Learning Representations of Partial Subgraphs by Intra- and Inter-Subgraph InfoMax'

## Installation

```bash
bash SGI/install.sh ${CUDA, optional, default is cu102.}
```

- If you have any trouble installing PyTorch Geometric, please install PyG's dependencies manually.
- Codes are tested with python `3.7.9` and `nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04` image.
- PYG's [FAQ](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#frequently-asked-questions) might be helpful.

## Basics
- The main train/test code is in `SGI/main.py`.
- If you want to see hyperparameter settings, refer to `SGI/args.yaml` and `SGI/arguments.py`.

## Run

```bash
python -u SGI/main.py \
    --dataset-name FNTN \
    --custom-key BIE2D2F64-ISI-X-GB-PGA \
    --gpu-ids 0 \
    --dataset-path /mnt/nas2/GNN-DATA/
```

### GPU Setting

There are three arguments for GPU settings (`--num-gpus-total`, `--num-gpus-to-use`, `--gpu-ids`).
Default values are from the author's machine, so we recommend you modify these values from `SGI/args.yaml` or by the command line.
- `--num-gpus-total` (default 4): The total number of GPUs in your machine.
- `--num-gpus-to-use` (default 1): The number of GPUs you want to use.
- `--gpu-deny-list` (default: `[0]`): The ids of GPUs you want to use.

### Datasets

#### Names (`--dataset-name`)

| Dataset         | `--dataset-name`              |
|-----------------|-------------------------------|
| FNTN            |  FNTN                         |
| EM-User         |  EMUser                       |
| HPO-Metab       |  HPOMetab                     |

#### Path (`--dataset-path`)

Download datasets in the supplementary material and put them into the specific path (`--dataset-path`).
```bash
root@5b592ce:~$ ls /mnt/nas2/GNN-DATA/
EMUSER  FNTN  HPOMETAB
```

### Models (`--custom-key`)

| Type            | FNTN                       | EMUser & HPOMetab       |
|-----------------|----------------------------|-------------------------|
| Intra-SGI       | BIE2D2F64-X-PGA            | E2D2F64-X               |
| Inter-SGI       | BISAGE-SHORT-ISI-X-GB-PGA  | SAGE-SHORT-ISI-X-GB     |
| Intra/Inter-SGI | BIE2D2F64-ISI-X-GB-PGA     | E2D2F64-ISI-X-GB        |


### Other Hyperparameters

See `SGI/args.yaml` or run `$ python SGI/main.py --help`.
