# PSI

Official Implementation of 'Partial Subgraph InfoMax (PSI)' from 'Models and Benchmarks for Representation Learning of Partially Observed Subgraphs', 31st ACM International Conference on Information and Knowledge Management (CIKM 2022, Short Papers Track).

## BibTeX

```
TBA
```

## Installation

```bash
bash PSI/install.sh ${CUDA, optional, default is cu102.}
```

- If you have any trouble installing PyTorch Geometric, please install PyG's dependencies manually.
- Codes are tested with python `3.7.9` and `nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04` image.
- PYG's [FAQ](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#frequently-asked-questions) might be helpful.

## Basics
- The main train/test code is in `PSI/main.py`.
- If you want to see hyperparameter settings, refer to `PSI/args.yaml` and `PSI/arguments.py`.

## Run

```bash
python -u PSI/main.py \
    --dataset-name FNTN \
    --custom-key BIE2D2F64-ISI-X-GB-PGA \
    --gpu-ids 0 \
    --dataset-path /mnt/nas2/GNN-DATA/
```

### GPU Setting

There are three arguments for GPU settings (`--num-gpus-total`, `--num-gpus-to-use`, `--gpu-ids`).
Default values are from the author's machine, so we recommend you modify these values from `PSI/args.yaml` or by the command line.
- `--num-gpus-total` (default 4): The total number of GPUs in your machine.
- `--num-gpus-to-use` (default 1): The number of GPUs you want to use.
- `--gpu-ids` (default: `[0]`): The ids of GPUs you want to use.

### Datasets

#### Names (`--dataset-name`)

| Dataset         | `--dataset-name`              |
|-----------------|-------------------------------|
| FNTN            |  FNTN                         |
| EM-User         |  EMUser                       |
| HPO-Metab       |  HPOMetab                     |

#### Path (`--dataset-path`)

Download [datasets](https://github.com/mims-harvard/SubGNN#prepare-data) and put them into the specific path (`--dataset-path`).
```bash
root@5b592ce:~$ ls /mnt/nas2/GNN-DATA/
EMUSER  FNTN  HPOMETAB
```

### Models (`--custom-key`)

| Type                     | FNTN                           | EMUser & HPOMetab                              |
|--------------------------|--------------------------------|------------------------------------------------|
| PS-DGI                   | BISAGE-SHORT-DGI-X-GB-PGA      | SAGE-SHORT-DGI-X-GB                            |
| PS-InfoGraph             | BISAGE-SHORT-ISI-X-GB-PGA      | SAGE-SHORT-ISI-X-GB                            |
| PS-MVGRL                 | BISAGE-SHORT-MVGRL-X-GB-PGA    | SAGE-SHORT-MVGRL-X-GB                          |
| PS-GraphCL               | BISAGE-SHORT-GRAPHCL3-X-GB-PGA | SAGE-SHORT-GRAPHCL3FB-X-GB (only for HPOMetab) |
| k-hop PSI                | BIE2D2F64-X-PGA                | E2D2F64-X                                      |
| k-hop PSI + PS-DGI       | BIE2D2F64-DGI-X-GB-PGA         | E2D2F64-DGI-X-GB                               |
| k-hop PSI + PS-InfoGraph | BIE2D2F64-ISI-X-GB-PGA         | E2D2F64-ISI-X-GB                               |


### Other Hyperparameters

See `PSI/args.yaml` or run `$ python PSI/main.py --help`.
