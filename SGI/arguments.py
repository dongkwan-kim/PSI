import os
import argparse
import json
from ruamel.yaml import YAML
from termcolor import cprint

from utils import create_hash


def get_args_key(args):
    return "-".join([args.model_name, args.dataset_name, args.custom_key])


def get_args(model_name, dataset_name, custom_key="", yaml_path=None, yaml_check=True) -> argparse.Namespace:

    yaml_path = yaml_path or os.path.join(os.path.dirname(os.path.realpath(__file__)), "args.yaml")

    custom_key = custom_key.split("+")[0]

    parser = argparse.ArgumentParser(description='Argument Parser for SGI and baselines.')

    parser.add_argument("--m", default="", type=str, help="Memo")

    # Dataset
    parser.add_argument("--dataset-path", default="/mnt/nas2/GNN-DATA", type=str)
    parser.add_argument("--dataset-id", default="0.0", type=str)
    parser.add_argument("--dataset-slice-type", default="num_edges", type=str,
                        choices=["num_edges", "random", "random_walk"])
    parser.add_argument("--dataset-slice-range-1", default=5, type=int)
    parser.add_argument("--dataset-slice-range-2", default=10, type=int)
    parser.add_argument("--train-dataset-slice-range-1", default=None, type=int)
    parser.add_argument("--train-dataset-slice-range-2", default=None, type=int)
    parser.add_argument("--dataset-slice-ratio", default=None, type=float)  # Use this if given, otherwise use range-1
    parser.add_argument("--dataset-num-slices", default=1, type=int)
    parser.add_argument("--dataset-val-ratio", default=0.15, type=int)
    parser.add_argument("--dataset-test-ratio", default=0.15, type=int)
    parser.add_argument("--dataset-debug", default=False, type=bool)
    parser.add_argument("--num-classes", default=4, type=int)

    # GPUs
    parser.add_argument("--use-gpu", default=True, type=bool)
    parser.add_argument("--num-gpus-total", default=4, type=int)
    parser.add_argument("--num-gpus-to-use", default=1, type=int)
    parser.add_argument("--gpu-ids", default=None, type=int, nargs="+")

    # Basics
    parser.add_argument("--checkpoint-dir", default="../checkpoints/")
    parser.add_argument("--log-dir", default="../lightning_logs/")
    parser.add_argument("--model-name", default=model_name)
    parser.add_argument("--dataset-name", default=dataset_name)
    parser.add_argument("--custom-key", default=custom_key)
    parser.add_argument("--save-model", default=True)
    parser.add_argument("--verbose", default=2)
    parser.add_argument("--dataset-seed", default=42)
    parser.add_argument("--model-seed", default=42)
    parser.add_argument("--model-debug", default=False)
    parser.add_argument("--debug-batch-idx", default=None, type=int)
    parser.add_argument("--accumulate-grad-batches", default=64)
    parser.add_argument("--use-tensorboard", default=True)
    parser.add_argument("--precision", default=32)
    parser.add_argument("--metric", default="accuracy", choices=["accuracy", "micro-f1"])
    parser.add_argument("--version", default="1.0")

    # Training
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--batch-size', default=1, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--eval-batch-size', default=None, type=int,
                        help='mini-batch size for evaluation. If not given, use batch_size')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument("--lambda-l2", default=0., type=float)
    parser.add_argument("--lambda-aux-x", default=0., type=float)
    parser.add_argument("--lambda-aux-e", default=0., type=float)
    parser.add_argument("--lambda-aux-isi", default=0., type=float)

    # Early stop
    parser.add_argument("--use-early-stop", default=False, type=bool)
    parser.add_argument("--early-stop-patience", default=5, type=int)
    parser.add_argument("--early-stop-min-delta", default=0.0, type=float)

    # Global graph
    parser.add_argument("--num-nodes-global", default=-1, type=int)
    parser.add_argument("--global-channels", default=32, type=int)
    parser.add_argument("--global-channel-type", default="Embedding", type=str,
                        choices=["Embedding", "Random", "Feature," "Pretrained"])

    # Data Sampler
    parser.add_argument("--data-sampler-num-hops", default=1, type=int)
    parser.add_argument("--data-sampler-neg-sample-ratio", default=1.0, type=float)
    parser.add_argument("--data-sampler-dropout-edges", default=0.0, type=float)
    parser.add_argument("--data-sampler-no-drop-pos-edges", default=False, type=bool)
    parser.add_argument("--data-use-obs-edge-only", default=False, type=bool)
    parser.add_argument("--data-sampler-balanced-sampling", default=True, type=bool)
    parser.add_argument("--data-sampler-shuffle", default=True, type=bool)
    parser.add_argument("--data-sampler-num-workers", default=0, type=int)
    parser.add_argument("--data-sampler-cache-hop-computation", default=False, type=bool)

    # Model Sampler
    parser.add_argument("--model-sampler-name", default=None, type=str)
    parser.add_argument("--model-sampler-kwargs", default="{}", type=json.loads)  # Dict

    # Model Encoder
    parser.add_argument("--gnn-name", default="GCNConv", type=str)
    parser.add_argument("--num-encoder-layers", default=2, type=int)
    parser.add_argument("--use-bn", default=False, type=bool)
    parser.add_argument("--activation", default="relu", type=str)
    parser.add_argument("--hidden-channels", default=64, type=int)
    parser.add_argument("--dropout-channels", default=0.2, type=float)
    parser.add_argument("--is-bidirectional", default=False, type=bool)
    parser.add_argument("--use-skip", default=True, type=bool)

    # Model Decoder
    parser.add_argument("--readout-name", default=None, type=str)
    parser.add_argument("--use-decoder", default=True, type=bool)
    parser.add_argument("--num-decoder-body-layers", default=1, type=int)
    parser.add_argument("--main-decoder-type", default="node", type=str, choices=["node", "edge"])
    parser.add_argument("--use-node-decoder", default=True)
    parser.add_argument("--use-edge-decoder", default=True)
    parser.add_argument("--obs-max-len", default=20)
    parser.add_argument("--is-obs-sequential", default=True)
    parser.add_argument("--pool-ratio", default=0.1)
    parser.add_argument("--use-pool-min-score", default=False)
    parser.add_argument("--use-inter-subgraph-infomax", default=False)
    parser.add_argument("--inter-subgraph-infomax-edge-type", default="global",
                        type=str, choices=["global", "subgraph"])

    # per-graph feature (e.g., text)
    parser.add_argument("--use-pergraph-attr", default=False, type=bool)
    parser.add_argument("--pergraph-encoder-type", default=None, type=str)
    parser.add_argument("--pergraph-channels", default=2000, type=int)
    parser.add_argument("--pergraph-hidden-channels", default=64, type=int)
    parser.add_argument("--vocab-size", default=-1, type=int)

    # Test
    parser.add_argument("--val-interval", default=10)

    # Tuning
    parser.add_argument("--use-pruner", default=True)

    # Experiment specific parameters loaded from .yamls
    with open(yaml_path) as args_file:
        args = parser.parse_args()
        args_key = "-".join([args.model_name, args.dataset_name or args.dataset_class, args.custom_key])
        try:
            args_from_yaml = dict(YAML().load(args_file)[args_key].items())
            if yaml_check:
                dest_list = [action.dest for action in parser._actions]
                for k in args_from_yaml:
                    if k not in dest_list:
                        cprint("Warning: {} is not pre-defined".format(k), "red")
            parser.set_defaults(**args_from_yaml)
        except KeyError:
            cprint("KeyError: there's no {} in yamls".format(args_key), "red")
            exit()

    # Update params from .yamls
    args = parser.parse_args()
    if args.model_debug:
        args.val_interval = 1
    return args


def get_important_args(_args: argparse.Namespace) -> dict:
    important_args = [
        "dataset_id", "dataset_slice_type",
        "dataset_slice_range_1", "dataset_slice_range_2", "dataset_slice_ratio",
        "train_dataset_slice_range_1", "train_dataset_slice_range_2",
        "dataset_num_slices", "dataset_val_ratio", "dataset_test_ratio", "dataset_debug",
        "data_sampler_num_hops", "data_sampler_neg_sample_ratio",
        "data_sampler_dropout_edges", "data_sampler_balanced_sampling",
        "data_sampler_shuffle", "data_use_obs_edge_only", "data_sampler_cache_hop_computation",
        "data_sampler_no_drop_pos_edges",
        "model_sampler_name", "model_sampler_kwargs",
        "model_name", "dataset_name", "custom_key", "dataset_seed", "model_seed", "model_debug",
        "accumulate_grad_batches",
        "lr", "batch_size", "lambda_l2", "lambda_aux_x", "lambda_aux_e", "lambda_aux_isi",
        "use_early_stop", "early_stop_patience", "early_stop_min_delta",
        "global_channel_type",
        "gnn_name", "num_encoder_layers", "activation", "hidden_channels", "dropout_channels", "is_bidirectional",
        "readout_name", "use_decoder", "num_decoder_body_layers",
        "main_decoder_type", "use_node_decoder", "use_edge_decoder",
        "obs_max_len", "is_obs_sequential", "pool_ratio", "use_pool_min_score",
        "use_inter_subgraph_infomax", "inter_subgraph_infomax_edge_type",
        "use_pergraph_attr", "pergraph_encoder_type", "pergraph_channels", "pergraph_hidden_channels",
    ]
    ret = {}
    for ia_key in important_args:
        if ia_key in _args.__dict__:
            ret[ia_key] = _args.__getattribute__(ia_key)
        else:
            cprint(f"Warning: {ia_key} is not used.", "red")
    return ret


def get_args_hash(_args, length=7, **kwargs):
    return create_hash({**get_important_args(_args), **kwargs})[:length]


def save_args(model_dir_path: str, _args: argparse.Namespace):

    if not os.path.isdir(model_dir_path):
        raise NotADirectoryError("Cannot save arguments, there's no {}".format(model_dir_path))

    with open(os.path.join(model_dir_path, "args.txt"), "w") as arg_file:
        for k, v in sorted(_args.__dict__.items()):
            arg_file.write("{}: {}\n".format(k, v))


def pprint_args(_args: argparse.Namespace):
    cprint("Args PPRINT: {}".format(get_args_key(_args)), "yellow")
    for k, v in sorted(_args.__dict__.items()):
        print("\t- {}: {}".format(k, v))


def pdebug_args(_args: argparse.Namespace, logger):
    logger.debug("Args LOGGING-PDEBUG: {}".format(get_args_key(_args)))
    for k, v in sorted(_args.__dict__.items()):
        logger.debug("\t- {}: {}".format(k, v))


if __name__ == '__main__':
    test_args = get_args("SGI", "FNTN", "TEST+MEMO")
    pprint_args(test_args)
    print("Type: {}".format(type(test_args)))
    print("Dict [get_important_args(test_args)]: {}".format(get_important_args(test_args)))
    print("Hash [get_args_hash(test_args)]: {}".format(get_args_hash(test_args)))
