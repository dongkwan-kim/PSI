import logging
import os
import shutil
import random
import time
from pprint import pprint
from typing import List, Tuple, Dict

from pytorch_lightning import Callback, seed_everything
from pytorch_lightning import loggers as pl_loggers

import torch
import numpy as np
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from termcolor import cprint

from arguments import get_args, get_args_key, pprint_args
from main import run_train


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append({metric: metric_val for metric, metric_val in trainer.callback_metrics.items()
                             if metric == self.monitor})


def suggest_hparams(args, trial, name_to_method_and_spaces: Dict[str, Tuple]):
    for name, method_and_spaces in name_to_method_and_spaces.items():
        _method, *_spaces = method_and_spaces
        method = {
            "categorical": trial.suggest_categorical,
            "int": trial.suggest_int,
            "uniform": trial.suggest_uniform,
            "loguniform": trial.suggest_loguniform,
            "discrete_uniform": trial.suggest_discrete_uniform,
        }[_method]
        suggested = method(name, *_spaces)
        setattr(args, name, suggested)
    return args, trial


def objective(trial):
    global tune_args  # tune_args in main lines.
    global search_config
    global METRIC_TO_MONITOR
    global ABLATION_GRID_SEARCH

    # Filenames for each trial must be made unique in order to access each checkpoint.
    args_key = get_args_key(tune_args)
    metrics_callback = MetricsCallback(monitor=METRIC_TO_MONITOR)
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor=METRIC_TO_MONITOR)

    if ABLATION_GRID_SEARCH:
        # Add randomness in suggest_hparams, since run_train has own seed_everything.
        random.seed(int(time.time()))
    tune_args, trial = suggest_hparams(tune_args, trial, search_config)

    results = run_train(
        tune_args,
        trainer_given_kwargs=dict(
            logger=pl_loggers.TensorBoardLogger(tune_args.log_dir, name=f"tune_{args_key}"),
            callbacks=[metrics_callback, pruning_callback],
        ),
        run_test=True,
        clean_ckpt=True,
    )
    trainer, model, callbacks = results["trainer"], results["model"], results["callbacks"]
    metrics = [cb for cb in callbacks if cb.__class__.__name__ == "MetricsCallback"][0].metrics

    try:
        metrics_list = [m[METRIC_TO_MONITOR].item() for m in metrics]  # tensor
    except AttributeError:  # primitive: int or float
        metrics_list = [m[METRIC_TO_MONITOR] for m in metrics]
    if METRIC_TO_MONITOR in ["val_acc", "val_f1"]:
        return max(metrics_list)
    elif METRIC_TO_MONITOR == "val_loss":
        return min(metrics_list)
    else:
        raise ValueError("Wrong metric: {}".format(METRIC_TO_MONITOR))


if __name__ == '__main__':

    ABLATION_GRID_SEARCH = False
    N_TRIALS = 50

    tune_args = get_args(
        model_name="PSI",
        dataset_name="FNTN",
        custom_key="BIE2D2F64-ISI-X-GB-PGA",  # BISAGE, SMALL-E
    )
    METRIC_TO_MONITOR = {"HPONeuro": "val_f1"}.get(tune_args.dataset_name, "val_acc")
    tune_args.verbose = 0  # Force args' verbose be 0
    tune_args.use_early_stop = False
    tune_args_key = get_args_key(tune_args)
    pprint_args(tune_args)

    # Logging
    hparams_dir = "../hparams"
    os.makedirs(hparams_dir, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    logger = optuna.logging._get_library_root_logger()
    logger.addHandler(logging.FileHandler(os.path.join(hparams_dir, f"{tune_args_key}.log"), mode="w"))

    if not ABLATION_GRID_SEARCH:
        search_config = {
            "lambda_l2": ("loguniform", 1e-7, 1e-3),
            # "lr": ("categorical", [0.001, 0.005])
        }
        if tune_args.use_decoder:
            if tune_args.use_node_decoder:
                search_config["lambda_aux_x"] = ("discrete_uniform", 0.00, 5.0, 0.01)
            if tune_args.use_edge_decoder:
                search_config["lambda_aux_e"] = ("discrete_uniform", 0.00, 5.0, 0.01)
            if tune_args.subgraph_infomax_type is not None:
                search_config["lambda_aux_isi"] = ("discrete_uniform", 0.00, 5.0, 0.01)
            if not tune_args.use_pool_min_score:
                if tune_args.dataset_name in ["HPONeuro", "HPOMetab"]:
                    search_config["pool_ratio"] = ("loguniform", 1e-4, 1e-2)
                else:
                    search_config["pool_ratio"] = ("loguniform", 1e-3, 1e-1)

            if tune_args.data_sampler_no_drop_pos_edges:
                search_config["data_sampler_dropout_edges"] = ("discrete_uniform", 0.95, 0.99, 0.01)
            elif tune_args.dataset_name in ["HPONeuro", "HPOMetab"]:
                search_config["data_sampler_dropout_edges"] = ("discrete_uniform", 0.55, 0.75, 0.05)
            else:
                search_config["data_sampler_dropout_edges"] = ("discrete_uniform", 0.4, 0.6, 0.05)

        elif tune_args.subgraph_infomax_type is not None:
            search_config["lambda_aux_isi"] = ("discrete_uniform", 0.00, 5.0, 0.01)

    else:  # ABLATION_GRID_SEARCH
        tune_args.log_dir = tune_args.log_dir.replace("lightning_logs", "lightning_logs_ablation")
        search_config = {
            "lambda_aux_x": ("categorical", [1.0, 2.0, 3.0]),
            "lambda_aux_isi": ("categorical", [1.0, 2.0, 3.0]),
        }
        if tune_args.dataset_name in ["FNTN"]:
            search_config["lambda_l2"] = ("categorical", [1e-4, 1e-3])
        else:
            search_config["lambda_l2"] = ("categorical", [1e-6, 1e-5])

        if tune_args.dataset_name in ["HPONeuro", "HPOMetab"]:
            search_config["pool_ratio"] = ("categorical", [1e-4, 1e-3])
        else:
            search_config["pool_ratio"] = ("categorical", [1e-3, 1e-2])
        tune_args.data_sampler_dropout_edges = 0.5
        N_TRIALS = np.product([len(v2) for v1, v2 in search_config.values()])  # search all spaces

    logger.info("-- HPARAM SEARCH CONFIG --")
    logger.info("- N_TRIALS: {}".format(N_TRIALS))
    logger.info("- ABLATION_GRID_SEARCH: {}".format(ABLATION_GRID_SEARCH))
    logger.info("- Search Space:")
    for k, v in search_config.items():
        logger.info(f"\t- {k}: {str(v)}")

    if all([v[0] == "categorical" for v in search_config.values()]):
        sampler = optuna.samplers.GridSampler({k: v[1] for k, v in search_config.items()})
        logger.info("Using GridSampler")
    else:
        sampler = optuna.samplers.TPESampler()

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=20000) if tune_args.use_pruner else optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=tune_args_key,
        direction="maximize" if METRIC_TO_MONITOR in ["val_acc", "val_f1"] else "minimize",
        sampler=sampler,
        pruner=pruner,
        storage=f'sqlite:///{hparams_dir}/{tune_args_key}.db',
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    best_trial = study.best_trial
    pprint(best_trial)

    print("  Value: {}".format(best_trial.value))

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))
