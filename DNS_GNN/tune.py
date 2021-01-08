import logging
import os
import shutil
from pprint import pprint
from typing import List, Tuple, Dict

from pytorch_lightning import Callback, seed_everything
from pytorch_lightning import loggers as pl_loggers

import torch

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

    # Filenames for each trial must be made unique in order to access each checkpoint.
    args_key = get_args_key(tune_args)
    metrics_callback = MetricsCallback(monitor=METRIC_TO_MONITOR)
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor=METRIC_TO_MONITOR)
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

    metrics_list = [m[METRIC_TO_MONITOR].item() for m in metrics]
    if METRIC_TO_MONITOR == "val_acc":
        return max(metrics_list)
    elif METRIC_TO_MONITOR == "val_loss":
        return min(metrics_list)
    else:
        raise ValueError("Wrong metric: {}".format(METRIC_TO_MONITOR))


if __name__ == '__main__':

    N_TRIALS = 500
    METRIC_TO_MONITOR = "val_acc"

    tune_args = get_args(
        model_name="DNS",
        dataset_name="FNTN",
        custom_key="BIE2D2F64-ISI-X",  # BISAGE, SMALL-E
    )
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

    search_config = {
        "lambda_l2": ("loguniform", 5e-5, 5e-3),
        # "lr": ("categorical", [0.001, 0.005])
    }
    if tune_args.use_decoder:
        if tune_args.use_node_decoder:
            search_config["lambda_aux_x"] = ("discrete_uniform", 0.00, 20.0, 0.01)
        if tune_args.use_edge_decoder:
            search_config["lambda_aux_e"] = ("discrete_uniform", 0.00, 20.0, 0.01)
        if tune_args.use_inter_subgraph_infomax:
            search_config["lambda_aux_isi"] = ("discrete_uniform", 0.00, 20.0, 0.001)
        if not tune_args.use_pool_min_score:
            search_config["pool_ratio"] = ("loguniform", 5e-4, 0.25)
        search_config["data_sampler_dropout_edges"] = ("discrete_uniform", 0.25, 0.75, 0.05)

    logger.info("-- HPARAM SEARCH CONFIG --")
    for k, v in search_config.items():
        logger.info(f"{k}: {str(v)}")

    if all([v[0] == "categorical" for v in search_config.values()]):
        sampler = optuna.samplers.GridSampler({k: v[1] for k, v in search_config.items()})
        logger.info("Using GridSampler")
    else:
        sampler = optuna.samplers.TPESampler()

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=20000) if tune_args.use_pruner else optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=tune_args_key,
        direction="maximize" if METRIC_TO_MONITOR == "val_acc" else "minimize",
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
