from collections import OrderedDict, defaultdict
from copy import deepcopy
from pprint import pprint
import os
from typing import List, Any, Dict

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.utilities.memory import garbage_collection_cuda
from termcolor import cprint
from torch.nn import functional as F
from torch import nn
from torch import optim
from pytorch_lightning.core.lightning import LightningModule

from arguments import get_args_key, get_args, pprint_args, get_args_hash
from data import DNSDataModule
from model import DNSNet
from utils import merge_or_update


class MainModel(LightningModule):

    def __init__(self, hparams, dataset: DNSDataModule):
        super().__init__()
        self.model = None  # see setup
        self.hparams = hparams
        self.dataset = dataset
        self.save_hyperparameters(hparams)

    def setup(self, stage: str = None):
        self.hparams.num_nodes_global = self.dataset.num_nodes_global
        self.hparams.num_classes = self.dataset.num_classes
        if self.hparams.model_name == "DNS" and self.hparams.version == "1.0":
            self.model = DNSNet(self.hparams)
        else:
            raise ValueError(f"Wrong model ({self.hparams.model_name}) or version ({self.hparams.version})")
        pprint(next(self.model.modules()))

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.lambda_l2)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        logits_g, train_loss, loss_part = self._get_logits_and_loss_and_parts(batch, batch_idx)
        return {
            "loss": train_loss,
            "train_acc_step": accuracy(logits_g, batch.y),
            "loss_part": loss_part,
        }

    def training_epoch_end(self, outputs) -> None:
        for k in outputs[0]["loss_part"]:
            self.log(f"train_{k}", torch.stack([output["loss_part"][k] for output in outputs]).mean())
        self.log("train_loss", torch.stack([output["loss"] for output in outputs]).mean())
        self.log("train_acc", torch.stack([output["train_acc_step"] for output in outputs]).mean())

    def validation_step(self, batch, batch_idx):
        logits_g, val_loss, _ = self._get_logits_and_loss_and_parts(batch, batch_idx)
        return {
            "val_loss": val_loss,
            "val_acc_step": accuracy(logits_g, batch.y),
        }

    def validation_epoch_end(self, outputs):
        self.log("val_loss", torch.stack([output["val_loss"] for output in outputs]).mean())
        self.log("val_acc", torch.stack([output["val_acc_step"] for output in outputs]).mean())

    def test_step(self, batch, batch_idx):
        logits_g, test_loss, _ = self._get_logits_and_loss_and_parts(batch, batch_idx)
        return {
            "test_loss": test_loss,
            "test_acc_step": accuracy(logits_g, batch.y),
        }

    def test_epoch_end(self, outputs):
        test_acc = torch.stack([output["test_acc_step"] for output in outputs]).mean()
        test_loss = torch.stack([output["test_loss"] for output in outputs]).mean()
        self.log("test_loss", test_loss)
        self.logger.log_metrics({"hp_metric": float(test_acc)})
        return {
            "test_loss": test_loss,
            "test_acc": test_acc,
        }

    def _get_logits_and_loss_and_parts(self, batch, batch_idx):
        if self.hparams.use_decoder:
            out = self._forward_with_dec(batch, batch_idx)
            logits_g = out["logits_g"]
            total_loss = out["total_loss"]
        else:
            out = {}
            logits_g = self(batch.x, batch.obs_x_idx, batch.edge_index_01, pergraph_attr=batch.pergraph_attr)
            total_loss = F.cross_entropy(logits_g, batch.y)
        return logits_g, total_loss, {k: v for k, v in out.items() if k not in ["logits_g", "total_loss"]}

    def _forward_with_dec(self, batch, batch_idx):
        # batch example:
        # Data(edge_index_01=[2, 686034], edge_index_2=[2, 262], labels_e=[786], labels_x=[526], mask_e=[686296],
        #      mask_x=[25868], obs_x_idx=[9], x=[25868], y=[1])
        # forward(x_idx, obs_x_idx, edge_index_01, edge_index_2)
        logits_g, dec_x, dec_e = self(
            batch.x, batch.obs_x_idx, batch.edge_index_01, batch.edge_index_2, batch.pergraph_attr,
        )
        if batch.mask_x is not None:
            dec_x = dec_x[batch.mask_x]
        if batch.mask_e is not None:
            dec_e = dec_e[batch.mask_e]

        total_loss = 0
        loss_g = F.cross_entropy(logits_g, batch.y)
        total_loss += loss_g
        o = {"total_loss": total_loss, "logits_g": logits_g, "loss_g": loss_g}
        if self.hparams.lambda_aux_x > 0 and batch.labels_x is not None:
            loss_x = F.cross_entropy(dec_x, batch.labels_x)
            total_loss += self.hparams.lambda_aux_x * loss_x
            o["loss_x"] = loss_x
        if self.hparams.lambda_aux_e > 0 and batch.labels_e is not None:
            loss_e = F.cross_entropy(dec_e, batch.labels_e)
            total_loss += self.hparams.lambda_aux_e * loss_e
            o["loss_e"] = loss_e
        return o


def run_train(args, trainer_given_kwargs=None, run_test=True, clean_ckpt=False):
    seed_everything(args.model_seed)

    dm = DNSDataModule(args, prepare_data_and_setup=True)
    model = MainModel(args, dm)
    callbacks = []
    args_key = get_args_key(args)

    if args.save_model:
        args_hash = get_args_hash(args)
        filepath = os.path.join(args.checkpoint_dir, args_key, "{epoch:03d}-{val_loss:.5f}-{val_acc:.2f}")
        checkpoint_callback = ModelCheckpoint(
            filepath=filepath,
            save_top_k=1,
            verbose=True if args.verbose > 0 else False,
            monitor='val_acc',
            mode='max',
            prefix=f"{args_hash}",
        )
    else:
        checkpoint_callback = False

    if args.use_early_stop:
        early_stop_callback = EarlyStopping(
            monitor='val_acc',
            min_delta=args.early_stop_min_delta,  # TODO: How to set this value.
            patience=args.early_stop_patience,
            verbose=True if args.verbose > 0 else False,
            mode="max",
        )
        callbacks.append(early_stop_callback)

    # False in HP-search
    logger = TensorBoardLogger(args.log_dir, name=args_key) if args.use_tensorboard else False
    gpus = args.gpu_ids if args.use_gpu else None
    precision = args.precision if args.use_gpu else 32

    trainer_kwargs = dict(
        gpus=gpus,
        deterministic=True,
        max_epochs=args.epochs,
        check_val_every_n_epoch=args.val_interval,
        reload_dataloaders_every_epoch=False,
        progress_bar_refresh_rate=1 if args.verbose > 0 else 0,
        logger=logger,
        checkpoint_callback=checkpoint_callback,  # can be changed in HP-search
        callbacks=callbacks if len(callbacks) > 0 else None,
        num_sanity_val_steps=0,
        fast_dev_run=args.model_debug,
        precision=precision,
    )
    if trainer_given_kwargs:
        trainer_kwargs = merge_or_update(trainer_kwargs, trainer_given_kwargs)

    trainer = Trainer(**trainer_kwargs)

    trainer.fit(model, dm)

    ret = {
        "trainer": trainer,
        "model": model,
        "callbacks": trainer_kwargs.get("callbacks", None),  # # for metrics_callback in HP-search
        "test_results": None,
    }

    if run_test:
        from main_test import main_test
        ret["test_results"]: Dict[str, torch.Tensor] = main_test(trainer)[0]  # 1st emt of len-1 list

    if clean_ckpt and trainer.checkpoint_callback.best_model_path:
        os.remove(trainer.checkpoint_callback.best_model_path)
        cprint(f"Removed: {trainer.checkpoint_callback.best_model_path}", "red")

    return ret


def run_train_multiple(num_runs, args, trainer_given_kwargs=None, run_test=True, clean_ckpt=False):
    ret_list = defaultdict(list)
    for r in range(num_runs):
        _args = deepcopy(args)
        _args.model_seed += r

        single_results = run_train(
            _args,
            trainer_given_kwargs=trainer_given_kwargs,
            run_test=run_test,
            clean_ckpt=clean_ckpt,
        )
        for k, v in single_results["test_results"].items():
            ret_list[k].append(v)

        # Make GPU empty.
        for k in single_results:
            single_results[k] = None
        garbage_collection_cuda()

    return ret_list


if __name__ == '__main__':
    main_args = get_args(
        model_name="DNS",
        dataset_name="FNTN",
        custom_key="SMALL-E",
    )
    pprint_args(main_args)

    main_results = run_train(
        main_args,
        run_test=True,
        clean_ckpt=True,
    )
    main_trainer = main_results["trainer"]
    main_model = main_results["model"]
