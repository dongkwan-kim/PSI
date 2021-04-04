from collections import OrderedDict, defaultdict
from copy import deepcopy
from pprint import pprint
import os
from typing import List, Any, Dict, Tuple

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.utilities.memory import garbage_collection_cuda
from termcolor import cprint
from torch.nn import functional as F
from torch import nn, Tensor
from torch import optim
from pytorch_lightning.core.lightning import LightningModule
from torch_geometric.data import Data
from sklearn.metrics import f1_score

from arguments import get_args_key, get_args, pprint_args, get_args_hash
from data import DNSDataModule
from model import DNSNet
from utils import merge_or_update, cprint_arg_conditionally


def _cac_kw():
    return dict(
        condition_func=lambda args: args[0].hparams.model_debug,
        filter_func=lambda arg: isinstance(arg, Data),
        out_func=lambda arg: "\nBatch on model_debug=True: {}".format(arg),
    )


def ga(b, attr_name):
    return getattr(b, attr_name, None)


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
            self.model = DNSNet(self.hparams, self.dataset.embedding)
        else:
            raise ValueError(f"Wrong model ({self.hparams.model_name}) or version ({self.hparams.version})")
        pprint(next(self.model.modules()))

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.lambda_l2)

    def forward(self, batch):
        return self.model(
            batch.x, batch.obs_x_index, batch.edge_index_01,
            edge_index_2=ga(batch, "edge_index_2"), pergraph_attr=ga(batch, "pergraph_attr"),
            batch=batch.batch,
            x_idx_isi=ga(batch, "x_pos_and_neg"), edge_index_isi=ga(batch, "edge_index_pos_and_neg"),
            batch_isi=ga(batch, "batch_pos_and_neg"), ptr_isi=ga(batch, "ptr_pos_and_neg"),
        )

    def loss_with_logits(self, logits, y) -> Tensor:
        if y.size(-1) == 1:
            return F.cross_entropy(logits, y)
        else:  # multi-label
            return F.binary_cross_entropy_with_logits(logits, y)

    def perf_ingredients_step(self, logits, y, prefix) -> Dict[str, Tensor]:
        if self.hparams.metric == "accuracy":
            return {f"{prefix}_acc_step": accuracy(logits, y)}
        elif self.hparams.metric == "micro-f1":
            return {f"{prefix}_logits": logits, f"{prefix}_y": y}
        else:
            raise ValueError(f"Wrong metric: {self.hparams.metric}")

    def perf_aggr(self, outputs: List, prefix) -> Tuple[str, Tensor]:
        if self.hparams.metric == "accuracy":
            acc_list = [output[f"{prefix}_acc_step"] for output in outputs]
            return f"{prefix}_acc", torch.stack(acc_list).mean()
        elif self.hparams.metric == "micro-f1":
            logits = torch.stack([output[f"{prefix}_logits"] for output in outputs]).squeeze()
            ys = torch.stack([output[f"{prefix}_y"] for output in outputs]).squeeze()
            pred, ys = (logits > 0.0).float().cpu().numpy(), ys.cpu().numpy()
            micro_f1 = f1_score(pred, ys, average="micro") if pred.sum() > 0 else 0
            return f"{prefix}_f1", micro_f1
        else:
            raise ValueError(f"Wrong metric: {self.hparams.metric}")

    @cprint_arg_conditionally(**_cac_kw())
    def training_step(self, batch, batch_idx):
        logits_g, train_loss, loss_part = self._get_logits_and_loss_and_parts(batch, batch_idx)
        return {
            "loss": train_loss,
            "loss_part": loss_part,
            **self.perf_ingredients_step(logits_g, batch.y, prefix="train"),
        }

    def training_epoch_end(self, outputs) -> None:
        for k in outputs[0]["loss_part"]:
            self.log(f"train_{k}", torch.stack([output["loss_part"][k] for output in outputs]).mean())
        self.log("train_loss", torch.stack([output["loss"] for output in outputs]).mean())
        # Below is the same as
        # self.log("train_acc", torch.stack([output["train_acc_step"] for output in outputs]).mean())
        self.log(*self.perf_aggr(outputs, prefix="train"))

    @cprint_arg_conditionally(**_cac_kw())
    def validation_step(self, batch, batch_idx):
        logits_g, val_loss, _ = self._get_logits_and_loss_and_parts(batch, batch_idx)
        return {
            "val_loss": val_loss,
            **self.perf_ingredients_step(logits_g, batch.y, prefix="val"),
        }

    def validation_epoch_end(self, outputs):
        self.log("val_loss", torch.stack([output["val_loss"] for output in outputs]).mean())
        self.log(*self.perf_aggr(outputs, prefix="val"))

    @cprint_arg_conditionally(**_cac_kw())
    def test_step(self, batch, batch_idx):
        logits_g, test_loss, _ = self._get_logits_and_loss_and_parts(batch, batch_idx)
        return {
            "test_loss": test_loss,
            **self.perf_ingredients_step(logits_g, batch.y, prefix="test"),
        }

    def test_epoch_end(self, outputs):
        perf_name, test_perf = self.perf_aggr(outputs, prefix="test")
        test_loss = torch.stack([output["test_loss"] for output in outputs]).mean()
        self.log("test_loss", test_loss)
        self.logger.log_metrics({"hp_metric": float(test_perf)})
        return {
            "test_loss": test_loss,
            perf_name: test_perf,
        }

    def _get_logits_and_loss_and_parts(self, batch, batch_idx):
        if self.hparams.use_decoder:
            out = self._forward_with_dec(batch, batch_idx)
            logits_g = out["logits_g"]
            total_loss = out["total_loss"]
        else:
            out = {}
            logits_g, _, _, loss_isi = self(batch)
            # F.ce(logits_g, batch.y) or F.bce_w/_logits(logits_g, batch.y)
            total_loss = self.loss_with_logits(logits_g, batch.y)
            if self.hparams.use_inter_subgraph_infomax and self.hparams.lambda_aux_isi > 0:
                total_loss += self.hparams.lambda_aux_isi * loss_isi
                out["loss_isi"] = loss_isi
        return logits_g, total_loss, {k: v for k, v in out.items() if k not in ["logits_g", "total_loss"]}

    def _forward_with_dec(self, batch, batch_idx):
        """
        :param batch:
            example (isi False)
                Data(edge_index_01=[2, 686034], edge_index_2=[2, 262], labels_e=[786], labels_x=[526],
                     mask_e_index=[686296], mask_x_index=[25868], obs_x_index=[9], x=[25868], y=[1])
            example (isi True)
                Data(edge_index_01=[2, 2099097], edge_index_isi=[2, 1019], labels_x=[250], mask_x_index=[250],
                     obs_x_index=[7], ptr_isi=[1], x=[82468], x_isi=[1016], y=[1])

        :param batch_idx:
        :return:
        """
        # forward args:
        #   x_idx, obs_x_index, edge_index_01, edge_index_2, pergraph_attr,
        #   x_idx_isi, edge_index_isi, ptr_isi
        logits_g, dec_x, dec_e, loss_isi = self(batch)
        if ga(batch, "mask_x_index") is not None:
            dec_x = dec_x[batch.mask_x_index]
        if ga(batch, "mask_e_index") is not None:
            dec_e = dec_e[batch.mask_e_index]

        total_loss = 0
        loss_g = self.loss_with_logits(logits_g, batch.y)
        total_loss += loss_g
        o = {"logits_g": logits_g, "loss_g": loss_g}
        if self.hparams.lambda_aux_x > 0 and ga(batch, "labels_x") is not None:
            loss_x = F.cross_entropy(dec_x, batch.labels_x)
            total_loss += self.hparams.lambda_aux_x * loss_x
            o["loss_x"] = loss_x
        if self.hparams.lambda_aux_e > 0 and ga(batch, "labels_e") is not None:
            loss_e = F.cross_entropy(dec_e, batch.labels_e)
            total_loss += self.hparams.lambda_aux_e * loss_e
            o["loss_e"] = loss_e
        if self.hparams.use_inter_subgraph_infomax and self.hparams.lambda_aux_isi > 0:
            if self.training:
                total_loss += self.hparams.lambda_aux_isi * loss_isi
                o["loss_isi"] = loss_isi
        o["total_loss"] = total_loss
        return o


def run_train(args, trainer_given_kwargs=None, run_test=True, clean_ckpt=False):
    seed_everything(args.model_seed)

    dm = DNSDataModule(args, prepare_data_and_setup=True)
    model = MainModel(args, dm)
    callbacks = []
    args_key = get_args_key(args)
    val_perf_name = "val_acc" if args.metric == "accuracy" else "val_f1"

    if args.save_model:
        args_hash = get_args_hash(args)
        filepath = os.path.join(args.checkpoint_dir, args_key,
                                "{epoch:03d}-{val_loss:.5f}-{" + val_perf_name + ":.2f}")
        checkpoint_callback = ModelCheckpoint(
            filepath=filepath,
            save_top_k=1,
            verbose=True if args.verbose > 0 else False,
            monitor=val_perf_name,
            mode='max',
            prefix=f"{args_hash}",
        )
    else:
        checkpoint_callback = False

    if args.use_early_stop:
        early_stop_callback = EarlyStopping(
            monitor=val_perf_name,
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
        accumulate_grad_batches=args.accumulate_grad_batches,
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


def run_train_multiple(num_runs, args,
                       trainer_given_kwargs=None, run_test=True, clean_ckpt=False,
                       change_model_seed=True, change_dataset_seed=True) -> Dict[str, List[float]]:
    """
    return example:
        defaultdict(<class 'list'>,
            {'test_acc': [0.0, 0.0, 0.0],
             'test_loss': [1.4073759317398071,
                           1.3666242361068726,
                           1.360438585281372]})
    """
    assert change_model_seed or change_dataset_seed

    test_list = defaultdict(list)
    for r in range(num_runs):
        _args = deepcopy(args)
        if change_model_seed:
            _args.model_seed += r
        if change_dataset_seed:
            _args.dataset_seed += r

        single_results = run_train(
            _args,
            trainer_given_kwargs=trainer_given_kwargs,
            run_test=run_test,
            clean_ckpt=clean_ckpt,
        )
        for k, v in single_results["test_results"].items():
            test_list[k].append(v)

        # Make GPU empty.
        for k in single_results:
            single_results[k] = None
        garbage_collection_cuda()

    return test_list


if __name__ == '__main__':

    MODE = "RUN_MULTIPLE"  # RUN_ONCE, RUN_MULTIPLE
    NUM_RUNS = 5

    main_args = get_args(
        model_name="DNS",
        dataset_name="HPONeuro",  # FNTN, HPOMetab, HPONeuro
        custom_key="E2D2F64-ISI-X-GB",  # BISAGE-SHORT, BIE2D2F64-ISI-X-PGA, E2D2F64-ISI-X-GB
    )
    pprint_args(main_args)
    cprint("MODE: {} (#={})".format(MODE, NUM_RUNS), "red")

    if MODE == "RUN_ONCE":
        main_results = run_train(
            main_args,
            run_test=True,
            clean_ckpt=True,
        )
        main_trainer = main_results["trainer"]
        main_model = main_results["model"]
    elif MODE == "RUN_MULTIPLE":
        main_results = run_train_multiple(
            num_runs=NUM_RUNS,
            args=main_args,
            trainer_given_kwargs=None, run_test=True, clean_ckpt=True,
            change_model_seed=True, change_dataset_seed=False,
        )
        pprint(main_results)
