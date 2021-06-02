import time
import os

from pytorch_lightning.callbacks import Callback
from termcolor import cprint

from arguments import get_args, pprint_args
from main import run_train


class TimerCallback(Callback):

    time_init_start = None
    time_fit_start = None
    time_epoch_start = None
    batch_count = 0

    def on_init_start(self, trainer):
        self.time_init_start = time.time()
        cprint("\nInit start", "green")

    def on_fit_start(self, trainer, pl_module):
        self.time_fit_start = time.time()
        cprint("\nFit start", "green")

    def on_train_epoch_start(self, trainer, pl_module):
        self.time_epoch_start = time.time()
        cprint("\nTraining epoch start", "green")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self.batch_count += 1
        cprint("\nBatch incremented: {}".format(self.batch_count), "green")

    def on_validation_epoch_start(self, trainer, pl_module):
        now = time.time()
        dt_init_start = now - self.time_init_start
        dt_fit_start = now - self.time_fit_start
        dt_epoch_start = now - self.time_epoch_start
        cprint("\n------------------", "green")
        try:
            cprint(pl_module.hparams.custom_key, "green")
        except AttributeError:
            pass
        cprint("Total #batch: {}".format(self.batch_count), "green")

        cprint("\n------------------", "green")
        cprint("init_start ~ ", "green")
        cprint("\t- Total time / epoch: {}".format(dt_init_start), "green")
        cprint("\t- Total time / batch: {}".format(dt_init_start / self.batch_count), "green")

        cprint("\n------------------", "green")
        cprint("fit_start ~ ", "green")
        cprint("\t- Total time / epoch: {}".format(dt_fit_start), "green")
        cprint("\t- Total time / batch: {}".format(dt_fit_start / self.batch_count), "green")

        cprint("\n------------------", "green")
        cprint("epoch_start ~ ", "green")
        cprint("\t- Total time / epoch: {}".format(dt_epoch_start), "green")
        cprint("\t- Total time / batch: {}".format(dt_epoch_start / self.batch_count), "green")

        cprint("\n------------------", "green")
        cprint("Exit before the validation", "green")
        exit()


if __name__ == '__main__':
    main_args = get_args(
        model_name="SGI",
        dataset_name="FNTN",  # FNTN, EMUser, HPOMetab
        custom_key="BIE2D2F64-ISI-X-GB-PGA",
        # E2D2F64-X, BIE2D2F64-X-PGA,
        # SAGE-SHORT-ISI-X-GB, BISAGE-SHORT-ISI-X-GB-PGA
        # E2D2F64-ISI-X-GB, BIE2D2F64-ISI-X-GB-PGA
    )

    # control variables
    main_args.data_sampler_num_workers = max(40, os.cpu_count())
    main_args.batch_size = 4
    main_args.accumulate_grad_batches = 1
    if main_args.use_decoder:
        main_args.data_sampler_dropout_edges = 0.5
        main_args.pool_ratio = 0.001

    main_args.log_dir = "../logs_tmp"
    # main_args.model_debug = True

    pprint_args(main_args)

    main_results = run_train(
        main_args,
        run_test=True,
        clean_ckpt=True,
        additional_callbacks=[TimerCallback()],
    )
