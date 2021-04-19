import os
import warnings
from typing import Dict, Iterable, List, Optional, Union

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import GPUAccelerator
from torch.utils.data import DataLoader

from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.cloud_io import load as pl_load


def main_test(
        trainer,
        model: Optional[LightningModule] = None,
        test_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        ckpt_path: Optional[str] = 'best',
        verbose: bool = True,
        datamodule: Optional[LightningDataModule] = None,
):
    r"""

    Separates from fit to make sure you never run on your test set until you want to.

    Args:
        ckpt_path: Either ``best`` or path to the checkpoint you wish to test.
            If ``None``, use the weights from the last epoch to test. Default to ``best``.

        datamodule: A instance of :class:`LightningDataModule`.

        model: The model to test.

        test_dataloaders: Either a single
            Pytorch Dataloader or a list of them, specifying validation samples.

        verbose: If True, prints the test results

    Returns:
        The final test result dictionary. If no test_epoch_end is defined returns a list of dictionaries
    """
    # --------------------
    # SETUP HOOK
    # --------------------
    trainer.verbose_test = verbose

    # If you supply a datamodule you can't supply train_dataloader or val_dataloaders
    if test_dataloaders and datamodule:
        raise MisconfigurationException(
            'You cannot pass test_dataloaders to trainer.test if you supply a datamodule'
        )

    # Attach datamodule to get setup/prepare_data added to model before the call to it below
    trainer.data_connector.attach_datamodule(model or trainer.get_model(), datamodule, 'test')

    if model is not None:
        results = trainer.__test_given_model(model, test_dataloaders)
    else:
        results = __main_test_using_best_weights(trainer, ckpt_path, test_dataloaders)

    trainer.teardown('test')

    return results


def __main_test_using_best_weights(trainer: Trainer, ckpt_path, test_dataloaders):
    model = trainer.get_model()

    # if user requests the best checkpoint but we don't have it, error
    if ckpt_path == 'best' and not trainer.checkpoint_callback.best_model_path:
        raise MisconfigurationException(
            'ckpt_path is "best", but ModelCheckpoint is not configured to save the best model.'
        )

    # load best weights
    if ckpt_path is not None:
        # ckpt_path is 'best' so load the best model
        if ckpt_path == 'best':
            ckpt_path = trainer.checkpoint_callback.best_model_path

        if len(ckpt_path) == 0:
            rank_zero_warn(
                f'.test() found no path for the best weights, {ckpt_path}. Please '
                f'specify a path for a checkpoint .test(ckpt_path=PATH)'
            )
            return {}
        if trainer.accelerator_backend is not None:
            trainer.accelerator_backend.barrier()

        ckpt = pl_load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt['state_dict'])

    # attach dataloaders
    if test_dataloaders is not None:
        trainer.data_connector.attach_dataloaders(model, test_dataloaders=test_dataloaders)

    # run tests
    trainer.tested_ckpt_path = ckpt_path
    trainer.testing = True
    os.environ['PL_TESTING_MODE'] = '1'
    trainer.model = model

    # results = trainer.fit(model)  # This not works. Why? I do not know.
    # ----------------------------
    trainer.accelerator_backend = trainer.accelerator_connector.select_accelerator()
    if not isinstance(trainer.accelerator_backend, GPUAccelerator):
        results = trainer.run_test()  # Instead, use this.
    else:  # With GPUs.
        # accelerator_backend.setup() wo/ .trainer.call_setup_hook(model)
        torch.cuda.set_device(trainer.accelerator_backend.trainer.root_gpu)
        model.cuda(trainer.accelerator_backend.trainer.root_gpu)
        model = trainer.accelerator_backend.trainer.precision_connector.connect(model)
        trainer.accelerator_backend.trainer.model = model
        trainer.accelerator_backend.test_loop = trainer.run_evaluation
        results = trainer.accelerator_backend.train_or_test()

    trainer.testing = False
    del os.environ['PL_TESTING_MODE']

    # teardown
    if trainer.is_function_implemented('teardown'):
        model_ref = trainer.get_model()
        model_ref.teardown('test')

    return results
