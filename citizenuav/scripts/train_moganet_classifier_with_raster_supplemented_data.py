import os

from argparse import ArgumentParser

import numpy as np

from plyer import notification
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from citizenuav.data import MixedDataModule
from citizenuav.io import write_params
from citizenuav.models import InatMogaNetClassifier

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="./lightning_logs")
    parser.add_argument("--patience", type=int, default=-1)
    parser.add_argument("--min_delta", type=float, default=0)
    parser.add_argument("--seed", type=int, default=np.random.rand())

    parser = MixedDataModule.add_dm_specific_args(parser)
    parser = InatMogaNetClassifier.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    write_params(args.log_dir, vars(args), "train_classifier")

    data_dir = args.data_dir
    log_dir = args.log_dir
    seed_everything(args.seed, workers=True)

    # if no subset is defined take all present species
    args.n_classes = len(next(os.walk(args.inat_dir))[1])

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    tb_logger = TensorBoardLogger(save_dir=log_dir)

    dict_args = vars(args)
    dm = MixedDataModule(**dict_args)
    model = InatMogaNetClassifier(**dict_args)

    callbacks = []
    if args.patience > 0:
        callbacks.append(
            EarlyStopping(monitor="val_cce", mode="min", patience=args.patience, verbose=True, min_delta=args.min_delta)
        )

    trainer = Trainer.from_argparse_args(args, logger=tb_logger)

    trainer.fit(model, dm)

    trainer.test(ckpt_path="best", dataloaders=dm.test_dataloader())

    notification.notify(title="Classifier Training", message="Training done.")
