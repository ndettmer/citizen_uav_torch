from argparse import ArgumentParser
from plyer import notification
import os
import numpy as np

from CitizenUAV.models import InatMogaNetClassifier
from CitizenUAV.data import InatDataModule
from CitizenUAV.io import write_params
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log_dir", type=str, default='./lightning_logs')
    parser.add_argument("--patience", type=int, default=-1)
    parser.add_argument("--min_delta", type=float, default=0)
    parser.add_argument("--raster_dir", type=str)
    parser.add_argument("--finetune_lr", type=float)
    parser.add_argument("--finetune_epochs", type=int)
    parser.add_argument("--finetune_sample_per_class", type=int)
    parser.add_argument("--seed", type=int, default=np.random.rand())

    parser = InatDataModule.add_dm_specific_args(parser)
    parser = InatMogaNetClassifier.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    write_params(args.log_dir, vars(args), 'train_classifier')

    data_dir = args.data_dir
    log_dir = args.log_dir
    seed_everything(args.seed, workers=True)

    # if no subset is defined take all present species
    args.n_classes = len(next(os.walk(args.data_dir))[1])

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    tb_logger = TensorBoardLogger(save_dir=log_dir)

    dict_args = vars(args)

    inat_dm = InatDataModule(**dict_args)

    dict_args['data_dir'] = dict_args.pop('raster_dir')
    dict_args['sample_per_class'] = dict_args.pop('finetune_sample_per_class')
    raster_dm = InatDataModule(**dict_args)

    model = InatMogaNetClassifier(**dict_args)

    callbacks = []
    if args.patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_cce",
                mode="min",
                patience=args.patience,
                verbose=True,
                min_delta=args.min_delta
            )
        )

    trainer = Trainer.from_argparse_args(args, logger=tb_logger)

    trainer.fit(model, inat_dm)

    trainer.test(ckpt_path='best', dataloaders=inat_dm.test_dataloader())

    # fine-tune
    model.lr = dict_args.pop('finetune_lr')
    args.max_epochs = dict_args.pop('finetune_epochs')
    trainer = Trainer.from_argparse_args(args, logger=tb_logger)
    trainer.fit(model, raster_dm)

    trainer.test(ckpt_path='best', dataloaders=raster_dm.test_dataloader())

    notification.notify(
        title="Classifier Training",
        message="Training done."
    )
