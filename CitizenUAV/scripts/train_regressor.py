from argparse import ArgumentParser
from CitizenUAV.data import InatDistDataModule
from CitizenUAV.processes import offline_augmentation
from CitizenUAV.models import InatRegressor
from pytorch_lightning import Trainer
from CitizenUAV.manage import train_regressor
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import os

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--patience", type=int, default=-1, required=False)
    parser.add_argument("--log_dir", type=str, default='./lightning_logs')

    parser = InatDistDataModule.add_dm_specific_args(parser)
    parser = InatRegressor.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    data_dir = args.data_dir
    log_dir = args.log_dir

    # offline_augmentation(data_dir, img_per_class)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    tb_logger = TensorBoardLogger(save_dir=log_dir)

    dict_args = vars(args)
    dm = InatDistDataModule(**dict_args)
    model = InatRegressor(**dict_args)

    callbacks = []
    if args.patience > 0:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=args.patience, verbose=True))

    trainer = Trainer.from_argparse_args(args, callbacks=callbacks, logger=tb_logger)

    trainer.fit(model, dm)
