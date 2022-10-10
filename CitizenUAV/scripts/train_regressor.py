from argparse import ArgumentParser
from CitizenUAV.data import InatDistDataModule, offline_augmentation
from CitizenUAV.models import InatRegressor
from pytorch_lightning import Trainer
from CitizenUAV.manage import train_regressor


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_per_class", type=int, default=None, required=False)

    parser = InatDistDataModule.add_dm_specific_args(parser)
    parser = InatRegressor.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    species = args.species
    data_dir = args.data_dir
    img_per_class = args.img_per_class

    offline_augmentation(data_dir, img_per_class)

    dict_args = vars(args)
    dm = InatDistDataModule(**dict_args)
    model = InatRegressor(**dict_args)
    trainer = Trainer.from_argparse_args(args)

    trainer.fit(model, dm)
