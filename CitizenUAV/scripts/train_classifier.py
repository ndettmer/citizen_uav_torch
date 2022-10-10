from argparse import ArgumentParser
from CitizenUAV.models import InatClassifier
from CitizenUAV.data import *
from pytorch_lightning import Trainer


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_per_class", type=int, default=None, required=False)

    parser = InatDataModule.add_dm_specific_args(parser)
    parser = InatClassifier.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    species = args.species
    data_dir = args.data_dir
    img_per_class = args.img_per_class

    for spec in species:
        download_data(spec, data_dir, img_per_class)

    offline_augmentation(data_dir, img_per_class)

    dict_args = vars(args)
    dm = InatDataModule(**dict_args)
    model = InatClassifier(**dict_args)
    trainer = Trainer.from_argparse_args(args)

    trainer.fit(model, dm)
