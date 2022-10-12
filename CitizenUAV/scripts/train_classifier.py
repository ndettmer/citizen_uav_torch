from argparse import ArgumentParser
from CitizenUAV.models import InatClassifier
from CitizenUAV.data import *
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_per_class", type=int, default=None, required=False)
    parser.add_argument("--log_dir", type=str, default='./lightning_logs')

    parser = InatDataModule.add_dm_specific_args(parser)
    parser = InatClassifier.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    species = args.species
    data_dir = args.data_dir
    img_per_class = args.img_per_class
    log_dir = args.log_dir

    args.n_classes = len(species)

    for spec in species:
        download_data(spec, data_dir, img_per_class)

    offline_augmentation(data_dir, img_per_class)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    tb_logger = TensorBoardLogger(save_dir=log_dir)

    dict_args = vars(args)
    dm = InatDataModule(**dict_args)
    model = InatClassifier(**dict_args)
    trainer = Trainer.from_argparse_args(args)
    trainer.logger = tb_logger

    trainer.fit(model, dm)
