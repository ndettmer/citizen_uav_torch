from argparse import ArgumentParser
from CitizenUAV.models import InatClassifier
from CitizenUAV.data import *
from CitizenUAV.processes import *
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_per_class", type=int, default=None, required=False)
    parser.add_argument("--log_dir", type=str, default='./lightning_logs')
    parser.add_argument("--patience", type=int, default=-1)
    parser.add_argument("--min_delta", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser = InatDataModule.add_dm_specific_args(parser)
    parser = InatClassifier.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    species = args.species
    data_dir = args.data_dir
    img_per_class = args.img_per_class
    log_dir = args.log_dir
    torch.manual_seed(args.seed)

    if not species:
        # if no subset is defined take all present species
        args.n_classes = len(next(os.walk(args.data_dir))[1])
    else:
        args.n_classes = len(species)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    tb_logger = TensorBoardLogger(save_dir=log_dir)

    dict_args = vars(args)
    dm = InatDataModule(**dict_args)
    model = InatClassifier(**dict_args)

    callbacks = []
    if args.patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=args.patience,
                verbose=True,
                min_delta=args.min_delta
            )
        )

    trainer = Trainer.from_argparse_args(args, logger=tb_logger)

    trainer.fit(model, dm)

    trainer.test(ckpt_path='best', dataloaders=dm.test_dataloader())
