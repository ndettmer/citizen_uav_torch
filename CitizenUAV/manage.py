from CitizenUAV.data import *
from CitizenUAV.models import *
import pytorch_lightning as pl


def train_classifier(args):
    species = args.species
    data_dir = args.data_dir
    img_per_class = args.img_per_class

    for spec in species:
        download_data(spec, data_dir, img_per_class)

    offline_augmentation(data_dir, img_per_class)

    dm = InatDataModule(args)
    model = InatClassifier(len(species), args)
    trainer = pl.Trainer(max_epochs=args.max_epochs)

    trainer.fit(model, dm)


def train_regressor(data_dir, species, img_per_class=None, max_epochs=None, img_size=128, backbone='resnet50',
                    batch_size=4):
    for spec in species:
        download_data(spec, data_dir, img_per_class)

    dm = InatDistAngDataModule(data_dir, species, img_size=img_size, batch_size=batch_size)
    model = InatRegressor(backbone)
    trainer = pl.Trainer(max_epochs=max_epochs)

    trainer.fit(model, dm)
