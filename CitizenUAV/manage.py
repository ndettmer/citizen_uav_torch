from CitizenUAV.data import *
from CitizenUAV.models import *
import pytorch_lightning as pl


def train_model(data_dir, species, img_per_class=None, max_epochs=None, img_size=128, backbone='resnet50'):
    for spec in species:
        download_data(spec, data_dir, img_per_class)

    offline_augmentation(data_dir, img_per_class)

    dm = InatDataModule(data_dir, species, img_size=img_size)
    model = InatClassifier(len(species), backbone)
    trainer = pl.Trainer(max_epochs=max_epochs)

    trainer.fit(model, dm)
