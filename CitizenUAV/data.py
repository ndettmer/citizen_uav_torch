import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms

import pandas as pd
from tqdm import tqdm
from PIL import Image
import pyinaturalist as pin

import numpy as np

from typing import Optional, Union
from io import BytesIO
import os

from CitizenUAV.transforms import QuadCrop


def download_data(species: str, output_dir: os.PathLike, max_images: Optional[int] = None):
    quality = "research"

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    metadata_path = os.path.join(output_dir, 'metadata.csv')
    if os.path.exists(metadata_path):
        metadata = pd.read_csv(metadata_path)
        metadata.reset_index(drop=True, inplace=True)
        metadata.set_index('photo_id', inplace=True)
    else:
        metadata = pd.DataFrame(columns=['species', 'obs_id', 'n_photos', 'label'])
        metadata.index.name = 'photo_id'

    response = pin.get_observations(
        taxon_name=species,
        quality_grade=quality,
        photos=True,
        page='all'
    )
    obss = pin.Observations.from_json_list(response)

    if not len(obss):
        print(f"No observations found for species {species}")
        return False

    images_stored = 0

    for obs in tqdm(obss):

        n_photos = len(obs.photos)

        for i, photo in enumerate(obs.photos):

            spec_dir = os.path.join(output_dir, species)
            if not os.path.isdir(spec_dir):
                os.makedirs(spec_dir)
            filename = f'{photo.id}.png'
            img_path = os.path.join(spec_dir, filename)
            if os.path.exists(img_path) and photo.id in metadata.index:
                continue

            fp = photo.open()
            img = Image.open(BytesIO(fp.data))
            img.save(img_path, 'png')

            row = [species, obs.id, n_photos, np.nan]
            metadata.loc[photo.id] = row

            images_stored += 1
            if max_images is not None and images_stored >= max_images:
                break

        metadata.species = metadata.species.astype('category')
        metadata.label = metadata.species.cat.codes
        metadata.to_csv(metadata_path)

        if max_images is not None and len(metadata[metadata.species == species]) >= max_images:
            return True

    return True


class InatDataModule(pl.LightningDataModule):

    # 10% test, split rest into 80% train and 20% val
    def __init__(self, data_dir: os.PathLike, species: Optional[Union[list | str]] = None, batch_size: int = 4,
                 split: tuple = (.72, .18, .1), balance: bool = True, img_size: int = 256):
        super().__init__()

        if species is None:
            species = []
        if isinstance(species, str):
            species = [species]

        InatDataModule.validate_split(split)
        self.split = split

        InatDataModule.validate_data_dir(data_dir)
        self.data_dir = data_dir

        self.metadata = self.read_metadata(species)
        self.batch_size = batch_size

        img_transforms = transforms.Compose([transforms.ToTensor(), QuadCrop(), transforms.Resize(img_size)])
        self.ds = ImageFolder(str(self.data_dir), transform=img_transforms)

        self.species = species

        if self.species and balance:
            self._balance_and_filter()
        elif self.species:
            self._filter_species()
        elif balance:
            self._balance_dataset()

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    @staticmethod
    def validate_split(split):
        if (split_sum := np.round(np.sum(split), 10)) != 1.:
            raise ValueError(f"The sum of split has to be 1. Got {split_sum}")
        return True

    @staticmethod
    def validate_data_dir(data_dir):
        if not os.path.isdir(data_dir):
            raise ValueError(f"data_dir={data_dir}: No such directory found.")
        if not os.path.exists(os.path.join(data_dir, "metadata.csv")):
            raise ValueError(f"No metadata.csv file found in {data_dir}")
        return True

    def read_metadata(self, species: Optional[list] = None):
        metadata = pd.read_csv(os.path.join(self.data_dir, "metadata.csv"))
        metadata.reset_index(inplace=True)
        metadata.drop(columns='index', inplace=True)
        metadata.species = metadata.species.astype('category')

        if len(species):
            metadata = metadata[metadata.species.isin(species)]

        return metadata

    def _balance_metadata(self):
        gb = self.metadata.groupby('label')
        balanced_metadata = gb.apply(lambda x: x.sample(gb.size().min()).reset_index(drop=True)).reset_index(drop=True)
        self.metadata = balanced_metadata

    def _balance_dataset(self):
        self._balance_metadata()
        file_paths = [os.path.join(self.data_dir, row.species, f"{row.photo_id}.png") for _, row in
                      self.metadata.iterrows()]
        idx = [i for i in range(len(self.ds)) if self.ds.samples[i][0] in file_paths]
        self._replace_ds(idx)

    def _filter_species(self):
        class_idx = [self.ds.class_to_idx[spec] for spec in self.species]
        idx = [i for i in range(len(self.ds)) if self.ds[i][1] in class_idx]
        self._replace_ds(idx)

    def _balance_and_filter(self):
        class_idx = [self.ds.class_to_idx[spec] for spec in self.species]

        # filter for species
        idx = [i for i in range(len(self.ds)) if self.ds[i][1] in class_idx]

        # balance classes
        self._balance_metadata()
        file_paths = [os.path.join(self.data_dir, row.species, f"{row.photo_id}.png") for _, row in
                      self.metadata.iterrows()]
        idx = [i for i in idx if self.ds.samples[i][0] in file_paths]
        self._replace_ds(idx)

    def _replace_ds(self, idx):
        old_targets = np.array(self.ds.targets)
        new_targets = old_targets[idx]
        new_ds = Subset(self.ds, idx)
        new_ds.targets = new_targets
        self.ds = new_ds

    def setup(self, stage: Optional[str] = None) -> None:

        # TODO: fill other classes somewhere else

        abs_split = list(np.floor(np.array(self.split)[:2] * len(self.metadata)).astype(np.int32))
        abs_split.append(len(self.metadata) - np.sum(abs_split))

        if stage == "fit" or stage is None:
            self.train_ds, self.val_ds, _ = random_split(self.ds, abs_split)

        if stage == "test" or stage is None:
            _, _, self.test_ds = random_split(self.ds, abs_split)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=True)
