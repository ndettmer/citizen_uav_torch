import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms

import pandas as pd
from PIL import Image

from typing import Optional, Union, Sequence
import os

from CitizenUAV.transforms import *
from CitizenUAV.utils import get_pid_from_path


def validate_split(split: tuple) -> bool:
    """
    Validate the split tuple.
    :param split: Split
    :return: True, if valid.
    """
    if (split_sum := np.round(np.sum(split), 10)) != 1.:
        raise ValueError(f"The sum of split has to be 1. Got {split_sum}")
    return True


def validate_data_dir(data_dir: os.PathLike) -> bool:
    """
    Validate the data directory.
    :param data_dir: Data directory path.
    :return: True, if valid.
    """
    if not os.path.isdir(data_dir):
        raise ValueError(f"data_dir={data_dir}: No such directory found.")
    if not os.path.exists(os.path.join(data_dir, "metadata.csv")) and not os.path.exists(
            os.path.join(data_dir, "distances.csv")):
        raise ValueError(f"No metadata.csv or distances.csv file found in {data_dir}")
    return True


class InatDataModule(pl.LightningDataModule):
    """
    Data module for the inaturalist image dataset based on previously downloaded images.
    """

    @staticmethod
    def add_dm_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("InatDataModule")
        parser.add_argument("--data_dir", type=str)
        parser.add_argument("--species", type=str, nargs='+', required=False)
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--split", type=tuple, default=(.72, .18, .1))
        parser.add_argument("--img_size", type=int, default=128, choices=[2 ** x for x in range(6, 10)])
        return parent_parser

    # 10% test, split rest into 80% train and 20% val by default
    def __init__(self, data_dir: os.PathLike, species: Optional[Union[list | str]] = None, batch_size: int = 4,
                 split: tuple = (.72, .18, .1), img_size: int = 128, min_distance: float = None,
                 **kwargs):
        """
        :param data_dir: Directory where the data lies.
        :param species: Species to consider.
        :param batch_size: Batch size.
        :param split: Split into train, validation, test.
        :param balance: If true, the dataset will be balanced.
        :param img_size: Quadratic image output size (length of an edge).
        """
        super().__init__()

        # Make sure species is a list.
        if species is None:
            species = []
        if isinstance(species, str):
            species = [species]

        # Make sure split sums up to 1.
        validate_split(split)
        self.split = split

        # Make sure the data directory is valid.
        validate_data_dir(data_dir)
        self.data_dir = data_dir

        self.metadata = self.read_metadata(species)
        self.batch_size = batch_size

        # Compose transformations for the output samples and create the dataset object.
        img_transforms = transforms.Compose([transforms.ToTensor(), QuadCrop(), transforms.Resize(img_size)])
        self.ds = ImageFolder(str(self.data_dir), transform=img_transforms)
        self.idx = range(len(self.ds))

        self.species = species
        self.min_distance = min_distance or 0.

        # Filter dataset for species and/or balance the dataset by removing samples from over-represented classes.
        #if self.species and balance:
        #    self._balance_and_filter()
        #elif self.species:
        #    self._filter_species()
        #elif balance:
        #    self._balance_dataset()

        if self.species:
            self._filter_species()
        if self.min_distance:
            self._filter_distance()

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def read_metadata(self, species: Optional[list] = None) -> pd.DataFrame:
        """
        Read the metadata DataFrame.
        :param species: Species to filter for.
        :return: The metadata DataFrame.
        """
        metadata = pd.read_csv(os.path.join(self.data_dir, "metadata.csv"))
        metadata.reset_index(inplace=True)
        metadata.drop(columns='index', inplace=True)
        metadata.species = metadata.species.astype('category')

        if len(species):
            metadata = metadata[metadata.species.isin(species)]

        return metadata

    def _filter_species(self):
        """
        Filter dataset for species to be considered.
        """
        class_idx = [self.ds.class_to_idx[spec] for spec in self.species]
        self.idx = [i for i in self.idx if self.ds[i][1] in class_idx]

    def _filter_distance(self):
        """Filter dataset for samples with a minimum distance."""
        if 'distance' not in self.metadata:
            raise KeyError("The samples have no acquisition distance assigned.")
        min_dist_subset = self.metadata[self.metadata.distance >= self.min_distance].index
        self.idx = [i for i in self.idx if get_pid_from_path(self.ds.samples[i][0]) in min_dist_subset]

    def _replace_ds(self, idx: Sequence[int]):
        """
        Replace dataset based on indices.
        :param idx: List of indices to keep.
        """
        old_targets = np.array(self.ds.targets)
        new_targets = old_targets[idx]
        new_samples = [self.ds.samples[i] for i in idx]
        new_ds = Subset(self.ds, idx)
        new_ds.targets = new_targets
        new_ds.samples = new_samples
        new_ds.classes = self.ds.classes
        self.ds = new_ds

    def setup(self, stage: Optional[str] = None) -> None:

        self._replace_ds(self.idx)

        # Calculate absolute number of samples from split percentages.
        abs_split = list(np.floor(np.array(self.split)[:2] * len(self.ds)).astype(np.int32))
        abs_split.append(len(self.ds) - np.sum(abs_split))

        self.train_ds, self.val_ds, self.test_ds = random_split(self.ds, abs_split)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.test_ds, batch_size=self.batch_size)


class InatDistDataset(Dataset):

    def __init__(self, data_dir, transform, use_normalized=True):
        super().__init__()
        self.transform = transform
        self.data_dir = data_dir
        df_path = os.path.join(data_dir, "distances.csv")
        df = pd.read_csv(df_path)

        self.min_dist = df.Distance.min()
        self.max_dist = df.Distance.max()

        # normalize between 0 and 1
        if 'standard_dist' not in df or ('standard_dist' in df and sum(df.standard_dist.isna())):
            df['standard_dist'] = \
                (df.Distance - self.min_dist) / (self.max_dist - self.min_dist)
            df.to_csv(df_path)

        self.use_normalized = use_normalized
        if self.use_normalized:
            self.targets = df.standard_dist.values.astype(np.float32)
            subset = df[['Image', 'standard_dist']]
        else:
            self.targets = df.Distance.values.astype(np.float32)
            subset = df[['Image', 'Distance']]
        self.samples = list(subset.itertuples(index=False, name=None))

    def __getitem__(self, idx):
        filename, y = self.samples[idx]
        filepath = os.path.join(self.data_dir, filename)
        img = Image.open(filepath)
        x = self.transform(img)
        if x.shape[0] == 1:
            x = torch.concat([x, x, x], dim=0)
        if x.shape[0] > 3:
            x_mean = x.mean(dim=0).unsqueeze(0)
            x = torch.concat([x_mean, x_mean, x_mean], dim=0)

        return x, y

    def __len__(self):
        return len(self.samples)


class InatDistDataModule(pl.LightningDataModule):

    @staticmethod
    def add_dm_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("InatDistDataModule")
        parser.add_argument("--data_dir", type=str)
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--split", type=tuple, default=(.72, .18, .1))
        parser.add_argument("--img_size", type=int, default=128, choices=[2 ** x for x in range(6, 10)])
        return parent_parser

    def __init__(self, data_dir: os.PathLike, batch_size: int = 4, split: tuple = (.72, .18, .1), img_size: int = 128,
                 **kwargs):
        """
        :param data_dir: Directory where the data lies.
        :param batch_size: Batch size.
        :param split: Split into train, validation, test.
        :param img_size: Quadratic image output size (length of an edge).
        """
        super().__init__()

        self.num_workers = os.cpu_count()

        validate_split(split)
        self.split = split

        validate_data_dir(data_dir)
        self.data_dir = data_dir

        self.batch_size = batch_size

        # Compose transformations for the output samples and create the dataset object.
        img_transforms = transforms.Compose([transforms.ToTensor(), QuadCrop(), transforms.Resize(img_size), Log10()])
        self.ds = InatDistDataset(str(self.data_dir), transform=img_transforms)

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Calculate absolute number of samples from split percentages.
        abs_split = list(np.floor(np.array(self.split)[:2] * len(self.ds)).astype(np.int32))
        abs_split.append(len(self.ds) - np.sum(abs_split))

        self.train_ds, self.val_ds, self.test_ds = random_split(self.ds, abs_split)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)
