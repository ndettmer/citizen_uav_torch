import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms

import pandas as pd
from tqdm import tqdm
from PIL import Image
import pyinaturalist as pin

from typing import Optional, Union
from io import BytesIO
import os
from collections import Counter

from CitizenUAV.transforms import *


def download_data(species: str, output_dir: os.PathLike, max_images: Optional[int] = None,
                  min_year: Optional[int] = 2010):
    """
    Download inaturalist image data for a certain species.
    :param species: species to collect data for
    :param output_dir: output directory
    :param max_images: maximum number of images to download
    :param min_year: year of the earliest observations to collect
    :return pd.DataFrame: collected metadata
    """
    quality = "research"

    # create directory
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # load or create metadata DataFrame
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    if os.path.exists(metadata_path):
        metadata = pd.read_csv(metadata_path)
        metadata.reset_index(drop=True, inplace=True)
        metadata.set_index('photo_id', inplace=True)
    else:
        metadata = pd.DataFrame(columns=['species', 'obs_id', 'n_photos', 'label'])
        metadata.index.name = 'photo_id'

    # collect data from inaturalist
    response = pin.get_observations(
        taxon_name=species,
        quality_grade=quality,
        photos=True,
        page='all',
        year=range(int(min_year), 2024)
    )
    obss = pin.Observations.from_json_list(response)

    # break process if no data was found
    if not len(obss):
        print(f"No observations found for species {species}")
        return False

    # create species directory if it doesn't exist
    spec_dir = os.path.join(output_dir, species)
    if not os.path.isdir(spec_dir):
        os.makedirs(spec_dir)

    images_stored = 0
    # iterate over observations
    p_bar = tqdm(obss)
    p_bar.set_description(f"Downloading data for species {species} ...")
    for obs in p_bar:

        n_photos = len(obs.photos)

        # iterate over images in observation
        for i, photo in enumerate(obs.photos):

            # set file name with photo id
            filename = f'{photo.id}.png'
            img_path = os.path.join(spec_dir, filename)

            # create entry in metadata
            row = [species, obs.id, n_photos, np.nan]
            if len(metadata.columns) > len(row):
                # deal with existing extended metadata
                row += [np.nan] * (len(metadata.columns) - len(row))
            metadata.loc[photo.id] = row

            # skip photo, if already downloaded
            if os.path.exists(img_path):
                continue

            # download image
            fp = photo.open()
            img = Image.open(BytesIO(fp.data))
            img.save(img_path, 'png')

            # count image and stop if enough images have been downloaded
            images_stored += 1
            if max_images is not None and images_stored >= max_images:
                break

        # format class label column in metadata
        metadata.species = metadata.species.astype('category')
        metadata.label = metadata.species.cat.codes
        metadata.to_csv(metadata_path)

        # stop whole procedure if enough images have been downloaded
        if max_images is not None and len(metadata[metadata.species == species]) >= max_images:
            return metadata

    return metadata


def extend_metadata(data_dir, consider_augmented=False):
    csv_path = os.path.join(data_dir, 'metadata.csv')
    ds = ImageFolder(data_dir, transform=transforms.ToTensor())
    metadata = pd.read_csv(csv_path)
    metadata.set_index('photo_id', inplace=True)

    idx = range(len(ds))
    if not consider_augmented:
        # skip augmented images
        idx = [i for i in idx if "_" not in os.path.basename(ds.samples[i][0])]

    max_vals = pd.Series(index=metadata.index, dtype='float32')
    min_vals = pd.Series(index=metadata.index, dtype='float32')
    mean_vals = pd.Series(index=metadata.index, dtype='float32')
    paths = pd.Series(index=metadata.index, dtype=str)
    heights = pd.Series(index=metadata.index, dtype='int32')
    widths = pd.Series(index=metadata.index, dtype='int32')
    contrasts = pd.Series(index=metadata.index, dtype='float32')
    saturations = pd.Series(index=metadata.index, dtype='float32')
    broken = pd.Series(index=metadata.index, dtype=bool)
    broken[:] = False

    for i in tqdm(idx):
        path, _ = ds.samples[i]
        filename = os.path.splitext(os.path.basename(path))[0]
        pid = int(filename)

        try:
            img, cls_idx = ds[i]
        except OSError:
            # skip and mark broken files
            broken[pid] = True
            print(f"Skipping broken file with id {pid} ...")
            continue

        cls = ds.classes[cls_idx]

        try:
            row = metadata.loc[pid]
        except KeyError:
            new_row = [cls, np.nan, np.nan, cls_idx]
            if len(metadata.columns) > len(new_row):
                new_row += [np.nan] * (len(metadata.columns) - len(new_row))
            metadata.loc[pid] = new_row
            row = metadata.loc[pid]
        if cls != row.species:
            raise ValueError(f"Classes {cls} and {metadata['species']} do not match for image {pid}!")

        max_val = float(torch.max(img).numpy())
        max_vals[pid] = max_val
        min_val = float(torch.min(img).numpy())
        min_vals[pid] = min_val
        mean_vals[pid] = float(torch.mean(img).numpy())
        channels, heights[pid], widths[pid], = img.size()
        contrast = max_val - min_val
        contrasts[pid] = contrast
        saturations[pid] = contrast / max_val
        paths[pid] = path

    # NaN entries might have been created. Replace them with the default value.
    broken.fillna(False, inplace=True)

    metadata['max_val'] = max_vals
    metadata['min_val'] = min_vals
    metadata['mean_val'] = mean_vals
    metadata['path'] = paths
    metadata['height'] = heights
    metadata['width'] = widths
    metadata['contrast'] = contrasts
    metadata['saturation'] = saturations
    metadata['broken'] = broken

    metadata.to_csv(csv_path)

    return metadata


def offline_augmentation(data_dir: os.PathLike, target_n):
    """
    Perform offline augmentation on present images.
    :param data_dir: The directory where the data lies.
    :param target_n: Target number of samples per class.
    :return: True, if the procedure succeeded.
    """
    # create dataset
    ds = ImageFolder(data_dir, transform=transforms.Compose([transforms.ToTensor(), QuadCrop()]))

    # count samples per class
    n_samples = dict(Counter(ds.targets))

    # define repertoire of augmentation techniques
    techniques = [RandomBrightness(), RandomContrast(), RandomSaturation()]

    # postprocessing including random flips
    do_anyway_after = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.ToPILImage()]

    # iterate over classes
    for cls, n in n_samples.items():
        pbar = tqdm(range(target_n - n))
        pbar.set_description(f"Augmenting for {ds.classes[cls]}")

        # only consider indices of current class for augmentation sampling
        cls_idx = [i for i in range(len(ds)) if ds.targets[i] == cls]

        for _ in pbar:
            # sample image to be modified
            rand_idx = np.random.choice(cls_idx)
            img, y = ds[rand_idx]

            # Apply each technique with a probability of 0.5 (don't confuse with internal random parameters).
            random_apply = transforms.RandomApply(techniques)

            # compose and apply modification pipeline
            transform = transforms.Compose([random_apply] + do_anyway_after)
            augmented = transform(img)

            # determine new file name
            orig_filepath = ds.samples[rand_idx][0]
            filepath = os.path.splitext(orig_filepath)[0]
            n_copies = 1
            # make sure the filename doesn't exist yet
            while os.path.exists(f"{filepath}_{n_copies}.png"):
                n_copies += 1
            filepath = f"{filepath}_{n_copies}.png"
            # save new image
            augmented.save(filepath, 'png')

    return True


class InatDataModule(pl.LightningDataModule):
    """
    Data module for the inaturalist image dataset based on previously downloaded images.
    """

    @staticmethod
    def add_dm_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("InatDataModule")
        parser.add_argument("--data_dir", type=str)
        parser.add_argument("--species", type=str, nargs='+', required=True)
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--split", type=tuple, default=(.72, .18, .1))
        parser.add_argument("--balance", type=bool, default=True)
        parser.add_argument("--img_size", type=int, default=128, choices=[2 ** x for x in range(6, 10)])
        return parent_parser

    # 10% test, split rest into 80% train and 20% val by default
    def __init__(self, data_dir: os.PathLike, species: Optional[Union[list | str]] = None, batch_size: int = 4,
                 split: tuple = (.72, .18, .1), balance: bool = True, img_size: int = 128, **kwargs):
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
        InatDataModule.validate_split(split)
        self.split = split

        # Make sure the data directory is valid.
        InatDataModule.validate_data_dir(data_dir)
        self.data_dir = data_dir

        self.metadata = self.read_metadata(species)
        self.batch_size = batch_size

        # Compose transformations for the output samples and create the dataset object.
        img_transforms = transforms.Compose([transforms.ToTensor(), QuadCrop(), transforms.Resize(img_size)])
        self.ds = ImageFolder(str(self.data_dir), transform=img_transforms)

        self.species = species

        # Filter dataset for species and/or balance the dataset by removing samples from over-represented classes.
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
    def validate_split(split: tuple) -> bool:
        """
        Validate the split tuple.
        :param split: Split
        :return: True, if valid.
        """
        if (split_sum := np.round(np.sum(split), 10)) != 1.:
            raise ValueError(f"The sum of split has to be 1. Got {split_sum}")
        return True

    @staticmethod
    def validate_data_dir(data_dir: os.PathLike) -> bool:
        """
        Validate the data directory.
        :param data_dir: Data directory path.
        :return: True, if valid.
        """
        if not os.path.isdir(data_dir):
            raise ValueError(f"data_dir={data_dir}: No such directory found.")
        if not os.path.exists(os.path.join(data_dir, "metadata.csv")):
            raise ValueError(f"No metadata.csv file found in {data_dir}")
        return True

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

    def _balance_metadata(self):
        """
        Balance the metadata DataFrame by removing samples from overrepresented classes.
        """
        gb = self.metadata.groupby('label')
        balanced_metadata = gb.apply(lambda x: x.sample(gb.size().min()).reset_index(drop=True)).reset_index(drop=True)
        self.metadata = balanced_metadata

    def _balance_dataset(self):
        """
        Balance the dataset based on a balanced metadata DataFrame.
        """
        self._balance_metadata()
        file_paths = [os.path.join(self.data_dir, row.species, f"{row.photo_id}.png") for _, row in
                      self.metadata.iterrows()]
        idx = [i for i in range(len(self.ds)) if self.ds.samples[i][0] in file_paths]
        self._replace_ds(idx)

    def _filter_species(self):
        """
        Filter dataset for species to be considered.
        """
        class_idx = [self.ds.class_to_idx[spec] for spec in self.species]
        idx = [i for i in range(len(self.ds)) if self.ds[i][1] in class_idx]
        self._replace_ds(idx)

    def _balance_and_filter(self):
        """
        Filter dataset for species to be considered and balance data based on a balanced metadata DataFrame.
        """
        class_idx = [self.ds.class_to_idx[spec] for spec in self.species]

        # filter for species
        idx = [i for i in range(len(self.ds)) if self.ds[i][1] in class_idx]

        # balance classes
        self._balance_metadata()
        file_paths = [os.path.join(self.data_dir, row.species, f"{row.photo_id}.png") for _, row in
                      self.metadata.iterrows()]
        idx = [i for i in idx if self.ds.samples[i][0] in file_paths]
        self._replace_ds(idx)

    def _replace_ds(self, idx: list):
        """
        Replace dataset based on indices.
        :param idx: List of indices to keep.
        """
        old_targets = np.array(self.ds.targets)
        new_targets = old_targets[idx]
        new_ds = Subset(self.ds, idx)
        new_ds.targets = new_targets
        self.ds = new_ds

    def setup(self, stage: Optional[str] = None) -> None:

        # Calculate absolute number of samples from split percentages.
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


class InatDistDataset(ImageFolder):

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        # TODO: overwrite y with distance label
        return x, y


class InatDistDataModule(InatDataModule):

    def __init__(self, data_dir: os.PathLike, species: Optional[Union[list | str]] = None, batch_size: int = 4,
                 split: tuple = (.72, .18, .1), balance: bool = True, img_size: int = 128):
        super().__init__(data_dir, species, batch_size, split, balance, img_size)

        # Compose transformations for the output samples and create the dataset object.
        # TODO: Check min and max value of images
        img_transforms = transforms.Compose([transforms.ToTensor(), QuadCrop(), transforms.Resize(img_size), Log10()])
        self.ds = InatDistDataset(str(self.data_dir), transform=img_transforms)
