import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torch.utils.data.dataset import T_co
from torchvision.datasets import ImageFolder
from torchvision import transforms

import pandas as pd
from PIL import Image
import rasterio as rio
import rasterio.mask
from rasterio.windows import Window
import fiona
from tqdm import tqdm

from typing import Optional, Union, Sequence
import os
from collections import Counter
import logging

from CitizenUAV.transforms import *
from CitizenUAV.utils import get_pid_from_path, read_inat_metadata, channel_mean_std


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
        parser.add_argument("--balance", type=bool, default=False, required=False)
        parser.add_argument("--min_distance", type=int, default=None, required=False)
        parser.add_argument("--sample_per_class", type=int, default=None, required=False)
        parser.add_argument("--normalize", type=bool, default=False, required=False)
        return parent_parser

    # 10% test, split rest into 80% train and 20% val by default
    def __init__(self, data_dir: os.PathLike, species: Optional[Union[list | str]] = None, batch_size: int = 4,
                 split: tuple = (.72, .18, .1), img_size: int = 128, min_distance: float = None, balance: bool = False,
                 sample_per_class: int = -1, normalize: bool = False, **kwargs):
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

        # TODO: fix dtype warning
        self.metadata = read_inat_metadata(self.data_dir)
        self.batch_size = batch_size
        self.normalize = normalize

        # Compose transformations for the output samples and create the dataset object.
        transforms_list = [transforms.ToTensor(), QuadCrop(), transforms.Resize(img_size)]

        img_transform = transforms.Compose(transforms_list)
        self.ds = ImageFolder(str(self.data_dir), transform=img_transform)
        self.idx = range(len(self.ds))

        self.species = species
        self.min_distance = min_distance or 0.
        self.balance = balance
        self.num_workers = os.cpu_count()
        self.sample_per_class = sample_per_class

        self._filter_broken_images()
        if self.species:
            self._filter_species()
        if self.min_distance:
            self._filter_distance()
        # Balancing should always be the last step of modifying the selection of samples!
        if self.balance or self.sample_per_class > 0:
            self._balance_dataset()

        self._replace_ds(self.idx)

        if self.normalize:
            self._add_normalize()

        # Calculate absolute number of samples from split percentages.
        abs_split = list(np.floor(np.array(self.split)[:2] * len(self.ds)).astype(np.int32))
        abs_split.append(len(self.ds) - np.sum(abs_split))

        self.train_ds, self.val_ds, self.test_ds = random_split(self.ds, abs_split)

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

    def _filter_broken_images(self):
        if 'image_okay' not in self.metadata and 'broken' not in self.metadata:
            logging.warning("No information about broken images in the metadata!")
        if 'image_okay' in self.metadata:
            okay_pids = self.metadata[self.metadata.image_okay].index
            self.idx = [i for i in self.idx if get_pid_from_path(self.ds.samples[i][0]) in okay_pids]
        if 'broken' in self.metadata:
            okay_pids = self.metadata[~self.metadata.broken].index
            self.idx = [i for i in self.idx if get_pid_from_path(self.ds.samples[i][0]) in okay_pids]

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

    def get_target_distribution(self):
        cleaned_targets = [self.ds.targets[i] for i in self.idx]
        return dict(Counter(cleaned_targets))

    def _balance_dataset(self):
        """Select an equal number of samples per class."""
        # check if balanced and determine minimum number of samples per class
        n_samples = self.get_target_distribution()
        min_n = min(n_samples.values())

        if self.sample_per_class > 0:
            min_n = min((min_n, self.sample_per_class))

        balanced = True
        for t, n in n_samples.items():
            if n != min_n:
                balanced = False

        tmp_targets = np.array(self.ds.targets)

        # Remove idx candidates from target selection, that have been removed from self.idx earlier.
        if len(tmp_targets) > len(self.idx):
            total_idx = set(range(len(self.ds)))
            tmp_idx = set(self.idx)
            idx_complement = total_idx ^ tmp_idx
            tmp_targets[np.array(list(idx_complement)).astype(int)] = -1

        # randomly choose samples from classes
        if not balanced:
            self.idx = list(np.concatenate(
                [np.random.choice(np.argwhere(tmp_targets == t).flatten(), size=min_n, replace=False) for t in
                 n_samples.keys()]))

        logging.info(f"Class sample distribution after balancing procedure: {self.get_target_distribution()}")

    def _replace_ds(self, idx: Sequence[int]) -> None:
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

    def _add_normalize(self):

        if isinstance(self.ds, Subset):
            normalize_exists = sum(['Normalize' in str(t) for t in self.ds.dataset.transform.transforms]) > 0
        elif isinstance(self.ds, ImageFolder):
            normalize_exists = sum(['Normalize' in str(t) for t in self.ds.transform.transforms]) > 0
        else:
            raise TypeError(f"The dataset has the wrong type: {type(self.ds)}")

        if not normalize_exists:
            means, stds = channel_mean_std(self.ds)
            norm = transforms.Normalize(means, stds)
            if isinstance(self.ds, Subset):
                self.ds.dataset.transform.transforms.append(norm)
            else:
                self.ds.transform.transforms.append(norm)

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)


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

        # Calculate absolute number of samples from split percentages.
        abs_split = list(np.floor(np.array(self.split)[:2] * len(self.ds)).astype(np.int32))
        abs_split.append(len(self.ds) - np.sum(abs_split))

        self.train_ds, self.val_ds, self.test_ds = random_split(self.ds, abs_split)

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)


class GTiffDataset(Dataset):
    """
    Pytorch dataset creating samples form a GeoTiff dataset using a moving window approach
    """

    # TODO: Distinguish between labeled and non-labeled dataset!
    # If I want to rely on labels, I need to crop the dataset to the labeled area. Else I can use the whole dataset.
    # Cropping also makes the memory consumption of the label masks manageable!

    def __init__(self, filename: Union[str | os.PathLike], shape_dir: Union[str | os.PathLike],
                 window_size: int = 128, stride: int = 1):
        """
        :param filename: Path to the GeoTiff file
        :param window_size: Edge length of the quadratic moving window
        :param stride: Step size of the moving window (no padding)
        """
        super().__init__()
        self.filename = filename
        self.window_size = window_size
        self.stride = stride
        # padding = 0
        self.rds = rio.open(filename)

        # load shape masks
        self.classes = []
        self.class_masks = []
        self.class_mask_transforms = []

        for root, directory, files in os.walk(shape_dir):
            for file in files:
                if os.path.splitext(file)[1] == ".shp":
                    cls = os.path.splitext(file.split("_")[-1])[0]
                    self.classes.append(cls)

                    # Save cropped masks and transforms instead of paths
                    # Then the retrieving the class coverage is just a simple lookup without any I/O operation.
                    # I need to transform between the cropped mask and the full area.
                    shape_path = os.path.join(shape_dir, file)
                    with fiona.open(shape_path) as shape_file:
                        shapes = [feature["geometry"] for feature in shape_file]

                    # Weirdly sometimes None gets into shapes, which leads to errors in the masking process.
                    # Prevent that!
                    shapes = [shape for shape in shapes if shape is not None]
                    shape_mask, shape_transform = rio.mask.mask(self.rds, shapes, crop=True)

                    shape_mask = shape_mask[3] > 0
                    self.class_masks.append(shape_mask)
                    self.class_mask_transforms.append(shape_transform)

        self.classes.append('soil')
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.n_windows_x = self._get_n_windows_x()
        self.n_windows_y = self._get_n_windows_y()

        # Minimum coverage of any data in a window to be considered as a sample
        self.min_cover_factor = 2/3
        self.min_cover = self.window_size ** 2 * self.min_cover_factor

        # Minimum coverage of a class in a window to be considered a sample for that class
        self.min_cls_cover_factor = .01
        self.min_cls_cover = self.window_size ** self.min_cls_cover_factor

        # only use bounding boxes that contain a minimum amount of data
        self.bb_path = f"{os.path.splitext(self.filename)[0]}_bbs_{self.window_size}-{self.stride}.npy"
        if not os.path.exists(self.bb_path):
            self.bbs = self._preselect_windows()
        else:
            self.bbs = np.load(self.bb_path)

        # cache targets
        self.targets_path = f"{os.path.splitext(self.filename)[0]}_targets_{self.window_size}-{self.stride}.npy"
        if not os.path.exists(self.targets_path):
            self.targets = [-1] * len(self.bbs)
        else:
            self.targets = np.load(self.targets_path)

    def __del__(self):
        self.rds.close()
        np.save(self.targets_path, self.targets)

    def _get_n_windows_x(self):
        """
        Simple but a little costly approach for determining the number of windows in the x axis.
        """
        n = 1
        x = 0
        while x + self.window_size < self.rds.width:
            x += self.stride
            n += 1
        return n

    def _get_n_windows_y(self):
        """
        Simple but a little costly approach for determining the number of windows in the y axis.
        """
        n = 1
        y = 0
        while y + self.window_size < self.rds.height:
            y += self.stride
            n += 1
        return n

    def _get_bounding_box_from_index(self, index):
        """
        Calculate the bounding box of the window with given index before filtering.
        :param index: window index
        """
        x_min = (index % self.n_windows_x) * self.stride
        y_min = (index // self.n_windows_x) * self.stride
        x_max = x_min + self.window_size
        y_max = y_min + self.window_size
        return x_min, x_max, y_min, y_max

    def _preselect_windows(self) -> np.ndarray:
        """
        Select windows worth showing and store them as a serialized numpy array.
        :return: Interesting bounding boxes.
        """
        bbs = []

        p_bar = tqdm(range(self.raw_len()))
        p_bar.set_description("Choosing windows to use")
        mask = self.get_mask_bool()

        for i in p_bar:
            bb = self._get_bounding_box_from_index(i)
            x_min, x_max, y_min, y_max = bb
            if not mask[x_min:x_max, y_min].any():
                # skip windows with empty first lines
                continue
            mask_window = mask[x_min:x_max, y_min:y_max]
            cover = np.sum(mask_window)
            if cover >= self.min_cover:
                bbs.append(np.array(bb, dtype=np.uint32))

        bbs = np.array(bbs)
        np.save(self.bb_path, bbs)
        return bbs

    def get_bb_data(self, bb: Union[tuple | np.ndarray]) -> torch.Tensor:
        """
        Get RGB data in the given bounding box
        :param bb: bounding box (x_min, x_max, y_min, y_max)
        :return: Tensor containing the data
        """
        x_min, x_max, y_min, y_max = bb
        window = Window.from_slices((x_min, x_max), (y_min, y_max))
        return torch.from_numpy(self.rds.read((1, 2, 3), window=window))

    def getitem_raw(self, index):
        """
        Get window by non-filtered index.
        :param index: Index of the window without filtering in self._preselect_windows()
        """
        bb = self._get_bounding_box_from_index(index)
        sample = self.get_bb_data(bb)
        target = self.get_bb_label(bb)

        return sample, target

    def __getitem__(self, index) -> T_co:
        bb = self.bbs[index]
        sample = self.get_bb_data(bb)

        # Caching of targets
        if self.targets[index] > -1:
            target = self.targets[index]
        else:
            target = self.get_bb_label(bb)
            self.targets[index] = target

        return sample, target

    def __len__(self):
        return self.bbs.shape[0]

    def get_bb_label(self, bb: Union[tuple, np.ndarray]) -> int:
        """
        Determine which of the labels should be assigned to the given bounding box (only one label is possible).
        :param bb: Bounding box.
        :return: label
        """
        coverages = self.get_all_bb_class_coverages(bb, True)

        if not (coverages >= self.min_cls_cover_factor).any():
            return self.class_to_idx['soil']

        return int(np.argmax(coverages))

    def get_bb_cls_coverage(self, bb: Union[tuple | np.ndarray], cls_idx: int, share: bool = False) \
            -> Union[int | float]:
        """
        Get the number of pixels in the given bounding box that are covered with the given class.
        :param bb: Boundig box.
        :param cls_idx: Class to determine the coverage for.
        :param share: If true, return percentual coverage, absolute coverage otherwise.
        :return: Class coverage in the window.
        """
        if cls_idx > len(self.classes) - 2:
            raise ValueError(f"Species with class index {cls_idx} does not exits. "
                             f"Please choose from {list(np.array(self.classes)[:-1])}")

        x_min, x_max, y_min, y_max = bb

        # get mask and transform
        cls_mask_cropped = self.class_masks[cls_idx]
        cls_mask_transform = self.class_mask_transforms[cls_idx]

        # map origin of cropped shape mask to geo-referenced system
        min_point_geo = cls_mask_transform * (0, 0)

        # map geo-referenced origin back to the non-cropped dataset
        min_point_ds = ~self.rds.transform * min_point_geo

        # Here the axes have to be switched for some reason
        min_point_ds = tuple(reversed(min_point_ds))

        # initialize class-specific shape mask
        cls_mask = np.zeros((self.rds.height, self.rds.width), dtype=bool)

        # get cropped cover
        idxs = np.argwhere(cls_mask_cropped)

        # map to non-cropped system
        idxs = (idxs + min_point_ds).astype(int)

        # apply cover
        cls_mask[idxs[:, 0], idxs[:, 1]] = True

        # decide which calculate numeric coverage
        coverage = int(np.sum(cls_mask[x_min:x_max, y_min:y_max]))
        if share:
            coverage /= self.window_size ** 2
        return coverage

    def get_all_bb_class_coverages(self, bb: Union[tuple | np.ndarray], share: bool = False):
        coverages = []
        for cls_idx in range(len(self.classes) - 1):
            coverage = self.get_bb_cls_coverage(bb, cls_idx, share)
            coverages.append(coverage)

        return np.array(coverages)

    def raw_len(self):
        return self.n_windows_x * self.n_windows_y

    def get_red(self):
        return self.rds.read(1)

    def get_green(self):
        return self.rds.read(2)

    def get_blue(self):
        return self.rds.read(3)

    def get_mask(self):
        return self.rds.read(4)

    def get_mask_bool(self):
        return self.rds.read(4) > 0


