from abc import ABC
from copy import deepcopy
from dataclasses import asdict, dataclass

import affine
import pytorch_lightning as pl
import yaml
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset, Dataset, IterableDataset
from torch.utils.data.dataset import T_co
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

import pandas as pd
from PIL import Image
import rasterio as rio
from rasterio.windows import Window
import fiona
from tqdm import tqdm

from typing import Optional, Union, Sequence, Tuple
from pathlib import Path
import os
from collections import Counter
import logging

from citizenuav.transforms import *
from citizenuav.io import get_pid_from_path, read_split_inat_metadata, empty_dir
from citizenuav.math import channel_mean_std


def validate_split(split: tuple) -> bool:
    """
    Validate the split tuple.
    :param split: Split
    :return: True, if valid.
    """
    if (split_sum := np.round(np.sum(split), 10)) != 1.:
        raise ValueError(f"The sum of split has to be 1. Got {split_sum}")
    return True


def validate_data_dir(data_dir: Union[str, Path]) -> bool:
    """
    Validate the data directory.
    :param data_dir: Data directory path.
    :return: True, if valid.
    """
    if not os.path.isdir(data_dir):
        raise ValueError(f"data_dir={data_dir}: No such directory found.")

    # Check if distance dataset
    if not os.path.exists(os.path.join(data_dir, "distances.csv")):

        # Has to be classification dataset
        # Check for metadata
        _, subdirs, _ = next(os.walk(data_dir))
        for d in subdirs:
            if not os.path.exists(os.path.join(data_dir, d, "metadata.csv")):
                raise ValueError(f"No metadata.csv or distances.csv file found in {data_dir}")

    return True


class InatImageFolderWithPath(ImageFolder):
    """
    Subclass of torchvision.datasets.ImageFolder that also outputs the path to an image and that makes image samples
    accessible by their assigned photo ID.
    """
    def __getitem__(self, item):
        img, t = super().__getitem__(item)
        path, _ = self.samples[item]
        return img, t, path

    def get_item_by_pid(self, pid: str):
        """
        Get item by photo ID.
        :param pid: photo ID
        :return: image tensor, target, path
        """
        for idx, (path, t) in enumerate(self.samples):
            if get_pid_from_path(path) == pid:
                return self[idx]

        return None


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
    def __init__(self, data_dir: Union[str, Path], species: Optional[Union[list, str]] = None, batch_size: int = 4,
                 split: tuple = (.72, .18, .1), img_size: int = 128, min_distance: float = None, balance: bool = False,
                 sample_per_class: int = -1, normalize: bool = False, return_path: bool = False, **kwargs):
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

        self.data_dir = data_dir

        self.metadata = read_split_inat_metadata(self.data_dir, species)
        if 'label' in self.metadata.columns:
            self.metadata.drop(columns=['label'], inplace=True)
        self.batch_size = batch_size
        self.normalize = normalize
        self.return_path = return_path

        # Compose transformations for the output samples and create the dataset object.
        transforms_list = [transforms.ToTensor(), QuadCrop(), transforms.Resize(img_size)]

        img_transform = transforms.Compose(transforms_list)
        if self.return_path:
            self.ds = InatImageFolderWithPath(str(self.data_dir), transform=img_transform)
        else:
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
        if self.balance or (self.sample_per_class is not None and self.sample_per_class > 0):
            self._balance_dataset()

        self._replace_ds(self.idx)

        if self.normalize:
            self.add_normalize()

        # Calculate absolute number of samples from split percentages.
        abs_split = list(np.floor(np.array(self.split)[:2] * len(self.ds)).astype(np.int32))
        abs_split.append(len(self.ds) - np.sum(abs_split))

        self.train_ds, self.val_ds, self.test_ds = random_split(self.ds, abs_split)

    def _filter_broken_images(self):
        """
        Exclude samples that have a non-accessible image file.
        """
        if 'image_okay' not in self.metadata and 'broken' not in self.metadata:
            logging.warning("No information about broken images in the metadata!")
        if 'image_okay' in self.metadata:
            okay_pids = self.metadata[self.metadata.image_okay].index
            self.idx = [i for i in self.idx if get_pid_from_path(self.ds.samples[i][0]) in okay_pids]
            # prioritize image_okay column
            return
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
        assert isinstance(self.ds, ImageFolder)

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
        self.idx = range(len(self.ds))

    def get_channel_mean_std(self) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Determine or load channel means and standard deviations for the data in data_dir.
        :return: Means and standard deviations as float tensors.
        """
        filename = os.path.join(self.data_dir, 'mean_std.yml')
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                data = yaml.safe_load(file)
            means = torch.FloatTensor(data['means'])
            stds = torch.FloatTensor(data['stds'])
        else:
            means, stds = channel_mean_std(self.ds)
            data = {
                'means': means.numpy().tolist(),
                'stds': stds.numpy().tolist()
            }
            with open(filename, 'w') as file:
                yaml.dump(data, file)

        return means, stds

    def get_normalize_module(self) -> transforms.Normalize:
        """
        Create a torchvision.transforms.Normalize module using the channel-wise means and standard deviations for
        the data in data_dir.
        :return: Normalize module.
        """
        means, stds = self.get_channel_mean_std()
        return transforms.Normalize(means, stds)

    def add_normalize(self, norm: Optional[nn.Module] = None):
        """
        Append a normalizing module to the end of the transforms for the dataset.
        """

        if isinstance(self.ds, Subset):
            normalize_exists = sum(['Normalize' in str(t) for t in self.ds.dataset.transform.transforms]) > 0
        elif isinstance(self.ds, ImageFolder):
            normalize_exists = sum(['Normalize' in str(t) for t in self.ds.transform.transforms]) > 0
        else:
            raise TypeError(f"The dataset has the wrong type: {type(self.ds)}")

        if not normalize_exists:
            if norm is None:
                norm = self.get_normalize_module()
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
    """
    Dataset for distance training data.
    """

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

    def __getitem__(self, idx) -> T_co:
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
    """
    Data module for distance training data.
    """

    @staticmethod
    def add_dm_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("InatDistDataModule")
        parser.add_argument("--data_dir", type=str)
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--split", type=tuple, default=(.72, .18, .1))
        parser.add_argument("--img_size", type=int, default=128, choices=[2 ** x for x in range(6, 10)])
        return parent_parser

    def __init__(self, data_dir: Path, batch_size: int = 4, split: tuple = (.72, .18, .1), img_size: int = 128,
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

    def __init__(self, filename: Union[str, Path], shape_dir: Optional[Union[str, Path]] = None,
                 window_size: int = 128, stride: int = 1, normalize: bool = False, means: Optional[tuple] = None,
                 stds: Optional[tuple] = None):
        """
        :param filename: Path to the GeoTiff file
        :param shape_dir: Path to the directory containing the shape files needed for determining class labels.
            If None, None will be returned as a target for each sample.
        :param window_size: Edge length of the quadratic moving window
        :param stride: Step size of the moving window (no padding)
        """
        super().__init__()
        self.filename = filename
        self.shape_dir = shape_dir
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize or (means is not None and stds is not None)
        # padding = 0
        self.rds = rio.open(filename)

        if self.normalize:
            if means is not None:
                norm_means = np.array(means)
            else:
                norm_means = np.array([
                    self.get_red().mean(),
                    self.get_green().mean(),
                    self.get_blue().mean()
                ])
            if stds is not None:
                norm_stds = np.array(stds)
            else:
                norm_stds = np.array([
                    self.get_red().std(),
                    self.get_green().std(),
                    self.get_blue().std()
                ])
            self.norm = transforms.Normalize(norm_means, norm_stds)

        # load shape masks
        self.classes = []
        self.class_masks = []
        self.class_mask_transforms = []
        self.labeled_area_cropped = None

        if self.shape_dir is not None:

            # get labeled-area
            prefix = os.path.basename(self.shape_dir)
            la_shapes, la_transform = self.get_shapes_from_file(f"{prefix}_labeled-area.shp", True)
            self.labeled_area_cropped = la_shapes
            self.labeled_area_transform = la_transform

            self.return_targets = True
            for root, directory, files in os.walk(self.shape_dir):
                for file in files:
                    if 'labeled-area' in file:
                        # skip mask for labeled area
                        continue
                    if os.path.splitext(file)[1] == ".shp":
                        cls = os.path.splitext(file.split("_")[-1])[0]
                        self.classes.append(cls)

                        # Save cropped masks and transforms instead of paths
                        shape_mask, shape_transform = self.get_shapes_from_file(file)
                        self.class_masks.append(shape_mask)
                        self.class_mask_transforms.append(shape_transform)

            # add soil mask
            self.classes.append('soil')
            # the soil mask is each pixel of the labeled area that is not labeled as any other class
            soil_mask = self.uncrop_mask(self.labeled_area_cropped, self.labeled_area_transform).copy()
            for cls_transform, cls_mask in zip(self.class_mask_transforms, self.class_masks):
                soil_mask &= ~self.uncrop_mask(cls_mask, cls_transform)

            # determine minimum point of the labeled area in the dataset coordinate system
            min_point_geo = self.labeled_area_transform * (0, 0)
            min_point_ds = ~self.rds.transform * min_point_geo
            min_point_ds = tuple(reversed(min_point_ds))

            # crop soil mask to the extent of the labeled area
            soil_mask_cropped = soil_mask[
                                int(min_point_ds[0]):int(min_point_ds[0]) + self.labeled_area_cropped.shape[0],
                                int(min_point_ds[1]):int(min_point_ds[1]) + self.labeled_area_cropped.shape[1]]
            self.class_masks.append(soil_mask_cropped.copy())
            self.class_mask_transforms.append(deepcopy(self.labeled_area_transform))
            del soil_mask
            del soil_mask_cropped

            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        else:
            self.return_targets = False

        self.n_windows_x = self._get_n_windows_x()
        self.n_windows_y = self._get_n_windows_y()

        # Minimum coverage of any data in a window to be considered as a sample
        self.min_cover_factor = 3 / 4
        self.min_cover = self.window_size ** 2 * self.min_cover_factor

        if self.return_targets:
            # Minimum coverage of a class in a window to be considered a sample for that class
            self.min_cls_cover_factor = .01
            self.min_cls_cover = self.window_size ** self.min_cls_cover_factor

        cache_filename = f"{os.path.splitext(self.filename)[0]}-{self.window_size}-{self.stride}{'-labeled' if self.shape_dir else ''}"

        # only use bounding boxes that contain a minimum amount of data
        self.bb_path = f"{cache_filename}_bbs.npy"
        if not os.path.exists(self.bb_path):
            self.bbs = self._preselect_windows()
        else:
            self.bbs = np.load(self.bb_path)

        if self.return_targets:
            # cache targets
            self.targets_path = f"{cache_filename}_targets.npy"
            if not os.path.exists(self.targets_path):
                self.targets = [-1] * len(self.bbs)
            else:
                self.targets = np.load(self.targets_path)

    def __del__(self):
        self.rds.close()
        if self.return_targets:
            np.save(self.targets_path, self.targets)

    def get_labeled_area(self) -> Union[np.ndarray, None]:
        """
        Load boolean mask for labeled pixels cropped to the dataset extent or None if no labels exist.
        :return: Boolean mask for labeled pixels or None.
        """
        if self.labeled_area_cropped is None:
            return None
        labeled_area = self.uncrop_mask(self.labeled_area_cropped, self.labeled_area_transform)
        return labeled_area

    def get_shapes_from_file(self, filename, crop_mask: bool = True) -> Tuple[np.ndarray, affine.Affine]:
        """
        Load label shapes from file and return as a boolean mask together with dataset transformation.
        :param filename: Name of the shape file. The complete path is created using the shape_dir parameter.
        :param crop_mask: If True, crop the label mask to its content.
        :return: Boolean label mask and transformation
        """
        # Then the retrieving the class coverage is just a simple lookup without any I/O operation.
        # I need to transform between the cropped mask and the full area.
        shape_path = os.path.join(self.shape_dir, filename)
        with fiona.open(shape_path) as shape_file:
            shapes = [feature["geometry"] for feature in shape_file]

        # Weirdly sometimes None gets into shapes, which leads to errors in the masking process.
        # Prevent that!
        shapes = [shape for shape in shapes if shape is not None]
        shape_mask, shape_transform = rio.mask.mask(self.rds, shapes, crop=crop_mask)

        shape_mask = shape_mask[3] > 0
        return shape_mask, shape_transform

    def _get_n_windows_x(self) -> int:
        """
        Simple but a little costly approach for determining the number of windows in the x axis.
        :return: Number of windows in x axis.
        """
        n = 1
        x = 0
        while x + self.window_size < self.rds.height:
            x += self.stride
            n += 1
        if x > self.rds.height:
            # We reached beyond the boundary
            n -= 1

        return n

    def _get_n_windows_y(self) -> int:
        """
        Simple but a little costly approach for determining the number of windows in the y axis.
        :return: Number of windows in y axis.
        """
        n = 1
        y = 0
        while y + self.window_size < self.rds.width:
            y += self.stride
            n += 1
        if y > self.rds.width:
            # We reached beyond the boundary
            n -= 1

        return n

    def _get_bounding_box_from_index(self, index) -> Tuple[int, int, int, int]:
        """
        Calculate the bounding box of the window with given index before filtering.
        :param index: window index
        :return x_min, x_max, y_min, y_max
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
        if self.labeled_area_cropped is not None:
            mask = self.get_labeled_area()
        else:
            mask = self.get_mask_bool()

        for i in p_bar:
            bb = self._get_bounding_box_from_index(i)
            x_min, x_max, y_min, y_max = bb
            if x_max >= mask.shape[0] or y_max >= mask.shape[1]:
                continue

            # skip windows with empty first lines
            if not mask[x_min, y_min:y_max].any():
                continue
            if not mask[x_min:x_max, y_min].any():
                continue

            mask_window = mask[x_min:x_max, y_min:y_max]
            cover = np.sum(mask_window)
            if cover >= self.min_cover:
                bbs.append(np.array(bb, dtype=np.uint32))

        bbs = np.array(bbs)
        np.save(self.bb_path, bbs)
        return bbs

    def get_bb_data(self, bb: Union[tuple, np.ndarray], normalize: Optional[bool] = None) -> torch.Tensor:
        """
        Get RGB data in the given bounding box
        :param bb: bounding box (x_min, x_max, y_min, y_max)
        :param normalize: Normalize the sample
        :return: Tensor containing the data
        """
        if normalize is None:
            normalize = self.normalize

        x_min, x_max, y_min, y_max = bb
        window = Window.from_slices((x_min, x_max), (y_min, y_max))
        tensor = torch.from_numpy(self.rds.read((1, 2, 3), window=window)).float() / 255.
        if normalize:
            tensor = self.norm(tensor)
        return tensor

    def getitem_raw(self, index) -> T_co:
        """
        Get window by non-filtered index.
        :param index: Index of the window without filtering in self._preselect_windows()
        :return: Sample image as a Tensor and target.
        """
        bb = self._get_bounding_box_from_index(index)
        sample = self.get_bb_data(bb)
        target = self.get_bb_label(bb)

        return sample, target

    def __getitem__(self, index) -> T_co:
        bb = self.bbs[index]
        sample = self.get_bb_data(bb)

        if self.return_targets:
            # Caching of targets
            if self.targets[index] > -1:
                target = self.targets[index]
            else:
                target = self.get_bb_label(bb)
                self.targets[index] = target
        else:
            target = 0

        return sample, target

    def get_non_normalized_item(self, index) -> T_co:
        """
        Get item without normalization.
        :return: Sample image and target.
        """
        bb = self.bbs[index]
        sample = self.get_bb_data(bb, False)

        if self.return_targets:
            # Caching of targets
            if self.targets[index] > -1:
                target = self.targets[index]
            else:
                target = self.get_bb_label(bb)
                self.targets[index] = target
        else:
            target = 0

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

    def get_pix_label(self, x: int, y: int) -> int:
        """
        Get label in non-geo-referenced position (x, y).
        :param x:
        :param y:
        :return: Label in position (x, y).
        """
        for cls_idx in range(len(self.classes) - 1):
            if self.get_cls_mask_in_bb((x, x+1, y, y+1), cls_idx).all():
                return cls_idx
        return self.class_to_idx['soil']

    def uncrop_mask(self, mask: np.ndarray, transform: affine.Affine) -> np.ndarray:
        """
        Transform a mask cropped to the content of that mask back to the large dataset.
        :param mask: Cropped mask.
        :param transform: Transform that was put out by rio.mask.mask(ds, shapes, crop=True)
        :return: The uncropped mask.
        """

        # map origin of cropped shape mask to geo-referenced system
        min_point_geo = transform * (0, 0)

        # map geo-referenced origin back to the non-cropped dataset
        min_point_ds = ~self.rds.transform * min_point_geo

        # Here the axes have to be swapped for some reason
        min_point_ds = tuple(reversed(min_point_ds))

        # initialize uncropped shape mask
        mask_uncropped = np.zeros((self.rds.height, self.rds.width), dtype=bool)

        # get cropped cover
        idxs = np.argwhere(mask)

        # map to non-cropped system
        idxs = (idxs + min_point_ds).astype(int)

        # apply cover
        mask_uncropped[idxs[:, 0], idxs[:, 1]] = True

        return mask_uncropped

    def get_cls_mask(self, cls_idx: int) -> np.ndarray:
        """
        Get uncropped class mask for given class.
        :param cls_idx: Index of wanted class.
        :return: boolean mask of class coverage.
        """
        # get mask and transform
        cls_mask_cropped = self.class_masks[cls_idx]
        cls_mask_transform = self.class_mask_transforms[cls_idx]

        cls_mask = self.uncrop_mask(cls_mask_cropped, cls_mask_transform)
        return cls_mask

    def get_cls_mask_in_bb(self, bb: Union[tuple, np.ndarray], cls_idx: int) -> np.ndarray:
        """
        Get class mask for given bounding box.
        :return: Class mask for bounding box.
        """
        x_min, x_max, y_min, y_max = bb

        cls_mask = self.get_cls_mask(cls_idx)

        return cls_mask[x_min:x_max, y_min:y_max]

    def get_bb_cls_coverage(self, bb: Union[tuple, np.ndarray], cls_idx: int, share: bool = False) \
            -> Union[int, float]:
        """
        Get the number of pixels in the given bounding box that are covered with the given class.
        :param bb: Bounding box.
        :param cls_idx: Class to determine the coverage for.
        :param share: If true, return percentual coverage, absolute coverage otherwise.
        :return: Class coverage in the window.
        """
        if cls_idx > len(self.classes) - 2:
            raise ValueError(f"Species with class index {cls_idx} does not exits. "
                             f"Please choose from {list(np.array(self.classes)[:-1])}")

        # calculate numeric coverage
        coverage = int(np.sum(self.get_cls_mask_in_bb(bb, cls_idx)))
        if share:
            coverage /= self.window_size ** 2
        return coverage

    def get_all_bb_class_coverages(self, bb: Union[tuple, np.ndarray], share: bool = False):
        """
        Collection of class coverages in given bounding box.
        :param bb: Bounding box coordinates x_min, x_max, y_min, y_max
        :param share: If True, the proportion of coverage will be returned, the absolute number of pixels otherwise.
        """
        coverages = []
        # soil is always the last class
        # here it is excluded
        for cls_idx in range(len(self.classes) - 1):
            coverage = self.get_bb_cls_coverage(bb, cls_idx, share)
            coverages.append(coverage)

        return np.array(coverages)

    def raw_len(self) -> int:
        return self.n_windows_x * self.n_windows_y

    def get_red(self) -> np.ndarray:
        return self.rds.read(1) / 255.

    def get_green(self) -> np.ndarray:
        return self.rds.read(2) / 255.

    def get_blue(self) -> np.ndarray:
        return self.rds.read(3) / 255.

    def get_mask(self) -> np.ndarray:
        return self.rds.read(4)

    def get_mask_bool(self) -> np.ndarray:
        return self.get_mask() > 0


class MixedDataModule(pl.LightningDataModule):
    """
    This data module is meant for including an amount of raster data samples into the training process.
    """

    @staticmethod
    def add_dm_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MixedDataModule")
        parser.add_argument("--data_dir", type=str)
        parser.add_argument("--inat_dir", type=str)
        parser.add_argument("--raster_dir", type=str)
        parser.add_argument("--n_inat_samples_per_class", type=int)
        parser.add_argument("--n_raster_samples_per_class", type=int)
        parser.add_argument("--batch_size", type=int, default=4, required=False)
        parser.add_argument("--split", type=float, nargs=3, default=(.72, .18, .1), required=False)
        parser.add_argument("--img_size", type=int, default=128, required=False, choices=[2 ** x for x in range(6, 10)])
        parser.add_argument("--balance", type=bool, default=False, required=False)
        parser.add_argument("--normalize", type=bool, default=True, required=False)

        return parent_parser

    def __init__(
            self,
            data_dir: Union[str, Path],
            inat_dir: Union[str, Path],
            raster_dir: Union[str, Path],
            n_inat_samples_per_class: int,
            n_raster_samples_per_class: int,
            batch_size: int = 4,
            split: tuple = (.72, .18, .1),
            img_size: int = 128,
            balance: bool = True,
            normalize: bool = True,
            return_path: bool = False,
            **kwargs
    ):
        super().__init__()

        # Make sure split sums up to 1.
        validate_split(split)
        self.split = split

        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.inat_dir = inat_dir
        self.n_inat_samples_per_class = n_inat_samples_per_class
        self.n_raster_samples_per_class = n_raster_samples_per_class
        self.raster_dir = raster_dir
        self.batch_size = batch_size
        self.balance = balance
        self.normalize = normalize
        self.return_path = return_path
        self.num_workers = os.cpu_count()

        # Compose transformations for the output samples and create the dataset object.
        self.img_transform = transforms.Compose([transforms.ToTensor(), QuadCrop(), transforms.Resize(img_size)])

        if not self.ready() or True:

            cache_file = os.path.join(self.data_dir, 'mean_std.yml')
            if os.path.exists(cache_file):
                os.remove(cache_file)

            # build mixed dataset
            inat_ds = ImageFolder(self.inat_dir, transform=self.img_transform)
            raster_ds = ImageFolder(self.raster_dir, transform=self.img_transform)

            # For now just cover the case that classes match exactly
            if inat_ds.class_to_idx != raster_ds.class_to_idx:
                raise KeyError(f"Classes of given datasets do not match! {inat_ds.classes}, {raster_ds.classes}")

            for cls_name in inat_ds.classes:
                for source in ['inat', 'raster']:
                    sink_class_path = os.path.join(self.data_dir, cls_name, source)
                    if not os.path.exists(sink_class_path):
                        os.makedirs(sink_class_path)
                    elif os.listdir(sink_class_path):
                        empty_dir(sink_class_path)

                    source_ds = eval(f"{source}_ds")
                    source_idx = range(len(source_ds))
                    n_source_samples_per_class = eval(f"self.n_{source}_samples_per_class")
                    if self.balance:
                        possible_samples_per_class = dict(Counter(source_ds.targets))[source_ds.class_to_idx[cls_name]]
                        n_source_samples_per_class = min([n_source_samples_per_class, possible_samples_per_class])

                    source_idx = [i for i in source_idx if source_ds.classes[source_ds.targets[i]] == cls_name]
                    source_idx = np.random.choice(source_idx, size=n_source_samples_per_class, replace=False)

                    p_bar = tqdm(source_idx)
                    p_bar.set_description(f"Preparing class {cls_name} from source {source}")
                    for i in p_bar:
                        img, t = source_ds[i]
                        img = to_pil_image(img)
                        img_source_path, _ = source_ds.samples[i]
                        img_name = f"{get_pid_from_path(img_source_path)}.png"
                        img_sink_path = os.path.join(sink_class_path, img_name)
                        img.save(img_sink_path, 'png')

        self.ds = ImageFolder(self.data_dir, self.img_transform)
        self.ds.raster = [os.path.basename(os.path.dirname(s[0])) == 'raster' for s in self.ds.samples]
        if normalize:
            self._add_normalize()

        idx = range(len(self.ds))
        inat_idx = [i for i in idx if self.ds.raster[i]]
        inat_split = list(np.floor(np.array(split)[:2] * len(inat_idx)).astype(np.int32))
        inat_split.append(len(inat_idx) - np.sum(inat_split))
        np.random.shuffle(inat_idx)
        inat_train_idx = inat_idx[:inat_split[0]]
        inat_val_idx = inat_idx[inat_split[0]:inat_split[0] + inat_split[1]]
        inat_test_idx = inat_idx[inat_split[0] + inat_split[1]:]

        raster_idx = list(set(list(idx)) ^ set(inat_idx))
        raster_split = list(np.floor(np.array(split)[:2] * len(raster_idx)).astype(np.int32))
        raster_split.append(len(raster_idx) - np.sum(raster_split))
        np.random.shuffle(raster_idx)
        raster_train_idx = raster_idx[:raster_split[0]]
        raster_val_idx = raster_idx[raster_split[0]:raster_split[0] + raster_split[1]]
        raster_test_idx = raster_idx[raster_split[0] + raster_split[1]:]

        train_idx = list(inat_train_idx) + list(raster_train_idx)
        self.train_ds = Subset(self.ds, train_idx)
        val_idx = list(inat_val_idx) + list(raster_val_idx)
        self.val_ds = Subset(self.ds, val_idx)
        test_idx = list(inat_test_idx) + list(raster_test_idx)
        self.test_ds = Subset(self.ds, test_idx)

    def ready(self):
        ready = True
        classes = ImageFolder(self.inat_dir).classes
        for cls_name in classes:
            for source in ['inat', 'raster']:
                sink_class_path = os.path.join(self.data_dir, cls_name, source)
                n_source_samples_per_class = eval(f"self.n_{source}_samples_per_class")
                if self.balance:
                    source_class_path = os.path.join(eval(f"self.{source}_dir"), cls_name)
                    possible_samples_per_class = len([fn for fn in os.listdir(source_class_path) if fn[-4:] == '.png'])
                    n_source_samples_per_class = min([n_source_samples_per_class, possible_samples_per_class])
                if not os.path.exists(sink_class_path) or len(os.listdir(sink_class_path)) < n_source_samples_per_class:
                    ready = False

        return ready

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def _add_normalize(self, norm: Optional[nn.Module] = None):
        if isinstance(self.ds, Subset):
            normalize_exists = sum(['Normalize' in str(t) for t in self.ds.dataset.transform.transforms]) > 0
        elif isinstance(self.ds, ImageFolder):
            normalize_exists = sum(['Normalize' in str(t) for t in self.ds.transform.transforms]) > 0
        else:
            raise TypeError(f"The dataset has the wrong type: {type(self.ds)}")

        if not normalize_exists:
            if norm is None:
                norm = self.get_normalize_module()
            if isinstance(self.ds, Subset):
                self.ds.dataset.transform.transforms.append(norm)
            else:
                self.ds.transform.transforms.append(norm)

    def get_normalize_module(self):
        means, stds = self.get_channel_mean_std()
        return transforms.Normalize(means, stds)

    def get_channel_mean_std(self):
        filename = os.path.join(self.data_dir, 'mean_std.yml')
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                data = yaml.safe_load(file)
            means = torch.FloatTensor(data['means'])
            stds = torch.FloatTensor(data['stds'])
        else:
            means, stds = channel_mean_std(self.ds)
            data = {
                'means': means.numpy().tolist(),
                'stds': stds.numpy().tolist()
            }
            with open(filename, 'w') as file:
                yaml.dump(data, file)

        return means, stds


class RasterValidationDataModule(pl.LightningDataModule):
    """
    This data module is meant for training a classifier on iNaturalist data and validate the model performance on
    another raster dataset (instance of GTiffDataset).
    """

    @staticmethod
    def add_dm_specific_args(parent_parser):
        inat_parser = InatDataModule.add_dm_specific_args(parent_parser)
        parser = inat_parser.add_argument_group("MixedDataModule")
        parser.add_argument("--val_data_dir", type=str, required=True)
        parser.add_argument("--val_size", type=float, required=False, default=1.)
        return parent_parser

    def __init__(self, **kwargs):
        super().__init__()

        self.val_data_dir = kwargs.pop('val_data_dir')
        self.val_size = kwargs.pop('val_size', 1.)
        normalize_mixed = kwargs.pop('normalize', False)
        kwargs['normalize'] = False

        # use iNat data for training only
        kwargs['split'] = (1., 0., 0.)

        self.inat_dm = InatDataModule(**kwargs)
        self.batch_size = self.inat_dm.batch_size
        self.num_workers = self.inat_dm.num_workers

        self.val_ds = ImageFolder(
            self.val_data_dir,
            transform=self.inat_dm.train_ds.dataset.dataset.transform
        )

        val_idx = np.random.choice(range(len(self.val_ds)), size=int(len(self.val_ds) * self.val_size), replace=False)
        self.val_ds = Subset(self.val_ds, val_idx)

        # normalize over both datasets
        if normalize_mixed:
            if normalize_mixed:
                self._add_normalize()

    def get_channel_mean_std(self):
        filename = os.path.join(self.inat_dm.data_dir, 'mean_std_mixed.yml')
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                data = yaml.safe_load(file)
            means = torch.FloatTensor(data['means'])
            stds = torch.FloatTensor(data['stds'])
        else:
            # calculate means and stds over both datasets combined
            means, stds = channel_mean_std(IterDataset(self.all_samples_gen))
            data = {
                'means': means.numpy().tolist(),
                'stds': stds.numpy().tolist()
            }
            with open(filename, 'w') as file:
                yaml.dump(data, file)

        return means, stds

    def all_samples_gen(self):
        for i in range(len(self.inat_dm.ds)):
            yield self.inat_dm.ds[i]
        for i in range(len(self.val_ds)):
            yield self.val_ds[i]

    def get_normalize_module(self):
        means, stds = self.get_channel_mean_std()
        return transforms.Normalize(means, stds)

    def _add_normalize(self):
        norm = self.get_normalize_module()
        self.inat_dm.add_normalize(norm)
        self.val_ds.dataset.transform.transforms.append(norm)

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.inat_dm.train_dataloader()

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        #return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)
        raise NotImplemented()


class IterDataset(IterableDataset):
    def __getitem__(self, index) -> T_co:
        """
        CAUTION: The index will not be respected here!
        """
        return next(self.generator)

    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator()


class CUAVDataClass(ABC):

    def dict(self):
        return asdict(self)

    def series(self):
        return pd.Series(self.dict())

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    @classmethod
    def from_series(cls, series: pd.Series):
        return cls.from_dict(dict(series))

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        preds = []
        for idx, row in df.iterrows():
            preds.append(cls.from_series(row))
        return preds


@dataclass
class PixPred(CUAVDataClass):
    x: int
    y: int
    prediction: int
    ds_path: str
    model_path: str
    target: Optional[int] = np.nan
    prediction_text: Optional[str] = None
    confidence: Optional[float] = np.nan


@dataclass
class BoxPred(CUAVDataClass):
    """
    Box prediction in a raster dataset.
    """
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    prediction: int
    ds_path: str
    model_path: str
    target: Optional[int] = np.nan
    prediction_text: Optional[str] = None
    confidence: Optional[float] = np.nan
