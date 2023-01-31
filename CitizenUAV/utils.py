from dataclasses import dataclass
import os
import sys
from datetime import datetime
from typing import Tuple, Union, Optional
from pathlib import Path
import logging
import yaml

import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm


def get_pid_from_path(path):
    """
    Takes a path to an INaturalist image file and removes the directories and file extension leaving only the photo ID.
    :param path: Path to the file
    :return int: INaturalist photo ID
    """
    filename = os.path.splitext(os.path.basename(path))[0]
    return str(filename)


def read_inat_metadata(data_dir):
    """
    Reading routine for metadata.csv file for an iNaturalist dataset.
    :param data_dir: The directory the data and its corresponding metadata lie in.
    :return pd.DataFrame: The metadata DataFrame.
    """
    metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    metadata.photo_id = metadata.photo_id.astype(str)
    metadata.set_index('photo_id', inplace=True)
    metadata = metadata[~metadata.index.duplicated(keep='first')]
    if 'Unnamed: 0' in metadata:
        metadata.drop(columns=['Unnamed: 0'], inplace=True)
    if 'distance' in metadata.columns:
        metadata.distance = metadata.distance.astype(float)
    if 'broken' in metadata.columns:
        metadata.broken = metadata.broken.astype(bool)
    if 'path' in metadata.columns:
        metadata.path = metadata.path.astype(str)
    return metadata


def read_split_inat_metadata(data_dir: Union[str, Path], species: Optional[list[str]] = None):
    _, dirs, _ = next(os.walk(data_dir))
    if not len(dirs):
        raise FileNotFoundError(f"No metadata.csv or other subdirectories found in {data_dir}.")
    logging.info("Found the following species:\n" + "\n".join(dirs))

    if species is not None and species:
        dirs = [d for d in dirs if d in species]

    dfs = []
    cols = []
    for d in dirs:
        subdir = os.path.join(data_dir, d)
        if not os.path.exists(os.path.join(subdir, 'metadata.csv')):
            df = read_split_inat_metadata(subdir)
            df.species = d
        else:
            df = read_inat_metadata(subdir)
            if len(df.columns) > len(cols):
                cols = df.columns
        dfs.append(df)

    for df in dfs:
        for col in cols:
            if col not in df.columns:
                df[col] = pd.Series(dtype=object)

    combined = pd.concat(dfs)
    if 'image_okay' in combined.columns:
        combined.image_okay = combined.image_okay.astype(bool)
        combined.image_okay.fillna(False)
    return combined


def store_split_inat_metadata(metadata: pd.DataFrame, data_dir: Union[str, Path]):
    for spec in metadata.species.unique():
        df = metadata[metadata.species == spec]
        assert df.index.name == 'photo_id'
        df.reset_index(inplace=True)
        try:
            df.to_csv(os.path.join(data_dir, spec, 'metadata.csv'), index=False)
            df.to_csv(os.path.join(data_dir, spec, 'metadata_backup.csv'), index=False)
        except KeyboardInterrupt:
            df.to_csv(os.path.join(data_dir, spec, 'metadata.csv'), index=False)
            df.to_csv(os.path.join(data_dir, spec, 'metadata_backup.csv'), index=False)
            sys.exit()


def channel_mean_std(ds: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the mean and standard deviation for each channel of an image dataset.
    :param ds: The image dataset.
    :return: The means and standard deviations.
    """
    dl = DataLoader(ds, batch_size=32, num_workers=1)

    pixel_counter = 0
    first_moment = torch.empty(3)
    second_moment = torch.empty(3)

    pbar = tqdm(dl)
    pbar.set_description("Calculating means and stds for dataset normalization")
    # TODO: on my current dataset for the red channel NaN is returned!
    for images, _ in tqdm(dl):
        b, c, h, w = images.shape
        n_batch_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_squares = torch.sum(images ** 2, dim=[0, 2, 3])
        first_moment = (pixel_counter * first_moment + sum_) / (pixel_counter + n_batch_pixels)
        second_moment = (pixel_counter * second_moment + sum_of_squares) / (pixel_counter + n_batch_pixels)
        pixel_counter += n_batch_pixels

    mean, std = first_moment, torch.sqrt(second_moment - first_moment ** 2)

    return mean, std


def gram_matrix(feature_maps: torch.Tensor) -> torch.Tensor:
    """
    Source: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    :param feature_maps: F_l, all feature maps of layer l
    :return: G_l the gram matrix of layer l
    """
    batch_size, n_filters, width, height = feature_maps.size()

    # resize F_XL into \hat F_XL
    features = feature_maps.view(batch_size * n_filters, width * height)

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(batch_size * n_filters * width * height)


def write_params(dest_dir: Union[str, Path], params: dict, func_name: Optional[str] = None):
    filename = os.path.join(dest_dir,
                            f'{datetime.now().strftime("%y-%m-%d_%H-%M")}{("_" + func_name + "_") if func_name is not None else ""}_parameters.yml')
    with open(filename, 'w') as outfile:
        return yaml.dump(params, outfile)


def get_center_of_bb(bb):
    x_min, x_max, y_min, y_max = bb

    x_off = x_max - x_min
    x_c = x_max - x_off // 2

    y_off = y_max - y_min
    y_c = y_max - y_off // 2

    return x_c, y_c


@dataclass
class BoxPred:
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    prediction: int
    ds_path: str
    model_path: str
