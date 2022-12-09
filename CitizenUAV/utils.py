import os
from typing import Tuple

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
    return metadata


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
