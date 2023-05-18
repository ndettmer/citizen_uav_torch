import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Tuple


def get_area_around_center(x_c: int, y_c: int, size: int) -> tuple[int, int, int, int]:
    if size < 2:
        return x_c, x_c + 1, y_c, y_c + 1

    min_side = size // 2
    max_side = size - min_side

    x_max = x_c + max_side
    x_min = x_c - min_side
    y_max = y_c + max_side
    y_min = y_c - min_side

    return x_min, x_max, y_min, y_max


def get_center_of_bb(bb: tuple[int, int, int, int]) -> tuple[int, int]:
    x_min, x_max, y_min, y_max = bb

    x_off = x_max - x_min
    x_c = x_max - x_off // 2

    y_off = y_max - y_min
    y_c = y_max - y_off // 2

    return x_c, y_c


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
    for images, *_ in tqdm(dl):
        b, c, h, w = images.shape
        n_batch_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_squares = torch.sum(images ** 2, dim=[0, 2, 3])
        first_moment = (pixel_counter * first_moment + sum_) / (pixel_counter + n_batch_pixels)
        second_moment = (pixel_counter * second_moment + sum_of_squares) / (pixel_counter + n_batch_pixels)
        pixel_counter += n_batch_pixels

    mean, std = first_moment, torch.sqrt(second_moment - first_moment ** 2)

    return mean, std
