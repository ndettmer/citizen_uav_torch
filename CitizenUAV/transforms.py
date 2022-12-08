from torchvision.transforms.functional import crop, adjust_brightness, adjust_saturation, adjust_contrast
import torch
from torch import nn
import numpy as np
import math


class QuadCrop(object):
    """
    Quadratic cropping to center square.
    """

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        :param img: Image
        :return: Quadratic image
        """
        small_dim = np.argmin(img.shape[1:]) + 1
        large_dim = np.argmax(img.shape[1:]) + 1
        crop_size = img.shape[small_dim]
        orig_size = img.shape[large_dim]
        offset = np.floor((orig_size - crop_size) / 2).astype(np.int32)
        top = 0 if small_dim == 1 else offset
        left = 0 if small_dim == 2 else offset
        return crop(img, top, left, crop_size, crop_size)

    def __repr__(self):
        return self.__class__.__name__+'()'


class RandomBrightness(object):
    """
    Adjust brightness by random.
    """

    def __init__(self, rand_range: float = .1):
        self.rand_range = rand_range

    def __call__(self, img: torch.Tensor):
        # TODO: check scaling again! I think the source range is wrong. np.random.rand() is in [0;1], not [-1;1]
        factor = np.random.rand() / 2. * self.rand_range + (1 - self.rand_range / 2.)
        return adjust_brightness(img, factor)


class RandomContrast(object):
    """
    Adjust contrast by random.
    """

    def __init__(self, rand_range: float = .1):
        self.rand_range = rand_range

    def __call__(self, img: torch.Tensor):
        factor = np.random.rand() / 2. * self.rand_range + (1 - self.rand_range / 2.)
        return adjust_contrast(img, factor)


class RandomSaturation(object):
    """
    Adjust saturation by random.
    """

    def __init__(self, rand_range: float = .1):
        self.rand_range = rand_range

    def __call__(self, img: torch.Tensor):
        factor = np.random.rand() / 2. * self.rand_range + (1 - self.rand_range / 2.)
        return adjust_saturation(img, factor)


class Log10(object):
    """
    Take the log10 of the image.
    """

    def __init__(self, out_max: int = 1., in_max: int = 1.):
        self.c = out_max / math.log(float(in_max + 1), 10)

    def __call__(self, img: torch.Tensor):
        return torch.log10(img + 1.) * self.c


class Clamp(object):
    """
    Transform version of torch.clamp()
    """

    def __init__(self, min=0, max=1):
        self.min = min
        self.max = max

    def __call__(self, img: torch.Tensor):
        return torch.clamp(img, min=self.min, max=self.max)


