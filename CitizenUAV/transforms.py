from torchvision.transforms.functional import crop
import torch
import numpy as np


class QuadCrop(object):

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
