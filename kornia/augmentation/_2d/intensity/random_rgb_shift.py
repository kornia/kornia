import enum
from typing import Dict, Optional, Tuple, Union, cast

import random
import torch
from torch import Tensor

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D


def generate_parameters(self) -> Dict[str, Tensor]:
    r_shift = random.uniform(self.r_shift_limit[0], self.r_shift_limit[1])
    g_shift = random.uniform(self.g_shift_limit[0], self.g_shift_limit[1])
    b_shift = random.uniform(self.b_shift_limit[0], self.b_shift_limit[1])

    return dict(r_shift=r_shift, g_shift=g_shift, b_shift=b_shift)

def shift_image(img, value):
    max_value = 255

    lut = torch.arange(0, max_value + 1).type(torch.float64)
    lut += value

    lut = torch.clamp(lut, 0, max_value).type(img.dtype)
    for i in range(max_value + 1):
        indices = torch.where(img == i)
        img[indices[0], indices[1], indices[2]] = lut[i]
    return img


def shift_rgb_uint8(image, r_shift, g_shift, b_shift):
    if r_shift == g_shift == b_shift:
        pass

    shifted = torch.empty_like(image)
    shifts = [r_shift, g_shift, b_shift]
    for i, shift in enumerate(shifts):
        # shifted[..., i] = input[..., i] + shift
        shifted[..., i] = shift_image(image[..., i], shift)

    return shifted


class RandomRGBShift(IntensityAugmentationBase2D):
    """

    Note:
        Input tensor must be float and normalized into [0, 1].

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> img = torch.rand(1, 1, 5, 5)
        >>> shift = RandomRGBShift()
        >>> shift(img)

    """
    # Note: Extra params, inplace=False in Torchvision.
    def __init__(
        self,
        r_shift_limit: int = 20,
        g_shift_limit: int = 20,
        b_shift_limit: int = 20,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
        return_transform: Optional[bool] = None,
    ) -> None:
        super().__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch, keepdim=keepdim)
        self.r_shift_limit = (-r_shift_limit, r_shift_limit)
        self.g_shift_limit = (-g_shift_limit, g_shift_limit)
        self.b_shift_limit = (-b_shift_limit, b_shift_limit)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], transform: Optional[Tensor] = None
    ) -> Tensor:
        return shift_rgb_uint8(input, params["r_shift"], params["g_shift"], params["b_shift"])