from typing import Any, Dict, Optional, Tuple

import torch

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.color import hls_to_rgb, rgb_to_hls
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK


class RandomSnow(IntensityAugmentationBase2D):
    r"""Generates snow effect on given tensor image or a batch tensor images.

    Args:
        snow_coefficient: A tuple of floats (lower and upper bound) between 0 and 1 that control
        the amount of snow to add to the image, the larger value corresponds to the more snow.
        brightness: A tuple of floats (lower and upper bound) greater than 1 that controls the
        brightness of the snow.
        same_on_batch: If True, apply the same transformation to each image in a batch. Default: False.
        p: Probability of applying the transformation. Default: 0.5.
        keepdim: Keep the output tensor with the same shape as input. Default: False.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:
        >>> inputs = torch.rand(2, 3, 4, 4)
        >>> snow = kornia.augmentation.RandomSnow(p=1.0, snow_coefficient=(0.1, 0.6), brightness=(1.0, 5.0))
        >>> output = snow(inputs)
        >>> output.shape
        torch.Size([2, 3, 4, 4])
    """

    def __init__(
        self,
        snow_coefficient: Tuple[float, float] = (0.5, 0.5),
        brightness: Tuple[float, float] = (2, 2),
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        KORNIA_CHECK(all(0 <= el <= 1 for el in snow_coefficient), "Snow coefficient values must be between 0 and 1.")
        KORNIA_CHECK(all(1 <= el for el in brightness), "Brightness values must be greater than 1.")

        self._param_generator = rg.PlainUniformGenerator(
            (snow_coefficient, "snow_coefficient", 0.5, (0.0, 1.0)), (brightness, "brightness", None, None)
        )

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        KORNIA_CHECK(input.shape[1] == 3, "Number of color channels should be 3.")
        KORNIA_CHECK(len(input.shape) in (3, 4), "Wrong input dimension.")

        if len(input.shape) == 3:
            input = input[None, :, :, :]
        input_HLS = rgb_to_hls(input)

        mask = torch.zeros_like(input_HLS)
        # Retrieve generated parameters
        snow_coefficient = params["snow_coefficient"].to(input)
        brightness = params["brightness"].to(input)
        snow_coefficient = snow_coefficient[:, None, None, None]
        brightness = brightness[:, None, None, None]

        mask[:, 1, :, :] = torch.where(input_HLS[:, 1, :, :] < snow_coefficient[:, 0, :, :], 1, 0)

        # Increase Light channel of the image by given brightness for areas based on snow coefficient.
        new_light = (input_HLS * mask * brightness).clamp(min=0.0, max=1.0)
        input_HLS = input_HLS * (1 - mask) + new_light

        output = hls_to_rgb(input_HLS)
        return output
