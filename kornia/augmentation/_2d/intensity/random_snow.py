from typing import Any, Dict, Optional

from torch import clamp

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.color import hls_to_rgb, rgb_to_hls
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK


class RandomSnow(IntensityAugmentationBase2D):
    r"""Generates snow effect on given tensor image or a batch tensor images.

    Args:
        snow_coefficient: A float between 0 and 1 that controls the amount of snow to add to the
        image, the larger value corresponds to the more snow.
        brightness: A float greater than 1 that controls the brightness of the snow.
        same_on_batch: If True, apply the same transformation to each image in a batch. Default: False.
        p: Probability of applying the transformation. Default: 0.5.
        keepdim: Keep the output tensor with the same shape as input. Default: False.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:
        >>> inputs = torch.rand(2, 3, 4, 4)
        >>> snow = kornia.augmentation.RandomSnow(p=1.0, snow_coefficient=0.2, brightness=2.0)
        >>> output = snow(inputs)
        >>> output.shape
        torch.Size([2, 3, 4, 4])
    """

    def __init__(
        self,
        snow_coefficient: float = 0.5,
        brightness: float = 2.5,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        KORNIA_CHECK(0 <= snow_coefficient <= 1, "Snow coefficient must be between 0 and 1.")

        snow_coefficient = snow_coefficient * 255 / 2 + 255 / 3

        self._param_generator = rg.PlainUniformGenerator(
            ((snow_coefficient, snow_coefficient), "snow_coefficient", None, None),
            ((brightness, brightness), "brightness", None, None),
        )

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        snow_coefficient = params["snow_coefficient"].to(input)[0]
        brightness = params["brightness"].to(input)[0]
        input_HLS = rgb_to_hls(input)

        # Increase Light channel of the image by given brightness for areas based on snow coefficient.
        new_light = input_HLS[:, :, 1][input_HLS[:, :, 1] < snow_coefficient]
        new_light = new_light * brightness

        # Setting value 255 for white pixels
        new_light = clamp(new_light, min=0, max=255)
        input_HLS[:, :, 1][input_HLS[:, :, 1] < snow_coefficient] = new_light

        output = hls_to_rgb(input_HLS)
        return output
