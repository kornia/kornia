from typing import Any, Dict, Optional, Tuple

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
        >>> snow = kornia.augmentation.RandomSnow(p=1.0, snow_coefficient=(0.1, 0.6), brightness=(1, 5))
        >>> output = snow(inputs)
        >>> output.shape
        torch.Size([2, 3, 4, 4])
    """

    def __init__(
        self,
        snow_coefficient: Tuple[float, float] = (0.0, 1.0),
        brightness: Tuple[float, float] = (1.0, 5.0),
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        KORNIA_CHECK(all([0 <= el <= 1 for el in snow_coefficient]), "Snow coefficient values must be between 0 and 1.")
        KORNIA_CHECK(all([1 <= el for el in brightness]), "Brightness values must be greater than 1.")

        self._param_generator = rg.PlainUniformGenerator(
            (snow_coefficient, "snow_coefficient", 0.5, (0.0, 1.0)), (brightness, "brightness", None, None)
        )

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        snow_coefficient = (params["snow_coefficient"].to(input)).mean()
        brightness = (params["brightness"].to(input)).mean()
        input_HLS = rgb_to_hls(input)

        # Increase Light channel of the image by given brightness for areas based on snow coefficient.
        new_light = input_HLS[:, :, 1][input_HLS[:, :, 1] < snow_coefficient]
        new_light = new_light * brightness

        new_light = new_light.clamp(min=0, max=1)
        input_HLS[:, :, 1][input_HLS[:, :, 1] < snow_coefficient] = new_light

        output = hls_to_rgb(input_HLS)
        return output
