from typing import Any, Dict, Optional

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.color import hls_to_rgb, rgb_to_hls
from kornia.core import Tensor


class RandomSnow(IntensityAugmentationBase2D):
    r"""Generates snow effect on given tensor image or a batch tensor images.

    .. image:: _static/ing/RandomSnow.png

    Args:


    Examples:
    """

    def __init__(
        self,
        snow_coefficient: int = 0.5,
        brightness: int = 2.5,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        if snow_coefficient < 0 or snow_coefficient > 1:
            raise ValueError("Snow coefficient must be between 0 and 1.")
        snow_coefficient *= 255 / 2
        snow_coefficient += 255 / 3

        self.flags = dict(snow_coefficient=snow_coefficient, brightness=brightness)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        snow_coefficient, brightness = flags["snow_coefficient"], flags["brightness"]

        input_HLS = rgb_to_hls(input)
        input_HLS[:, :, 1][input_HLS[:, :, 1] < snow_coefficient] *= brightness
        input_HLS[:, :, 1][input_HLS[:, :, 1] > 255] = 255
        output = hls_to_rgb(input_HLS)
        return output
