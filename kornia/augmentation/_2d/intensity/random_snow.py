from typing import Any, Dict, Optional

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.color import hls_to_rgb, rgb_to_hls
from kornia.core import Tensor


class RandomSnow(IntensityAugmentationBase2D):
    r"""Generates snow effect on given tensor image or a batch tensor images.

    Args:
        snow_coefficient (float): A float between 0 and 1 that controls the amount of snow to add to the
        image, the larger value corresponds to the more snow.
        brightness (float): A float greater than 1 that controls the brightness of the snow.
        same_on_batch (bool): If True, apply the same transformation to each image in a batch. Default: False.
        p (float): Probability of applying the transformation. Default: 0.5.
        keepdim (bool): Keep the output tensor with the same shape as input. Default: False.

    Returns:
        torch.Tensor: The snowed image with shape (C, H, W).

    Examples:
        >>> import torch
        >>> import kornia
        >>> rng = torch.manual_seed(0)
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
        input_HLS[:, :, 1][
            input_HLS[:, :, 1] < snow_coefficient
        ] *= brightness  # Increase Light channel of the image by given brightness for areas based on snow coefficient
        input_HLS[:, :, 1][input_HLS[:, :, 1] > 255] = 255  # Setting value 255 for white pixels
        output = hls_to_rgb(input_HLS)
        return output
