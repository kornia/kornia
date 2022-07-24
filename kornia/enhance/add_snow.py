import torch
from kornia.color import rgb_to_hls, hls_to_rgb
from kornia.testing import KORNIA_CHECK_IS_COLOR, KORNIA_CHECK_IS_TENSOR


def add_snow(image: torch.Tensor, snow_coef: torch.Tensor, brightness_coef: torch.Tensor) -> torch.Tensor:
    """Add snow to the image.

    Snow is added in the form of bleach of some pixels.
    """

    KORNIA_CHECK_IS_TENSOR(image)
    KORNIA_CHECK_IS_COLOR(image, f"with shape {image.shape}")

    snow_coef *= 0.5  # = 255 / 2
    snow_coef += 0.33  # = 255 / 3

    hls = rgb_to_hls(image)
    hls[:, 1, :, :][hls[:, 1, :, :] < snow_coef] = hls[:, 1, :, :][hls[:, 1, :, :] < snow_coef] * brightness_coef
    hls[:, 1, :, :][hls[:, 1, :, :] > 1] = 1

    rgb = hls_to_rgb(hls)
    return rgb
