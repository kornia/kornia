from typing import Optional

import torch
import torch.nn.functional as TF

from kornia.core import Tensor
from kornia.testing import KORNIA_CHECK_SHAPE
from kornia.utils.misc import reduce_first_dims


def grayscale_to_rgb(gray: Tensor) -> Tensor:
    r"""Convert a grayscale image to RGB version of image.

    .. image:: _static/img/grayscale_to_rgb.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        gray: grayscale image to be converted to RGB with shape :math:`(*, 1, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> gray = torch.randn(2, 1, 4, 5)
        >>> rgb = grayscale_to_rgb(gray) # 2x3x4x5
    """
    KORNIA_CHECK_SHAPE(gray, ["*", "1", "H", "W"])
    gray, shape = reduce_first_dims(gray, keep_last_dims=3)
    weights = gray.new_ones(3, 1, 1, 1)
    rgb = TF.conv2d(gray, weights, bias=None)
    rgb = rgb.view(*shape[:-3], 3, shape[-2], shape[-1])
    return rgb


def rgb_to_grayscale(rgb: Tensor, rgb_weights: Optional[Tensor] = None) -> Tensor:
    r"""Convert a RGB image to grayscale version of image.

    .. image:: _static/img/rgb_to_grayscale.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        rgb: RGB image to be converted to grayscale with shape :math:`(*, 3, H, W)`.
        rgb_weights: Weights that will be applied on each channel (RGB).
            The sum of the weights should add up to one.
    Returns:
        grayscale version of the image with shape :math:`(*, 1, H, W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       color_conversions.html>`__.

    Example:
        >>> rgb = torch.rand(2, 3, 4, 5)
        >>> gray = rgb_to_grayscale(rgb) # 2x1x4x5
    """
    KORNIA_CHECK_SHAPE(rgb, ["*", "3", "H", "W"])
    rgb, shape = reduce_first_dims(rgb, keep_last_dims=3)

    if rgb_weights is None:
        # 8 bit images
        if rgb.dtype == torch.uint8:
            rgb_weights = torch.tensor([76, 150, 29], device=rgb.device, dtype=torch.uint8)
        # floating point images
        elif rgb.dtype in (torch.float16, torch.float32, torch.float64):
            rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=rgb.device, dtype=rgb.dtype)
        else:
            raise TypeError(f"Unknown data type: {rgb.dtype}")
    else:
        # is tensor that we make sure is in the same device/dtype
        rgb_weights = rgb_weights.to(rgb)

    rgb_weights = rgb_weights.view(1, 3, 1, 1)
    gray = TF.conv2d(rgb, rgb_weights, bias=None)
    return gray.view(*shape[:-3], 1, shape[-2], shape[-1])


def bgr_to_grayscale(bgr: torch.Tensor) -> torch.Tensor:
    r"""Convert a BGR image to grayscale.

    The image data is assumed to be in the range of (0, 1). First flips to RGB, then converts.

    Args:
        bgr: BGR image to be converted to grayscale with shape :math:`(B, 3, H, W)`.

    Returns:
        grayscale version of the image with shape :math:`(B, 1, H, W)`.

    Example:
        >>> bgr = torch.rand(2, 3, 4, 5)
        >>> gray = bgr_to_grayscale(bgr) # 2x1x4x5
    """
    KORNIA_CHECK_SHAPE(bgr, ["*", "3", "H", "W"])
    bgr, shape = reduce_first_dims(bgr, keep_last_dims=3)

    # 8 bit images
    if bgr.dtype == torch.uint8:
        bgr_weights = torch.tensor([29, 150, 76], device=bgr.device, dtype=torch.uint8)
    # floating point images
    elif bgr.dtype in (torch.float16, torch.float32, torch.float64):
        bgr_weights = torch.tensor([0.114, 0.587, 0.299], device=bgr.device, dtype=bgr.dtype)
    else:
        raise TypeError(f"Unknown data type: {bgr.dtype}")

    bgr_weights = bgr_weights.view(1, 3, 1, 1)
    gray = TF.conv2d(bgr, bgr_weights, bias=None)
    return gray.view(*shape[:-3], 1, shape[-2], shape[-1])
