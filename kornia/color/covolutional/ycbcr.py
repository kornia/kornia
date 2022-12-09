from typing import Tuple

import torch
import torch.nn.functional as F

from kornia.core import Tensor, tensor
from kornia.testing import KORNIA_CHECK_SHAPE
from kornia.utils.misc import reduce_first_dims

_RGB2Y_WEIGHTS = tensor([[[[0.299]], [[0.587]], [[0.114]]]])  # (3, 1, 1, 1)


_RGB2YCBCR_WEIGHTS = tensor(
    [
        [[[0.2990000]], [[0.5870000]], [[0.1140000]]],
        [[[-0.168636]], [[-0.331068]], [[0.4997040]]],
        [[[0.4998130]], [[-0.418531]], [[-0.081282]]],
    ]
)  # (3, 3, 1, 1)


_RGB2YCBCR_BIAS = tensor([0.0, 0.5, 0.5])  # (3, )


_YCBCR2RGB_WEIGHTS = tensor(
    [[[[1.0]], [[0.0000]], [[1.4030]]], [[[1.0]], [[-0.344]], [[-0.714]]], [[[1.0]], [[1.7730]], [[0.0000]]]]
)  # (3, 3, 1, 1)


_YCBCR2RGB_BIAS = tensor([-0.7015, 0.529, -0.8865])  # (3, )


def rgb_to_ycbcr(rgb: Tensor) -> Tensor:
    r"""Convert an RGB image to YCbCr.

    .. image:: _static/img/rgb_to_ycbcr.png

    Args:
        rgb: RGB Image to be converted to YCbCr with shape :math:`(*, 3, H, W)`.

    Returns:
        YCbCr version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> rgb = torch.rand(2, 3, 4, 5)
        >>> ycbcr = rgb_to_ycbcr(rgb)  # 2x3x4x5
    """
    KORNIA_CHECK_SHAPE(rgb, ["*", "3", "H", "W"])
    rgb, shape = reduce_first_dims(rgb, keep_last_dims=3, return_shape=True)
    ycbcr = F.conv2d(rgb, _RGB2YCBCR_WEIGHTS.type_as(rgb), bias=_RGB2YCBCR_BIAS.type_as(rgb))
    return ycbcr.view(*shape[:-3], 3, shape[-2], shape[-1])


def rgb_to_y(rgb: Tensor) -> Tensor:
    r"""Convert an RGB image to Y.

    Args:
        rgb: RGB Image to be converted to Y with shape :math:`(*, 3, H, W)`.

    Returns:
        Y version of the image with shape :math:`(*, 1, H, W)`.

    Examples:
        >>> rgb = torch.rand(2, 3, 4, 5)
        >>> y = rgb_to_y(rgb)  # 2x1x4x5
    """
    KORNIA_CHECK_SHAPE(rgb, ["*", "3", "H", "W"])
    rgb, shape = reduce_first_dims(rgb, keep_last_dims=3, return_shape=True)
    y = F.conv2d(rgb, _RGB2Y_WEIGHTS.type_as(rgb), bias=None)
    return y.view(*shape[:-3], 1, shape[-2], shape[-1])


def ycbcr_to_rgb(ycbcr: Tensor) -> Tensor:
    r"""Convert an YCbCr image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        ycbcr: YCbCr Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> ycbcr = torch.rand(2, 3, 4, 5)
        >>> rgb = ycbcr_to_rgb(ycbcr)  # 2x3x4x5
    """
    KORNIA_CHECK_SHAPE(ycbcr, ["*", "3", "H", "W"])
    ycbcr, shape = reduce_first_dims(ycbcr, keep_last_dims=3, return_shape=True)
    rgb = F.conv2d(ycbcr, _YCBCR2RGB_WEIGHTS.type_as(ycbcr), bias=_YCBCR2RGB_BIAS.type_as(ycbcr))
    return rgb.view(*shape[:-3], 3, shape[-2], shape[-1])
