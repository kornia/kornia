from typing import Tuple

import torch
import torch.nn.functional as F

from kornia.core import Tensor, tensor
from kornia.testing import KORNIA_CHECK_SHAPE
from kornia.utils.misc import reduce_first_dims


_RGB2YUV_WEIGHTS = tensor(
    [
        [[[0.29900]], [[0.58700]], [[0.11400]]],
        [[[-0.1470]], [[-0.2890]], [[0.43600]]],
        [[[0.61500]], [[-0.5150]], [[-0.1000]]],
    ]
)  # 3x3x1x1


_YUV2RGB_WEIGHTS = tensor(
    [[[[1.0]], [[0.0000]], [[1.1400]]], [[[1.0]], [[-0.396]], [[-0.581]]], [[[1.0]], [[2.0290]], [[0.0000]]]]
)  # 3x3x1x1


def rgb_to_yuv(rgb: Tensor) -> Tensor:
    r"""Convert an RGB image to YUV.

    .. image:: _static/img/rgb_to_yuv.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        rgb: RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
        YUV version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> rgb = torch.rand(2, 3, 4, 5)
        >>> yuv = rgb_to_yuv(rgb)  # 2x3x4x5
    """
    KORNIA_CHECK_SHAPE(rgb, ["*", "3", "H", "W"])
    rgb, shape = reduce_first_dims(rgb, keep_last_dims=3, return_shape=True)

    weights = _RGB2YUV_WEIGHTS.type_as(rgb)
    yuv = F.conv2d(rgb, weights, bias=None)
    return yuv.view(*shape[:-3], 3, shape[-2], shape[-1])


def rgb_to_yuv420(rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Convert an RGB image to YUV 420 (subsampled).

    The image data is assumed to be in the range of (0, 1). Input need to be padded to be evenly divisible by 2
    horizontal and vertical. This function will output chroma siting (0.5,0.5)

    Args:
        rgb: RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
        A Tensor containing the Y plane with shape :math:`(*, 1, H, W)`
        A Tensor containing the UV planes with shape :math:`(*, 2, H/2, W/2)`

    Example:
        >>> rgb = torch.rand(2, 3, 4, 6)
        >>> yuv420 = rgb_to_yuv420(rgb)  # (2x1x4x6, 2x2x2x3)
    """
    KORNIA_CHECK_SHAPE(rgb, ["*", "3", "H", "W"])
    rgb, shape = reduce_first_dims(rgb, keep_last_dims=3, return_shape=True)

    if rgb.shape[-2] % 2 == 1 or rgb.shape[-1] % 2 == 1:
        raise ValueError(f"Input H&W must be evenly divisible by 2. Got {rgb.shape}")

    weights = _RGB2YUV_WEIGHTS.type_as(rgb)
    yuv = F.conv2d(rgb, weights, bias=None)
    y = yuv[..., :1, :, :]
    uv = F.avg_pool2d(yuv[..., 1:3, :, :], (2, 2))

    y = y.view(*shape[:-3], 1, shape[-2], shape[-1])
    uv = uv.view(*shape[:-3], 2, shape[-2] // 2, shape[-1] // 2)

    return y, uv


def rgb_to_yuv422(rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Convert an RGB image to YUV 422 (subsampled).

    The image data is assumed to be in the range of (0, 1). Input need to be padded to be evenly divisible by 2
    vertical. This function will output chroma siting (0.5)

    Args:
        rgb: RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
       A Tensor containing the Y plane with shape :math:`(*, 1, H, W)`
       A Tensor containing the UV planes with shape :math:`(*, 2, H, W/2)`

    Example:
        >>> rgb = torch.rand(2, 3, 4, 6)
        >>> yuv422 = rgb_to_yuv422(rgb)  # (2x1x4x6, 2x1x4x3)
    """
    KORNIA_CHECK_SHAPE(rgb, ["*", "3", "H", "W"])
    rgb, shape = reduce_first_dims(rgb, keep_last_dims=3, return_shape=True)

    if rgb.shape[-2] % 2 == 1 or rgb.shape[-1] % 2 == 1:
        raise ValueError(f"Input H&W must be evenly divisible by 2. Got {rgb.shape}")

    weights = _RGB2YUV_WEIGHTS.type_as(rgb)
    yuv = F.conv2d(rgb, weights, bias=None)
    y = yuv[..., :1, :, :]
    uv = F.avg_pool2d(yuv[..., 1:3, :, :], (1, 2))

    y = y.view(*shape[:-3], 1, shape[-2], shape[-1])
    uv = uv.view(*shape[:-3], 2, shape[-2], shape[-1] // 2)

    return y, uv


def yuv_to_rgb(yuv: torch.Tensor) -> torch.Tensor:
    r"""Convert an YUV image to RGB.

    The image data is assumed to be in the range of (0, 1) for luma and (-0.5, 0.5) for chroma.

    Args:
        yuv: YUV Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> yuv = torch.rand(2, 3, 4, 5)
        >>> rgb = yuv_to_rgb(yuv)  # 2x3x4x5
    """
    KORNIA_CHECK_SHAPE(yuv, ["*", "3", "H", "W"])
    yuv, shape = reduce_first_dims(yuv, keep_last_dims=3, return_shape=True)

    weights = _YUV2RGB_WEIGHTS.type_as(yuv)
    rgb = F.conv2d(yuv, weights, bias=None)
    return rgb.view(*shape[:-3], 3, shape[-2], shape[-1])


def yuv420_to_rgb(y: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    r"""Convert an YUV420 image to RGB.

    The image data is assumed to be in the range of (0, 1) for luma and (-0.5, 0.5) for chroma.
    Input need to be padded to be evenly divisible by 2 horizontal and vertical.
    This function assumed chroma siting is (0.5, 0.5)

    Args:
        y: Y (luma) Image plane to be converted to RGB with shape :math:`(*, 1, H, W)`.
        uv: UV (chroma) Image planes to be converted to RGB with shape :math:`(*, 2, H/2, W/2)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> y = torch.rand(2, 1, 4, 6)
        >>> uv = torch.rand(2, 2, 2, 3)
        >>> rgb = yuv420_to_rgb(y, uv)  # 2x3x4x6
    """
    KORNIA_CHECK_SHAPE(uv, ["*", "2", "h", "w"])
    uv, uv_shape = reduce_first_dims(uv, keep_last_dims=3, return_shape=True)

    KORNIA_CHECK_SHAPE(y, ["*", "1", str(2 * uv_shape[-2]), str(2 * uv_shape[-1])])
    y, y_shape = reduce_first_dims(y, keep_last_dims=3, return_shape=True)

    # first upsample
    yuv444image = torch.cat([y, uv.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)], dim=-3)

    # then convert the yuv444 tensor
    weights = _YUV2RGB_WEIGHTS.type_as(yuv444image)
    rgb = F.conv2d(yuv444image, weights, bias=None)

    return rgb.view(*y_shape[:-3], 3, y_shape[-2], y_shape[-1])


def yuv422_to_rgb(y: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    r"""Convert an YUV422 image to RGB.

    The image data is assumed to be in the range of (0, 1) for luma and (-0.5, 0.5) for chroma.
    Input need to be padded to be evenly divisible by 2 vertical. This function assumed chroma siting is (0.5)

    Args:
        y: Y (luma) Image plane to be converted to RGB with shape :math:`(*, 1, H, W)`.
        uv: UV (luma) Image planes to be converted to RGB with shape :math:`(*, 2, H, W/2)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> y = torch.rand(2, 1, 4, 6)
        >>> uv = torch.rand(2, 2, 2, 3)
        >>> rgb = yuv420_to_rgb(y, uv)  # 2x3x4x5
    """
    KORNIA_CHECK_SHAPE(uv, ["*", "2", "H", "w"])
    uv, uv_shape = reduce_first_dims(uv, keep_last_dims=3, return_shape=True)

    KORNIA_CHECK_SHAPE(y, ["*", "1", str(uv_shape[-2]), str(2 * uv_shape[-1])])
    y, y_shape = reduce_first_dims(y, keep_last_dims=3, return_shape=True)

    # first upsample
    yuv444image = torch.cat([y, uv.repeat_interleave(2, dim=-1)], dim=-3)

    # then convert the yuv444 tensor
    weights = _YUV2RGB_WEIGHTS.type_as(yuv444image)
    rgb = F.conv2d(yuv444image, weights, bias=None)

    return rgb.view(*y_shape[:-3], 3, y_shape[-2], y_shape[-1])
