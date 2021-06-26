from typing import Tuple

import torch
import torch.nn as nn

from .rgb import linear_rgb_to_rgb, rgb_to_linear_rgb
from .xyz import rgb_to_xyz, xyz_to_rgb

"""
The RGB to Luv color transformations were translated from scikit image's rgb2luv and luv2rgb

https://github.com/scikit-image/scikit-image/blob/a48bf6774718c64dade4548153ae16065b595ca9/skimage/color/colorconv.py

"""


def rgb_to_luv(image: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    r"""Converts a RGB image to Luv.

    .. image:: _static/img/rgb_to_luv.png

    The image data is assumed to be in the range of :math:`[0, 1]`. Luv
    color is computed using the D65 illuminant and Observer 2.

    Args:
        image: RGB Image to be converted to Luv with shape :math:`(*, 3, H, W)`.
        eps: for numerically stability when dividing.

    Returns:
        Luv version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_luv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))

    # Convert from sRGB to Linear RGB
    lin_rgb = rgb_to_linear_rgb(image)

    xyz_im: torch.Tensor = rgb_to_xyz(lin_rgb)

    x: torch.Tensor = xyz_im[..., 0, :, :]
    y: torch.Tensor = xyz_im[..., 1, :, :]
    z: torch.Tensor = xyz_im[..., 2, :, :]

    threshold = 0.008856
    L: torch.Tensor = torch.where(y > threshold, 116.0 * torch.pow(y.clamp(min=threshold), 1.0 / 3.0) - 16.0, 903.3 * y)

    # Compute reference white point
    xyz_ref_white: Tuple[float, float, float] = (0.95047, 1.0, 1.08883)
    u_w: float = (4 * xyz_ref_white[0]) / (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2])
    v_w: float = (9 * xyz_ref_white[1]) / (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2])

    u_p: torch.Tensor = (4 * x) / (x + 15 * y + 3 * z + eps)
    v_p: torch.Tensor = (9 * y) / (x + 15 * y + 3 * z + eps)

    u: torch.Tensor = 13 * L * (u_p - u_w)
    v: torch.Tensor = 13 * L * (v_p - v_w)

    out = torch.stack([L, u, v], dim=-3)

    return out


def luv_to_rgb(image: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    r"""Converts a Luv image to RGB.

    Args:
        image: Luv image to be converted to RGB with shape :math:`(*, 3, H, W)`.
        eps: for numerically stability when dividing.

    Returns:
        Luv version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = luv_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))

    L: torch.Tensor = image[..., 0, :, :]
    u: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :]

    # Convert from Luv to XYZ
    y: torch.Tensor = torch.where(L > 7.999625, torch.pow((L + 16) / 116, 3.0), L / 903.3)

    # Compute white point
    xyz_ref_white: Tuple[float, float, float] = (0.95047, 1.0, 1.08883)
    u_w: float = (4 * xyz_ref_white[0]) / (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2])
    v_w: float = (9 * xyz_ref_white[1]) / (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2])

    a: torch.Tensor = u_w + u / (13 * L + eps)
    d: torch.Tensor = v_w + v / (13 * L + eps)
    c: torch.Tensor = 3 * y * (5 * d - 3)

    z: torch.Tensor = ((a - 4) * c - 15 * a * d * y) / (12 * d + eps)
    x: torch.Tensor = -(c / (d + eps) + 3.0 * z)

    xyz_im: torch.Tensor = torch.stack([x, y, z], -3)

    rgbs_im: torch.Tensor = xyz_to_rgb(xyz_im)

    # Convert from RGB Linear to sRGB
    rgb_im = linear_rgb_to_rgb(rgbs_im)

    return rgb_im


class RgbToLuv(nn.Module):
    r"""Converts an image from RGB to Luv.

    The image data is assumed to be in the range of :math:`[0, 1]`. Luv
    color is computed using the D65 illuminant and Observer 2.

    Returns:
        Luv version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> luv = RgbToLuv()
        >>> output = luv(input)  # 2x3x4x5

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

        [2] https://www.easyrgb.com/en/math.php

        [3] http://www.poynton.com/ColorFAQ.html
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgb_to_luv(image)


class LuvToRgb(nn.Module):
    r"""Converts an image from Luv to RGB.

    Returns:
        RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = LuvToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    References:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

        [2] https://www.easyrgb.com/en/math.php

        [3] http://www.poynton.com/ColorFAQ.html
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return luv_to_rgb(image)
