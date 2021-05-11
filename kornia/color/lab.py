from typing import Tuple

import torch
import torch.nn as nn

from .xyz import rgb_to_xyz, xyz_to_rgb
from .rgb import rgb_to_linear_rgb, linear_rgb_to_rgb

"""
The RGB to Lab color transformations were translated from scikit image's rgb2lab and lab2rgb

https://github.com/scikit-image/scikit-image/blob/a48bf6774718c64dade4548153ae16065b595ca9/skimage/color/colorconv.py

"""


def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    r"""Converts a RGB image to Lab.

    The image data is assumed to be in the range of :math:`[0, 1]`. Lab
    color is computed using the D65 illuminant and Observer 2.

    Args:
        image (torch.Tensor): RGB Image to be converted to Lab with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: Lab version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_lab(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    # Convert from sRGB to Linear RGB
    lin_rgb = rgb_to_linear_rgb(image)

    xyz_im: torch.Tensor = rgb_to_xyz(lin_rgb)

    # normalize for D65 white point
    xyz_ref_white = torch.tensor(
        [0.95047, 1., 1.08883], device=xyz_im.device, dtype=xyz_im.dtype)[..., :, None, None]
    xyz_normalized = torch.div(xyz_im, xyz_ref_white)

    threshold = 0.008856
    power = torch.pow(xyz_normalized.clamp(min=threshold), 1 / 3.)
    scale = 7.787 * xyz_normalized + 4. / 29.
    xyz_int = torch.where(xyz_normalized > threshold, power, scale)

    x: torch.Tensor = xyz_int[..., 0, :, :]
    y: torch.Tensor = xyz_int[..., 1, :, :]
    z: torch.Tensor = xyz_int[..., 2, :, :]

    L: torch.Tensor = (116. * y) - 16.
    a: torch.Tensor = 500. * (x - y)
    _b: torch.Tensor = 200. * (y - z)

    out: torch.Tensor = torch.stack([L, a, _b], dim=-3)

    return out


def lab_to_rgb(image: torch.Tensor, clip: bool = True) -> torch.Tensor:
    r"""Converts a Lab image to RGB.

    Args:
        image (torch.Tensor): Lab image to be converted to RGB with shape :math:`(*, 3, H, W)`.
        clip (bool): Whether to apply clipping to insure output RGB values in range :math:`[0, 1]`. Default is True

    Returns:
        torch.Tensor: Lab version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = lab_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    L: torch.Tensor = image[..., 0, :, :]
    a: torch.Tensor = image[..., 1, :, :]
    _b: torch.Tensor = image[..., 2, :, :]

    fy = (L + 16.) / 116.
    fx = (a / 500.) + fy
    fz = fy - (_b / 200.)

    # if color data out of range: Z < 0
    fz = fz.clamp(min=0.)

    fxyz = torch.stack([fx, fy, fz], dim=-3)

    # Convert from Lab to XYZ
    power = torch.pow(fxyz, 3.0)
    scale = (fxyz - 4. / 29.) / 7.787
    xyz = torch.where(fxyz > .2068966, power, scale)

    # For D65 white point
    xyz_ref_white = torch.tensor(
        [0.95047, 1., 1.08883], device=xyz.device, dtype=xyz.dtype)[..., :, None, None]
    xyz_im = xyz * xyz_ref_white

    rgbs_im: torch.Tensor = xyz_to_rgb(xyz_im)

    # https://github.com/richzhang/colorization-pytorch/blob/66a1cb2e5258f7c8f374f582acc8b1ef99c13c27/util/util.py#L107
#     rgbs_im = torch.where(rgbs_im < 0, torch.zeros_like(rgbs_im), rgbs_im)

    # Convert from RGB Linear to sRGB
    rgb_im = linear_rgb_to_rgb(rgbs_im)

    # Clip to 0,1 https://www.w3.org/Graphics/Color/srgb
    if clip:
        rgb_im = torch.clamp(rgb_im, min=0., max=1.)

    return rgb_im


class RgbToLab(nn.Module):
    r"""Converts an image from RGB to Lab.

    The image data is assumed to be in the range of :math:`[0, 1]`. Lab
    color is computed using the D65 illuminant and Observer 2.

    Returns:
        torch.Tensor: Lab version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> lab = RgbToLab()
        >>> output = lab(input)  # 2x3x4x5

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

        [2] https://www.easyrgb.com/en/math.php

        [3] https://github.com/torch/image/blob/dc061b98fb7e946e00034a5fc73e883a299edc7f/generic/image.c#L1467
    """

    def __init__(self) -> None:
        super(RgbToLab, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgb_to_lab(image)


class LabToRgb(nn.Module):
    r"""Converts an image from Lab to RGB.

    Returns:
        torch.Tensor: RGB version of the image. Range may not be in :math:`[0, 1]`.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = LabToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    References:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

        [2] https://www.easyrgb.com/en/math.php

        [3] https://github.com/torch/image/blob/dc061b98fb7e946e00034a5fc73e883a299edc7f/generic/image.c#L1518
    """

    def __init__(self) -> None:
        super(LabToRgb, self).__init__()

    def forward(self, image: torch.Tensor, clip: bool = True) -> torch.Tensor:
        return lab_to_rgb(image, clip)
