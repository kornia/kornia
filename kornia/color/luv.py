from typing import Tuple
import torch
import torch.nn as nn

from .xyz import rgb_to_xyz, xyz_to_rgb


class RgbToLuv(nn.Module):
    r"""Converts an image from RGB to Luv

    The image data is assumed to be in the range of :math:`[0, 1]`. Luv
    color is computed using the D65 white point.

    args:
        image (torch.Tensor): RGB image to be converted to Luv.
    returns:
        torch.Tensor: Luv version of the image.
    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`
    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> luv = kornia.color.RgbToLuv()
        >>> output = luv(input)  # 2x3x4x5
    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
        [2] https://www.easyrgb.com/en/math.php
        [3] http://www.poynton.com/ColorFAQ.html
    """

    def __init__(self) -> None:

        super(RgbToLuv, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore

        return rgb_to_luv(image)


class LuvToRgb(nn.Module):
    r"""Converts an image from Luv to RGV

    args:
        image (torch.Tensor): Luv image to be converted to RGB.
    returns:
        torch.Tensor: RGB version of the image.
    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`
    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = kornia.color.LuvToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    References:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
        [2] https://www.easyrgb.com/en/math.php
        [3] http://www.poynton.com/ColorFAQ.html
    """

    def __init__(self) -> None:

        super(LuvToRgb, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore

        return luv_to_rgb(image)


def rgb_to_luv(image: torch.Tensor) -> torch.Tensor:
    r"""Converts a RGB image to Luv.

    See :class:`~kornia.color.RgbToLuv` for details.

    Args:
        image (torch.Tensor): RGB image
    Returns:
        torch.Tensor : Luv image
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    rs: torch.Tensor = torch.where(r > 0.04045, torch.pow(((r + 0.055) / 1.055), 2.4), r / 12.92)
    gs: torch.Tensor = torch.where(g > 0.04045, torch.pow(((g + 0.055) / 1.055), 2.4), g / 12.92)
    bs: torch.Tensor = torch.where(b > 0.04045, torch.pow(((b + 0.055) / 1.055), 2.4), b / 12.92)

    image_s = torch.stack((rs, gs, bs), dim=-3)

    xyz_im: torch.Tensor = rgb_to_xyz(image_s)

    x: torch.Tensor = xyz_im[..., 0, :, :]
    y: torch.Tensor = xyz_im[..., 1, :, :]
    z: torch.Tensor = xyz_im[..., 2, :, :]

    L: torch.Tensor = torch.where(torch.gt(y, 0.008856),
                                  116. * torch.pow(y, 1. / 3.) - 16.,
                                  903.3 * y)

    eps: float = torch.finfo(torch.float64).eps

    xyz_ref_white: Tuple[float, float, float] = (.95047, 1., 1.08883)
    u_w: float = (4 * xyz_ref_white[0]) / (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2])
    v_w: float = (9 * xyz_ref_white[1]) / (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2])
    u_p: torch.Tensor = (4 * x) / (x + 15 * y + 3 * z + eps)
    v_p: torch.Tensor = (9 * y) / (x + 15 * y + 3 * z + eps)

    u: torch.Tensor = 13 * L * (u_p - u_w)
    v: torch.Tensor = 13 * L * (v_p - v_w)

    out = torch.stack((L, u, v), dim=-3)

    return out


def luv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Converts a Luv image to RGB.

    See :class:`~kornia.color.LuvToRgb` for details.

    Args:
        image (torch.Tensor): Luv image
    Returns:
        torch.Tensor : RGB image
    """
    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    L: torch.Tensor = image[..., 0, :, :]
    u: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :]

    y: torch.Tensor = torch.where(L > 7.999625,
                                  torch.pow((L + 16) / 116, 3.0),
                                  L / 903.3)

    xyz_ref_white: Tuple[float, float, float] = (0.95047, 1., 1.08883)
    u_w: float = (4 * xyz_ref_white[0]) / (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2])
    v_w: float = (9 * xyz_ref_white[1]) / (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2])

    eps: float = torch.finfo(torch.float64).eps

    a: torch.Tensor = u_w + u / (13 * L + eps)
    d: torch.Tensor = v_w + v / (13 * L + eps)
    c: torch.Tensor = 3 * y * (5 * d - 3)

    z: torch.Tensor = ((a - 4) * c - 15 * a * d * y) / (12 * d + eps)
    x: torch.Tensor = -(c / (d + eps) + 3. * z)

    xyz_im: torch.Tensor = torch.stack((x, y, z), -3)

    rgbs_im: torch.Tensor = xyz_to_rgb(xyz_im)

    rs: torch.Tensor = rgbs_im[..., 0, :, :]
    gs: torch.Tensor = rgbs_im[..., 1, :, :]
    bs: torch.Tensor = rgbs_im[..., 2, :, :]

    r: torch.Tensor = torch.where(rs > 0.0031308, 1.055 * torch.pow(rs, 1 / 2.4) - 0.055, 12.92 * rs)
    g: torch.Tensor = torch.where(gs > 0.0031308, 1.055 * torch.pow(gs, 1 / 2.4) - 0.055, 12.92 * gs)
    b: torch.Tensor = torch.where(bs > 0.0031308, 1.055 * torch.pow(bs, 1 / 2.4) - 0.055, 12.92 * bs)

    rgb_im: torch.Tensor = torch.stack((r, g, b), dim=-3)

    return rgb_im
