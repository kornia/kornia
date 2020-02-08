import torch
import torch.nn as nn

from .xyz import rgb_to_xyz, xyz_to_rgb


class RgbToLuv(nn.Module):
    r"""Converts an image from RGB to Luv

    The image data is assumed to be in the range of :math:`[0, 1]`.

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
        [2] http://www.poynton.com/ColorFAQ.html
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

    xyz_im: torch.Tensor = rgb_to_xyz(image)

    x: torch.Tensor = xyz_im[..., 0, :, :]
    y: torch.Tensor = xyz_im[..., 1, :, :]
    z: torch.Tensor = xyz_im[..., 2, :, :]

    L: torch.Tensor = torch.where(y > 0.008856,
                                  116 * torch.pow(y, 1 / 3) - 16,
                                  903.3 * y)

    eps: float = torch.finfo(torch.float32).eps

    u_p: torch.Tensor = (4 * x) / (x + 15 * y + 3 * z + eps)
    v_p: torch.Tensor = (9 * y) / (x + 15 * y + 3 * z + eps)

    u: torch.Tensor = 13 * L * (u_p - 0.19793943)
    v: torch.Tensor = 13 * L * (v_p - 0.46831096)

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

    L: torch.Tensor = image[..., 0, :, :]
    u: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :]

    y: torch.Tensor = torch.where(L > 7.99961,
                                  torch.pow((L + 16) / 116, 3.0),
                                  L / 903.3)

    eps = torch.finfo(torch.float32).eps

    u_p: torch.Tensor = (u / (13. * L + eps)) + 0.19793943
    v_p: torch.Tensor = (v / (13. * L + eps)) + 0.46831096

    x: torch.Tensor = y * (9. * u_p) / (4. * v_p + eps)
    z: torch.Tensor = y * ((-3. * u_p - 20. * v_p + 12) / (4. * v_p + eps))

    xyz_im: torch.Tensor = torch.stack((x, y, z), -3)

    rgb_im: torch.Tensor = rgb_to_xyz(xyz_im)

    return rgb_im
