import math

import torch
import torch.nn as nn

import kornia
from kornia.constants import pi


def rgb_to_hls(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a RGB image to HLS.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image (torch.Tensor): RGB image to be converted to HLS with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: HLS version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_hls(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    maxc: torch.Tensor = image.max(-3)[0]
    minc: torch.Tensor = image.min(-3)[0]

    imax: torch.Tensor = image.max(-3)[1]

    l: torch.Tensor = (maxc + minc) / 2  # luminance

    deltac: torch.Tensor = maxc - minc

    s: torch.Tensor = torch.where(l < 0.5, deltac / (maxc + minc), deltac /
                                  (torch.tensor(2.) - (maxc + minc)))  # saturation

    hi: torch.Tensor = torch.zeros_like(deltac)

    hi[imax == 0] = (((g - b) / deltac) % 6)[imax == 0]
    hi[imax == 1] = (((b - r) / deltac) + 2)[imax == 1]
    hi[imax == 2] = (((r - g) / deltac) + 4)[imax == 2]

    h: torch.Tensor = 2. * math.pi * (60. * hi) / 360.  # hue [0, 2*pi]

    image_hls: torch.Tensor = torch.stack([h, l, s], dim=-3)

    # JIT indexing is not supported before 1.6.0 https://github.com/pytorch/pytorch/issues/38962
    # image_hls[torch.isnan(image_hls)] = 0.
    image_hls = torch.where(
        torch.isnan(image_hls), torch.tensor(0., device=image_hls.device, dtype=image_hls.dtype), image_hls)

    return image_hls


def hls_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a HLS image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image (torch.Tensor): HLS image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = hls_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    h: torch.Tensor = image[..., 0, :, :] * 360 / (2 * math.pi)
    l: torch.Tensor = image[..., 1, :, :]
    s: torch.Tensor = image[..., 2, :, :]

    kr = (0 + h / 30) % 12
    kg = (8 + h / 30) % 12
    kb = (4 + h / 30) % 12
    a = s * torch.min(l, torch.tensor(1.) - l)

    ones_k = torch.ones_like(kr)

    fr: torch.Tensor = l - a * torch.max(torch.min(torch.min(kr - torch.tensor(3.),
                                                             torch.tensor(9.) - kr), ones_k), -1 * ones_k)
    fg: torch.Tensor = l - a * torch.max(torch.min(torch.min(kg - torch.tensor(3.),
                                                             torch.tensor(9.) - kg), ones_k), -1 * ones_k)
    fb: torch.Tensor = l - a * torch.max(torch.min(torch.min(kb - torch.tensor(3.),
                                                             torch.tensor(9.) - kb), ones_k), -1 * ones_k)

    out: torch.Tensor = torch.stack([fr, fg, fb], dim=-3)

    return out


class RgbToHls(nn.Module):
    r"""Convert an image from RGB to HLS.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        torch.Tensor: HLS version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> hls = RgbToHls()
        >>> output = hls(input)  # 2x3x4x5
    """

    def __init__(self) -> None:
        super(RgbToHls, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgb_to_hls(image)


class HlsToRgb(nn.Module):
    r"""Convert an image from HLS to RGB.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        torch.Tensor: RGB version of the image.

    Shape:
        - input: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Reference:
        https://en.wikipedia.org/wiki/HSL_and_HSV

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = HlsToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    def __init__(self) -> None:
        super(HlsToRgb, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return hls_to_rgb(image)
