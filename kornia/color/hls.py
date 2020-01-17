import torch
import torch.nn as nn
import kornia
from kornia.geometry import pi


class HlsToRgb(nn.Module):
    r"""Convert image from HLS to Rgb
    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): HLS image to be converted to RGB.

    returns:
        torch.tensor: RGB version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    reference:
        https://en.wikipedia.org/wiki/HSL_and_HSV

    Examples::

        >>> import torch
        >>> import kornia
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = kornia.color.HlsToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    """

    def __init__(self) -> None:
        super(HlsToRgb, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return hls_to_rgb(image)


def hls_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an HLS image to RGB
    The image data is assumed to be in the range of (0, 1).

    Args:
        input (torch.Tensor): HLS Image to be converted to RGB.


    Returns:
        torch.Tensor: RGB version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    h: torch.Tensor = image[..., 0, :, :] * 360 / (2 * pi)
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
    r"""Convert image from RGB to HLS
    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): RGB image to be converted to HLS.

    returns:
        torch.tensor: HLS version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples::

        >>> import torch
        >>> import kornia
        >>> input = torch.rand(2, 3, 4, 5)
        >>> hls = kornia.color.RgbToHls()
        >>> output = hls(input)  # 2x3x4x5

    """

    def __init__(self) -> None:
        super(RgbToHls, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return rgb_to_hls(image)


def rgb_to_hls(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to HLS
    The image data is assumed to be in the range of (0, 1).

    Args:
        input (torch.Tensor): RGB Image to be converted to HLS.


    Returns:
        torch.Tensor: HLS version of the image.
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

    h: torch.Tensor = 2. * pi * (60. * hi) / 360.  # hue [0, 2*pi]

    image_hls: torch.Tensor = torch.stack([h, l, s], dim=-3)

    image_hls[torch.isnan(image_hls)] = 0.

    return image_hls
