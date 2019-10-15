import torch
import torch.nn as nn
import cv2
import kornia


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

    Examples::

        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = kornia.color.HlsToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    """

    def __init__(self) -> None:
        super(HlsToRgb, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return hls_to_rgb(image)


def hls_to_rgb(image):
    r"""Convert an HSV image to RGB
    The image data is assumed to be in the range of (0, 1).

    Args:
        input (torch.Tensor): RGB Image to be converted to HSV.


    Returns:
        torch.Tensor: HSV version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    h: torch.Tensor = image[..., 0, :, :] * 360
    l: torch.Tensor = image[..., 1, :, :]
    s: torch.Tensor = image[..., 2, :, :]

    kr = (0 + h / 30) % 12
    kg = (8 + h / 30) % 12
    kb = (4 + h / 30) % 12
    a = s * torch.min(l, 1 - l)

    fr = l - a * torch.max(torch.min(torch.min(kr - 3, 9 - kr),torch.ones_like(kr)), -1 * torch.ones_like(kr))
    fg = l - a * torch.max(torch.min(torch.min(kg - 3, 9 - kg),torch.ones_like(kg)), -1 * torch.ones_like(kr))
    fb = l - a * torch.max(torch.min(torch.min(kb - 3, 9 - kb),torch.ones_like(kb)), -1 * torch.ones_like(kr))

    out: torch.Tensor = torch.stack([fr, fg, fb],dim=-3)

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

        >>> input = torch.rand(2, 3, 4, 5)
        >>> hsv = kornia.color.RgbToHls()
        >>> output = hsv(input)  # 2x3x4x5

    """

    def __init__(self) -> None:
        super(RgbToHls, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return rgb_to_hls(image)


def rgb_to_hls(image):
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

    Imax: torch.Tensor = image.max(-3)[1]
    Imin: torch.Tensor = image.min(-3)[1]

    l: torch.Tensor = (maxc + minc) / 2  # luminance

    deltac: torch.Tensor = maxc - minc
    s: torch.Tensor = torch.zeros_like(deltac)
    s[l < 0.5]: torch.Tensor = (deltac[l < 0.5]) / (maxc[l < 0.5] + minc[l < 0.5])
    s[l >= 0.5]: torch.Tensor = (deltac[l >= 0.5]) / (2 - (maxc[l >= 0.5] + minc[l >= 0.5]))  # Saturation

    hi = torch.zeros_like(deltac)
    hi[Imax == 0] = ((g[Imax == 0] - b[Imax == 0]) / deltac[Imax == 0]) % 6
    hi[Imax == 1] = ((b[Imax == 1] - r[Imax == 1]) / deltac[Imax == 1]) + 2
    hi[Imax == 2] = ((r[Imax == 2] - g[Imax == 2]) / deltac[Imax == 2]) + 4
    h = (60 * hi) / 360

    return torch.stack([h, l, s], dim=-3)
