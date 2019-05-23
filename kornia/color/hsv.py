import torch
import torch.nn as nn


class HsvToRgb(nn.Module):
    r"""Convert image from HSV to Rgb
    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): RGB image to be converted to HSV.

    returns:
        torch.tensor: HSV version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    """

    def __init__(self) -> None:
        super(HsvToRgb, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return hsv_to_rgb(image)


def hsv_to_rgb(image):
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

    h: torch.Tensor = image[..., 0, :, :]
    s: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :]

    hi: torch.Tensor = torch.floor(h * 6)
    f: torch.Tensor = h * 6 - hi
    p: torch.Tensor = v * (1 - s)
    q: torch.Tensor = v * (1 - f * s)
    t: torch.Tensor = v * (1 - (1 - f) * s)

    out: torch.Tensor = torch.stack([hi, hi, hi], dim=-3) % 6

    out[out == 0]: torch.Tensor = torch.stack((v, t, p), dim=-3)[out == 0]
    out[out == 1]: torch.Tensor = torch.stack((q, v, p), dim=-3)[out == 1]
    out[out == 2]: torch.Tensor = torch.stack((p, v, t), dim=-3)[out == 2]
    out[out == 3]: torch.Tensor = torch.stack((p, q, v), dim=-3)[out == 3]
    out[out == 4]: torch.Tensor = torch.stack((t, p, v), dim=-3)[out == 4]
    out[out == 5]: torch.Tensor = torch.stack((v, p, q), dim=-3)[out == 5]

    return out


class RgbToHsv(nn.Module):
    r"""Convert image from RGB to HSV.

    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): RGB image to be converted to HSV.

    returns:
        torch.tensor: HSV version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    """

    def __init__(self) -> None:
        super(RgbToHsv, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return rgb_to_hsv(image)


def rgb_to_hsv(image):
    r"""Convert an RGB image to HSV.

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

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    maxc: torch.Tensor = image.max(-3)[0]
    minc: torch.Tensor = image.min(-3)[0]

    v: torch.Tensor = maxc  # brightness

    deltac: torch.Tensor = maxc - minc
    s: torch.Tensor = deltac / v  # saturation

    # avoid division by zero
    deltac: torch.Tensor = torch.where(
        deltac == 0, torch.ones_like(deltac), deltac)

    rc: torch.Tensor = (maxc - r) / deltac
    gc: torch.Tensor = (maxc - g) / deltac
    bc: torch.Tensor = (maxc - b) / deltac

    maxg: torch.Tensor = g == maxc
    maxr: torch.Tensor = r == maxc

    h: torch.Tensor = 4.0 + gc - rc
    h[maxg]: torch.Tensor = 2.0 + rc[maxg] - bc[maxg]
    h[maxr]: torch.Tensor = bc[maxr] - gc[maxr]
    h[minc == maxc]: torch.Tensor = 0.0

    h: torch.Tensor = (h / 6.0) % 1.0

    return torch.stack([h, s, v], dim=-3)
