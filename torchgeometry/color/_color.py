import torch
import torch.nn as nn


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


class RgbToBgr(nn.Module):
    r"""Convert image from RGB to BGR.

    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): RGB image to be converted to BGR

    returns:
        torch.tensor: BGR version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    """

    def __init__(self) -> None:
        super(RgbToBgr, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return rgb_to_bgr(image)


def rgb_to_bgr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a RGB image to BGR.

    Args:
        input (torch.Tensor): RGB Image to be converted to BGR.

    Returns:
        torch.Tensor: BGR version of the image.
    """

    return bgr_to_rgb(image)


class BgrToRgb(nn.Module):
    r"""Convert image from BGR to RGB.

    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): BGR image to be converted to RGB.

    returns:
        torch.tensor: RGB version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    """

    def __init__(self) -> None:
        super(BgrToRgb, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return bgr_to_rgb(image)


def bgr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a BGR image to RGB.

    Args:
        input (torch.Tensor): BGR Image to be converted to RGB.

    Returns:
        torch.Tensor: RGB version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W).Got {}"
                         .format(image.shape))

    # flip image channels
    out: torch.Tensor = image.flip(-3)

    return out


class RgbToGrayscale(nn.Module):
    r"""convert image to grayscale version of image.

    the image data is assumed to be in the range of (0, 1).

    args:
        input (torch.Tensor): image to be converted to grayscale.

    returns:
        torch.Tensor: grayscale version of the image.

    shape:
        - input: :math:`(*, 3, H, W)`
        - output: :math:`(*, 1, H, W)`

    Examples::

        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = tgm.image.RgbToGrayscale()
        >>> output = gray(input)  # 2x1x4x5
    """

    def __init__(self) -> None:
        super(RgbToGrayscale, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return rgb_to_grayscale(input)


def rgb_to_grayscale(input: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to grayscale.

    See :class:`~torchgeometry.color.RgbToGrayscale` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(input)))

    if len(input.shape) < 3 and input.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(input.shape))

    # https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
    r, g, b = torch.chunk(input, chunks=3, dim=-3)
    gray: torch.Tensor = 0.299 * r + 0.587 * g + 0.110 * b
    return gray


class Normalize(nn.Module):
    r"""Normalize a tensor image or a batch of tensor images
    with mean and standard deviation. Input must be a tensor of shape (C, H, W)
    or a batch of tensors (*, C, H, W).
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels,
    this transform will normalize each channel of the input ``torch.*Tensor``
    i.e. ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (torch.Tensor): Mean for each channel.
        std (torch.Tensor): Standard deviation for each channel.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:

        super(Normalize, self).__init__()

        self.mean: torch.Tensor = mean
        self.std: torch.Tensor = std

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return normalize(input, self.mean, self.std)

    def __repr__(self):
        repr = '(mean={0}, std={1})'.format(self.mean, self.std)
        return self.__class__.__name__ + repr


def normalize(data: torch.Tensor, mean: torch.Tensor,
              std: torch.Tensor) -> torch.Tensor:
    r"""Normalise the image with channel-wise mean and standard deviation.

    Args:
        data (torch.Tensor): The image tensor to be normalised.
        mean (torch.Tensor): Mean for each channel.
        std (torch.Tensor): Standard deviations for each channel.

        Returns:
        Tensor: The normalised image tensor.
    """

    if not torch.is_tensor(data):
        raise TypeError('data should be a tensor. Got {}'.format(type(data)))

    if not torch.is_tensor(mean):
        raise TypeError('mean should be a tensor. Got {}'.format(type(mean)))

    if not torch.is_tensor(std):
        raise TypeError('std should be a tensor. Got {}'.format(type(std)))

    if mean.shape[0] != data.shape[-3] and mean.shape[:2] != data.shape[:2]:
        raise ValueError('mean lenght and number of channels do not match')

    if std.shape[0] != data.shape[-3] and std.shape[:2] != data.shape[:2]:
        raise ValueError('std lenght and number of channels do not match')

    mean = mean[..., :, None, None].to(data.device)
    std = std[..., :, None, None].to(data.device)

    out: torch.Tensor = (data - mean) / std

    return out
