import torch
import torch.nn as nn


class RgbToYcbcr(torch.nn.Module):
    r"""Convert image from RGB to YCbCr.
    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): RGB iRGB image to be converted to YCbCr.

    returns:
        torch.Tensor: YCbCr version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    """

    def __init__(self) -> None:
        super(RgbToYcbcr, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return rgb_to_ycbcr(image)


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a RGB image to YCbCr.
    See :class:`~kornia.color.RgbToYcbcr` for details.

    Args:
        input (torch.Tensor): RGB Image to be converted to YCbCr.

    Returns:
        torch.Tensor: YCbCr version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W).Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    y: torch.Tensor = 0.29900 * r + 0.58700 * g + 0.11400 * b
    cb: torch.Tensor = -0.168736 * r - 0.331264 * g + 0.50000 * b + (128. / 255)
    cr: torch.Tensor = 0.50000 * r - 0.418688 * g - 0.081312 * b + (128. / 255)

    out: torch.Tensor = torch.stack([y, cb, cr], dim=-3)

    return out


class YcbcrToRgb(torch.nn.Module):
    r"""Convert image from YCbCr to RGB.
    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): YCbCr image to be converted to RGB.

    returns:
        torch.Tensor: RGB version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    """

    def __init__(self) -> None:
        super(YcbcrToRgb, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return ycbcr_to_rgb(image)


def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a YCbCr image to RGB.
    See :class:`~kornia.color.YcbcrToRgb` for details.

    Args:
        input (torch.Tensor): YCbCr Image to be converted to RGB.

    Returns:
        torch.Tensor: RGB version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W).Got {}"
                         .format(image.shape))

    y: torch.Tensor = image[..., 0, :, :]
    cb: torch.Tensor = image[..., 1, :, :]
    cr: torch.Tensor = image[..., 2, :, :]

    cb_: torch.Tensor = cb - 128. / 255
    cr_: torch.Tensor = cr - 128. / 255

    r: torch.Tensor = y + 1.40200 * cr_
    g: torch.Tensor = y - 0.344136 * cb_ - 0.714136 * cr_
    b: torch.Tensor = y + 1.77200 * cb_

    out: torch.Tensor = torch.stack([r, g, b], dim=-3)

    return out
