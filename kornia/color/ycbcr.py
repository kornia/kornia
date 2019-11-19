import torch
import torch.nn as nn


class YcbcrToRgb(nn.Module):
    r"""Convert image from YCbCr to Rgb
    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): YCbCr image to be converted to RGB.

    returns:
        torch.tensor: RGB version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples::

        >>> import torch
        >>> import kornia
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = kornia.color.YcbcrToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    """

    def __init__(self) -> None:
        super(YcbcrToRgb, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return ycbcr_to_rgb(image)


def ycbcr_to_rgb(image):
    r"""Convert an YCbCr image to RGB
    The image data is assumed to be in the range of (0, 1).

    Args:
        input (torch.Tensor): YCbCr Image to be converted to RGB.


    Returns:
        torch.Tensor: RGB version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    y = image[..., 0, :, :]
    cb = image[..., 1, :, :]
    cr = image[..., 2, :, :]

    delta = .5

    r: torch.Tensor = y + 1.403 * (cr - delta)
    g: torch.Tensor = y - .714 * (cr - delta) - .344 * (cb - delta)
    b: torch.Tensor = y + 1.773 * (cb - delta)
    return torch.stack((r, g, b), -3)


class RgbToYcbcr(nn.Module):
    r"""Convert image from RGB to YCnCr
    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): RGB image to be converted to YCbCr.

    returns:
        torch.tensor: YCbCr version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples::

        >>> import torch
        >>> import kornia
        >>> input = torch.rand(2, 3, 4, 5)
        >>> ycbcr = kornia.color.RgbToYcbcr()
        >>> output = ycbcr(input)  # 2x3x4x5

    """

    def __init__(self) -> None:
        super(RgbToYcbcr, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return rgb_to_ycbcr(image)


def rgb_to_ycbcr(image):
    r"""Convert an RGB image to YCbCr.

    Args:
        input (torch.Tensor): RGB Image to be converted to YCbCr.

    Returns:
        torch.Tensor: YCbCr version of the image.
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

    delta = .5
    y: torch.Tensor = .299 * r + .587 * g + .114 * b
    cb: torch.Tensor = (b - y) * .564 + delta
    cr: torch.Tensor = (r - y) * .713 + delta
    return torch.stack((y, cb, cr), -3)
