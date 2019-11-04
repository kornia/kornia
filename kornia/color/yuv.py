
import torch
import torch.nn as nn
import torch.nn.functional as F


class RgbToYuv(nn.Module):
    r"""Convert image from RGB to YUV
    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): RGB image to be converted to YUV.

    returns:
        torch.tensor: YUV version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples::

        >>> input = torch.rand(2, 3, 4, 5)
        >>> yuv = kornia.color.RgbToYuv()
        >>> output = yuv(input)  # 2x3x4x5

    Reference::
        [1] https://es.wikipedia.org/wiki/YUV#RGB_a_Y'UV
    """

    def __init__(self) -> None:
        super(RgbToYuv, self).__init__()

    def forward(  # type: ignore
            self, input: torch.Tensor) -> torch.Tensor:
        return rgb_to_yuv(input)


def rgb_to_yuv(input: torch.Tensor):
    r"""Convert an RGB image to YUV
    The image data is assumed to be in the range of (0, 1).

    Args:
        input (torch.Tensor): RGB Image to be converted to YUV.

    Returns:
        torch.Tensor: YUV version of the image.

    See :class:`~kornia.color.RgbToYuv` for details."""
    r, g, b = torch.chunk(input, chunks=3, dim=-3)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.147 * r - 0.289 * g + 0.436 * b
    v = 0.615 * r - 0.515 * g - 0.100 * b
    return torch.cat((y, u, v), -3)


class YuvToRgb(nn.Module):
    r"""Convert image from YUV to RGB
    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): YUV image to be converted to RGB.

    returns:
        torch.tensor: RGB version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples::

        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = kornia.color.YuvToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    def __init__(self) -> None:
        super(YuvToRgb, self).__init__()

    def forward(  # type: ignore
            self, input: torch.Tensor) -> torch.Tensor:
        return yuv_to_rgb(input)


def yuv_to_rgb(input: torch.Tensor):
    r"""Convert an YUV image to RGB
    The image data is assumed to be in the range of (0, 1).

    Args:
        input (torch.Tensor): YUV Image to be converted to RGB.

    Returns:
        torch.Tensor: RGB version of the image.

    See :class:`~kornia.color.YuvToRgb` for details."""
    y, u, v = torch.chunk(input, chunks=3, dim=-3)
    r = y + 1.14 * v  # coefficient for g is 0
    g = y + -0.396 * u - 0.581 * v
    b = y + 2.029 * u  # coefficient for b is 0
    return torch.cat((r, g, b), -3)
