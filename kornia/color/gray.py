import torch
import torch.nn as nn
from .rgb import bgr_to_rgb


class RgbToGrayscale(nn.Module):
    r"""convert RGB image to grayscale version of image.

    the image data is assumed to be in the range of (0, 1).

    args:
        input (torch.Tensor): RGB image to be converted to grayscale.

    returns:
        torch.Tensor: grayscale version of the image.

    shape:
        - input: :math:`(*, 3, H, W)`
        - output: :math:`(*, 1, H, W)`

    reference:
        https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

    Examples::

        >>> import torch
        >>> import kornia
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = kornia.color.RgbToGrayscale()
        >>> output = gray(input)  # 2x1x4x5
    """

    def __init__(self) -> None:
        super(RgbToGrayscale, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return rgb_to_grayscale(input)


def rgb_to_grayscale(input: torch.Tensor) -> torch.Tensor:
    r"""Convert a RGB image to grayscale.

    See :class:`~kornia.color.RgbToGrayscale` for details.

    Args:
        input (torch.Tensor): RGB image to be converted to grayscale.

    Returns:
        torch.Tensor: Grayscale version of the image.
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(input)))

    if len(input.shape) < 3 and input.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(input.shape))

    r, g, b = torch.chunk(input, chunks=3, dim=-3)
    gray: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    return gray


class BgrToGrayscale(nn.Module):
    r"""convert BGR image to grayscale version of image.

    the image data is assumed to be in the range of (0, 1).

    args:
        input (torch.Tensor): BGR image to be converted to grayscale.

    returns:
        torch.Tensor: grayscale version of the image.

    shape:
        - input: :math:`(*, 3, H, W)`
        - output: :math:`(*, 1, H, W)`

    reference:
        https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

    Examples::

        >>> import torch
        >>> import kornia
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = kornia.color.BgrToGrayscale()
        >>> output = gray(input)  # 2x1x4x5
    """

    def __init__(self) -> None:
        super(BgrToGrayscale, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return bgr_to_grayscale(input)


def bgr_to_grayscale(input: torch.Tensor) -> torch.Tensor:
    r"""Convert a BGR image to grayscale.

    See :class:`~kornia.color.BgrToGrayscale` for details.

    Args:
        input (torch.Tensor): BGR image to be converted to grayscale.

    Returns:
        torch.Tensor: Grayscale version of the image.
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(input)))

    if len(input.shape) < 3 and input.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(input.shape))

    input_rgb = bgr_to_rgb(input)
    gray: torch.Tensor = rgb_to_grayscale(input_rgb)
    return gray
