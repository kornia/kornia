import torch
import torch.nn as nn

import kornia


class RgbToHsv(nn.Module):
    r"""Convert an image from RGB to HSV.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        torch.tensor: HSV version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> hsv = RgbToHsv()
        >>> output = hsv(input)  # 2x3x4x5
    """

    def __init__(self) -> None:
        super(RgbToHsv, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return kornia.color.hsv.rgb_to_hsv(image)


class HsvToRgb(nn.Module):
    r"""Convert an image from HSV to RGB.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        torch.Tensor: RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = HsvToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    def __init__(self) -> None:
        super(HsvToRgb, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return kornia.color.hsv.hsv_to_rgb(image)
