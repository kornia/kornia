import torch
import torch.nn as nn

import kornia


class RgbToYcbcr(nn.Module):
    r"""Convert an image from RGB to YCbCr.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        torch.Tensor: YCbCr version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> ycbcr = RgbToYcbcr()
        >>> output = ycbcr(input)  # 2x3x4x5
    """

    def __init__(self) -> None:
        super(RgbToYcbcr, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return kornia.color.ycbcr.rgb_to_ycbcr(image)


class YcbcrToRgb(nn.Module):
    r"""Convert an image from YCbCr to Rgb.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        torch.Tensor: RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = YcbcrToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    def __init__(self) -> None:
        super(YcbcrToRgb, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return kornia.color.ycbcr.ycbcr_to_rgb(image)
