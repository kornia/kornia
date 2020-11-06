import torch
import torch.nn as nn

import kornia


class RgbToXyz(nn.Module):
    r"""Converts an image from RGB to XYZ.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        torch.Tensor: XYZ version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> xyz = RgbToXyz()
        >>> output = xyz(input)  # 2x3x4x5

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
    """

    def __init__(self) -> None:
        super(RgbToXyz, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return kornia.color.xyz.rgb_to_xyz(image)


class XyzToRgb(nn.Module):
    r"""Converts an image from XYZ to RGB.

    Returns:
        torch.Tensor: RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = XyzToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
    """

    def __init__(self) -> None:
        super(XyzToRgb, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return kornia.color.xyz.xyz_to_rgb(image)
