from typing import Tuple

import torch
import torch.nn as nn

import kornia

"""
The RGB to Luv color transformations were translated from scikit image's rgb2luv and luv2rgb

https://github.com/scikit-image/scikit-image/blob/a48bf6774718c64dade4548153ae16065b595ca9/skimage/color/colorconv.py

"""


class RgbToLuv(nn.Module):
    r"""Converts an image from RGB to Luv.

    The image data is assumed to be in the range of :math:`[0, 1]`. Luv
    color is computed using the D65 illuminant and Observer 2.

    Returns:
        torch.Tensor: Luv version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> luv = RgbToLuv()
        >>> output = luv(input)  # 2x3x4x5

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

        [2] https://www.easyrgb.com/en/math.php

        [3] http://www.poynton.com/ColorFAQ.html
    """

    def __init__(self) -> None:
        super(RgbToLuv, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return kornia.color.luv.rgb_to_luv(image)


class LuvToRgb(nn.Module):
    r"""Converts an image from Luv to RGB.

    Returns:
        torch.Tensor: RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = LuvToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    References:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

        [2] https://www.easyrgb.com/en/math.php

        [3] http://www.poynton.com/ColorFAQ.html
    """

    def __init__(self) -> None:
        super(LuvToRgb, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return kornia.color.luv.luv_to_rgb(image)
