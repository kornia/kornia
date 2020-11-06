import torch
import torch.nn as nn

import kornia


class RgbToYuv(nn.Module):
    r"""Convert an image from RGB to YUV.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        torch.Tensor: YUV version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> yuv = RgbToYuv()
        >>> output = yuv(input)  # 2x3x4x5

    Reference::
        [1] https://es.wikipedia.org/wiki/YUV#RGB_a_Y'UV
    """

    def __init__(self) -> None:
        super(RgbToYuv, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return kornia.color.yuv.rgb_to_yuv(input)


class YuvToRgb(nn.Module):
    r"""Convert an image from YUV to RGB.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        torch.Tensor: RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = YuvToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    def __init__(self) -> None:
        super(YuvToRgb, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return kornia.color.yuv.yuv_to_rgb(input)
