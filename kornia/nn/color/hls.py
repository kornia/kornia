import torch
import torch.nn as nn

import kornia


class RgbToHls(nn.Module):
    r"""Convert an image from RGB to HLS.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        torch.Tensor: HLS version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> hls = RgbToHls()
        >>> output = hls(input)  # 2x3x4x5
    """

    def __init__(self) -> None:
        super(RgbToHls, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return kornia.color.hls.rgb_to_hls(image)


class HlsToRgb(nn.Module):
    r"""Convert an image from HLS to RGB.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        torch.Tensor: RGB version of the image.

    Shape:
        - input: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Reference:
        https://en.wikipedia.org/wiki/HSL_and_HSV

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = HlsToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    def __init__(self) -> None:
        super(HlsToRgb, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return kornia.color.hls.hls_to_rgb(image)
