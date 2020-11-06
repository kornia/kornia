import torch
import torch.nn as nn


class RgbToGrayscale(nn.Module):
    r"""Module to convert a RGB image to grayscale version of image.

    The image data is assumed to be in the range of (0, 1).

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 1, H, W)`

    reference:
        https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = RgbToGrayscale()
        >>> output = gray(input)  # 2x1x4x5
    """

    def __init__(self) -> None:
        super(RgbToGrayscale, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        from kornia.color.gray import rgb_to_grayscale
        return rgb_to_grayscale(image)


class BgrToGrayscale(nn.Module):
    r"""Module to convert a BGR image to grayscale version of image.

    The image data is assumed to be in the range of (0, 1). First flips to RGB, then converts.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 1, H, W)`

    reference:
        https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = BgrToGrayscale()
        >>> output = gray(input)  # 2x1x4x5
    """

    def __init__(self) -> None:
        super(BgrToGrayscale, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        from kornia.color.gray import bgr_to_grayscale
        return bgr_to_grayscale(image)
