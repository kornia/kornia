import warnings

import torch
import torch.nn as nn

from kornia.color.rgb import bgr_to_rgb


def rgb_to_grayscale(
    image: torch.Tensor, rgb_weights: torch.Tensor = torch.tensor([0.299, 0.587, 0.114])
) -> torch.Tensor:
    r"""Convert a RGB image to grayscale version of image.

    .. image:: _static/img/rgb_to_grayscale.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB image to be converted to grayscale with shape :math:`(*,3,H,W)`.
        rgb_weights: Weights that will be applied on each channel (RGB).
            The sum of the weights should add up to one.
    Returns:
        grayscale version of the image with shape :math:`(*,1,H,W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       color_conversions.html>`__.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = rgb_to_grayscale(input) # 2x1x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))

    if not isinstance(rgb_weights, torch.Tensor):
        raise TypeError("rgb_weights is not a torch.Tensor. Got {}".format(type(rgb_weights)))

    if rgb_weights.shape[-1] != 3:
        raise ValueError("rgb_weights must have a shape of (*, 3). Got {}".format(rgb_weights.shape))

    r: torch.Tensor = image[..., 0:1, :, :]
    g: torch.Tensor = image[..., 1:2, :, :]
    b: torch.Tensor = image[..., 2:3, :, :]
    image_is_float: bool = torch.is_floating_point(image)
    if not image_is_float:
        warnings.warn("Input image is not of float dtype. Got {}".format(image.dtype))
    if (image.dtype != rgb_weights.dtype) and not image_is_float:
        raise TypeError(
            "Input image and rgb_weights should be of same dtype. Got {} and {}".format(image.dtype, rgb_weights.dtype)
        )
    w_tmp: torch.Tensor = rgb_weights.to(image.device, image.dtype)
    gray: torch.Tensor = w_tmp[..., 0] * r + w_tmp[..., 1] * g + w_tmp[..., 2] * b
    return gray


def bgr_to_grayscale(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a BGR image to grayscale.

    The image data is assumed to be in the range of (0, 1). First flips to RGB, then converts.

    Args:
        image: BGR image to be converted to grayscale with shape :math:`(*,3,H,W)`.

    Returns:
        grayscale version of the image with shape :math:`(*,1,H,W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = bgr_to_grayscale(input) # 2x1x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))

    image_rgb = bgr_to_rgb(image)
    gray: torch.Tensor = rgb_to_grayscale(image_rgb)
    return gray


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

    def __init__(self, rgb_weights: torch.Tensor = torch.tensor([0.299, 0.587, 0.114])) -> None:
        super(RgbToGrayscale, self).__init__()
        self.rgb_weights = rgb_weights

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return rgb_to_grayscale(image, rgb_weights=self.rgb_weights)


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

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return bgr_to_grayscale(image)
