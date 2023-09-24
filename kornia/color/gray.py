from __future__ import annotations

import keras_core as keras  # type: ignore
from keras_core import KerasTensor as kTensor

from kornia.color.rgb import bgr_to_rgb
from kornia.core import Module, Tensor, concatenate
from kornia.core.check import KORNIA_CHECK_IS_TENSOR


def _weighted_sum_channels_kernel(r: kTensor, g: kTensor, b: kTensor, w_r: float, w_g: float, w_b: float) -> kTensor:
    return w_r * r + w_g * g + w_b * b


def grayscale_to_rgb(image: Tensor) -> Tensor:
    r"""Convert a grayscale image to RGB version of image.

    .. image:: _static/img/grayscale_to_rgb.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: grayscale image tensor to be converted to RGB with shape :math:`(*,1,H,W)`.

    Returns:
        RGB version of the image with shape :math:`(*,3,H,W)`.

    Example:
        >>> input = torch.randn(2, 1, 4, 5)
        >>> gray = grayscale_to_rgb(input) # 2x3x4x5
    """
    KORNIA_CHECK_IS_TENSOR(image)

    if len(image.shape) < 3 or image.shape[-3] != 1:
        raise ValueError(f"Input size must have a shape of (*, 1, H, W). Got {image.shape}.")

    return concatenate([image, image, image], -3)


def grayscale_from_rgb(
    image: Tensor, rgb_weights: tuple[float, float, float] | None = None, channels_axis: int = -3
) -> Tensor:
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
       See a working example `here <https://kornia.github.io/tutorials/nbs/color_conversions.html>`__.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = rgb_to_grayscale(input) # 2x1x4x5
    """
    # KORNIA_CHECK_NUM_CHANNELS(image, 3, axis=channels_axis)

    if rgb_weights is None:
        # floating point images
        if "float32" in str(image.dtype):
            rgb_weights = (0.299, 0.587, 0.114)
        # 8 bit images
        elif "uint8" in str(image.dtype):
            rgb_weights = (76, 150, 29)
        else:
            raise TypeError(f"Unknown data type: {image.dtype}")

    # unpack the color image channels with RGB order
    rgb: tuple[Tensor, Tensor, Tensor] = keras.ops.split(image, 3, axis=channels_axis)

    # compute the weighted sum of the channels
    image_grayscale = _weighted_sum_channels_kernel(*rgb, *rgb_weights)

    return image_grayscale


def bgr_to_grayscale(image: Tensor) -> Tensor:
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
    KORNIA_CHECK_IS_TENSOR(image)

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    image_rgb: Tensor = bgr_to_rgb(image)
    return rgb_to_grayscale(image_rgb)


class GrayscaleToRgb(Module):
    r"""Module to convert a grayscale image to RGB version of image.

    The image data is assumed to be in the range of (0, 1).

    Shape:
        - image: :math:`(*, 1, H, W)`
        - output: :math:`(*, 3, H, W)`

    reference:
        https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

    Example:
        >>> input = torch.rand(2, 1, 4, 5)
        >>> rgb = GrayscaleToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    def forward(self, image: Tensor) -> Tensor:
        return grayscale_to_rgb(image)


class RgbToGrayscale(Module):
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

    def __init__(self, rgb_weights: tuple[float, float, float] | None = None) -> None:
        super().__init__()
        self.rgb_weights = rgb_weights or (0.299, 0.587, 0.114)

    def forward(self, image: Tensor) -> Tensor:
        return rgb_to_grayscale(image, rgb_weights=self.rgb_weights)


class BgrToGrayscale(Module):
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

    def forward(self, image: Tensor) -> Tensor:
        return bgr_to_grayscale(image)


# aliases

rgb_to_grayscale = grayscale_from_rgb
