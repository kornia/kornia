"""Module for grayscale conversions and utilities."""
from __future__ import annotations

import kornia.core as kornia_core
from kornia.color.rgb import bgr_to_rgb
from kornia.core import Module, Tensor, concatenate
from kornia.core.check import KORNIA_CHECK_IS_TENSOR


def _weighted_sum_channels_kernel(r: Tensor, g: Tensor, b: Tensor, w_r: float, w_g: float, w_b: float) -> Tensor:
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
    image: Tensor, rgb_weights: tuple[float, float, float] | None = None, channels_axis: int | None = None
) -> Tensor:
    r"""Convert a RGB image to grayscale version of image.

    .. image:: _static/img/rgb_to_grayscale.png

    The image data is assumed to be in the range of (0, 1) in float or (0, 255) in uint8.

    Args:
        image: RGB image to be converted to grayscale with shape :math:`(*,3,H,W)` or :math:`(*, H, W, 3)`.
            You need to specify the channel axis using `channels_axis` if the image shape is :math:`(*, H, W, 3)`.
        rgb_weights: Weights that will be applied on each channel (RGB).
            The sum of the weights should add up to one.
            If None, the standard weights are used (0.299, 0.587, 0.114).
        channels_axis: The axis corresponding to the channels dimension. If `None` is used, the channel will be defined
            by `kornia_core.channels_axis()`, which relies into keras `keras.backend.image_data_format()` flag.

    Returns:
        grayscale version of the image with shape :math:`(*,1,H,W)` or :math:`(*, H, W, 1)`.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/color_conversions.html>`__.

    Example:
        >>> rgb = torch.rand(3, 4, 5)  # try also with jax, numpy, or tensorflow
        >>> gray = grayscale_from_rgb(rgb) # 1x4x5
        >>> tuple(gray.shape)
        (1, 4, 5)
    """
    # KORNIA_CHECK_NUM_CHANNELS(image, 3, axis=channels_axis)

    channels_axis = kornia_core.channels_axis() if channels_axis is None else channels_axis

    if rgb_weights is None:
        # floating point images
        if "float" in str(image.dtype):
            rgb_weights = (0.299, 0.587, 0.114)
        # 8 bit images
        elif "uint8" in str(image.dtype):
            rgb_weights = (76, 150, 29)
        else:
            raise TypeError(f"Unknown data type: {image.dtype}")

    # unpack the color image channels with RGB order
    rgb: tuple[Tensor, Tensor, Tensor] = kornia_core.ops.split(image, 3, axis=channels_axis)

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
