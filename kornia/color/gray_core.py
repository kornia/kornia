from __future__ import annotations

# import jax.numpy as jnp
import keras_core as keras

# import numpy as np
# import tensorflow as tf
# TODO: import from korani.core.ops
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_TYPE
from kornia.image import Image
from kornia.image.base import ColorSpace, ImageLayout, PixelFormat

# TODO: add once are finished
__all__ = ["bgr_from_rgb", "grayscale_from_rgb", "grayscale_from_bgr", "rgb_from_bgr", "rgb_from_grayscale"]


# kernels


def _flip_image_channel_kernel(image_data: Tensor, axis: int) -> Tensor:
    return keras.ops.flip(image_data, axis=axis)


def _replicate_image_channel_kernel(image_data: Tensor, axis: int, num_replicas: int) -> Tensor:
    return keras.ops.concatenate(num_replicas * [image_data], axis=axis)


def _weighted_sum_channels_kernel(r, g, b, w_r, w_g, w_b) -> Tensor:
    return w_r * r + w_g * g + w_b * b


# API


def rgb_from_bgr(image: Image) -> Image:
    """Convert a BGR image to RGB version of image.

    .. image:: _static/img/bgr_to_rgb.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: Image to be converted to RGB.

    Returns:
        Image: RGB version of the image.
    """
    KORNIA_CHECK_TYPE(image, Image, f"Not an Image type. Got {type(image)}")

    KORNIA_CHECK(
        image.pixel_format.color_space == ColorSpace.BGR,
        f"Input image must be in RGB. Got {image.pixel_format.color_space}",
    )

    KORNIA_CHECK(image.channels == 3, f"Input size must have a shape of (*, 3, H, W). Got {image.shape}.")

    # flip image channels
    image_bgr_data = _flip_image_channel_kernel(image.data, axis=image.channels_idx)

    image_bgr_pixel_format = PixelFormat(color_space=ColorSpace.RGB, bit_depth=image.pixel_format.bit_depth)

    return Image(image_bgr_data, image_bgr_pixel_format, image.layout)


def bgr_from_rgb(image: Image) -> Image:
    """Convert a RGB image to RGB version of image.

    .. image:: _static/img/bgr_to_rgb.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: Image to be converted to BGR.

    Returns:
        Image: RGB version of the image.
    """
    KORNIA_CHECK_TYPE(image, Image, f"Not an Image type. Got {type(image)}")

    KORNIA_CHECK(
        image.pixel_format.color_space == ColorSpace.RGB,
        f"Input image must be in RGB. Got {image.pixel_format.color_space}",
    )

    KORNIA_CHECK(image.channels == 3, f"Input size must have a shape of (*, 3, H, W). Got {image.shape}.")

    # flip image channels
    image_bgr_data = _flip_image_channel_kernel(image.data, axis=image.channels_idx)

    image_bgr_pixel_format = PixelFormat(color_space=ColorSpace.BGR, bit_depth=image.pixel_format.bit_depth)

    return Image(image_bgr_data, image_bgr_pixel_format, image.layout)


def rgb_from_grayscale(image: Image) -> Image:
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
    KORNIA_CHECK_TYPE(image, Image, f"Not an Image type. Got {type(image)}")

    KORNIA_CHECK(
        image.pixel_format.color_space == ColorSpace.GRAY,
        f"Input image must be in grayscale. Got {image.pixel_format.color_space}",
    )

    KORNIA_CHECK(image.channels == 1, f"Input size must have a shape of (*, 1, H, W). Got {image.shape}.")

    image_rbg_data = _replicate_image_channel_kernel(image.data, axis=image.channels_idx, num_replicas=3)

    image_rgb_layout = ImageLayout(
        image_size=image.layout.image_size, channels_order=image.layout.channels_order, channels=3
    )

    image_rgb_pixel_format = PixelFormat(ColorSpace.RGB, image.pixel_format.bit_depth)

    return Image(image_rbg_data, image_rgb_pixel_format, image_rgb_layout)


def grayscale_from_rgb(image: Image, rgb_weights: list[int | float] | None = None) -> Image:
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
    KORNIA_CHECK_TYPE(image, Image, f"Not an Image type. Got {type(image)}")

    KORNIA_CHECK(
        image.pixel_format.color_space == ColorSpace.RGB,
        f"Input image must be in grayscale. Got {image.pixel_format.color_space}",
    )

    KORNIA_CHECK(image.channels == 3, f"Input size must have a shape of (*, 1, H, W). Got {image.shape}.")

    if rgb_weights is None:
        # 8 bit images
        if image.pixel_format.bit_depth == 8:
            rgb_weights = [76, 150, 29]
        elif image.pixel_format.bit_depth in [16, 32, 64]:
            rgb_weights = [0.299, 0.587, 0.114]

    # unpack the color image channels with RGB order
    r = image.get_channel(0)
    g = image.get_channel(1)
    b = image.get_channel(2)

    # compute the weighted sum of the channels
    w_r, w_g, w_b = keras.ops.split(rgb_weights, 3, axis=0)

    image_grayscale_data = _weighted_sum_channels_kernel(r, g, b, w_r, w_g, w_b)

    image_grayscale_layout = ImageLayout(
        image_size=image.layout.image_size, channels_order=image.layout.channels_order, channels=1
    )

    image_grayscale_pixel_format = PixelFormat(ColorSpace.GRAY, image.pixel_format.bit_depth)

    return Image(image_grayscale_data, image_grayscale_pixel_format, image_grayscale_layout)


def grayscale_from_bgr(image):
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
    KORNIA_CHECK_TYPE(image, Image, f"Not an Image type. Got {type(image)}")

    KORNIA_CHECK(
        image.pixel_format.color_space == ColorSpace.BGR,
        f"Input image must be in BRG. Got {image.pixel_format.color_space}",
    )

    KORNIA_CHECK(image.channels == 3, f"Input size must have a shape of (*, 3, H, W). Got {image.shape}.")

    return grayscale_from_rgb(rgb_from_bgr(image))


# TODO: Layers should be automatically generated from the functions above.

# class GrayscaleToRgb(keras.layers.Layer):
#    r"""Module to convert a grayscale image to RGB version of image.
#
#    The image data is assumed to be in the range of (0, 1).
#
#    Shape:
#        - image: :math:`(*, 1, H, W)`
#        - output: :math:`(*, 3, H, W)`
#
#    reference:
#        https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
#
#    Example:
#        >>> input = torch.rand(2, 1, 4, 5)
#        >>> rgb = GrayscaleToRgb()
#        >>> output = rgb(input)  # 2x3x4x5
#    """
#
#    def __init__(self):
#        super().__init__()
#
#    def call(self, inputs):
#        return grayscale_to_rgb(inputs)
#
#
# class RgbToGrayscale(keras.layers.Layer):
#    r"""Module to convert a RGB image to grayscale version of image.
#
#    The image data is assumed to be in the range of (0, 1).
#
#    Shape:
#        - image: :math:`(*, 3, H, W)`
#        - output: :math:`(*, 1, H, W)`
#
#    reference:
#        https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
#
#    Example:
#        >>> input = torch.rand(2, 3, 4, 5)
#        >>> gray = RgbToGrayscale()
#        >>> output = gray(input)  # 2x1x4x5
#    """
#
#    def __init__(self, rgb_weights=None):
#        super().__init__()
#        # TODO: add support for different weights
#        # if rgb_weights is None:
#        #     # 8 bit images
#        #     if str(image.dtype)[-5:] == "uint8":
#        #         rgb_weights = keras.ops.convert_to_tensor([76, 150, 29], dtype=image.dtype)
#        #     elif str(image.dtype)[-7:-2] in ["float16", "float32", "float64"]:
#        #         rgb_weights = keras.ops.convert_to_tensor([0.299, 0.587, 0.114], dtype=image.dtype)
#        self.rgb_weights = rgb_weights
#
#    def forward(self, image):
#        return rgb_to_grayscale(image, rgb_weights=self.rgb_weights)
#
#
# class BgrToGrayscale(keras.layers.Layer):
#    r"""Module to convert a BGR image to grayscale version of image.
#
#    The image data is assumed to be in the range of (0, 1). First flips to RGB, then converts.
#
#    Shape:
#        - image: :math:`(*, 3, H, W)`
#        - output: :math:`(*, 1, H, W)`
#
#    reference:
#        https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
#
#    Example:
#        >>> input = torch.rand(2, 3, 4, 5)
#        >>> gray = BgrToGrayscale()
#        >>> output = gray(input)  # 2x1x4x5
#    """
#
#    def __init__(self):
#        super().__init__()
#
#    def call(self, inputs):
#        return bgr_to_grayscale(inputs)
#
