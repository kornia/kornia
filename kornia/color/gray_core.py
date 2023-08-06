from __future__ import annotations

# import jax.numpy as jnp
import keras_core as keras

# import numpy as np
# import tensorflow as tf
# TODO: import from korani.core.ops
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK
from kornia.image import Image
from kornia.image.base import ColorSpace, ImageLayout, PixelFormat

# TODO: add once are finished
__all__ = ["grayscale_to_rgb"]

# def KORNIA_CHECK_IS_TENSOR(x: object, msg: str | None = None, raises: bool = True):
#    if not isinstance(x, [tf.Tensor, torch.Tensor, np.ndarray, jnp.array]):
#        if raises:
#            raise TypeError(f"Not a Tensor type. Got: {type(x)}.\n{msg}")
#        return False
#    return True


def bgr_to_rgb(image):
    # if not isinstance(image, [tf.Tensor, torch.Tensor, np.ndarray, jnp.array]):
    #    raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W).Got {image.shape}")

    # flip image channels
    out = keras.ops.flip(image, axis=-3)
    return out


# TODO: figure a way so that we can receive also raw tensors


def _grayscale_to_rgb_kernel(image_data: Tensor, axis: int) -> Tensor:
    return keras.ops.concatenate(3 * [image_data], axis=axis)


def grayscale_to_rgb(image: Image) -> Image:
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

    # KORNIA_CHECK_IS_TENSOR(image)

    KORNIA_CHECK(
        image.pixel_format.color_space == ColorSpace.GRAY,
        f"Input image must be in grayscale. Got {image.pixel_format.color_space}",
    )

    KORNIA_CHECK(image.channels == 1, f"Input size must have a shape of (*, 1, H, W). Got {image.shape}.")

    image_rbg_data = _grayscale_to_rgb_kernel(image.data, axis=image.channels_idx)

    image_rgb_layout = ImageLayout(
        image_size=image.layout.image_size, channels_order=image.layout.channels_order, channels=3
    )

    image_rgb_pixel_format = PixelFormat(ColorSpace.RGB, image.pixel_format.bit_depth)

    return Image(image_rbg_data, image_rgb_pixel_format, image_rgb_layout)


def rgb_to_grayscale(image, rgb_weights=None):
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
    # KORNIA_CHECK_IS_TENSOR(image)

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if rgb_weights is None:
        # 8 bit images
        if str(image.dtype)[-5:] == "uint8":
            rgb_weights = keras.ops.convert_to_tensor([76, 150, 29], dtype=image.dtype)
        elif str(image.dtype)[-7:-2] in ["float16", "float32", "float64"]:
            rgb_weights = keras.ops.convert_to_tensor([0.299, 0.587, 0.114], dtype=image.dtype)

    # unpack the color image channels with RGB order
    r = image[..., 0:1, :, :]
    g = image[..., 1:2, :, :]
    b = image[..., 2:3, :, :]

    w_r, w_g, w_b = keras.ops.split(rgb_weights, 3, axis=0)
    return w_r * r + w_g * g + w_b * b


def bgr_to_grayscale(image):
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
    # KORNIA_CHECK_IS_TENSOR(image)

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    image_rgb = bgr_to_rgb(image)
    return rgb_to_grayscale(image_rgb)


class GrayscaleToRgb(keras.layers.Layer):
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

    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return grayscale_to_rgb(inputs)


class RgbToGrayscale(keras.layers.Layer):
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

    def __init__(self, rgb_weights=None):
        super().__init__()
        # TODO: add support for different weights
        # if rgb_weights is None:
        #     # 8 bit images
        #     if str(image.dtype)[-5:] == "uint8":
        #         rgb_weights = keras.ops.convert_to_tensor([76, 150, 29], dtype=image.dtype)
        #     elif str(image.dtype)[-7:-2] in ["float16", "float32", "float64"]:
        #         rgb_weights = keras.ops.convert_to_tensor([0.299, 0.587, 0.114], dtype=image.dtype)
        self.rgb_weights = rgb_weights

    def forward(self, image):
        return rgb_to_grayscale(image, rgb_weights=self.rgb_weights)


class BgrToGrayscale(keras.layers.Layer):
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

    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return bgr_to_grayscale(inputs)
