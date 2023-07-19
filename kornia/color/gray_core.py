from __future__ import annotations

import keras_core as keras
import torch
import tensorflow as tf
import numpy as np
import jax.numpy as jnp

def KORNIA_CHECK_IS_TENSOR(x: object, msg: str | None = None, raises: bool = True):
    if not isinstance(x, [tf.Tensor, torch.Tensor, np.ndarray, jnp.array]):
        if raises:
            raise TypeError(f"Not a Tensor type. Got: {type(x)}.\n{msg}")
        return False
    return True

def bgr_to_rgb(image):
    if not isinstance(image, [tf.Tensor, torch.Tensor, np.ndarray, jnp.array]):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W).Got {image.shape}")

    # flip image channels
    out = keras.ops.flip(image, axis=-3)
    return out

def grayscale_to_rgb(image):
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
        raise ValueError(f"Input size must have a shape of (*, 1, H, W). " f"Got {image.shape}.")

    return keras.ops.concatenate([image, image, image], -3)

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
    KORNIA_CHECK_IS_TENSOR(image)

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if rgb_weights is None:
        # 8 bit images
        backend = keras.backend.backend()
        if backend == "numpy":
            if image.dtype == np.uint8:
                rgb_weights = np.array([76, 150, 29], dtype=np.uint8)
            elif image.dtype in (np.float16, np.float32, np.float64):
                rgb_weights = np.array([0.299, 0.587, 0.114], dtype=image.dtype)
            else:
                raise TypeError(f"Unknown data type: {image.dtype}")
        elif backend == "jax":
            if image.dtype == jnp.uint8:
                rgb_weights = jnp.array([76, 150, 29], dtype=jnp.uint8)
            elif image.dtype in (jnp.float16, jnp.float32, jnp.float64):
                rgb_weights = jnp.array([0.299, 0.587, 0.114], dtype=image.dtype)
            else:
                raise TypeError(f"Unknown data type: {image.dtype}")
        elif backend == "torch":
            if image.dtype == torch.uint8:
                rgb_weights = torch.tensor([76, 150, 29], dtype=torch.uint8)
            elif image.dtype in (torch.float16, torch.float32, torch.float64):
                rgb_weights = torch.array([0.299, 0.587, 0.114], dtype=image.dtype)
            else:
                raise TypeError(f"Unknown data type: {image.dtype}")
        elif backend == "tensorflow":
            if image.dtype == tf.uint8:
                rgb_weights = tf.convert_to_tensor([76, 150, 29], dtype=tf.uint8)
            elif image.dtype in (tf.float16, tf.float32, tf.float64):
                rgb_weights = tf.convert_to_tensor([0.299, 0.587, 0.114], dtype=image.dtype)
            else:
                raise TypeError(f"Unknown data type: {image.dtype}")
        # is tensor that we make sure is in the same device/dtype

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
    KORNIA_CHECK_IS_TENSOR(image)

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
        if rgb_weights is None:
            backend = keras.backend.backend()
            if backend == "numpy":
                rgb_weights = np.array([0.299, 0.587, 0.114])
            elif backend == "jax":
                rgb_weights = jnp.array([0.299, 0.587, 0.114])
            elif backend == "torch":
                rgb_weights = torch.tensor([0.299, 0.587, 0.114])
            elif backend == "tensorflow":
                rgb_weights = tf.convert_to_tensor([0.299, 0.587, 0.114])
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