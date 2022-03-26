from typing import Optional

from kornia.color.rgb import bgr_to_rgb
from kornia.core import Module, Tensor, concatenate
from kornia.core.image import Image, ImageColor


def grayscale_to_rgb(image: Tensor) -> Tensor:
    r"""Convert a grayscale image to RGB version of image.

    .. image:: _static/img/grayscale_to_rgb.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: grayscale image tensor to be converted to RGB with shape :math:`(*,1,H,W)`.

    Returns:
        RGB version of the image with shape :math:`(*,3,H,W)`.

    Example:
        >>> import torch
        >>> input = torch.randn(2, 1, 4, 5)
        >>> gray = grayscale_to_rgb(input) # 2x3x4x5
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not an image or tensor. " f"Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 1:
        raise ValueError(f"Input size must have a shape of (*, 1, H, W). " f"Got {image.shape}.")

    return concatenate([image, image, image], dim=-3)


def rgb_to_grayscale(image: Tensor, rgb_weights: Optional[Tensor] = None) -> Tensor:
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
        >>> import torch
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = rgb_to_grayscale(input) # 2x1x4x5
    """
    if not isinstance(image, (Tensor, Image)):
        raise TypeError(f"Input type is not a Image or Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if rgb_weights is None:
        rgb_weights = Tensor([0.299, 0.587, 0.114])
        rgb_weights = rgb_weights.to(image.device, image.dtype)

    if not isinstance(rgb_weights, Tensor):
        raise TypeError(f"rgb_weights is not a torch.Tensor. Got {type(rgb_weights)}")

    if rgb_weights.shape[-1] != 3:
        raise ValueError(f"rgb_weights must have a shape of (*, 3). Got {rgb_weights.shape}")

    # the rgb channels
    r = image[..., 0:1, :, :]
    g = image[..., 1:2, :, :]
    b = image[..., 2:3, :, :]

    # the weights to apply to each channel
    w_r = rgb_weights[0:1]
    w_g = rgb_weights[1:2]
    w_b = rgb_weights[2:3]

    output = r * w_r + g * w_g + b * w_b

    if hasattr(output, "color"):
        output.color = ImageColor.GRAY32

    return output


def bgr_to_grayscale(image: Tensor) -> Tensor:
    r"""Convert a BGR image to grayscale.

    The image data is assumed to be in the range of (0, 1). First flips to RGB, then converts.

    Args:
        image: BGR image to be converted to grayscale with shape :math:`(*,3,H,W)`.

    Returns:
        grayscale version of the image with shape :math:`(*,1,H,W)`.

    Example:
        >>> import torch
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = bgr_to_grayscale(input) # 2x1x4x5
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not an image or tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    image_rgb = bgr_to_rgb(image)
    image_bgr = rgb_to_grayscale(image_rgb)  # type: ignore[arg-type]

    return image_bgr


class GrayscaleToRgb(Module):
    r"""Module to convert a grayscale image to RGB version of image.

    The image data is assumed to be in the range of (0, 1).

    Shape:
        - image: :math:`(*, 1, H, W)`
        - output: :math:`(*, 3, H, W)`

    reference:
        https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

    Example:
        >>> import torch
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
        >>> import torch
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = RgbToGrayscale()
        >>> output = gray(input)  # 2x1x4x5
    """

    def __init__(self, rgb_weights: Optional[Tensor] = None) -> None:
        super().__init__()
        if rgb_weights is None:
            rgb_weights = Tensor([0.299, 0.587, 0.114])
        self.rgb_weights = rgb_weights

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
        >>> import torch
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = BgrToGrayscale()
        >>> output = gray(input)  # 2x1x4x5
    """

    def forward(self, image: Tensor) -> Tensor:
        return bgr_to_grayscale(image)
