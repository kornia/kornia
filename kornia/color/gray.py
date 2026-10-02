# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

from typing import ClassVar, Optional

import torch
from torch import nn

from kornia.color.rgb import bgr_to_rgb
from kornia.core.check import KORNIA_CHECK_IS_TENSOR


def grayscale_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a grayscale image to RGB version of image.

    .. image:: _static/img/grayscale_to_rgb.png

    The image data is assumed to be in the range of (0, 1).

    Convention:
        The singleton channel at axis -3 is repeated into R, G and B. Leading dimensions are preserved.

    Args:
        image: grayscale image torch.Tensor to be converted to RGB with shape :math:`(*,1,H,W)`.

    Returns:
        RGB version of the image with shape :math:`(*,3,H,W)`.

    Example:
        >>> input = torch.randn(2, 1, 4, 5)
        >>> gray = grayscale_to_rgb(input) # 2x3x4x5

    """
    KORNIA_CHECK_IS_TENSOR(image)

    if len(image.shape) < 3 or image.shape[-3] != 1:
        raise ValueError(f"Input size must have a shape of (*, 1, H, W). Got {image.shape}.")

    shape = list(image.shape)
    shape[-3] = 3
    # Use expand and clone to create independent memory storage.
    return image.expand(*shape).clone()


def rgb_to_grayscale(image: torch.Tensor, rgb_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""Convert a RGB image to grayscale version of image.

    .. image:: _static/img/rgb_to_grayscale.png

    The image data is assumed to be in the range of (0, 1).

    Convention:
        RGB is channel-first at axis -3 and the result has one channel. Default weights are
        ITU-R BT.601 luma weights (0.299, 0.587, 0.114) in RGB order.

    .. warning::
        On integer input, explicit float ``rgb_weights`` truncate to zero and integer weights wrap
        (`#5109 <https://github.com/kornia/kornia/issues/5109>`_).

    Args:
        image: RGB image to be converted to grayscale with shape :math:`(*,3,H,W)`.
        rgb_weights: Weights that will be applied on each channel (RGB).
            The sum of the weights should add up to one.

    Returns:
        grayscale version of the image with shape :math:`(*,1,H,W)`.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/color_conversions.html>`__.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = rgb_to_grayscale(input) # 2x1x4x5

    """
    KORNIA_CHECK_IS_TENSOR(image)

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if rgb_weights is None:
        # 8 bit images: OpenCV's fixed-point COLOR_RGB2GRAY weights, 0.299, 0.587 and 0.114 scaled by 2**15 and
        # summing to 2**15, so a flat grey maps to itself and the result matches cv2.cvtColor bit for bit. A uint8
        # multiply wraps (white came back as 1, pure red as 180), so accumulate in int32 and round back.
        if image.dtype == torch.uint8:
            r, g, b = image.unbind(dim=-3)
            acc = r.to(torch.int32) * 9798 + g.to(torch.int32) * 19235 + b.to(torch.int32) * 3735
            return ((acc + 16384) // 32768).to(dtype=torch.uint8).unsqueeze(-3)
        # floating point images
        if image.dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype)
        else:
            raise TypeError(f"Unknown data type: {image.dtype}")
    else:
        # is torch.Tensor that we make sure is in the same device/dtype
        rgb_weights = rgb_weights.to(image)

    # Unpack channels (View, don't copy)
    r, g, b = image.unbind(dim=-3)
    w_r, w_g, w_b = rgb_weights.unbind()

    # Accumulate results
    out = r * w_r
    out = torch.addcmul(out, g, w_g)
    out = torch.addcmul(out, b, w_b)

    # Restore channel dim
    return out.unsqueeze(-3)


def bgr_to_grayscale(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a BGR image to grayscale.

    The image data is assumed to be in the range of (0, 1). First flips to RGB, then converts.

    Convention:
        BGR is channel-first at axis -3. This uses the RGB luma weights after reversing channel order.

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

    image_rgb: torch.Tensor = bgr_to_rgb(image)
    return rgb_to_grayscale(image_rgb)


class GrayscaleToRgb(nn.Module):
    r"""nn.Module to convert a grayscale image to RGB version of image.

    See the Convention block on :func:`grayscale_to_rgb`.

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

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 1, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert a grayscale tensor to RGB.

        Args:
            image: Input tensor with shape :math:`(*, 1, H, W)`.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``1`` is a single grayscale channel, and ``H``/``W`` are height and width.

        Returns:
            RGB tensor with shape :math:`(*, 3, H, W)`.
        """
        return grayscale_to_rgb(image)


class RgbToGrayscale(nn.Module):
    r"""nn.Module to convert a RGB image to grayscale version of image.

    See the Convention block on :func:`rgb_to_grayscale`.

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

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 1, -1, -1]

    def __init__(self, rgb_weights: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        # None stays None: rgb_to_grayscale then picks the weights for the image dtype. Float32 defaults stored here
        # were cast to the image, so a uint8 image got the weights [0, 0, 0] and came back all zeros, and a float64
        # image got float32-rounded weights (#5109).
        self.rgb_weights = rgb_weights

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert an RGB tensor to grayscale.

        Args:
            image: Input tensor with shape :math:`(*, 3, H, W)`.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` corresponds to RGB channels, and ``H``/``W`` are height and width.

        Returns:
            Grayscale tensor with shape :math:`(*, 1, H, W)`, computed with the module's RGB weights.
        """
        return rgb_to_grayscale(image, rgb_weights=self.rgb_weights)


class BgrToGrayscale(nn.Module):
    r"""nn.Module to convert a BGR image to grayscale version of image.

    See the Convention block on :func:`bgr_to_grayscale`.

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

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 1, -1, -1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert a BGR tensor to grayscale.

        Args:
            image: Input tensor with shape :math:`(*, 3, H, W)`.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` corresponds to BGR channels, and ``H``/``W`` are height and width.

        Returns:
            Grayscale tensor with shape :math:`(*, 1, H, W)`.
        """
        return bgr_to_grayscale(image)