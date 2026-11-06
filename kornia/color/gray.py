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
from kornia.core.ops import _device_literal
from kornia.core.utils import _to_device_nonblocking


def grayscale_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a grayscale image to RGB.

    ``rgb = grayscale_to_rgb(image)`` turns a ``(*, 1, H, W)`` grayscale image into a three-channel one.

    .. image:: _static/img/grayscale_to_rgb.png

    Convention:
        The singleton channel at axis -3 is repeated into R, G and B.

    Args:
        image: grayscale image with shape ``(*, 1, H, W)``.

    Returns:
        RGB image with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`rgb_to_grayscale`: the other direction, which weights the channels and cannot be undone.
        - :class:`GrayscaleToRgb`: the module form.

    Example:
        >>> input = torch.randn(2, 1, 4, 5)
        >>> gray = grayscale_to_rgb(input) # 2x3x4x5

    References:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

    """
    KORNIA_CHECK_IS_TENSOR(image)

    if len(image.shape) < 3 or image.shape[-3] != 1:
        raise ValueError(f"Input size must have a shape of (*, 1, H, W). Got {image.shape}.")

    shape = list(image.shape)
    shape[-3] = 3
    return image.expand(*shape).clone()


def rgb_to_grayscale(image: torch.Tensor, rgb_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""Convert an RGB image to grayscale.

    ``gray = rgb_to_grayscale(image)`` returns the weighted sum of the R, G and B channels of a ``(*, 3, H, W)`` image
    as one channel, ``(*, 1, H, W)``.

    .. image:: _static/img/rgb_to_grayscale.png

    Convention:
        Default weights are ITU-R BT.601 luma, ``(0.299, 0.587, 0.114)`` in RGB order, applied to the values as given:
        the sRGB curve is not removed first.

    .. warning::
        Explicit ``rgb_weights`` expect a floating image: they are cast to the image's dtype, so on an integer image
        float weights truncate to zero and integer weights wrap. With the default weights a uint8 image is converted
        exactly, bit for bit as ``cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)`` does.

    Args:
        image: RGB image with shape ``(*, 3, H, W)``.
        rgb_weights: tensor of the R, G and B weights with shape ``(3,)``. Non-negative weights that sum to 1 keep
            the output in the range of the input.

    Returns:
        Grayscale image with shape ``(*, 1, H, W)``.

    See Also:
        - :func:`bgr_to_grayscale`: takes a BGR image, with the default weights only.
        - :func:`grayscale_to_rgb`: repeats the channel back into three.
        - :class:`RgbToGrayscale`: the module form, with ``rgb_weights`` stored.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/color_conversions.html>`__.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = rgb_to_grayscale(input) # 2x1x4x5

    References:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

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
            # Filled on the device in eager mode: ``torch.tensor(..., device=cuda)`` copies from the host and waits
            # for every queued kernel. The weights round from the same doubles either way. TorchScript, which cannot
            # compile ``_constant_tensor``, keeps the host literal.
            if torch.jit.is_scripting():
                rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype)
            else:
                rgb_weights = _device_literal([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype)
        else:
            raise TypeError(f"Unknown data type: {image.dtype}")
    else:
        # is torch.Tensor that we make sure is in the same device/dtype
        rgb_weights = _to_device_nonblocking(rgb_weights, image.device, image.dtype)

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

    ``gray = bgr_to_grayscale(image)`` converts a ``(*, 3, H, W)`` BGR image to ``(*, 1, H, W)`` with the default
    weights of :func:`rgb_to_grayscale`.

    Args:
        image: BGR image with shape ``(*, 3, H, W)``.

    Returns:
        Grayscale image with shape ``(*, 1, H, W)``.

    See Also:
        - :func:`rgb_to_grayscale`: takes an RGB image and accepts ``rgb_weights``.
        - :class:`BgrToGrayscale`: the module form.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = bgr_to_grayscale(input) # 2x1x4x5

    References:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

    """
    KORNIA_CHECK_IS_TENSOR(image)

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    image_rgb: torch.Tensor = bgr_to_rgb(image)
    return rgb_to_grayscale(image_rgb)


class GrayscaleToRgb(nn.Module):
    r"""Convert a grayscale image to RGB (module form).

    ``out = GrayscaleToRgb()(image)`` is :func:`grayscale_to_rgb`.

    Convention:
        - See the convention block of :func:`grayscale_to_rgb`.

    Shape:
        - image: ``(*, 1, H, W)``
        - output: ``(*, 3, H, W)``

    See Also:
        - :func:`grayscale_to_rgb`: the one-call form.
        - :class:`RgbToGrayscale`: the other direction.

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
            image: Input tensor with shape ``(*, 1, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``1`` is a single grayscale channel, and ``H``/``W`` are height and width.

        Returns:
            RGB tensor with shape ``(*, 3, H, W)``.
        """
        return grayscale_to_rgb(image)


class RgbToGrayscale(nn.Module):
    r"""Convert an RGB image to grayscale with stored weights (module form).

    ``conv = RgbToGrayscale(rgb_weights)`` stores the weights, and ``out = conv(image)`` is :func:`rgb_to_grayscale`
    with them. The default ``None`` keeps the default weights of the function.

    Convention:
        - See the convention block of :func:`rgb_to_grayscale`.

    Args:
        rgb_weights: tensor of the R, G and B weights with shape ``(3,)``, or ``None``.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 1, H, W)``

    See Also:
        - :func:`rgb_to_grayscale`: the one-call form.
        - :class:`BgrToGrayscale`: takes a BGR image, with the default weights only.
        - :class:`GrayscaleToRgb`: the other direction.

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
            image: Input tensor with shape ``(*, 3, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` corresponds to RGB channels, and ``H``/``W`` are height and width.

        Returns:
            Grayscale tensor with shape ``(*, 1, H, W)``, computed with the module's RGB weights.
        """
        return rgb_to_grayscale(image, rgb_weights=self.rgb_weights)


class BgrToGrayscale(nn.Module):
    r"""Convert a BGR image to grayscale (module form).

    ``out = BgrToGrayscale()(image)`` is :func:`bgr_to_grayscale`.

    Convention:
        - See the convention block of :func:`rgb_to_grayscale`.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 1, H, W)``

    See Also:
        - :func:`bgr_to_grayscale`: the one-call form.
        - :class:`RgbToGrayscale`: takes an RGB image and accepts ``rgb_weights``.

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
            image: Input tensor with shape ``(*, 3, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` corresponds to BGR channels, and ``H``/``W`` are height and width.

        Returns:
            Grayscale tensor with shape ``(*, 1, H, W)``.
        """
        return bgr_to_grayscale(image)
