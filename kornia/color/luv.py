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

"""The RGB to Luv color transformations were translated from scikit image's rgb2luv and luv2rgb.

https://github.com/scikit-image/scikit-image/blob/a48bf6774718c64dade4548153ae16065b595ca9/skimage/color/colorconv.py
"""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import nn

from kornia.core.utils import _autocast_disabled

from .rgb import linear_rgb_to_rgb, rgb_to_linear_rgb
from .xyz import rgb_to_xyz, xyz_to_rgb


def rgb_to_luv(image: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    r"""Convert an sRGB image to Luv.

    ``luv = rgb_to_luv(image)`` converts a ``(*, 3, H, W)`` sRGB image in :math:`[0, 1]` to CIE Luv with the D65 white
    point and the 2° observer. L is in :math:`[0, 100]`.

    .. image:: _static/img/rgb_to_luv.png

    Convention:
        Float16 and bfloat16 inputs are computed in float32 and rounded to the input dtype at return.
        Channels are ``L``, ``u``, ``v`` at axis -3. The sRGB curve is removed inside, so pass sRGB, not linear RGB.
        The scale is that of OpenCV's float32 Luv, not of its 8-bit Luv, and the values match skimage ``rgb2luv``.

    Args:
        image: sRGB image with shape ``(*, 3, H, W)``.
        eps: added to the divisor ``x + 15 y + 3 z``. ``eps=0`` gives NaN in U and V on black pixels.

    Returns:
        Luv image with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`luv_to_rgb`: the inverse.
        - :class:`RgbToLuv`: the module form, which has no ``eps`` and uses the default.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_luv(input)  # 2x3x4x5

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if image.dtype in (torch.float16, torch.bfloat16):
        if torch.jit.is_scripting():
            return _rgb_to_luv(image.float(), eps).to(image.dtype)
        # Color matrices use einsum/convolution, which an enclosing autocast would downcast again.
        with _autocast_disabled(image.device.type):
            return _rgb_to_luv(image.float(), eps).to(image.dtype)
    return _rgb_to_luv(image, eps)


def _rgb_to_luv(image: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute the conversion in the working dtype."""
    # Convert from sRGB to Linear RGB
    lin_rgb = rgb_to_linear_rgb(image)

    xyz_im: torch.Tensor = rgb_to_xyz(lin_rgb)

    x: torch.Tensor = xyz_im[..., 0, :, :]
    y: torch.Tensor = xyz_im[..., 1, :, :]
    z: torch.Tensor = xyz_im[..., 2, :, :]

    threshold = 0.008856
    L: torch.Tensor = torch.where(
        y > threshold,
        116.0 * torch.pow(y.clamp(min=threshold), 1.0 / 3.0) - 16.0,
        903.3 * y,
    )

    # Compute reference white point
    xyz_ref_white: tuple[float, float, float] = (0.95047, 1.0, 1.08883)
    u_w: float = (4 * xyz_ref_white[0]) / (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2])
    v_w: float = (9 * xyz_ref_white[1]) / (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2])

    u_p: torch.Tensor = (4 * x) / (x + 15 * y + 3 * z + eps)
    v_p: torch.Tensor = (9 * y) / (x + 15 * y + 3 * z + eps)

    u: torch.Tensor = 13 * L * (u_p - u_w)
    v: torch.Tensor = 13 * L * (v_p - v_w)

    return torch.stack([L, u, v], dim=-3)


def luv_to_rgb(image: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    r"""Convert a Luv image to sRGB.

    ``rgb = luv_to_rgb(image)`` converts a ``(*, 3, H, W)`` CIE Luv image with the D65 white point and the 2° observer
    to sRGB. L is expected in :math:`[0, 100]`.

    Convention:
        Float16 and bfloat16 inputs are computed in float32 and rounded to the input dtype at return.
        Channels are ``L``, ``u``, ``v`` at axis -3. The result is not clipped, so out-of-gamut Luv values give RGB
        values outside :math:`[0, 1]`.

    Args:
        image: Luv image with shape ``(*, 3, H, W)``.
        eps: added to the divisors of the inverse formula. ``eps=0`` gives NaN on black pixels.

    Returns:
        sRGB image with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`rgb_to_luv`: the inverse.
        - :class:`LuvToRgb`: the module form, which has no ``eps`` and uses the default.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = luv_to_rgb(input)  # 2x3x4x5

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if image.dtype in (torch.float16, torch.bfloat16):
        if torch.jit.is_scripting():
            return _luv_to_rgb(image.float(), eps).to(image.dtype)
        # Color matrices use einsum/convolution, which an enclosing autocast would downcast again.
        with _autocast_disabled(image.device.type):
            return _luv_to_rgb(image.float(), eps).to(image.dtype)
    return _luv_to_rgb(image, eps)


def _luv_to_rgb(image: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute the conversion in the working dtype."""
    L: torch.Tensor = image[..., 0, :, :]
    u: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :]

    # Convert from Luv to XYZ
    y: torch.Tensor = torch.where(L > 7.999625, torch.pow((L + 16) / 116, 3.0), L / 903.3)

    # Compute white point
    xyz_ref_white: tuple[float, float, float] = (0.95047, 1.0, 1.08883)
    u_w: float = (4 * xyz_ref_white[0]) / (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2])
    v_w: float = (9 * xyz_ref_white[1]) / (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2])

    a: torch.Tensor = u_w + u / (13 * L + eps)
    d: torch.Tensor = v_w + v / (13 * L + eps)
    c: torch.Tensor = 3 * y * (5 * d - 3)

    z: torch.Tensor = ((a - 4) * c - 15 * a * d * y) / (12 * d + eps)
    x: torch.Tensor = -(c / (d + eps) + 3.0 * z)

    xyz_im: torch.Tensor = torch.stack([x, y, z], -3)

    rgbs_im: torch.Tensor = xyz_to_rgb(xyz_im)

    # Convert from RGB Linear to sRGB
    return linear_rgb_to_rgb(rgbs_im)


class RgbToLuv(nn.Module):
    r"""Convert an image from sRGB to Luv (module form).

    ``out = RgbToLuv()(image)`` is :func:`rgb_to_luv` with its default ``eps``.

    Convention:
        - See the convention block of :func:`rgb_to_luv`.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 3, H, W)``

    See Also:
        - :func:`rgb_to_luv`: the one-call form, with an ``eps`` argument.
        - :class:`LuvToRgb`: the inverse.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> luv = RgbToLuv()
        >>> output = luv(input)  # 2x3x4x5

    References:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

        [2] https://www.easyrgb.com/en/math.php

        [3] http://www.poynton.com/ColorFAQ.html

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert an RGB tensor to Luv.

        Args:
            image: Input tensor with shape ``(*, 3, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` corresponds to RGB channels, and ``H``/``W`` are height and width.

        Returns:
            Luv tensor with shape ``(*, 3, H, W)``.
        """
        return rgb_to_luv(image)


class LuvToRgb(nn.Module):
    r"""Convert an image from Luv to sRGB (module form).

    ``out = LuvToRgb()(image)`` is :func:`luv_to_rgb` with its default ``eps``.

    Convention:
        - See the convention block of :func:`luv_to_rgb`.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 3, H, W)``

    See Also:
        - :func:`luv_to_rgb`: the one-call form, with an ``eps`` argument.
        - :class:`RgbToLuv`: the inverse.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = LuvToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    References:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

        [2] https://www.easyrgb.com/en/math.php

        [3] http://www.poynton.com/ColorFAQ.html

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert a Luv tensor to RGB.

        Args:
            image: Input tensor with shape ``(*, 3, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` corresponds to Luv channels, and ``H``/``W`` are height and width.

        Returns:
            RGB tensor with shape ``(*, 3, H, W)``.
        """
        return luv_to_rgb(image)
