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

"""The RGB to Lab color transformations were translated from scikit image's rgb2lab and lab2rgb.

https://github.com/scikit-image/scikit-image/blob/a48bf6774718c64dade4548153ae16065b595ca9/skimage/color/colorconv.py
"""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import nn

from kornia.core.utils import _autocast_disabled

from .rgb import linear_rgb_to_rgb, rgb_to_linear_rgb
from .xyz import rgb_to_xyz, xyz_to_rgb


def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an sRGB image to Lab.

    ``lab = rgb_to_lab(image)`` converts a ``(*, 3, H, W)`` sRGB image in :math:`[0, 1]` to CIE Lab with the D65 white
    point and the 2° observer. L is in :math:`[0, 100]`, and a and b lie within :math:`[-128, 127]`.

    .. image:: _static/img/rgb_to_lab.png

    Convention:
        Float16 and bfloat16 inputs are computed in float32 and rounded to the input dtype at return.
        Channels are ``L``, ``a``, ``b`` at axis -3. The sRGB curve is removed inside, so pass sRGB, not linear RGB.
        The scale is that of OpenCV's float32 Lab, not of its 8-bit Lab, where L is scaled to 0..255 and a and b are
        offset by 128. Values match skimage ``rgb2lab``.
        OpenCV 5's float ``COLOR_RGB2Lab`` quantizes its input, so it differs by less than 1 in each channel.

    Args:
        image: sRGB image with shape ``(*, 3, H, W)``.

    Returns:
        Lab image with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`lab_to_rgb`: the inverse.
        - :class:`RgbToLab`: the module form.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_lab(input)  # 2x3x4x5

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if image.dtype in (torch.float16, torch.bfloat16):
        if torch.jit.is_scripting():
            return _rgb_to_lab(image.float()).to(image.dtype)
        # Color matrices use einsum/convolution, which an enclosing autocast would downcast again.
        with _autocast_disabled(image.device.type):
            return _rgb_to_lab(image.float()).to(image.dtype)
    return _rgb_to_lab(image)


def _rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    """Compute the conversion in the working dtype."""
    # Convert from sRGB to Linear RGB
    lin_rgb = rgb_to_linear_rgb(image)

    xyz_im: torch.Tensor = rgb_to_xyz(lin_rgb)

    # F.normalize for D65 white point
    xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883], device=xyz_im.device, dtype=xyz_im.dtype)[..., :, None, None]
    xyz_normalized = torch.div(xyz_im, xyz_ref_white)

    threshold = 0.008856
    power = torch.pow(xyz_normalized.clamp(min=threshold), 1 / 3.0)
    scale = 7.787 * xyz_normalized + 4.0 / 29.0
    xyz_int = torch.where(xyz_normalized > threshold, power, scale)

    x: torch.Tensor = xyz_int[..., 0, :, :]
    y: torch.Tensor = xyz_int[..., 1, :, :]
    z: torch.Tensor = xyz_int[..., 2, :, :]

    L: torch.Tensor = (116.0 * y) - 16.0
    a: torch.Tensor = 500.0 * (x - y)
    _b: torch.Tensor = 200.0 * (y - z)

    out: torch.Tensor = torch.stack([L, a, _b], dim=-3)

    return out


def lab_to_rgb(image: torch.Tensor, clip: bool = True) -> torch.Tensor:
    r"""Convert a Lab image to sRGB.

    ``rgb = lab_to_rgb(image)`` converts a ``(*, 3, H, W)`` CIE Lab image with the D65 white point and the 2° observer
    to sRGB. L is expected in :math:`[0, 100]`, and a and b in :math:`[-128, 127]`.

    Convention:
        Float16 and bfloat16 inputs are computed in float32 and rounded to the input dtype at return.
        Channels are ``L``, ``a``, ``b`` at axis -3.

    Args:
        image: Lab image with shape ``(*, 3, H, W)``.
        clip: clamp the result to :math:`[0, 1]`. With ``False``, out-of-gamut Lab values give RGB values outside it.

    Returns:
        sRGB image with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`rgb_to_lab`: the inverse.
        - :class:`LabToRgb`: the module form, which takes ``clip`` when called.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = lab_to_rgb(input)  # 2x3x4x5

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if image.dtype in (torch.float16, torch.bfloat16):
        if torch.jit.is_scripting():
            return _lab_to_rgb(image.float(), clip).to(image.dtype)
        # Color matrices use einsum/convolution, which an enclosing autocast would downcast again.
        with _autocast_disabled(image.device.type):
            return _lab_to_rgb(image.float(), clip).to(image.dtype)
    return _lab_to_rgb(image, clip)


def _lab_to_rgb(image: torch.Tensor, clip: bool) -> torch.Tensor:
    """Compute the conversion in the working dtype."""
    L: torch.Tensor = image[..., 0, :, :]
    a: torch.Tensor = image[..., 1, :, :]
    _b: torch.Tensor = image[..., 2, :, :]

    fy = (L + 16.0) / 116.0
    fx = (a / 500.0) + fy
    fz = fy - (_b / 200.0)

    # if color data out of range: Z < 0
    fz = fz.clamp(min=0.0)

    fxyz = torch.stack([fx, fy, fz], dim=-3)

    # Convert from Lab to XYZ
    power = torch.pow(fxyz, 3.0)
    scale = (fxyz - 4.0 / 29.0) / 7.787
    xyz = torch.where(fxyz > 0.2068966, power, scale)

    # For D65 white point
    xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883], device=xyz.device, dtype=xyz.dtype)[..., :, None, None]
    xyz_im = xyz * xyz_ref_white

    rgbs_im: torch.Tensor = xyz_to_rgb(xyz_im)

    # https://github.com/richzhang/colorization-pytorch/blob/66a1cb2e5258f7c8f374f582acc8b1ef99c13c27/util/util.py#L107
    #     rgbs_im = torch.where(rgbs_im < 0, torch.zeros_like(rgbs_im), rgbs_im)

    # Convert from RGB Linear to sRGB
    rgb_im = linear_rgb_to_rgb(rgbs_im)

    # Clip to 0,1 https://www.w3.org/Graphics/Color/srgb
    if clip:
        rgb_im = torch.clamp(rgb_im, min=0.0, max=1.0)

    return rgb_im


class RgbToLab(nn.Module):
    r"""Convert an image from sRGB to Lab (module form).

    ``out = RgbToLab()(image)`` is :func:`rgb_to_lab`.

    Convention:
        - See the convention block of :func:`rgb_to_lab`.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 3, H, W)``

    See Also:
        - :func:`rgb_to_lab`: the one-call form.
        - :class:`LabToRgb`: the inverse.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> lab = RgbToLab()
        >>> output = lab(input)  # 2x3x4x5

    References:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

        [2] https://www.easyrgb.com/en/math.php

        [3] https://github.com/torch/image/blob/dc061b98fb7e946e00034a5fc73e883a299edc7f/generic/image.c#L1467

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert an RGB tensor to Lab.

        Args:
            image: Input tensor with shape ``(*, 3, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` corresponds to RGB channels, and ``H``/``W`` are height and width.

        Returns:
            Lab tensor with shape ``(*, 3, H, W)``.
        """
        return rgb_to_lab(image)


class LabToRgb(nn.Module):
    r"""Convert an image from Lab to sRGB (module form).

    ``out = LabToRgb()(image, clip)`` is :func:`lab_to_rgb`. ``clip`` is an argument of the call, not of the
    constructor.

    Convention:
        - See the convention block of :func:`lab_to_rgb`.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 3, H, W)``

    See Also:
        - :func:`lab_to_rgb`: the one-call form.
        - :class:`RgbToLab`: the inverse.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = LabToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    References:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

        [2] https://www.easyrgb.com/en/math.php

        [3] https://github.com/torch/image/blob/dc061b98fb7e946e00034a5fc73e883a299edc7f/generic/image.c#L1518

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, image: torch.Tensor, clip: bool = True) -> torch.Tensor:
        """Convert a Lab tensor to RGB.

        Args:
            image: Input tensor with shape ``(*, 3, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` corresponds to Lab channels, and ``H``/``W`` are height and width.
            clip: If ``True``, clamp output values to :math:`[0, 1]`.

        Returns:
            RGB tensor with shape ``(*, 3, H, W)``.
        """
        return lab_to_rgb(image, clip)
