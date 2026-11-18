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

from typing import ClassVar

import torch
from torch import nn

from kornia.color.utils import _apply_linear_transformation
from kornia.core.check import KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE


def rgb_to_xyz(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a linear RGB image to XYZ.

    ``xyz = rgb_to_xyz(image)`` converts a ``(*, 3, H, W)`` linear RGB image to CIE XYZ with the sRGB primaries and the
    D65 white point. It does not remove the sRGB curve: for an sRGB image, pass the result of
    :func:`rgb_to_linear_rgb`.

    .. image:: _static/img/rgb_to_xyz.png

    Convention:
        A matrix transform of the channels at axis -3. OpenCV's ``COLOR_RGB2XYZ`` does the same, and skimage
        ``rgb2xyz`` linearizes its input first.

    Args:
        image: linear RGB image with shape ``(*, 3, H, W)``.

    Returns:
        XYZ image with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`xyz_to_rgb`: the inverse.
        - :func:`rgb_to_linear_rgb`: the step before this one.
        - :class:`RgbToXyz`: the module form.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_xyz(input)  # 2x3x4x5

    """
    KORNIA_CHECK_IS_TENSOR(image)
    KORNIA_CHECK_SHAPE(image, ["*", "3", "H", "W"])

    # CIE RGB to XYZ Matrix (D65 White Point)
    dtype = image.dtype if image.is_floating_point() else torch.float32
    kernel = torch.tensor(
        [
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227],
        ],
        device=image.device,
        dtype=dtype,
    )

    # Apply Optimized Linear Transformation
    return _apply_linear_transformation(image, kernel)


def xyz_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an XYZ image to linear RGB.

    ``rgb = xyz_to_rgb(image)`` converts a ``(*, 3, H, W)`` CIE XYZ image with the D65 white point to linear RGB with
    the sRGB primaries. For an sRGB result, apply :func:`linear_rgb_to_rgb` to it.

    Convention:
        A matrix transform of the channels at axis -3. The result is not clipped.

    Args:
        image: XYZ image with shape ``(*, 3, H, W)``.

    Returns:
        Linear RGB image with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`rgb_to_xyz`: the inverse.
        - :func:`linear_rgb_to_rgb`: the step after this one.
        - :class:`XyzToRgb`: the module form.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = xyz_to_rgb(input)  # 2x3x4x5

    """
    KORNIA_CHECK_IS_TENSOR(image)
    KORNIA_CHECK_SHAPE(image, ["*", "3", "H", "W"])

    # CIE XYZ to RGB Matrix (D65 White Point)
    dtype = image.dtype if image.is_floating_point() else torch.float32
    kernel = torch.tensor(
        [
            [3.2404813432005266, -1.5371515162713185, -0.4985363261688878],
            [-0.9692549499965682, 1.8759900014898907, 0.0415559265582928],
            [0.0556466391351772, -0.2040413383665112, 1.0573110696453443],
        ],
        device=image.device,
        dtype=dtype,
    )

    # Apply Optimized Linear Transformation
    return _apply_linear_transformation(image, kernel)


class RgbToXyz(nn.Module):
    r"""Convert an image from linear RGB to XYZ (module form).

    ``out = RgbToXyz()(image)`` is :func:`rgb_to_xyz`.

    Convention:
        - See the convention block of :func:`rgb_to_xyz`.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 3, H, W)``

    See Also:
        - :func:`rgb_to_xyz`: the one-call form.
        - :class:`XyzToRgb`: the inverse.
        - :class:`RgbToLinearRgb`: removes the sRGB curve before this module.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> xyz = RgbToXyz()
        >>> output = xyz(input)  # 2x3x4x5

    References:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert an RGB tensor to XYZ.

        Args:
            image: Input tensor with shape ``(*, 3, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` corresponds to RGB channels, and ``H``/``W`` are height and width.

        Returns:
            XYZ tensor with shape ``(*, 3, H, W)``.
        """
        return rgb_to_xyz(image)


class XyzToRgb(nn.Module):
    r"""Convert an image from XYZ to linear RGB (module form).

    ``out = XyzToRgb()(image)`` is :func:`xyz_to_rgb`.

    Convention:
        - See the convention block of :func:`xyz_to_rgb`.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 3, H, W)``

    See Also:
        - :func:`xyz_to_rgb`: the one-call form.
        - :class:`RgbToXyz`: the inverse.
        - :class:`LinearRgbToRgb`: applies the sRGB curve after this module.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = XyzToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    References:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert an XYZ tensor to RGB.

        Args:
            image: Input tensor with shape ``(*, 3, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` corresponds to XYZ channels, and ``H``/``W`` are height and width.

        Returns:
            RGB tensor with shape ``(*, 3, H, W)``.
        """
        return xyz_to_rgb(image)
