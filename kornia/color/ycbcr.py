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

from kornia.core import ImageModule as Module
from kornia.core import Tensor


def rgb_to_ycbcr(image: Tensor) -> Tensor:
    r"""Convert an RGB image to YCbCr.

    .. image:: _static/img/rgb_to_ycbcr.png

    Args:
        image: RGB Image to be converted to YCbCr with shape :math:`(*, 3, H, W)`.

    Returns:
        YCbCr version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_ycbcr(input)  # 2x3x4x5

    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    input_shape = image.shape
    h, w = image.shape[-2:]
    image = image.reshape(-1, 3, h, w)

    y_weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype)
    y = torch.einsum("c,...chw->...hw", y_weights, image).unsqueeze(-3)

    r_slice = image[..., 0:1, :, :]
    b_slice = image[..., 2:3, :, :]
    delta = torch.tensor(0.5, device=image.device, dtype=image.dtype)
    cb_weight = torch.tensor(0.564, device=image.device, dtype=image.dtype)
    cr_weight = torch.tensor(0.713, device=image.device, dtype=image.dtype)

    cb = torch.addcmul(delta, b_slice - y, cb_weight)
    cr = torch.addcmul(delta, r_slice - y, cr_weight)
    return torch.cat([y, cb, cr], dim=-3).reshape(input_shape)


def rgb_to_y(image: Tensor) -> Tensor:
    r"""Convert an RGB image to Y.

    Args:
        image: RGB Image to be converted to Y with shape :math:`(*, 3, H, W)`.

    Returns:
        Y version of the image with shape :math:`(*, 1, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_y(input)  # 2x1x4x5

    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: Tensor = image[..., 0:1, :, :]
    g: Tensor = image[..., 1:2, :, :]
    b: Tensor = image[..., 2:3, :, :]

    y: Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    return y


def ycbcr_to_rgb(image: Tensor) -> Tensor:
    r"""Convert an YCbCr image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: YCbCr Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = ycbcr_to_rgb(input)  # 2x3x4x5

    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    input_shape = image.shape
    h, w = image.shape[-2:]
    image = image.reshape(-1, 3, h, w)

    ycbcr_to_rgb_matrix = torch.tensor(
        [
            [1.0, 0.0, 1.403],
            [1.0, -0.344, -0.714],
            [1.0, 1.773, 0.0],
        ],
        device=image.device,
        dtype=image.dtype,
    )

    delta = 0.5
    shift = torch.tensor([0.0, delta, delta], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)

    shifted_image = image - shift
    rgb_image = torch.einsum("ij, ...jhw -> ...ihw", ycbcr_to_rgb_matrix, shifted_image)
    return rgb_image.clamp(0, 1).reshape(input_shape)


class RgbToYcbcr(Module):
    r"""Convert an image from RGB to YCbCr.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        YCbCr version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> ycbcr = RgbToYcbcr()
        >>> output = ycbcr(input)  # 2x3x4x5

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, image: Tensor) -> Tensor:
        return rgb_to_ycbcr(image)


class YcbcrToRgb(Module):
    r"""Convert an image from YCbCr to Rgb.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = YcbcrToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, image: Tensor) -> Tensor:
        return ycbcr_to_rgb(image)
