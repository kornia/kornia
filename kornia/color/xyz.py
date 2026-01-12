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
import torch.nn.functional as F


def rgb_to_xyz(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a RGB image to XYZ.

    .. image:: _static/img/rgb_to_xyz.png

    Args:
        image: RGB Image to be converted to XYZ with shape :math:`(*, 3, H, W)`.

    Returns:
         XYZ version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_xyz(input)  # 2x3x4x5

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    # CIE RGB to XYZ Matrix (D65 White Point)
    kernel = torch.tensor(
        [
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227],
        ],
        device=image.device,
        dtype=image.dtype if image.is_floating_point() else torch.float32,
    )

    # Apply Optimized Linear Transformation
    return _apply_linear_transformation(image, kernel)



def xyz_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a XYZ image to RGB.

    Args:
        image: XYZ Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = xyz_to_rgb(input)  # 2x3x4x5

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    # CIE XYZ to RGB Matrix (D65 White Point)
    kernel = torch.tensor(
        [
            [3.2404813432005266, -1.5371515162713185, -0.4985363261688878],
            [-0.9692549499965682, 1.8759900014898907, 0.0415559265582928],
            [0.0556466391351772, -0.2040413383665112, 1.0573110696453443],
        ],
        device=image.device,
        dtype=image.dtype if image.is_floating_point() else torch.float32,
    )

    # Apply Optimized Linear Transformation
    return _apply_linear_transformation(image, kernel)


def _apply_linear_transformation(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply a 3x3 linear color transformation with device-aware optimization.

    Args:
        image: Input image tensor with shape :math:`(*, 3, H, W)`.
        kernel: Transformation matrix with shape :math:`(3, 3)` applied along the channel
            dimension.

    Returns:
        Tensor with the same shape as ``image`` containing the transformed values.
    """
    # Handle Integer inputs by casting to float
    if image.is_floating_point():
        dtype = image.dtype
    else:
        dtype = torch.float32
    
    image_compute = image.to(dtype)
    # NOTE: kernel is already created with the correct dtype above, no need to cast again.

    # BRANCH 1: CPU (Einsum)
    if image.device.type == "cpu":
        out = torch.einsum("...chw,oc->...ohw", image_compute, kernel)

    # BRANCH 2: GPU/Accelerators (Conv2d)
    # NOTE: We assume all non-CPU devices (CUDA, MPS, XPU) provide optimized conv2d kernels.
    else:
        # Reshape for conv2d: (B*..., C, H, W)
        input_shape = image_compute.shape
        # Flatten arbitrary batch dimensions: (*, 3, H, W) -> (-1, 3, H, W)
        input_flat = image_compute.reshape(-1, 3, input_shape[-2], input_shape[-1])
        
        # Reshape kernel: (3, 3) -> (3, 3, 1, 1)
        weight = kernel.view(3, 3, 1, 1)
        
        out_flat = F.conv2d(input_flat, weight)
        
        # Unflatten back to original shape
        out = out_flat.reshape(input_shape)

    return out.contiguous()


class RgbToXyz(nn.Module):
    r"""Convert an image from RGB to XYZ.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        XYZ version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> xyz = RgbToXyz()
        >>> output = xyz(input)  # 2x3x4x5

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgb_to_xyz(image)


class XyzToRgb(nn.Module):
    r"""Converts an image from XYZ to RGB.

    Returns:
        RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = XyzToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return xyz_to_rgb(image)
