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

import math

import torch
from torch import nn

from kornia.core import Tensor
from kornia.filters import filter2d, filter3d
from kornia.utils import create_meshgrid, create_meshgrid3d
from kornia.filters import filter2d
from kornia.geometry.grid import create_meshgrid


def distance_transform(image: Tensor, kernel_size: int = 3, h: float = 0.35) -> Tensor:
    r"""Approximates the Euclidean distance transform of images/volumes using cascaded convolution operations.

    The value at each pixel/voxel represents the distance to the nearest non-zero element.
    It uses the method described in :cite:`pham2021dtlayer`.
    The transformation is applied independently across the channel dimension.

    Args:
        image: Image or volume with shape :math:`(B,C,H,W)` or :math:`(B,C,D,H,W)`.
        kernel_size: size of the convolution kernel. Must be an odd number.
        h: value that influence the approximation of the min function.

    Returns:
        tensor with the same shape as input.

    Example:
        >>> # 2D example:
        >>> tensor = torch.zeros(1, 1, 5, 5)
        >>> tensor[:,:, 1, 2] = 1
        >>> dt = distance_transform(tensor)
        >>> # 3D example:
        >>> volume = torch.zeros(1, 1, 5, 5, 5)
        >>> volume[:, :, 2, 2, 2] = 1
        >>> dt = distance_transform(volume)

    """
    if not isinstance(image, Tensor):
        raise TypeError(f"image type is not a torch.Tensor. Got {type(image)}")

    if not image.is_floating_point():
        raise TypeError("image must be a floating point tensor")

    dim = len(image.shape)
    if dim not in (4, 5):
        raise ValueError(f"Invalid image shape, we expect 4D or 5D. Got: {image.shape}")

    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("kernel_size must be an odd integer >= 3")

    device = image.device
    dtype = image.dtype
    k_half = kernel_size // 2

    if dim == 4:
        n_iters = math.ceil(max(image.shape[2], image.shape[3]) / k_half)
        grid = create_meshgrid(kernel_size, kernel_size, False, device, dtype)
        grid = grid - k_half

        dist = torch.hypot(grid[0, ..., 0], grid[0, ..., 1])
        kernel = torch.exp(-dist / h).unsqueeze(0)

        def conv_op(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
            return filter2d(x, k, border_type="replicate")
    else:
        n_iters = math.ceil(max(image.shape[2:]) / k_half)
        grid = create_meshgrid3d(kernel_size, kernel_size, kernel_size, False, device, dtype)
        grid = grid - k_half
        dist = torch.norm(grid[0], p=2, dim=-1)
        kernel = torch.exp(-dist / h).unsqueeze(0)

        def conv_op(x: Tensor, k: Tensor) -> Tensor:
            return filter3d(x, k, border_type="replicate")

    out = torch.zeros_like(image)
    boundary = image.clone()
    signal_ones = torch.ones_like(boundary)

    for i in range(n_iters):
        cdt = conv_op(boundary, kernel)
        cdt = -h * torch.log(cdt)

        cdt = torch.nan_to_num(cdt, nan=0.0, posinf=0.0, neginf=0.0)

        mask = cdt > 0
        if not mask.any():
            break

        offset: int = i * k_half
        out = out + (offset + cdt) * mask.to(dtype=out.dtype)
        boundary = torch.where(mask, signal_ones, boundary)

    return out


class DistanceTransform(nn.Module):
    r"""Module that approximates the Euclidean distance transform of images/volumes using convolutions.

    Args:
        kernel_size: size of the convolution kernel.
        h: value that influence the approximation of the min function.

    """

    def __init__(self, kernel_size: int = 3, h: float = 0.35) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.h = h

    def forward(self, image: Tensor) -> Tensor:
        # Reshape multi-channel inputs to batch dimension to ensure independent processing
        if image.shape[1] > 1:
            # Dynamically determine spatial dimensions (works for H,W or D,H,W)
            spatial_dims = image.shape[2:]
            image_in = image.view(-1, 1, *spatial_dims)
        else:
            image_in = image

        return distance_transform(image_in, self.kernel_size, self.h).view_as(image)
