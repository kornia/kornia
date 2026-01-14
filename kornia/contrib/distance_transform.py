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

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE, KORNIA_CHECK_TYPE
from kornia.filters import filter2d, filter3d
from kornia.geometry.grid import create_meshgrid, create_meshgrid3d


def _distance_transform_2d_impl(image: torch.Tensor, kernel_size: int, h: float) -> torch.Tensor:
    device = image.device
    dtype = image.dtype
    k_half = kernel_size // 2

    n_iters = math.ceil(max(image.shape[2], image.shape[3]) / k_half)
    grid = create_meshgrid(kernel_size, kernel_size, False, device, dtype)
    grid = grid - k_half

    dist = torch.hypot(grid[0, ..., 0], grid[0, ..., 1])
    kernel = torch.exp(-dist / h).unsqueeze(0)

    out = torch.zeros_like(image)
    boundary = image.clone()
    signal_ones = torch.ones_like(boundary)

    for i in range(n_iters):
        cdt = filter2d(boundary, kernel, border_type="replicate")
        cdt = -h * torch.log(cdt)
        cdt = torch.nan_to_num(cdt, nan=0.0, posinf=0.0, neginf=0.0)

        mask = cdt > 0
        if not mask.any():
            break

        offset: int = i * k_half
        out = out + (offset + cdt) * mask.to(dtype=out.dtype)
        boundary = torch.where(mask, signal_ones, boundary)

    return out


def _distance_transform_3d_impl(image: torch.Tensor, kernel_size: int, h: float) -> torch.Tensor:
    device = image.device
    dtype = image.dtype
    k_half = kernel_size // 2

    n_iters = math.ceil(max(image.shape[2:]) / k_half)
    grid = create_meshgrid3d(kernel_size, kernel_size, kernel_size, False, device, dtype)
    grid = grid - k_half
    dist = torch.norm(grid[0], p=2, dim=-1)
    kernel = torch.exp(-dist / h).unsqueeze(0)

    out = torch.zeros_like(image)
    boundary = image.clone()
    signal_ones = torch.ones_like(boundary)

    for i in range(n_iters):
        cdt = filter3d(boundary, kernel, border_type="replicate")
        cdt = -h * torch.log(cdt)
        cdt = torch.nan_to_num(cdt, nan=0.0, posinf=0.0, neginf=0.0)

        mask = cdt > 0
        if not mask.any():
            break

        offset: int = i * k_half
        out = out + (offset + cdt) * mask.to(dtype=out.dtype)
        boundary = torch.where(mask, signal_ones, boundary)

    return out


def distance_transform(image: torch.Tensor, kernel_size: int = 3, h: float = 0.35) -> torch.Tensor:
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
    # Validation using KORNIA_CHECK API
    KORNIA_CHECK_IS_TENSOR(image)
    KORNIA_CHECK(image.is_floating_point(), "image must be a floating point tensor")

    KORNIA_CHECK(image.ndim in (4, 5), f"Invalid image shape, we expect BxCxHxW or BxCxDxHxW. Got: {image.shape}")

    if image.ndim == 4:
        KORNIA_CHECK_SHAPE(image, ["B", "C", "H", "W"])
    else:
        KORNIA_CHECK_SHAPE(image, ["B", "C", "D", "H", "W"])

    # dtype / param checks
    KORNIA_CHECK_TYPE(kernel_size, int, "kernel_size must be an int")
    KORNIA_CHECK(kernel_size % 2 != 0 and kernel_size >= 3, "kernel_size must be an odd integer >= 3")
    KORNIA_CHECK(h > 0, f"h must be a positive float, got {h}")

    if image.ndim == 4:
        return _distance_transform_2d_impl(image, kernel_size, h)

    return _distance_transform_3d_impl(image, kernel_size, h)


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

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Reshape multi-channel inputs to batch dimension to ensure independent processing
        if image.shape[1] > 1:
            # Dynamically determine spatial dimensions (works for H,W or D,H,W)
            spatial_dims = image.shape[2:]
            # Use reshape to handle non-contiguous tensors safely
            image_in = image.reshape(-1, 1, *spatial_dims)
        else:
            image_in = image

        return distance_transform(image_in, self.kernel_size, self.h).view_as(image)
