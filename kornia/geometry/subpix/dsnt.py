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

r"""Implementation of "differentiable spatial to numerical" (soft-argmax) operations.

As described in the paper "Numerical Coordinate Regression with Convolutional Neural Networks" by Nibali et al.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from kornia.core.check import KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.geometry.grid import create_meshgrid


def _validate_batched_image_tensor_input(tensor: torch.Tensor) -> None:
    KORNIA_CHECK_IS_TENSOR(tensor)
    KORNIA_CHECK_SHAPE(tensor, ["B", "C", "H", "W"])


def spatial_softmax2d(input: torch.Tensor, temperature: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""Apply the Softmax function over features in each image channel.

    Note that this function behaves differently to :py:class:`torch.nn.Softmax2d`, which
    instead applies Softmax over features at each spatial location.

    Args:
        input: the input torch.tensor with shape :math:`(B, N, H, W)`.
        temperature: factor to apply to input, adjusting the "smoothness" of the output distribution.

    Returns:
       a 2D probability distribution per image channel with shape :math:`(B, N, H, W)`.

    Examples:
        >>> heatmaps = torch.tensor([[[
        ... [0., 0., 0.],
        ... [0., 0., 0.],
        ... [0., 1., 2.]]]])
        >>> spatial_softmax2d(heatmaps)
        tensor([[[[0.0585, 0.0585, 0.0585],
                  [0.0585, 0.0585, 0.0585],
                  [0.0585, 0.1589, 0.4319]]]])

    """
    _validate_batched_image_tensor_input(input)

    batch_size, channels, height, width = input.shape
    if temperature is None:
        temperature = torch.tensor(1.0)
    temperature = temperature.to(device=input.device, dtype=input.dtype)
    x = input.view(batch_size, channels, -1)

    x_soft = F.softmax(x * temperature, dim=-1)

    return x_soft.view(batch_size, channels, height, width)


def spatial_expectation2d(input: torch.Tensor, normalized_coordinates: bool = True) -> torch.Tensor:
    r"""Compute the expectation of coordinate values using spatial probabilities.

    The input heatmap is assumed to represent a valid spatial probability distribution,
    which can be achieved using :func:`~kornia.geometry.subpixel.spatial_softmax2d`.

    Args:
        input: the input torch.tensor representing dense spatial probabilities with shape :math:`(B, N, H, W)`.
        normalized_coordinates: whether to return the coordinates normalized in the range
            of :math:`[-1, 1]`. Otherwise, it will return the coordinates in the range of the input shape.

    Returns:
       expected value of the 2D coordinates with shape :math:`(B, N, 2)`. Output order of the coordinates is (x, y).

    Examples:
        >>> heatmaps = torch.tensor([[[
        ... [0., 0., 0.],
        ... [0., 0., 0.],
        ... [0., 1., 0.]]]])
        >>> spatial_expectation2d(heatmaps, False)
        tensor([[[1., 2.]]])

    """
    _validate_batched_image_tensor_input(input)

    batch_size, channels, height, width = input.shape

    # Create coordinates grid.
    grid = create_meshgrid(height, width, normalized_coordinates, input.device)
    grid = grid.to(input.dtype)

    pos_x = grid[..., 0].reshape(-1)
    pos_y = grid[..., 1].reshape(-1)

    input_flat = input.view(batch_size, channels, -1)

    # Compute the expectation of the coordinates.
    expected_y = torch.sum(pos_y * input_flat, -1, keepdim=True)
    expected_x = torch.sum(pos_x * input_flat, -1, keepdim=True)

    output = torch.cat([expected_x, expected_y], -1)

    return output.view(batch_size, channels, 2)  # BxNx2


def render_gaussian2d(
    mean: torch.Tensor, std: torch.Tensor, size: tuple[int, int], normalized_coordinates: bool = True
) -> torch.Tensor:
    r"""Render the PDF of a 2D Gaussian distribution.

    Args:
        mean: the mean location of the Gaussian to render, :math:`(\mu_x, \mu_y)`. Shape: :math:`(*, 2)`.
        std: the standard deviation of the Gaussian to render, :math:`(\sigma_x, \sigma_y)`.
            Shape :math:`(*, 2)`. Should be able to be broadcast with `mean`.
        size: the (height, width) of the output image.
        normalized_coordinates: whether ``mean`` and ``std`` are assumed to use coordinates normalized
            in the range of :math:`[-1, 1]`. Otherwise, coordinates are assumed to be in the range of the output shape.

    Returns:
        torch.tensor including rendered points with shape :math:`(*, H, W)`.

    """
    if not (std.dtype == mean.dtype and std.device == mean.device):
        raise TypeError("Expected inputs to have the same dtype and device")

    height, width = size
    dtype = mean.dtype
    device = mean.device

    # Create coordinates vectors.
    if normalized_coordinates:
        xs = torch.linspace(-1, 1, width, device=device, dtype=dtype)
        ys = torch.linspace(-1, 1, height, device=device, dtype=dtype)
    else:
        xs = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
        ys = torch.linspace(0, height - 1, height, device=device, dtype=dtype)

    mu_x = mean[..., 0].unsqueeze(-1)
    mu_y = mean[..., 1].unsqueeze(-1)
    sigma_x = std[..., 0].unsqueeze(-1)
    sigma_y = std[..., 1].unsqueeze(-1)

    # Gaussian PDF = exp(-(x - \mu)^2 / (2 \sigma^2))
    #              = exp(dists * ks),
    #                torch.where dists = (x - \mu)^2 and ks = -1 / (2 \sigma^2)

    # dists <- (x - \mu)^2
    dist_x_sq = (xs - mu_x) ** 2
    dist_y_sq = (ys - mu_y) ** 2

    # ks <- -1 / (2 \sigma^2)
    k_x = -0.5 * torch.reciprocal(sigma_x**2)
    k_y = -0.5 * torch.reciprocal(sigma_y**2)

    # Assemble the 2D Gaussian.
    gauss_x = torch.exp(dist_x_sq * k_x)
    gauss_y = torch.exp(dist_y_sq * k_y)

    # Rescale so that values sum to one.
    gauss_x = gauss_x / (gauss_x.sum(dim=-1, keepdim=True) + 1e-8)
    gauss_y = gauss_y / (gauss_y.sum(dim=-1, keepdim=True) + 1e-8)

    return gauss_y.unsqueeze(-1) * gauss_x.unsqueeze(-2)
