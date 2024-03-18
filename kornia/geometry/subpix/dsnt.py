r"""Implementation of "differentiable spatial to numerical" (soft-argmax) operations, as described in the paper
"Numerical Coordinate Regression with Convolutional Neural Networks" by Nibali et al."""

from __future__ import annotations

import torch

from kornia.core import Tensor, concatenate, softmax
from kornia.core.check import KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.utils.grid import create_meshgrid


def _validate_batched_image_tensor_input(tensor: Tensor) -> None:
    KORNIA_CHECK_IS_TENSOR(tensor)
    KORNIA_CHECK_SHAPE(tensor, ["B", "C", "H", "W"])


def spatial_softmax2d(input: Tensor, temperature: Tensor = torch.tensor(1.0)) -> Tensor:
    r"""Apply the Softmax function over features in each image channel.

    Note that this function behaves differently to :py:class:`torch.nn.Softmax2d`, which
    instead applies Softmax over features at each spatial location.

    Args:
        input: the input tensor with shape :math:`(B, N, H, W)`.
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
    temperature = temperature.to(device=input.device, dtype=input.dtype)
    x = input.view(batch_size, channels, -1)

    x_soft = softmax(x * temperature, dim=-1)

    return x_soft.view(batch_size, channels, height, width)


def spatial_expectation2d(input: Tensor, normalized_coordinates: bool = True) -> Tensor:
    r"""Compute the expectation of coordinate values using spatial probabilities.

    The input heatmap is assumed to represent a valid spatial probability distribution,
    which can be achieved using :func:`~kornia.geometry.subpixel.spatial_softmax2d`.

    Args:
        input: the input tensor representing dense spatial probabilities with shape :math:`(B, N, H, W)`.
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

    output = concatenate([expected_x, expected_y], -1)

    return output.view(batch_size, channels, 2)  # BxNx2


def _safe_zero_division(numerator: Tensor, denominator: Tensor, eps: float = 1e-32) -> Tensor:
    return numerator / torch.clamp(denominator, min=eps)


def render_gaussian2d(mean: Tensor, std: Tensor, size: tuple[int, int], normalized_coordinates: bool = True) -> Tensor:
    r"""Render the PDF of a 2D Gaussian distribution.

    Args:
        mean: the mean location of the Gaussian to render, :math:`(\mu_x, \mu_y)`. Shape: :math:`(*, 2)`.
        std: the standard deviation of the Gaussian to render, :math:`(\sigma_x, \sigma_y)`.
          Shape :math:`(*, 2)`. Should be able to be broadcast with `mean`.
        size: the (height, width) of the output image.
        normalized_coordinates: whether ``mean`` and ``std`` are assumed to use coordinates normalized
          in the range of :math:`[-1, 1]`. Otherwise, coordinates are assumed to be in the range of the output shape.

    Returns:
        tensor including rendered points with shape :math:`(*, H, W)`.
    """
    if not (std.dtype == mean.dtype and std.device == mean.device):
        raise TypeError("Expected inputs to have the same dtype and device")

    height, width = size

    # Create coordinates grid.
    grid = create_meshgrid(height, width, normalized_coordinates, mean.device)
    grid = grid.to(mean.dtype)
    pos_x = grid[..., 0].view(height, width)
    pos_y = grid[..., 1].view(height, width)

    # Gaussian PDF = exp(-(x - \mu)^2 / (2 \sigma^2))
    #              = exp(dists * ks),
    #                where dists = (x - \mu)^2 and ks = -1 / (2 \sigma^2)

    # dists <- (x - \mu)^2
    dist_x = (pos_x - mean[..., 0, None, None]) ** 2
    dist_y = (pos_y - mean[..., 1, None, None]) ** 2

    # ks <- -1 / (2 \sigma^2)
    k_x = -0.5 * torch.reciprocal(std[..., 0, None, None])
    k_y = -0.5 * torch.reciprocal(std[..., 1, None, None])

    # Assemble the 2D Gaussian.
    exps_x = torch.exp(dist_x * k_x)
    exps_y = torch.exp(dist_y * k_y)
    gauss = exps_x * exps_y

    # Rescale so that values sum to one.
    val_sum = gauss.sum(-2, keepdim=True).sum(-1, keepdim=True)
    gauss = _safe_zero_division(gauss, val_sum)

    return gauss
