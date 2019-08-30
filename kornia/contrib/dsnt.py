r"""Implementation of "differentiable spatial to numerical" (soft-argmax)
operations, as described in the paper "Numerical Coordinate Regression with
Convolutional Neural Networks" by Nibali et al.
"""

from typing import Any, Tuple

import torch
import torch.nn.functional as F
from torch import finfo  # type: ignore

from kornia.utils.grid import create_meshgrid


def _validate_batched_image_tensor_input(tensor):
    if not torch.is_tensor(tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))
    if not len(tensor.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(tensor.shape))


def spatial_softmax_2d(
        input: torch.Tensor,
        temperature: torch.Tensor = torch.tensor(1.0),
) -> torch.Tensor:
    r"""Applies the Softmax function over features in each image channel.

    Note that this function behaves differently to `torch.nn.Softmax2d`, which
    instead applies Softmax over features at each spatial location.

    Returns a 2D probability distribution per image channel.

    Arguments:
        input (torch.Tensor): the input tensor.
        temperature (torch.Tensor): factor to apply to input, adjusting the
          "smoothness" of the output distribution. Default is 1.

    Shape:
        - Input: :math:`(B, N, H, W)`
        - Output: :math:`(B, N, H, W)`
    """
    _validate_batched_image_tensor_input(input)

    batch_size, channels, height, width = input.shape
    x: torch.Tensor = input.view(batch_size, channels, -1)

    x_soft: torch.Tensor = F.softmax(x * temperature, dim=-1)

    return x_soft.view(batch_size, channels, height, width)


def spatial_softargmax_2d(
        input: torch.Tensor,
        normalized_coordinates: bool = True,
) -> torch.Tensor:
    r"""Computes the 2D soft-argmax of a given input heatmap.

    The input heatmap is assumed to represent a valid spatial probability
    distribution, which can be achieved using
    :class:`~kornia.contrib.dsnt.spatial_softmax_2d`.

    Returns the index of the maximum 2D coordinates of the given heatmap.
    The output order of the coordinates is (x, y).

    Arguments:
        input (torch.Tensor): the input tensor.
        normalized_coordinates (bool): whether to return the
          coordinates normalized in the range of [-1, 1]. Otherwise,
          it will return the coordinates in the range of the input shape.
          Default is True.

    Shape:
        - Input: :math:`(B, N, H, W)`
        - Output: :math:`(B, N, 2)`

    Examples:
        >>> heatmaps = torch.tensor([[[
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 1., 0.]]]])
        >>> coords = spatial_softargmax_2d(heatmaps, False)
        tensor([[[1.0000, 2.0000]]])
    """
    _validate_batched_image_tensor_input(input)

    batch_size, channels, height, width = input.shape

    # Create coordinates grid.
    grid: torch.Tensor = create_meshgrid(
        height, width, normalized_coordinates)
    grid = grid.to(device=input.device, dtype=input.dtype)

    pos_x: torch.Tensor = grid[..., 0].reshape(-1)
    pos_y: torch.Tensor = grid[..., 1].reshape(-1)

    input_flat: torch.Tensor = input.view(batch_size, channels, -1)

    # Compute the expectation of the coordinates.
    expected_y: torch.Tensor = torch.sum(pos_y * input_flat, -1, keepdim=True)
    expected_x: torch.Tensor = torch.sum(pos_x * input_flat, -1, keepdim=True)

    output: torch.Tensor = torch.cat([expected_x, expected_y], -1)

    return output.view(batch_size, channels, 2)  # BxNx2


def _safe_zero_division(
        numerator: torch.Tensor,
        denominator: torch.Tensor,
) -> torch.Tensor:
    eps: float = finfo(numerator.dtype).tiny
    return numerator / torch.clamp(denominator, min=eps)


def render_gaussian_2d(
        mean: torch.Tensor,
        std: torch.Tensor,
        size: Tuple[int, int],
        normalized_coordinates: bool = True,
):
    r"""Renders the PDF of a 2D Gaussian distribution.

    Arguments:
        mean (torch.Tensor): the mean location of the Gaussian to render,
          :math:`(\mu_x, \mu_y)`.
        std (torch.Tensor): the standard deviation of the Gaussian to render,
          :math:`(\sigma_x, \sigma_y)`.
        size (list): the (height, width) of the output image.
        normalized_coordinates: whether `mean` and `std` are assumed to use
          coordinates normalized in the range of [-1, 1]. Otherwise,
          coordinates are assumed to be in the range of the output shape.
          Default is True.

    Shape:
        - `mean`: :math:`(*, 2)`
        - `std`: :math:`(*, 2)`. Should be able to be broadcast with `mean`.
        - Output: :math:`(*, H, W)`
    """
    mean_dtype: Any = mean.dtype
    if not mean_dtype.is_floating_point:
        raise TypeError("Expected `mean` to have floating point values. Got {}"
                        .format(mean.dtype))
    if not (std.dtype == mean.dtype and std.device == mean.device):
        raise TypeError("Expected inputs to have the same dtype and device")
    height, width = size

    # Create coordinates grid.
    grid: torch.Tensor = create_meshgrid(height, width, normalized_coordinates)
    grid = grid.to(device=mean.device, dtype=mean.dtype)
    pos_x: torch.Tensor = grid[..., 0].view(height, width)
    pos_y: torch.Tensor = grid[..., 1].view(height, width)

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
