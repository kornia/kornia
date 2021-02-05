from typing import Tuple

import torch
import torch.nn as nn

from kornia.utils import _extract_device_dtype

__all__ = [
    "get_tps_parameters",
    "warp_points_tps",
    "warp_img_tensor_tps"
]

# utilities for computing thin plate spline transforms


def _pair_square_euclidean(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Compute the pairwise squared euclidean distance matrices (B, N, M) between two tensors
        with shapes (B, N, C) and (B, M, C)."""
    # ||t1-t2||^2 = (t1-t2)^T(t1-t2) = t1^T*t1 + t2^T*t2 - 2*t1^T*t2
    t1_sq: torch.Tensor = tensor1.mul(tensor1).sum(dim=-1, keepdim=True)
    t2_sq: torch.Tensor = tensor2.mul(tensor2).sum(dim=-1, keepdim=True).transpose(1, 2)
    t1_t2: torch.Tensor = tensor1.matmul(tensor2.transpose(1, 2))
    square_dist: torch.Tensor = -2 * t1_t2 + t1_sq + t2_sq
    square_dist: torch.Tensor = square_dist.clamp(min=0)  # handle possible numerical errors
    return square_dist


def _kernel_distance(squared_distances: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute the TPS kernel distance function: r^2*log(r), where r is the euclidean distance.
    Since log(r) = 1/2*log(r^2), this function takes the squared distance matrix and calculates
    0.5 * r^2 * log(r^2)."""
    # r^2 * log(r) = 1/2 * r^2 * log(r^2)
    return 0.5 * squared_distances * squared_distances.add(eps).log()


def get_tps_parameters(points_src: torch.Tensor, points_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the TPS transform parameters that warp source points to target points.

    The input to this function is a tensor of (x, y) source points (B, N, 2) and a corresponding tensor of target
    (x, y) points (B, N, 2).

    The return value is a tuple containing two weight tensors. The first weight tensor contains the kernel weights
    (B, N, 2); the second tensor contains the affine weights (B, 3, 2). In both cases, the last dimension contains
    the weights for the x-transform and y-transform respectively.

    Args:
        points_src (torch.Tensor): batch of source points (B, N, 2) as (x, y) coordinate vectors
        points_target (torch.Tensor): batch of target points (B, N, 2) as (x, y) coordinate vectors

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: the (B, N, 2) kernel weights and (B, 3, 2) affine weights
    """

    device, dtype = _extract_device_dtype([points_src, points_target])
    batch_size, num_points = points_src.shape[:2]

    # set up and solve linear system
    # [K   P] [w] = [dst]
    # [P^T 0] [a]   [ 0 ]
    pair_distance: torch.Tensor = _pair_square_euclidean(points_src, points_target)
    k_matrix: torch.Tensor = _kernel_distance(pair_distance)

    zero_mat: torch.Tensor = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
    one_mat: torch.Tensor = torch.ones(batch_size, num_points, 1, device=device, dtype=dtype)
    dest_with_zeros: torch.Tensor = torch.cat((points_target, zero_mat[:, :, :2]), 1)
    p_matrix: torch.Tensor = torch.cat((one_mat, points_src), -1)
    p_matrix_t: torch.Tensor = torch.cat((p_matrix, zero_mat), 1).transpose(1, 2)
    l_matrix: torch.Tensor = torch.cat((k_matrix, p_matrix), -1)
    l_matrix = torch.cat((l_matrix, p_matrix_t), 1)

    weights, _ = torch.solve(dest_with_zeros, l_matrix)
    kernel_weights: torch.Tensor = weights[:, :-3]
    affine_weights: torch.Tensor = weights[:, -3:]

    return (kernel_weights, affine_weights)


def warp_points_tps(points_src: torch.Tensor, kernel_centers: torch.Tensor,
                    kernel_weights: torch.Tensor, affine_weights: torch.Tensor) -> torch.Tensor:
    r"""Warp a tensor of coordinate points using the thin plate spline defined by kernel points, kernel weights, and
    affine weights.

    The source points should be a (B, N, 2) tensor of (x, y) coordinates. The kernel centers are a (B, K, 2) tensor
    of (x, y) coordinates. The kernel weights are a (B, K, 2) tensor, and the affine weights are a (B, 3, 2) tensor.
    For the weight tensors, tensor[..., 0] contains the weights for the x-transform and tensor[..., 1] the weights
    for the y-transform.

    Returns a (B, N, 2) tensor containing the warped source points.

    Args:
        points_src (torch.Tensor): tensor of source points (B, N, 2)
        kernel_centers (torch.Tensor): tensor of kernel center points (B, K, 2)
        kernel_weights (torch.Tensor): tensor of kernl weights (B, K, 2)
        affine_weights (torch.Tensor): tensor of affine weights (B, 3, 2)

    Returns:
        torch.Tensor: The warped source points, from applying the TPS transform.
    """
    # f_{x|y}(v) = a_0 + [a_x a_y].v + \sum_i w_i * U(||v-u_i||)
    pair_distance: torch.Tensor = _pair_square_euclidean(points_src, kernel_centers)
    k_matrix: torch.Tensor = _kernel_distance(pair_distance)

    # broadcast the kernel distance matrix against the x and y weights to compute the x and y
    # transforms simultaneously
    warped: torch.Tensor = (
        k_matrix[..., None].mul(kernel_weights[:, None]).sum(-2) +
        points_src[..., None].mul(affine_weights[:, None, 1:]).sum(-2) +
        affine_weights[:, None, 0]
    )

    return warped


def warp_img_tensor_tps(tensor: torch.Tensor, kernel_centers: torch.Tensor, kernel_weights: torch.Tensor,
                        affine_weights: torch.Tensor, align_corners: bool = False) -> torch.Tensor:
    r"""Warp an image tensor according to the thin plate spline transform defined by kernel centers,
    kernel weights, and affine weights.

    The transform is applied to each pixel coordinate in the output image to obtain a point in the input
    image for interpolation of the output pixel. So the TPS parameters should correspond to a warp from
    output space to input space.

    The input tensor is a (B, C, H, W) tensor. The kernel centers, kernel weight and affine weights are the
    same as in ``warp_points_tps``.

    Returns a warped image tensor with the same shape as the input.

    Args:
        tensor (torch.Tensor): input image tensor (B, C, H, W)
        kernel_centers (torch.Tensor): kernel center points (B, K, 2)
        kernel_weights (torch.Tensor): tensor of kernl weights (B, K, 2)
        affine_weights (torch.Tensor): tensor of affine weights (B, 3, 2)

    Returns:
        torch.Tensor: warped image tensor (B, C, H, W)
    """
    device, dtype = _extract_device_dtype([tensor, kernel_centers, kernel_weights, affine_weights])
    batch_size, _, h, w = tensor.shape
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device=device, dtype=dtype),
                            torch.linspace(-1, 1, w, device=device, dtype=dtype))
    coords: torch.Tensor = torch.stack([xs, ys], -1).view(-1, 2)
    coords = torch.stack([coords] * batch_size, 0)  # expand to batch dimension
    warped: torch.Tensor = warp_points_tps(coords, kernel_centers, kernel_weights, affine_weights)
    warped = warped.view(-1, h, w, 2)
    warped_image: torch.Tensor = nn.functional.grid_sample(tensor, warped, align_corners=align_corners)

    return warped_image
