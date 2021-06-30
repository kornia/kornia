from typing import Tuple

import torch
import torch.nn as nn

from kornia.utils import create_meshgrid

__all__ = ["get_tps_transform", "warp_points_tps", "warp_image_tps"]

# utilities for computing thin plate spline transforms


def _pair_square_euclidean(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    r"""Compute the pairwise squared euclidean distance matrices :math:`(B, N, M)` between two tensors
    with shapes (B, N, C) and (B, M, C)."""
    # ||t1-t2||^2 = (t1-t2)^T(t1-t2) = t1^T*t1 + t2^T*t2 - 2*t1^T*t2
    t1_sq: torch.Tensor = tensor1.mul(tensor1).sum(dim=-1, keepdim=True)
    t2_sq: torch.Tensor = tensor2.mul(tensor2).sum(dim=-1, keepdim=True).transpose(1, 2)
    t1_t2: torch.Tensor = tensor1.matmul(tensor2.transpose(1, 2))
    square_dist: torch.Tensor = -2 * t1_t2 + t1_sq + t2_sq
    square_dist = square_dist.clamp(min=0)  # handle possible numerical errors
    return square_dist


def _kernel_distance(squared_distances: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Compute the TPS kernel distance function: :math:`r^2 log(r)`, where `r` is the euclidean distance.
    Since :math:`\log(r) = 1/2 \log(r^2)`, this function takes the squared distance matrix and calculates
    :math:`0.5 r^2 log(r^2)`."""
    # r^2 * log(r) = 1/2 * r^2 * log(r^2)
    return 0.5 * squared_distances * squared_distances.add(eps).log()


def get_tps_transform(points_src: torch.Tensor, points_dst: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the TPS transform parameters that warp source points to target points.

    The input to this function is a tensor of :math:`(x, y)` source points :math:`(B, N, 2)` and a corresponding
    tensor of target :math:`(x, y)` points :math:`(B, N, 2)`.

    Args:
        points_src: batch of source points :math:`(B, N, 2)` as :math:`(x, y)` coordinate vectors.
        points_dst: batch of target points :math:`(B, N, 2)` as :math:`(x, y)` coordinate vectors.

    Returns:
        :math:`(B, N, 2)` tensor of kernel weights and :math:`(B, 3, 2)`
            tensor of affine weights. The last dimension contains the x-transform and y-transform weights
            as seperate columns.

    Example:
        >>> points_src = torch.rand(1, 5, 2)
        >>> points_dst = torch.rand(1, 5, 2)
        >>> kernel_weights, affine_weights = get_tps_transform(points_src, points_dst)

    .. note::
        This function is often used in conjuntion with :func:`warp_points_tps`, :func:`warp_image_tps`.
    """
    if not isinstance(points_src, torch.Tensor):
        raise TypeError(f"Input points_src is not torch.Tensor. Got {type(points_src)}")

    if not isinstance(points_dst, torch.Tensor):
        raise TypeError(f"Input points_dst is not torch.Tensor. Got {type(points_dst)}")

    if not len(points_src.shape) == 3:
        raise ValueError(f"Invalid shape for points_src, expected BxNx2. Got {points_src.shape}")

    if not len(points_dst.shape) == 3:
        raise ValueError(f"Invalid shape for points_dst, expected BxNx2. Got {points_dst.shape}")

    device, dtype = points_src.device, points_src.dtype
    batch_size, num_points = points_src.shape[:2]

    # set up and solve linear system
    # [K   P] [w] = [dst]
    # [P^T 0] [a]   [ 0 ]
    pair_distance: torch.Tensor = _pair_square_euclidean(points_src, points_dst)
    k_matrix: torch.Tensor = _kernel_distance(pair_distance)

    zero_mat: torch.Tensor = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
    one_mat: torch.Tensor = torch.ones(batch_size, num_points, 1, device=device, dtype=dtype)
    dest_with_zeros: torch.Tensor = torch.cat((points_dst, zero_mat[:, :, :2]), 1)
    p_matrix: torch.Tensor = torch.cat((one_mat, points_src), -1)
    p_matrix_t: torch.Tensor = torch.cat((p_matrix, zero_mat), 1).transpose(1, 2)
    l_matrix: torch.Tensor = torch.cat((k_matrix, p_matrix), -1)
    l_matrix = torch.cat((l_matrix, p_matrix_t), 1)

    weights, _ = torch.solve(dest_with_zeros, l_matrix)
    kernel_weights: torch.Tensor = weights[:, :-3]
    affine_weights: torch.Tensor = weights[:, -3:]

    return (kernel_weights, affine_weights)


def warp_points_tps(
    points_src: torch.Tensor, kernel_centers: torch.Tensor, kernel_weights: torch.Tensor, affine_weights: torch.Tensor
) -> torch.Tensor:
    r"""Warp a tensor of coordinate points using the thin plate spline defined by kernel points, kernel weights,
    and affine weights.

    The source points should be a :math:`(B, N, 2)` tensor of :math:`(x, y)` coordinates. The kernel centers are
    a :math:`(B, K, 2)` tensor of :math:`(x, y)` coordinates. The kernel weights are a :math:`(B, K, 2)` tensor,
    and the affine weights are a :math:`(B, 3, 2)` tensor.  For the weight tensors, tensor[..., 0] contains the
    weights for the x-transform and tensor[..., 1] the weights for the y-transform.

    Args:
        points_src: tensor of source points :math:`(B, N, 2)`.
        kernel_centers: tensor of kernel center points :math:`(B, K, 2)`.
        kernel_weights: tensor of kernl weights :math:`(B, K, 2)`.
        affine_weights: tensor of affine weights :math:`(B, 3, 2)`.

    Returns:
        The :math:`(B, N, 2)` tensor of warped source points, from applying the TPS transform.

    Example:
        >>> points_src = torch.rand(1, 5, 2)
        >>> points_dst = torch.rand(1, 5, 2)
        >>> kernel_weights, affine_weights = get_tps_transform(points_src, points_dst)
        >>> warped = warp_points_tps(points_src, points_dst, kernel_weights, affine_weights)
        >>> warped_correct = torch.allclose(warped, points_dst)

    .. note::
        This function is often used in conjuntion with :func:`get_tps_transform`.
    """
    if not isinstance(points_src, torch.Tensor):
        raise TypeError(f"Input points_src is not torch.Tensor. Got {type(points_src)}")

    if not isinstance(kernel_centers, torch.Tensor):
        raise TypeError(f"Input kernel_centers is not torch.Tensor. Got {type(kernel_centers)}")

    if not isinstance(kernel_weights, torch.Tensor):
        raise TypeError(f"Input kernel_weights is not torch.Tensor. Got {type(kernel_weights)}")

    if not isinstance(affine_weights, torch.Tensor):
        raise TypeError(f"Input affine_weights is not torch.Tensor. Got {type(affine_weights)}")

    if not len(points_src.shape) == 3:
        raise ValueError(f"Invalid shape for points_src, expected BxNx2. Got {points_src.shape}")

    if not len(kernel_centers.shape) == 3:
        raise ValueError(f"Invalid shape for kernel_centers, expected BxNx2. Got {kernel_centers.shape}")

    if not len(kernel_weights.shape) == 3:
        raise ValueError(f"Invalid shape for kernel_weights, expected BxNx2. Got {kernel_weights.shape}")

    if not len(affine_weights.shape) == 3:
        raise ValueError(f"Invalid shape for affine_weights, expected BxNx2. Got {affine_weights.shape}")

    # f_{x|y}(v) = a_0 + [a_x a_y].v + \sum_i w_i * U(||v-u_i||)
    pair_distance: torch.Tensor = _pair_square_euclidean(points_src, kernel_centers)
    k_matrix: torch.Tensor = _kernel_distance(pair_distance)

    # broadcast the kernel distance matrix against the x and y weights to compute the x and y
    # transforms simultaneously
    k_mul_kernel = k_matrix[..., None].mul(kernel_weights[:, None]).sum(-2)
    points_mul_affine = points_src[..., None].mul(affine_weights[:, None, 1:]).sum(-2)
    warped: torch.Tensor = k_mul_kernel + points_mul_affine + affine_weights[:, None, 0]

    return warped


def warp_image_tps(
    image: torch.Tensor,
    kernel_centers: torch.Tensor,
    kernel_weights: torch.Tensor,
    affine_weights: torch.Tensor,
    align_corners: bool = False,
) -> torch.Tensor:
    r"""Warp an image tensor according to the thin plate spline transform defined by kernel centers,
    kernel weights, and affine weights.

    .. image:: _static/img/warp_image_tps.png

    The transform is applied to each pixel coordinate in the output image to obtain a point in the input
    image for interpolation of the output pixel. So the TPS parameters should correspond to a warp from
    output space to input space.

    The input `image` is a :math:`(B, C, H, W)` tensor. The kernel centers, kernel weight and affine weights
    are the same as in `warp_points_tps`.

    Args:
        image: input image tensor :math:`(B, C, H, W)`.
        kernel_centers: kernel center points :math:`(B, K, 2)`.
        kernel_weights: tensor of kernl weights :math:`(B, K, 2)`.
        affine_weights: tensor of affine weights :math:`(B, 3, 2)`.
        align_corners: interpolation flag used by `grid_sample`.

    Returns:
        warped image tensor :math:`(B, C, H, W)`.

    Example:
        >>> points_src = torch.rand(1, 5, 2)
        >>> points_dst = torch.rand(1, 5, 2)
        >>> image = torch.rand(1, 3, 32, 32)
        >>> # note that we are getting the reverse transform: dst -> src
        >>> kernel_weights, affine_weights = get_tps_transform(points_dst, points_src)
        >>> warped_image = warp_image_tps(image, points_src, kernel_weights, affine_weights)

    .. note::
        This function is often used in conjuntion with :func:`get_tps_transform`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input image is not torch.Tensor. Got {type(image)}")

    if not isinstance(kernel_centers, torch.Tensor):
        raise TypeError(f"Input kernel_centers is not torch.Tensor. Got {type(kernel_centers)}")

    if not isinstance(kernel_weights, torch.Tensor):
        raise TypeError(f"Input kernel_weights is not torch.Tensor. Got {type(kernel_weights)}")

    if not isinstance(affine_weights, torch.Tensor):
        raise TypeError(f"Input affine_weights is not torch.Tensor. Got {type(affine_weights)}")

    if not len(image.shape) == 4:
        raise ValueError(f"Invalid shape for image, expected BxCxHxW. Got {image.shape}")

    if not len(kernel_centers.shape) == 3:
        raise ValueError(f"Invalid shape for kernel_centers, expected BxNx2. Got {kernel_centers.shape}")

    if not len(kernel_weights.shape) == 3:
        raise ValueError(f"Invalid shape for kernel_weights, expected BxNx2. Got {kernel_weights.shape}")

    if not len(affine_weights.shape) == 3:
        raise ValueError(f"Invalid shape for affine_weights, expected BxNx2. Got {affine_weights.shape}")

    device, dtype = image.device, image.dtype
    batch_size, _, h, w = image.shape
    coords: torch.Tensor = create_meshgrid(h, w, device=device).to(dtype=dtype)
    coords = coords.reshape(-1, 2).expand(batch_size, -1, -1)
    warped: torch.Tensor = warp_points_tps(coords, kernel_centers, kernel_weights, affine_weights)
    warped = warped.view(-1, h, w, 2)
    warped_image: torch.Tensor = nn.functional.grid_sample(image, warped, align_corners=align_corners)

    return warped_image
