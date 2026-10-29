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

r"""Fit a thin-plate spline to pairs of 2D points, and warp points or images with it.

Points are ``(B, N, 2)`` tensors of ``(x, y)`` coordinates. :func:`get_tps_transform` and :func:`warp_points_tps`
take them in whatever frame the caller uses and normalize nothing; :func:`warp_image_tps` needs them in the normalized
frame of ``grid_sample``. :func:`get_tps_transform` returns kernel weights ``(B, N, 2)`` and affine weights
``(B, 3, 2)``, with the x and y transforms as columns, for radial kernels centred on its **second** argument, which the
warps take as ``kernel_centers``. A warp then computes ``f(p) = a_0 + p @ A + sum_i w_i U(|p - c_i|)`` with
:math:`U(r) = r^2 \log r`, ``[a_0; A]`` the affine weights and ``c_i`` the kernel centres. Bookstein's thin-plate
spline centres them on the first argument instead, so these warps differ from it between and away from the landmarks
(`#6404 <https://github.com/kornia/kornia/issues/6404>`_).

Which function to use:

- Fit the spline that maps source points onto target points: :func:`get_tps_transform`.
- Move points with it: :func:`warp_points_tps`. Warp an image with it: :func:`warp_image_tps`, which samples the
  input at the spline of each output pixel, so fit it from target to source, in normalized coordinates.
- For one global matrix, use :func:`~kornia.geometry.transform.warp_perspective` or
  :func:`~kornia.geometry.transform.warp_affine`; for random spline warps in an augmentation pipeline,
  :class:`~kornia.augmentation.RandomThinPlateSpline`; for a smoothed random displacement field,
  :func:`~kornia.geometry.transform.elastic_transform2d`.
"""

from __future__ import annotations

import torch

from kornia.core.utils import _torch_grid_sample, _torch_solve_cast
from kornia.geometry.grid import create_meshgrid

__all__ = ["get_tps_transform", "warp_image_tps", "warp_points_tps"]

# utilities for computing thin plate spline transforms


def _pair_square_euclidean(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    r"""Compute the pairwise squared euclidean distance matrices ``(B, N, M)`` between two tensors.

    Tensors with shapes (B, N, C) and (B, M, C).
    """
    # ||t1-t2||^2 = (t1-t2)^T(t1-t2) = t1^T*t1 + t2^T*t2 - 2*t1^T*t2
    t1_sq: torch.Tensor = tensor1.mul(tensor1).sum(dim=-1, keepdim=True)
    t2_sq: torch.Tensor = tensor2.mul(tensor2).sum(dim=-1, keepdim=True).transpose(1, 2)
    t1_t2: torch.Tensor = tensor1.matmul(tensor2.transpose(1, 2))
    square_dist: torch.Tensor = -2 * t1_t2 + t1_sq + t2_sq
    return square_dist.clamp(min=0)  # handle possible numerical errors


def _kernel_distance(squared_distances: torch.Tensor) -> torch.Tensor:
    r"""Compute the TPS kernel distance function: :math:`r^2 log(r)`, where `r` is the euclidean distance.

    Since
    :math: `\log(r) = 1/2 \log(r^2)`, this function takes the squared distance matrix and calculates
    :math: `0.5 r^2 log(r^2)`.
    """
    safe = torch.where(squared_distances > 0, squared_distances, torch.ones_like(squared_distances))
    return torch.where(
        squared_distances > 0,
        0.5 * squared_distances * safe.log(),
        torch.zeros_like(squared_distances),
    )


def get_tps_transform(points_src: torch.Tensor, points_dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Fit a thin-plate-spline warp that maps source points exactly onto target points.

    ``kernel_weights, affine_weights = get_tps_transform(points_src, points_dst)`` takes ``(B, N, 2)`` point pairs and
    returns the ``(B, N, 2)`` kernel weights and ``(B, 3, 2)`` affine weights of the spline whose kernels are centred on
    ``points_dst``: ``warp_points_tps(points_src, points_dst, kernel_weights, affine_weights)`` returns ``points_dst``.
    Unlike Bookstein's thin-plate spline, whose kernels are centred on the points where it interpolates
    (``points_src`` here), the kernels sit on ``points_dst``, so the warp differs between and away from the landmarks
    (`#6404 <https://github.com/kornia/kornia/issues/6404>`_). To warp an image, fit the reverse mapping and use
    :func:`warp_image_tps` (see Convention); for a homography from four point pairs, use
    :func:`get_perspective_transform`.

    Convention:
        - ``points_src``/``points_dst``: ``(B, N, 2)`` in ``(x, y)`` order, in whatever
          coordinate frame the caller supplies; no normalization is applied
        - returns kernel weights ``(B, N, 2)`` and affine weights ``(B, 3, 2)``; the
          identity mapping gives zero kernel weights and an identity affine up to solver round-off, and any pairs
          related by an affine map give zero kernel weights, except a constant map of more than three points, whose
          equal ``points_dst`` make the system singular
        - fewer than three points make the linear system singular and ``torch.linalg.solve`` raises; points all on one
          line make it singular too, but round-off can hide that, and the solve then returns huge affine weights that
          throw points off that line far away and may miss the landmarks themselves
        - the tensor passed as the **second** argument is the ``kernel_centers`` that the warp
          function must receive with these weights. :func:`warp_points_tps` is composed in the
          natural ``(points_src, points_dst)`` order; :func:`warp_image_tps` samples from output
          space back into input space, so it is composed **reversed**, with both point sets in the
          normalized frame given in its Convention block::

              kernel_weights, affine_weights = get_tps_transform(points_dst, points_src)
              warped = warp_image_tps(image, kernel_centers=points_src,
                                      kernel_weights=kernel_weights, affine_weights=affine_weights)

    Args:
        points_src: batch of source points ``(B, N, 2)`` as :math:`(x, y)` coordinate vectors.
        points_dst: batch of target points ``(B, N, 2)`` as :math:`(x, y)` coordinate vectors.

    Returns:
        ``(B, N, 2)`` torch.Tensor of kernel weights and ``(B, 3, 2)``
            torch.Tensor of affine weights. The last dimension contains the x-transform and y-transform weights
            as separate columns. Row 0 of the affine weights is the translation and rows 1 and 2 the matrix ``A``
            that multiplies a point as ``p @ A``.

    See Also:
        - :func:`warp_points_tps`: applies the fitted spline to points, with ``points_dst`` as ``kernel_centers``.
        - :func:`warp_image_tps`: applies a spline fitted from target to source to an image.
        - :func:`get_perspective_transform`: the ``(B, 3, 3)`` homography of four point pairs, one global matrix.

    Example:
        >>> points_src = torch.rand(1, 5, 2)
        >>> points_dst = torch.rand(1, 5, 2)
        >>> kernel_weights, affine_weights = get_tps_transform(points_src, points_dst)

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

    weights = _torch_solve_cast(l_matrix, dest_with_zeros)
    kernel_weights: torch.Tensor = weights[:, :-3]
    affine_weights: torch.Tensor = weights[:, -3:]

    return kernel_weights, affine_weights


def warp_points_tps(
    points_src: torch.Tensor, kernel_centers: torch.Tensor, kernel_weights: torch.Tensor, affine_weights: torch.Tensor
) -> torch.Tensor:
    r"""Move 2D points by a thin-plate spline given as kernel centres and weights.

    ``warped = warp_points_tps(points_src, kernel_centers, kernel_weights, affine_weights)`` takes ``(B, N, 2)`` points
    and the ``(B, K, 2)`` kernel centres, ``(B, K, 2)`` kernel weights and ``(B, 3, 2)`` affine weights of a spline, and
    returns the ``(B, N, 2)`` warped points in the same frame, unclamped. With the weights of
    ``get_tps_transform(points_src, points_dst)``, pass ``points_dst`` as ``kernel_centers``: that function centres the
    kernels there, unlike Bookstein's thin-plate spline (`#6404 <https://github.com/kornia/kornia/issues/6404>`_). In
    the weight tensors, ``[..., 0]`` holds the weights of the x transform and ``[..., 1]`` those of the y transform. To
    warp an image, use :func:`warp_image_tps`.

    Convention:
        - points: ``(B, N, 2)`` (or ``(B, K, 2)`` for kernel centers) in ``(x, y)``
          order; no normalization is applied, and the output is in the same frame, not clamped
        - ``kernel_centers`` must be the second argument passed to :func:`get_tps_transform`
          (see its Convention block)

    Args:
        points_src: torch.Tensor of source points ``(B, N, 2)``.
        kernel_centers: torch.Tensor of kernel center points ``(B, K, 2)``.
        kernel_weights: torch.Tensor of kernel weights ``(B, K, 2)``.
        affine_weights: torch.Tensor of affine weights ``(B, 3, 2)``.

    Returns:
        The ``(B, N, 2)`` torch.Tensor of warped source points, from applying the TPS transform.

    See Also:
        - :func:`get_tps_transform`: fits the weights from point pairs.
        - :func:`warp_image_tps`: applies a spline to every pixel of an image, in normalized coordinates.
        - :func:`~kornia.geometry.linalg.transform_points`: moves points by one ``(B, 3, 3)`` matrix instead.

    Example:
        >>> points_src = torch.rand(1, 5, 2)
        >>> points_dst = torch.rand(1, 5, 2)
        >>> kernel_weights, affine_weights = get_tps_transform(points_src, points_dst)
        >>> warped = warp_points_tps(points_src, points_dst, kernel_weights, affine_weights)
        >>> warped_correct = torch.allclose(warped, points_dst)

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
    padding_mode: str = "zeros",
) -> torch.Tensor:
    r"""Warp images by a thin-plate spline that maps each output pixel to the input pixel it samples.

    ``out = warp_image_tps(image, kernel_centers, kernel_weights, affine_weights)`` takes images ``(B, C, H, W)`` and a
    spline as in :func:`warp_points_tps` (one per image, or a batch of 1 for all), applies the spline to the normalized
    coordinates of every output pixel and returns the ``(B, C, H, W)`` images bilinearly sampled there. The spline
    therefore maps output space to input space: to move control points ``points_src`` of the input onto
    ``points_dst``, fit ``get_tps_transform(points_dst, points_src)`` and pass ``points_src`` as ``kernel_centers``,
    with both point sets in normalized coordinates. The kernels then sit on ``points_src``, whereas Bookstein's
    thin-plate spline would centre them on ``points_dst``, the output points where it interpolates
    (`#6404 <https://github.com/kornia/kornia/issues/6404>`_). For one global matrix, use
    :func:`~kornia.geometry.transform.warp_perspective`; for random spline warps in an augmentation pipeline,
    :class:`~kornia.augmentation.RandomThinPlateSpline`.

    .. image:: _static/img/warp_image_tps.png

    Convention:
        - image: ``(B, C, H, W)``; the weights come from :func:`get_tps_transform` called
          **reversed**, ``get_tps_transform(points_dst, points_src)``, and ``kernel_centers`` is
          ``points_src`` (see its Convention block)
        - all control points live in the normalized ``grid_sample`` frame selected by
          ``align_corners``: the half-pixel :math:`x_{norm} = (2x+1)/W - 1` at the default
          ``False``, the corner-aligned :math:`x_{norm} = 2x/(W-1) - 1` at ``True``; values outside
          :math:`[-1, 1]` sample the padding. Pixel-space control points silently give a wrong
          warp of the right shape
        - align_corners: ``False`` by default (the matrix warps default to ``True``)

    Args:
        image: input image torch.Tensor ``(B, C, H, W)``.
        kernel_centers: kernel center points ``(B, K, 2)``.
        kernel_weights: torch.Tensor of kernel weights ``(B, K, 2)``.
        affine_weights: torch.Tensor of affine weights ``(B, 3, 2)``.
        align_corners: interpolation flag used by `grid_sample`.
        padding_mode: padding flag used by `grid_sample`.

    Returns:
        warped image torch.Tensor ``(B, C, H, W)``.

    See Also:
        - :func:`get_tps_transform`: fits the spline; call it with the target points first for this warp.
        - :func:`warp_points_tps`: applies the same spline to points.
        - :class:`~kornia.augmentation.RandomThinPlateSpline`: random spline warps, in an augmentation pipeline.
        - :func:`~kornia.geometry.transform.elastic_transform2d`: a smoothed random displacement field instead of
          control points.

    Example:
        >>> points_src = torch.rand(1, 5, 2)
        >>> points_dst = torch.rand(1, 5, 2)
        >>> image = torch.rand(1, 3, 32, 32)
        >>> # note that we are getting the reverse transform: dst -> src
        >>> kernel_weights, affine_weights = get_tps_transform(points_dst, points_src)
        >>> warped_image = warp_image_tps(image, points_src, kernel_weights, affine_weights)

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

    batch_size, _, h, w = image.shape
    coords: torch.Tensor = create_meshgrid(
        h, w, device=image.device, dtype=image.dtype, normalized_coordinates=True, align_corners=align_corners
    )
    coords = coords.reshape(-1, 2).expand(batch_size, -1, -1)
    warped: torch.Tensor = warp_points_tps(coords, kernel_centers, kernel_weights, affine_weights)
    warped = warped.view(-1, h, w, 2)
    warped_image: torch.Tensor = _torch_grid_sample(
        image, warped, padding_mode=padding_mode, align_corners=align_corners
    )

    return warped_image
