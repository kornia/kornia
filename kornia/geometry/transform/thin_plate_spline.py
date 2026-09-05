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

import warnings

import torch
from torch import nn

from kornia.core.utils import _torch_solve_cast
from kornia.geometry.grid import create_meshgrid

__all__ = ["get_tps_transform", "warp_image_tps", "warp_points_tps"]

# utilities for computing thin plate spline transforms


def _pair_square_euclidean(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    r"""Compute the pairwise squared euclidean distance matrices :math:`(B, N, M)` between two tensors.

    Tensors with shapes (B, N, C) and (B, M, C).
    """
    # ||t1-t2||^2 = (t1-t2)^T(t1-t2) = t1^T*t1 + t2^T*t2 - 2*t1^T*t2
    t1_sq: torch.Tensor = tensor1.mul(tensor1).sum(dim=-1, keepdim=True)
    t2_sq: torch.Tensor = tensor2.mul(tensor2).sum(dim=-1, keepdim=True).transpose(1, 2)
    t1_t2: torch.Tensor = tensor1.matmul(tensor2.transpose(1, 2))
    square_dist: torch.Tensor = -2 * t1_t2 + t1_sq + t2_sq
    square_dist = square_dist.clamp(min=0)  # handle possible numerical errors
    return square_dist


def _kernel_distance(squared_distances: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Compute the TPS kernel distance function: :math:`r^2 log(r)`, where `r` is the euclidean distance.

    Since
    :math: `\log(r) = 1/2 \log(r^2)`, this function takes the squared distance matrix and calculates
    :math: `0.5 r^2 log(r^2)`.
    """
    # r^2 * log(r) = 1/2 * r^2 * log(r^2)
    return 0.5 * squared_distances * squared_distances.add(eps).log()


def get_tps_transform(points_src: torch.Tensor, points_dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the TPS transform parameters that warp source points to target points.

    The input to this function is a torch.Tensor of :math:`(x, y)` source points :math:`(B, N, 2)` and a corresponding
    torch.Tensor of target :math:`(x, y)` points :math:`(B, N, 2)`.

    Convention:
        - ``points_src``/``points_dst``: :math:`(B, N, 2)` in ``(x, y)`` order, in
          whatever coordinate frame the caller supplies — the function performs no
          normalization of its own
        - returns kernel weights :math:`(B, N, 2)` and affine weights :math:`(B, 3, 2)`,
          consumed by :func:`warp_points_tps`/:func:`warp_image_tps`; the identity
          mapping (``points_src == points_dst``) yields kernel weights and an affine
          that are mathematically zero/identity, realized only up to linear-solver
          (LU) round-off — not bit-exact in general, and the residual size is
          dtype- and backend-dependent; float16 currently produces NaN weights
        - neither :func:`warp_points_tps` nor :func:`warp_image_tps` calls this
          function — the caller composes them explicitly; whichever tensor is
          passed as this function's **second** positional argument is
          ``kernel_centers`` for the returned ``kernel_weights``/``affine_weights``
          pair, and must be passed as ``kernel_centers`` again to the warp
          function; :func:`warp_points_tps` is typically composed in the
          natural ``(points_src, points_dst)`` order, while :func:`warp_image_tps`
          is typically composed **reversed** — ``get_tps_transform(points_dst,
          points_src)`` — since image warping samples from output space back
          into input space; the coordinate frames for that composition depend on
          :func:`warp_image_tps`'s grid mode. Its legacy default uses a
          corner-aligned ``points_dst`` output lattice and the ``align_corners``
          frame for ``points_src``. With ``use_correct_grid=True``, both point sets
          use the frame selected by ``align_corners`` — see that function's
          Convention block for both formulas::

              kernel_weights, affine_weights = get_tps_transform(points_dst, points_src)
              warped = warp_image_tps(image, kernel_centers=points_src,
                                      kernel_weights=kernel_weights, affine_weights=affine_weights,
                                      use_correct_grid=True)

    Args:
        points_src: batch of source points :math:`(B, N, 2)` as :math:`(x, y)` coordinate vectors.
        points_dst: batch of target points :math:`(B, N, 2)` as :math:`(x, y)` coordinate vectors.

    Returns:
        :math:`(B, N, 2)` torch.Tensor of kernel weights and :math:`(B, 3, 2)`
            torch.Tensor of affine weights. The last dimension contains the x-transform and y-transform weights
            as separate columns.

    Example:
        >>> points_src = torch.rand(1, 5, 2)
        >>> points_dst = torch.rand(1, 5, 2)
        >>> kernel_weights, affine_weights = get_tps_transform(points_src, points_dst)

    .. note::
        This function is often used in conjunction with :func:`warp_points_tps`, :func:`warp_image_tps`.

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
    r"""Warp a torch.Tensor of coordinate points using the thin plate spline defined by arguments.

    The source points should be a :math:`(B, N, 2)` torch.Tensor of :math:`(x, y)` coordinates. The kernel centers are
    a :math:`(B, K, 2)` torch.Tensor of :math:`(x, y)` coordinates. The kernel weights are a :math:`(B, K, 2)`
    torch.Tensor, and the affine weights are a :math:`(B, 3, 2)` torch.Tensor.  For the weight tensors,
    torch.Tensor[..., 0] contains the weights for the x-transform and torch.Tensor[..., 1] the weights
    for the y-transform.

    Convention:
        - points: :math:`(B, N, 2)` (or :math:`(B, K, 2)` for kernel centers) in
          ``(x, y)`` order, in whatever coordinate frame the input points are already
          in — this function performs no normalization of its own
        - ``kernel_centers`` must be the ``points_dst`` argument (the second
          positional argument) passed to :func:`get_tps_transform`; see its
          Convention block for the full binding rule and recipe
        - returns points warped in that same coordinate frame/units as the input, with
          no renormalization; the output is **not** clamped — a valid TPS transform can
          map an in-range point outside that frame (e.g. warping ``(0.75, 0)`` through a
          transform that is exactly a 2x affine scale returns ``(1.5, 0)``, outside
          :math:`[-1, 1]`)

    Args:
        points_src: torch.Tensor of source points :math:`(B, N, 2)`.
        kernel_centers: torch.Tensor of kernel center points :math:`(B, K, 2)`.
        kernel_weights: torch.Tensor of kernel weights :math:`(B, K, 2)`.
        affine_weights: torch.Tensor of affine weights :math:`(B, 3, 2)`.

    Returns:
        The :math:`(B, N, 2)` torch.Tensor of warped source points, from applying the TPS transform.

    Example:
        >>> points_src = torch.rand(1, 5, 2)
        >>> points_dst = torch.rand(1, 5, 2)
        >>> kernel_weights, affine_weights = get_tps_transform(points_src, points_dst)
        >>> warped = warp_points_tps(points_src, points_dst, kernel_weights, affine_weights)
        >>> warped_correct = torch.allclose(warped, points_dst)

    .. note::
        This function is often used in conjunction with :func:`get_tps_transform`.

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
    use_correct_grid: bool = False,
) -> torch.Tensor:
    r"""Warp an image torch.Tensor according to the thin plate spline transform defined by arguments.

    .. image:: _static/img/warp_image_tps.png

    The transform is applied to each pixel coordinate in the output image to obtain a point in the input
    image for interpolation of the output pixel. So the TPS parameters should correspond to a warp from
    output space to input space.

    The input `image` is a :math:`(B, C, H, W)` torch.Tensor. The kernel centers, kernel weight and affine weights
    are the same as in `warp_points_tps`.

    Convention:
        - image: :math:`(B, C, H, W)`; kernel/affine weights as returned by
          :func:`get_tps_transform` called **reversed** —
          ``get_tps_transform(points_dst, points_src)`` — see its Convention block
        - ``kernel_centers`` must be the tensor passed as the **second** positional
          argument to that reversed :func:`get_tps_transform` call (i.e.
          ``points_src``); see :func:`get_tps_transform`'s Convention block for the
          full binding rule and recipe
        - control points use normalized ``grid_sample`` coordinates, where
          ``[-1, 1]`` is the in-bounds extent (values outside are allowed, subject
          to ``padding_mode``) rather than a validity bound
        - with ``use_correct_grid=True``, both arguments to the reversed
          :func:`get_tps_transform` call use the frame selected by
          ``align_corners``: at ``False`` this is the half-pixel mapping
          :math:`x_{norm} = (2x+1)/W - 1`; at ``True`` this is the corner-aligned
          mapping :math:`x_{norm} = 2x/(W-1) - 1`
        - the legacy ``use_correct_grid=False`` mode always uses a corner-aligned
          destination/output lattice, while the source/input side follows
          ``align_corners``; this mismatch is preserved temporarily for backward
          compatibility and emits a deprecation warning when ``align_corners=False``
          - pixel-space (unnormalized) control points on either side silently
            produce a wrong warp of the correct shape, with no error raised
        - align_corners: ``False`` by default
        - padding_mode: ``'zeros'`` by default
        - use_correct_grid: ``False`` during the deprecation window; opt in with
          ``True`` so the output grid honors ``align_corners``

    .. warning::
        The legacy output grid used by ``use_correct_grid=False`` is deprecated
        when ``align_corners=False``. Set ``use_correct_grid=True`` to adopt the
        corrected convention before the default changes in a future minor release.

    Args:
        image: input image torch.Tensor :math:`(B, C, H, W)`.
        kernel_centers: kernel center points :math:`(B, K, 2)`.
        kernel_weights: torch.Tensor of kernel weights :math:`(B, K, 2)`.
        affine_weights: torch.Tensor of affine weights :math:`(B, 3, 2)`.
        align_corners: interpolation flag used by `grid_sample`.
        padding_mode: padding flag used by `grid_sample`.
        use_correct_grid: whether the output grid should honor ``align_corners``.

    Returns:
        warped image torch.Tensor :math:`(B, C, H, W)`.

    Example:
        >>> points_src = torch.rand(1, 5, 2)
        >>> points_dst = torch.rand(1, 5, 2)
        >>> image = torch.rand(1, 3, 32, 32)
        >>> # note that we are getting the reverse transform: dst -> src
        >>> kernel_weights, affine_weights = get_tps_transform(points_dst, points_src)
        >>> warped_image = warp_image_tps(
        ...     image, points_src, kernel_weights, affine_weights, use_correct_grid=True
        ... )

    .. note::
        This function is often used in conjunction with :func:`get_tps_transform`.

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
    if not align_corners and not use_correct_grid and not torch.jit.is_scripting():
        # Python warnings cannot be captured in a compiled graph. Eager callers
        # receive the migration warning; compiled calls keep the legacy math.
        if not torch.compiler.is_compiling():
            warnings.warn(
                "warp_image_tps currently preserves its legacy corner-aligned output grid when "
                "align_corners=False. Set use_correct_grid=True to make the output grid honor "
                "align_corners. The default will change after the documented deprecation window.",
                DeprecationWarning,
                stacklevel=2,
            )

    if use_correct_grid:
        identity = torch.eye(2, 3, device=image.device, dtype=image.dtype).unsqueeze(0)
        coords: torch.Tensor = nn.functional.affine_grid(identity, (1, 1, h, w), align_corners=align_corners)
    else:
        coords = create_meshgrid(h, w, device=image.device, dtype=image.dtype, normalized_coordinates=True)
    coords = coords.reshape(-1, 2).expand(batch_size, -1, -1)
    warped: torch.Tensor = warp_points_tps(coords, kernel_centers, kernel_weights, affine_weights)
    warped = warped.view(-1, h, w, 2)
    warped_image: torch.Tensor = nn.functional.grid_sample(
        image, warped, padding_mode=padding_mode, align_corners=align_corners
    )

    return warped_image
