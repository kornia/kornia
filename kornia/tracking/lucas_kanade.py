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

"""Sparse, single-scale translation tracking with inverse-compositional LK."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from kornia.core._small_linalg import _adjugate_2x2
from kornia.filters import spatial_gradient
from kornia.geometry.transform import remap

__all__ = ["track_points_lk"]


def _patches(image: Tensor, points: Tensor, offsets: Tensor) -> Tensor:
    batch, count = points.shape[:2]
    size = offsets.shape[0]
    grid = points[:, :, None, None, :] + offsets
    samples = remap(
        image,
        grid[..., 0].reshape(batch, count * size, size),
        grid[..., 1].reshape(batch, count * size, size),
        align_corners=True,
    )
    return samples.reshape(batch, image.shape[1], count, size * size).permute(0, 2, 3, 1)


def _inside(points: Tensor, height: int, width: int, radius: int) -> Tensor:
    return (
        torch.isfinite(points).all(-1)
        & (points[..., 0] >= radius)
        & (points[..., 0] <= width - radius - 1)
        & (points[..., 1] >= radius)
        & (points[..., 1] <= height - radius - 1)
    )


def track_points_lk(
    image_prev: Tensor,
    image_next: Tensor,
    points_prev: Tensor,
    points_next: Tensor | None = None,
    *,
    window_size: int = 15,
    max_iterations: int = 30,
    epsilon: float = 1e-3,
    min_eigenvalue: float = 1e-4,
) -> tuple[Tensor, Tensor, Tensor]:
    """Track sparse points between grayscale images using single-scale Lucas--Kanade.

    This translation-only inverse-compositional solver follows :cite:`BakerMatthews2004`,
    building on :cite:`LucasKanade1981`. It samples bilinear patches and reuses the
    reference-image central-difference gradients and 2x2 normal matrix. The paper is
    available at https://publications.ri.cmu.edu/lucas-kanade-20-years-on-a-unifying-framework.

    Args:
        image_prev: Reference grayscale image of shape (B,1,H,W).
        image_next: Next image with the same shape, device and dtype. Float32 and float64
            are supported. Intensities in [0,1] are recommended; no rescaling is performed.
        points_prev: Reference pixel coordinates of shape (B,N,2), in (x,y) order.
        points_next: Optional absolute initial next-frame coordinates, of the same shape.
            Defaults to points_prev; provide an estimate for larger displacements.
        window_size: Odd patch side length, at least 3. Images must be at least two pixels larger.
        max_iterations: Positive iteration budget. Updates are vectorized over all points.
        epsilon: Positive finite convergence threshold on the update length, in pixels.
        min_eigenvalue: Nonnegative finite minimum eigenvalue of the mean gradient normal
            matrix. This threshold depends on the intensity scale.

    Returns:
        Tracked coordinates (B,N,2), a boolean validity mask (B,N), and final mean squared
        patch errors (B,N). Invalid tracks return their original coordinates (both zero if
        the original point is nonfinite), and positive infinite error.

    Note:
        A track is valid only if it has two-dimensional texture, retains full patch support
        and converges within the budget. The reference patch also needs a one-pixel gradient
        halo. Validity does not establish visibility or detect occlusion. Any nonfinite image
        pixel invalidates that image pair; nonfinite points invalidate only their own track.
        Invalid operands are sanitized before sampling and solving. Gradients are supported
        through valid tracks away from interpolation, stopping and validity boundaries.
        This is a local, single-scale method; it does not build a pyramid.

    See Also:
        :class:`kornia.tracking.HomographyTracker` tracks a planar target with a stateful
        homography estimator. This function is a stateless primitive for supplied points.
        :class:`~kornia.geometry.transform.image_registrator.ImageRegistrator` estimates a global image transform.
    """
    for name, value in (
        ("image_prev", image_prev),
        ("image_next", image_next),
        ("points_prev", points_prev),
        ("points_next", points_next),
    ):
        if value is None and name == "points_next":
            continue
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a Tensor.")
        if value.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"{name} must have dtype float32 or float64.")
        if value.dtype != image_prev.dtype or value.device != image_prev.device:
            raise ValueError("All inputs must have the same dtype and device.")
    if image_prev.ndim != 4 or image_prev.shape[1] != 1 or image_next.shape != image_prev.shape:
        raise ValueError("Images must have identical (B, 1, H, W) shapes.")
    batch, _, height, width = image_prev.shape
    if points_prev.ndim != 3 or points_prev.shape[0] != batch or points_prev.shape[-1] != 2:
        raise ValueError("points_prev must have shape (B, N, 2).")
    if points_next is not None and points_next.shape != points_prev.shape:
        raise ValueError("points_next must have the same shape as points_prev.")
    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer of at least 3.")
    if min(height, width) < window_size + 2:
        raise ValueError("Images must be at least window_size + 2 in each dimension.")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")
    if not math.isfinite(epsilon) or epsilon <= 0 or not math.isfinite(min_eigenvalue) or min_eigenvalue < 0:
        raise ValueError("epsilon must be positive and min_eigenvalue nonnegative; both must be finite.")

    finite_points = torch.isfinite(points_prev).all(-1)
    original = torch.where(finite_points[..., None], points_prev, torch.zeros_like(points_prev))
    if batch == 0 or points_prev.shape[1] == 0:
        # Empty slices keep connections to every input without reading uninitialized data.
        zero = image_prev.flatten()[:0].sum() + image_next.flatten()[:0].sum()
        if points_next is not None:
            zero = zero + points_next.flatten()[:0].sum()
        return original + zero, finite_points, original.sum(-1) * 0 + zero

    image_ok = torch.isfinite(image_prev).flatten(1).all(1) & torch.isfinite(image_next).flatten(1).all(1)
    prev = torch.where(image_ok[:, None, None, None], image_prev, torch.zeros_like(image_prev))
    nxt = torch.where(image_ok[:, None, None, None], image_next, torch.zeros_like(image_next))
    radius = window_size // 2
    center = points_prev.new_tensor([width // 2, height // 2])
    eligible = image_ok[:, None] & _inside(points_prev, height, width, radius + 1)
    reference_points = torch.where(eligible[..., None], original, center)
    axis = torch.arange(-radius, radius + 1, device=prev.device, dtype=prev.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    offsets = torch.stack((xx, yy), -1)
    reference = _patches(prev, reference_points, offsets)[..., 0]
    gradient = _patches(spatial_gradient(prev, mode="diff", normalized=True)[:, 0], reference_points, offsets)
    normal = gradient.transpose(-1, -2) @ gradient / (window_size * window_size)
    # The discrete conditioning decision is not part of the differentiated solve.
    a, b, d = normal.detach()[..., 0, 0], normal.detach()[..., 0, 1], normal.detach()[..., 1, 1]
    trace = a + d
    smallest = (trace - torch.sqrt((a - d).square() + 4 * b.square())) * 0.5
    eligible = eligible & torch.isfinite(normal).all(-1).all(-1)
    eligible = (
        eligible
        & (smallest >= min_eigenvalue)
        & (a * d - b.square() > 8 * torch.finfo(prev.dtype).eps * trace.square())
    )
    safe_normal = torch.where(eligible[..., None, None], normal, torch.eye(2, device=prev.device, dtype=prev.dtype))
    adjugate, determinant = _adjugate_2x2(safe_normal)
    inverse = adjugate / determinant[..., None, None]

    guess = original if points_next is None else points_next
    eligible = eligible & _inside(guess, height, width, radius)
    current = torch.where(eligible[..., None], guess, center)
    converged = torch.zeros_like(eligible)
    for _ in range(max_iterations):
        active = eligible & ~converged
        residual = reference - _patches(nxt, current, offsets)[..., 0]
        rhs = (gradient * residual[..., None]).mean(-2)
        delta = (inverse @ rhs[..., None])[..., 0]
        candidate = current + torch.where(active[..., None], delta, torch.zeros_like(delta))
        step_ok = _inside(candidate, height, width, radius)
        eligible = eligible & step_ok
        converged = converged | (active & step_ok & (delta.square().sum(-1) <= epsilon * epsilon))
        current = torch.where(eligible[..., None], candidate, center)
    valid = eligible & converged
    final_points = torch.where(valid[..., None], current, center)
    error = (reference - _patches(nxt, final_points, offsets)[..., 0]).square().mean(-1)
    return (
        torch.where(valid[..., None], current, original),
        valid,
        torch.where(valid, error, torch.full_like(error, float("inf"))),
    )
