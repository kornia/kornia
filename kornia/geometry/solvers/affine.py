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

"""Affine transformations estimated from point correspondences."""

from __future__ import annotations

from typing import Optional

import torch

__all__ = ["find_affine"]


def _check_points(points1: torch.Tensor, points2: torch.Tensor, min_points: int, batched: bool) -> None:
    if points1.shape != points2.shape:
        raise ValueError(f"points1 and points2 must have the same shape. Got {points1.shape} and {points2.shape}.")
    if (points1.ndim != 3 if batched else points1.ndim < 2) or points1.shape[-1] != 2:
        prefix = "(B, N, 2)" if batched else "(..., N, 2)"
        raise ValueError(f"points must have shape {prefix}. Got {points1.shape}.")
    if points1.shape[-2] < min_points:
        raise ValueError(f"At least {min_points} correspondences are required. Got {points1.shape[-2]}.")
    if points1.device != points2.device:
        raise ValueError("points1 and points2 must be on the same device.")
    if not points1.is_floating_point() or not points2.is_floating_point():
        raise TypeError("points1 and points2 must be floating point tensors.")
    if points1.dtype != points2.dtype:
        raise TypeError("points1 and points2 must have the same dtype.")


def _prepare_weights(weights: Optional[torch.Tensor], points: torch.Tensor, batch_shape: torch.Size) -> torch.Tensor:
    if weights is None:
        return torch.ones((*batch_shape, points.shape[-2]), device=points.device, dtype=points.dtype)
    if weights.shape != (*batch_shape, points.shape[-2]):
        raise ValueError(f"weights must have shape {(*batch_shape, points.shape[-2])}. Got {weights.shape}.")
    if weights.device != points.device:
        raise ValueError("weights and points must be on the same device.")
    if not weights.is_floating_point():
        raise TypeError("weights must be a floating point tensor.")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("weights must be finite and non-negative.")
    return weights


def _fit_affine_components(
    points1: torch.Tensor, points2: torch.Tensor, weights: torch.Tensor, center: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit ``points2 = A @ points1 + t`` using a guarded weighted normal equation."""
    work_dtype = torch.float32 if points1.dtype in (torch.float16, torch.bfloat16) else points1.dtype
    x, y = points1.to(work_dtype), points2.to(work_dtype)
    # Weight rescaling leaves the least-squares problem unchanged and avoids
    # underflow/overflow in the normal equation for extreme input weights. Do
    # this before casting weights to the point arithmetic dtype: float32 weights
    # may exceed the finite range of float16 inputs.
    weight_dtype = torch.promote_types(work_dtype, weights.dtype)
    weights_for_scaling = weights.to(weight_dtype)
    max_weight = weights_for_scaling.amax(dim=-1, keepdim=True)
    normalized_weights = weights_for_scaling / torch.where(max_weight > 0, max_weight, torch.ones_like(max_weight))
    w = normalized_weights.to(work_dtype)
    support = w.sum(dim=-1, keepdim=True)
    safe_support = torch.where(support > 0, support, torch.ones_like(support))
    if center:
        x_mean = (x * w[..., None]).sum(dim=-2) / safe_support
        y_mean = (y * w[..., None]).sum(dim=-2) / safe_support
    else:
        x_mean = torch.zeros_like(x[..., 0, :])
        y_mean = torch.zeros_like(y[..., 0, :])

    xc, yc = x - x_mean[..., None, :], y - y_mean[..., None, :]
    # Scaling makes the rank criterion relative to the data and keeps normal equations conditioned.
    # Unlike an RMS scale, max-absolute scaling never evaluates sqrt at its singular point.
    nonzero_weight = (w > 0)[..., None]
    x_scale = torch.where(nonzero_weight, xc.abs(), torch.zeros_like(xc)).amax(dim=(-2, -1), keepdim=True)
    y_scale = torch.where(nonzero_weight, yc.abs(), torch.zeros_like(yc)).amax(dim=(-2, -1), keepdim=True)
    one = torch.ones_like(x_scale)
    x_scale = torch.where(x_scale > 0, x_scale, one)
    y_scale = torch.where(y_scale > 0, y_scale, one)
    xn, yn = xc / x_scale, yc / y_scale

    normal = torch.einsum("...n,...ni,...nj->...ij", w, xn, xn)
    cross = torch.einsum("...n,...ni,...nj->...ij", w, yn, xn)
    a, b, c, d = normal[..., 0, 0], normal[..., 0, 1], normal[..., 1, 0], normal[..., 1, 1]
    det = a * d - b * c
    trace = a + d
    eps = torch.finfo(work_dtype).eps
    valid = (support[..., 0] > 0) & (det > eps * trace.square())
    safe_det = torch.where(valid, det, torch.ones_like(det))
    inverse = torch.stack((d, -b, -c, a), dim=-1).reshape_as(normal) / safe_det[..., None, None]
    linear_normalized = cross @ inverse
    linear = linear_normalized * (y_scale / x_scale)
    translation = y_mean - torch.matmul(linear, x_mean[..., None]).squeeze(-1)
    nan_linear = torch.full_like(linear, float("nan"))
    nan_translation = torch.full_like(translation, float("nan"))
    return torch.where(valid[..., None, None], linear, nan_linear), torch.where(
        valid[..., None], translation, nan_translation
    )


def _find_affine_linear(
    points1: torch.Tensor, points2: torch.Tensor, weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Estimate origin-fixed 2D linear maps from batched point correspondences.

    This internal primitive supports arbitrary leading batch dimensions and at least two
    non-collinear correspondences. It is used by AdaLAM, where the seed correspondence
    fixes the translation before fitting the local affinity.
    """
    _check_points(points1, points2, 1, batched=False)
    weights = _prepare_weights(weights, points1, points1.shape[:-2])
    linear, _ = _fit_affine_components(points1, points2, weights, center=False)
    return linear.to(points1.dtype)


def find_affine(points1: torch.Tensor, points2: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""Estimate 2D affine transformations from point correspondences.

    Fits the weighted least-squares model :math:`\mathbf{y}=A\mathbf{x}+\mathbf{t}`
    independently for each batch. The calculation centers and isotropically normalizes
    each point set before solving its 2-by-2 normal equation. Inputs must have the
    same floating-point dtype and device; half inputs are accumulated in float32.

    Args:
        points1: Source points with shape :math:`(B, N, 2)`, where :math:`N \geq 3`.
        points2: Destination points with shape :math:`(B, N, 2)`.
        weights: Optional non-negative finite correspondence weights with shape
            :math:`(B, N)`.

    Returns:
        Homogeneous affine transformations with shape :math:`(B, 3, 3)`. Degenerate
        batches without three non-collinear, positive-weight source points contain
        NaNs, allowing robust estimators to reject singular fits as invalid models.
    """
    _check_points(points1, points2, 3, batched=True)
    weights = _prepare_weights(weights, points1, points1.shape[:-2])
    linear, translation = _fit_affine_components(points1, points2, weights, center=True)
    result = torch.zeros((*points1.shape[:-2], 3, 3), device=points1.device, dtype=linear.dtype)
    result[..., :2, :2] = linear
    result[..., :2, 2] = translation
    result[..., 2, 2] = 1
    return result.to(points1.dtype)
