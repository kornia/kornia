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
    # Weight values are not checked: reading them back would sync the host on accelerators, twice per call in RANSAC's
    # local optimization, and break fullgraph compilation. The contract is in find_affine's docstring.
    return weights


def _fit_affine_components(
    points1: torch.Tensor, points2: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit ``points2 = A @ points1 + t`` using a guarded weighted normal equation."""
    # Explicit casts alone do not prevent autocast from lowering matrix multiplications again.
    with torch.autocast(device_type=points1.device.type, enabled=False):
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
        x_mean = (x * w[..., None]).sum(dim=-2) / safe_support
        y_mean = (y * w[..., None]).sum(dim=-2) / safe_support

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
        # Rotate the source onto the principal axes of its weighted scatter. For a scatter elongated along any
        # other direction, det = a * d - b * c cancels, and its relative error, about aspect**2 * eps, scales the
        # whole linear map. The fit is invariant to this rotation, so the detached angle loses no gradient and
        # atan2(0, 0) of an isotropic scatter is never differentiated.
        scatter = ((xn * w[..., None]).transpose(-1, -2) @ xn).detach()
        angle = 0.5 * torch.atan2(2 * scatter[..., 0, 1], scatter[..., 0, 0] - scatter[..., 1, 1])
        rotation = torch.stack((angle.cos(), angle.sin(), -angle.sin(), angle.cos()), dim=-1).reshape_as(scatter)
        xn = xn @ rotation.transpose(-1, -2)

        normal = (xn * w[..., None]).transpose(-1, -2) @ xn
        cross = (yn * w[..., None]).transpose(-1, -2) @ xn
        a, b, c, d = normal[..., 0, 0], normal[..., 0, 1], normal[..., 1, 0], normal[..., 1, 1]
        det = a * d - b * c
        trace = a + d
        eps = torch.finfo(work_dtype).eps
        valid = (support[..., 0] > 0) & (det > eps * trace.square())
        safe_det = torch.where(valid, det, torch.ones_like(det))
        inverse = torch.stack((d, -b, -c, a), dim=-1).reshape_as(normal) / safe_det[..., None, None]
        linear_normalized = cross @ inverse @ rotation
        linear = linear_normalized * (y_scale / x_scale)
        translation = y_mean - torch.matmul(linear, x_mean[..., None]).squeeze(-1)
        nan_linear = torch.full_like(linear, float("nan"))
        nan_translation = torch.full_like(translation, float("nan"))
        return torch.where(valid[..., None, None], linear, nan_linear), torch.where(
            valid[..., None], translation, nan_translation
        )


def _find_affine_linear(points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
    """Estimate origin-fixed 2D linear maps ``points2 = points1 @ A^T`` by least squares.

    Solves the 2x2 normal equation ``A (X^T X) = Y^T X`` directly for points of shape ``(..., N, 2)``. Unlike
    :func:`find_affine`, it neither centers, scales nor rotates the points, so it suits well-scaled, well-spread
    points: AdaLAM refits its local affinities with it, on coordinates relative to the seed correspondence and
    normalized by the neighborhood radius, where zero padding rows contribute nothing. Rank-deficient fits are
    NaN under the relative criterion ``det(X^T X) <= eps * trace(X^T X)**2``.
    """
    _check_points(points1, points2, 1, batched=False)
    with torch.autocast(device_type=points1.device.type, enabled=False):
        work_dtype = torch.float32 if points1.dtype in (torch.float16, torch.bfloat16) else points1.dtype
        x, y = points1.to(work_dtype), points2.to(work_dtype)
        normal = x.transpose(-1, -2) @ x
        a, b, c, d = normal[..., 0, 0], normal[..., 0, 1], normal[..., 1, 0], normal[..., 1, 1]
        det = a * d - b * c
        valid = det > torch.finfo(work_dtype).eps * (a + d).square()
        safe_det = torch.where(valid, det, torch.ones_like(det))
        inverse = torch.stack((d, -b, -c, a), dim=-1).reshape_as(normal) / safe_det[..., None, None]
        linear = (y.transpose(-1, -2) @ x) @ inverse
        return torch.where(valid[..., None, None], linear, torch.full_like(linear, float("nan"))).to(points1.dtype)


def _find_affine_linear_pair(points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
    """Solve origin-fixed 2D linear maps exactly from two correspondences.

    The minimal case of :func:`_find_affine_linear` for points of shape ``(..., 2, 2)``: two correspondences
    determine the map, so the 2x2 system is solved directly rather than through the normal equation, which
    squares its condition number. Rank-deficient samples are NaN under the same relative criterion,
    ``det(X^T X) <= eps * trace(X^T X)**2``, written without squaring as ``|det X| <= sqrt(eps) * ||X||_F**2``.
    """
    _check_points(points1, points2, 2, batched=False)
    if points1.shape[-2] != 2:
        raise ValueError(f"Exactly 2 correspondences are required. Got {points1.shape[-2]}.")
    with torch.autocast(device_type=points1.device.type, enabled=False):
        work_dtype = torch.float32 if points1.dtype in (torch.float16, torch.bfloat16) else points1.dtype
        x, y = points1.to(work_dtype), points2.to(work_dtype)
        a, b, c, d = x[..., 0, 0], x[..., 0, 1], x[..., 1, 0], x[..., 1, 1]
        det = a * d - b * c
        valid = det.abs() > torch.finfo(work_dtype).eps ** 0.5 * x.square().sum(dim=(-2, -1))
        safe_det = torch.where(valid, det, torch.ones_like(det))
        # x @ A^T = y, so A^T = adj(x) @ y / det(x).
        adjugate = torch.stack((d, -b, -c, a), dim=-1).reshape_as(x)
        linear = (adjugate @ y).transpose(-1, -2) / safe_det[..., None, None]
        return torch.where(valid[..., None, None], linear, torch.full_like(linear, float("nan"))).to(points1.dtype)


def _find_affine_triplet(
    points1: torch.Tensor, points2: torch.Tensor, weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Solve 2D affine transformations exactly from three correspondences, RANSAC's minimal samples.

    Points have shape ``(B, 3, 2)`` and the result is homogeneous ``(B, 3, 3)``, like :func:`find_affine`. The linear
    part maps the differences to the first correspondence, by :func:`_find_affine_linear_pair`, and the translation
    follows from the first correspondence; collinear samples are NaN under that function's rank criterion. Positive
    weights do not change an exact fit, so ``weights``, part of RANSAC's solver interface, are not used.
    """
    _check_points(points1, points2, 3, batched=True)
    if points1.shape[-2] != 3:
        raise ValueError(f"Exactly 3 correspondences are required. Got {points1.shape[-2]}.")
    with torch.autocast(device_type=points1.device.type, enabled=False):
        work_dtype = torch.float32 if points1.dtype in (torch.float16, torch.bfloat16) else points1.dtype
        x, y = points1.to(work_dtype), points2.to(work_dtype)
        origin1, origin2 = x[:, :1], y[:, :1]
        linear = _find_affine_linear_pair(x[:, 1:] - origin1, y[:, 1:] - origin2)
        translation = origin2.transpose(-1, -2) - linear @ origin1.transpose(-1, -2)
        bottom = torch.tensor([0.0, 0.0, 1.0], device=x.device, dtype=work_dtype).expand(x.shape[0], 1, 3)
        return torch.cat((torch.cat((linear, translation), dim=-1), bottom), dim=-2).to(points1.dtype)


def find_affine(points1: torch.Tensor, points2: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""Estimate 2D affine transformations from point correspondences.

    Fits the weighted least-squares model :math:`\mathbf{y}=A\mathbf{x}+\mathbf{t}`
    independently for each batch. The calculation centers and isotropically normalizes
    each point set before solving its 2-by-2 normal equation. Inputs must have the
    same floating-point dtype and device; half inputs are accumulated in float32.
    Ambient autocast is disabled during fitting to preserve this arithmetic precision.

    Args:
        points1: Source points with shape :math:`(B, N, 2)`, where :math:`N \geq 3`.
        points2: Destination points with shape :math:`(B, N, 2)`.
        weights: Optional correspondence weights with shape :math:`(B, N)`. They must be
            finite and non-negative; other values are not detected and give undefined
            results. Only relative weights matter.

    Returns:
        Homogeneous affine transformations with shape :math:`(B, 3, 3)`. Degenerate
        batches without three non-collinear, positive-weight source points contain
        NaNs, allowing robust estimators to reject singular fits as invalid models.
    """
    _check_points(points1, points2, 3, batched=True)
    weights = _prepare_weights(weights, points1, points1.shape[:-2])
    linear, translation = _fit_affine_components(points1, points2, weights)
    result = torch.zeros((*points1.shape[:-2], 3, 3), device=points1.device, dtype=linear.dtype)
    result[..., :2, :2] = linear
    result[..., :2, 2] = translation
    result[..., 2, 2] = 1
    return result.to(points1.dtype)
