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

"""Zero-skew camera intrinsic initialization from planar homographies."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from kornia.core._small_linalg import _adjugate_3x3
from kornia.image import ImageSize

__all__ = ["intrinsics_from_homographies"]


def _constraint(a: Tensor, b: Tensor) -> Tensor:
    return torch.stack(
        (
            a[..., 0] * b[..., 0],
            a[..., 1] * b[..., 1],
            a[..., 2] * b[..., 0] + a[..., 0] * b[..., 2],
            a[..., 2] * b[..., 1] + a[..., 1] * b[..., 2],
            a[..., 2] * b[..., 2],
        ),
        -1,
    )


def _conic(vh: Tensor) -> Tensor:
    b = vh[:, -1]
    b = b * torch.where(b[:, 4:] >= 0, 1.0, -1.0)
    zero = torch.zeros_like(b[:, 0])
    return torch.stack((b[:, 0], zero, b[:, 2], zero, b[:, 1], b[:, 3], b[:, 2], b[:, 3], b[:, 4]), -1).reshape(
        -1, 3, 3
    )


def intrinsics_from_homographies(
    homographies: Tensor,
    image_size: tuple[int, int] | ImageSize,
    weights: Tensor | None = None,
    *,
    degeneracy_rtol: float = 1e-2,
) -> tuple[Tensor, Tensor]:
    """Initialize zero-skew pinhole camera intrinsics from multiple planar homographies.

    Implements the linear initialization stage of :cite:`zhang2000calibration`,
    https://doi.org/10.1109/34.888718. Zero skew is imposed in the linear system, leaving
    four intrinsic parameters to estimate. The five-column constraint system estimates
    ``B = K^{-T} K^{-1}`` with ``B12 = 0``; a Cholesky factorization recovers the intrinsic
    matrix. This factorization and numerical conditioning are implementation choices.

    Args:
        homographies: Plane-to-pixel homographies (B,V,3,3), with at least two views of
            the same metric plane in diverse orientations. Each view may have an arbitrary
            nonzero homogeneous scale, including a negative scale. Float32 and float64
            are supported. Nonfinite or singular views with nonzero weight invalidate
            their camera; zero-weight views are ignored, including nonfinite entries.
        image_size: True image (height,width), as positive integers or an :class:`~kornia.image.ImageSize`
            with integer dimensions (Python integers or scalar integer tensors), shared
            by all cameras. It sets the image-coordinate
            normalization and therefore affects the algebraic-error objective and noisy
            estimates. The principal point is estimated, not fixed to the image center.
        weights: Optional nonnegative finite per-view weights (B,V), with the same dtype
            and device as ``homographies``. The sum of squared constraint residuals for
            each view is multiplied by its weight. Zero removes a view; at least two
            positive-weight views are needed per camera. None gives equal weights.
        degeneracy_rtol: Relative rank and smallest-subspace separation threshold in (0,1).
            The dtype-independent default is 0.01: both sigma4/sigma1 and
            (sigma4-sigma5)/sigma1 must exceed it, with descending singular values of the
            normalized weighted system. This flags weak viewing geometry, not calibration
            accuracy. See the calibration guide for the tilt/noise sweep motivating it.

    Returns:
        A tuple ``(intrinsics, valid)``. Intrinsics have shape (B,3,3), positive focal
        lengths, zero skew and final row (0,0,1), on the input dtype and device. ``valid``
        is a boolean tensor (B,) on the same device. Invalid cameras return the identity
        matrix as a placeholder, not a calibration; always check the mask.

    Raises:
        TypeError: For unsupported tensor types or dtypes.
        ValueError: For invalid shapes, image size, weights or threshold. Data-dependent
            calibration failures set ``valid=False`` without failing other cameras.

    Note:
        This is an initializer for an ideal, distortion-free, zero-skew camera, not a full
        calibration pipeline. It estimates neither distortion nor extrinsics and cannot
        identify every violation of that camera model. Board axes must use the same length
        unit; arbitrary projective or anisotropic board coordinates change the solution.
        Two suitably diverse views suffice under zero skew; more views are recommended
        with noise. The threshold is a heuristic and cannot guarantee accurate intrinsics.

        CPU and CUDA float32 inputs compute in float64; MPS retains float32. The small
        constraint matrix is decomposed directly, without squaring its condition number
        through normal equations. Local gradients are supported for well-conditioned
        inputs with distinct singular values; they are not globally stable at degeneracy.
        Invalid cameras and zero-weight views have zero gradients. Calibration diagnostics
        use tensor masks without Python data-dependent branches. Weight-value validation
        raises eagerly and may cause a compile graph break when weights are supplied.
        TorchScript and ONNX export are not promised.
    """
    if not isinstance(homographies, Tensor):
        raise TypeError("homographies must be a Tensor.")
    if homographies.dtype not in (torch.float32, torch.float64):
        raise TypeError("homographies must have dtype float32 or float64.")
    if homographies.ndim != 4 or homographies.shape[-2:] != (3, 3):
        raise ValueError("homographies must have shape (B,V,3,3).")
    batch, views = homographies.shape[:2]
    if batch < 1 or views < 2:
        raise ValueError("homographies require a nonempty batch and at least two views.")
    dimensions = (image_size.height, image_size.width) if isinstance(image_size, ImageSize) else image_size
    if not isinstance(dimensions, tuple) or len(dimensions) != 2:
        raise ValueError("image_size must contain positive integer height and width.")
    size: list[int] = []
    for dimension in dimensions:
        if isinstance(dimension, Tensor):
            if dimension.ndim != 0 or dimension.dtype not in (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                raise ValueError("image_size tensors must be scalar integers.")
            dimension = int(dimension.item())
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError("image_size must contain positive integer height and width.")
        size.append(dimension)
    height, width = size
    if not math.isfinite(degeneracy_rtol) or not 0 < degeneracy_rtol < 1:
        raise ValueError("degeneracy_rtol must be finite and in (0,1).")
    if weights is not None:
        if not isinstance(weights, Tensor) or weights.dtype != homographies.dtype:
            raise TypeError("weights must be a Tensor with the homographies dtype.")
        if weights.shape != (batch, views) or weights.device != homographies.device:
            raise ValueError("weights must have shape (B,V) and the homographies device.")
        if not bool((torch.isfinite(weights) & (weights >= 0)).all()):
            raise ValueError("weights must be finite and nonnegative.")

    work_dtype = torch.float32 if homographies.device.type == "mps" else torch.float64
    work = homographies.to(work_dtype)
    view_weights = torch.ones_like(work[..., 0, 0]) if weights is None else weights.to(work_dtype)
    active = view_weights > 0
    eye = torch.eye(3, dtype=work_dtype, device=homographies.device)
    finite = torch.isfinite(work).all(dim=(-2, -1))
    # Sanitize BEFORE normalization: even a masked NaN can poison a backward pass.
    work = torch.where((active & finite)[..., None, None], work, eye)
    gauge = work.detach().abs().amax(dim=(-2, -1), keepdim=True)
    _, determinant = _adjugate_3x3(work.detach() / torch.where(gauge > 0, gauge, 1.0))
    view_valid = finite & (determinant != 0)
    valid = ((~active) | view_valid).all(-1) & (active.sum(-1) >= 2)
    work = torch.where(view_valid[..., None, None], work, eye)
    h = work[..., :2]
    scale = 2.0 / max(height, width)
    conditioning = h.new_tensor(
        [[scale, 0, -scale * (width - 1) / 2], [0, scale, -scale * (height - 1) / 2], [0, 0, 1]]
    )
    inverse_conditioning = h.new_tensor([[1 / scale, 0, (width - 1) / 2], [0, 1 / scale, (height - 1) / 2], [0, 0, 1]])
    # Scale before taking a norm to avoid overflow for arbitrary homogeneous gauges.
    magnitude = h.detach().abs().amax(dim=(-2, -1), keepdim=True)
    h = conditioning @ (h / magnitude)
    h = h / torch.linalg.vector_norm(h, dim=(-2, -1), keepdim=True)
    first, second = h[..., 0], h[..., 1]
    system = torch.stack((_constraint(first, second), _constraint(first, first) - _constraint(second, second)), -2)
    # A common weight scale is irrelevant to this homogeneous least-squares problem.
    weight_scale = view_weights.detach().amax(-1, keepdim=True)
    view_weights = view_weights / torch.where(weight_scale > 0, weight_scale, 1.0)
    root_weights = torch.where(active, torch.where(active, view_weights, 1.0).sqrt(), 0.0)
    system = system * root_weights[..., None, None]
    system = system.reshape(batch, 2 * views, 5)
    if views == 2:
        # A zero row retains the 4x5 nullspace in a reduced 5x5 SVD, including its
        # derivative. The extra vector of a full 4x5 SVD has no PyTorch gradient.
        system = torch.cat((system, system.new_zeros(batch, 1, 5)), -2)
    # Check a detached copy first so repeated/zero singular values in invalid cameras
    # never enter the differentiable SVD. A final output mask alone is insufficient.
    _, diagnostics, diagnostic_vh = torch.linalg.svd(system.detach(), full_matrices=False)
    valid = valid & torch.isfinite(diagnostics).all(-1) & (diagnostics[:, 0] > 0)
    valid = valid & (diagnostics[:, 3] > degeneracy_rtol * diagnostics[:, 0])
    valid = valid & (diagnostics[:, 3] - diagnostics[:, 4] > degeneracy_rtol * diagnostics[:, 0])
    _, info = torch.linalg.cholesky_ex(_conic(diagnostic_vh))
    valid = valid & (info == 0)
    fallback = torch.diag(system.new_tensor([5.0, 4.0, 3.0, 2.0, 1.0]))
    fallback = torch.cat((fallback, system.new_zeros(system.shape[-2] - 5, 5)), -2)
    system = torch.where(valid[:, None, None], system, fallback)
    _, _, vh = torch.linalg.svd(system, full_matrices=False)

    conic = torch.where(valid[:, None, None], _conic(vh), eye)
    factor, _ = torch.linalg.cholesky_ex(conic)
    eye = eye.expand(batch, -1, -1)
    normalized_k = torch.linalg.solve_triangular(factor.transpose(-1, -2), eye, upper=True)
    intrinsics = inverse_conditioning @ normalized_k
    intrinsics = intrinsics / intrinsics[:, 2:3, 2:3]
    intrinsics = intrinsics.to(homographies.dtype)
    valid = valid & torch.isfinite(intrinsics).all(dim=(-2, -1))
    return torch.where(valid[:, None, None], intrinsics, eye.to(homographies.dtype)), valid
