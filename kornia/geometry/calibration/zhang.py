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

__all__ = ["init_camera_intrinsics_zhang"]


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


def init_camera_intrinsics_zhang(
    homographies: Tensor,
    image_size: tuple[int, int],
    *,
    degeneracy_rtol: float | None = None,
) -> Tensor:
    """Initialize zero-skew pinhole camera intrinsics from multiple planar homographies.

    Implements the linear constraints in Zhang, *A Flexible New Technique for Camera
    Calibration*, IEEE TPAMI 22(11), 2000, https://doi.org/10.1109/34.888718.
    Zero skew is imposed in the linear system, leaving four intrinsic parameters to estimate.

    Args:
        homographies: Plane-to-pixel homographies (B,V,3,3), with at least three views of
            the same metric plane in diverse orientations. Each view may have an arbitrary
            nonzero homogeneous scale, including a negative scale. Float32 and float64
            are supported; all entries must be finite.
        image_size: Positive integer (height,width). This provides a common isotropic
            numerical preconditioner only; the principal point is estimated, not fixed to
            the image center.
        degeneracy_rtol: Relative rank and smallest-subspace separation threshold in (0,1).
            Defaults to ten times the working-dtype epsilon times max(2*V,5). This detects
            numerical degeneracy, not calibration accuracy.

    Returns:
        Intrinsic matrices (B,3,3), with positive focal lengths, zero skew and final row
        (0,0,1), on the input dtype and device.

    Raises:
        ValueError: For invalid shapes, nonfinite inputs, insufficiently diverse views,
            or constraints that do not yield a positive definite intrinsic matrix.
            One invalid batch member fails the entire call.

    Note:
        This is an initializer for an ideal, distortion-free, zero-skew camera, not a full
        calibration pipeline. It estimates neither distortion nor extrinsics and cannot
        identify every violation of that camera model. Board axes must use the same length
        unit; arbitrary projective or anisotropic board coordinates change the solution.

        CPU and CUDA float32 inputs compute in float64; MPS retains float32. The small
        constraint matrix is decomposed directly, without squaring its condition number
        through normal equations. Local gradients are supported for well-conditioned
        inputs with a separated smallest singular direction; they are not globally stable
        at degeneracy. Strict data-dependent diagnostics permit compile graph breaks.
        Full-graph compilation, TorchScript and ONNX export are not promised.
    """
    if not isinstance(homographies, Tensor):
        raise TypeError("homographies must be a Tensor.")
    if homographies.dtype not in (torch.float32, torch.float64):
        raise TypeError("homographies must have dtype float32 or float64.")
    if homographies.ndim != 4 or homographies.shape[-2:] != (3, 3):
        raise ValueError("homographies must have shape (B,V,3,3).")
    batch, views = homographies.shape[:2]
    if batch < 1 or views < 3:
        raise ValueError("homographies require a nonempty batch and at least three views.")
    if len(image_size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in image_size):
        raise ValueError("image_size must contain positive integer height and width.")
    if degeneracy_rtol is not None and (not math.isfinite(degeneracy_rtol) or not 0 < degeneracy_rtol < 1):
        raise ValueError("degeneracy_rtol must be finite and in (0,1).")
    if not bool(torch.isfinite(homographies).all()):
        raise ValueError("homographies must contain only finite values.")

    work_dtype = torch.float32 if homographies.device.type == "mps" else torch.float64
    work = homographies.to(work_dtype)
    gauge = work.detach().abs().amax(dim=(-2, -1), keepdim=True)
    if bool((gauge == 0).any()):
        raise ValueError("degenerate zero homography in a batch member.")
    _, determinant = _adjugate_3x3(work.detach() / gauge)
    if bool((determinant == 0).any()):
        raise ValueError("degenerate singular homography in a batch member.")
    h = work[..., :2]
    height, width = image_size
    scale = 2.0 / max(height, width)
    conditioning = h.new_tensor(
        [[scale, 0, -scale * (width - 1) / 2], [0, scale, -scale * (height - 1) / 2], [0, 0, 1]]
    )
    inverse_conditioning = h.new_tensor([[1 / scale, 0, (width - 1) / 2], [0, 1 / scale, (height - 1) / 2], [0, 0, 1]])
    # Scale before taking a norm to avoid overflow for arbitrary homogeneous gauges.
    magnitude = h.detach().abs().amax(dim=(-2, -1), keepdim=True)
    if bool((magnitude == 0).any()):
        raise ValueError("degenerate homography: the first two columns are zero in a batch member.")
    h = conditioning @ (h / magnitude)
    h = h / torch.linalg.vector_norm(h, dim=(-2, -1), keepdim=True)
    first, second = h[..., 0], h[..., 1]
    system = torch.stack((_constraint(first, second), _constraint(first, first) - _constraint(second, second)), -2)
    system = system.reshape(batch, 2 * views, 5)
    # The generic SVD helper builds full U and may move large MPS batches to CPU.
    # Reduced SVD here retains the small (2V,5) factorization on the requested device.
    _, singular, vh = torch.linalg.svd(system, full_matrices=False)
    diagnostics = singular.detach()
    rtol = degeneracy_rtol
    if rtol is None:
        rtol = 10 * max(2 * views, 5) * torch.finfo(work_dtype).eps
    good = torch.isfinite(diagnostics).all(-1) & (diagnostics[:, 0] > 0)
    good = good & (diagnostics[:, 3] > rtol * diagnostics[:, 0])
    good = good & (diagnostics[:, 3] - diagnostics[:, 4] > rtol * diagnostics[:, 0])
    if not bool(good.all()):
        failed = (~good).nonzero().flatten().tolist()
        raise ValueError(
            f"degenerate calibration constraints in batch members {failed}; use diverse plane orientations."
        )

    b = vh[:, -1]
    b = b * torch.where(b[:, 4:] >= 0, 1.0, -1.0)
    zero = torch.zeros_like(b[:, 0])
    conic = torch.stack((b[:, 0], zero, b[:, 2], zero, b[:, 1], b[:, 3], b[:, 2], b[:, 3], b[:, 4]), -1)
    conic = conic.reshape(batch, 3, 3)
    factor, info = torch.linalg.cholesky_ex(conic)
    if bool((info != 0).any()):
        failed = (info != 0).nonzero().flatten().tolist()
        raise ValueError(f"Non-positive-definite calibration constraints in batch members {failed}.")
    eye = torch.eye(3, dtype=work_dtype, device=homographies.device).expand(batch, -1, -1)
    normalized_k = torch.linalg.solve_triangular(factor.transpose(-1, -2), eye, upper=True)
    intrinsics = inverse_conditioning @ normalized_k
    intrinsics = intrinsics / intrinsics[:, 2:3, 2:3]
    if not bool(torch.isfinite(intrinsics.to(homographies.dtype)).all()):
        raise ValueError("Estimated intrinsics are not finite in the input dtype.")
    return intrinsics.to(homographies.dtype)
