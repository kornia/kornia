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

"""Batched kernels of RANSAC's Levenberg-Marquardt pipeline for fundamental matrices and homographies.

The minimal solvers share their building blocks with the public estimators: the partial-pivoted LU null space
(:func:`~kornia.geometry.solvers.homogeneous._null_space_lu`, also behind :func:`~kornia.geometry.epipolar.run_7point`,
the eight-point case of :func:`~kornia.geometry.epipolar.run_8point` and the minimal case of
:func:`~kornia.geometry.homography.find_homography_dlt`), the epipolar design rows and the seven-point cubic. What
stays here is what the public functions' contracts rule out in a sampling loop: correspondences are normalized once
per call rather than per sample, models stay in that normalized frame at unit Frobenius norm instead of being scaled
to ``F[2, 2] = 1``, absent candidates are NaN rather than padded, and everything runs under ``torch.no_grad`` with
closed forms (the rank-2 projection, the cubic) whose ``clamp``-guarded ``sqrt``/``acos`` would not give portable
gradients (#4229). Residuals of many models on one set of correspondences are scored from one matrix product with
per-correspondence monomials. The refiners are Levenberg-Marquardt iterations batched over models, in the spirit of
PoseLib's (Larsson and contributors, https://github.com/PoseLib/PoseLib) ``refine_fundamental`` and
``refine_homography``: fundamental matrices are parametrized by the SVD-based factorization of Bartoli and Sturm,
"Nonlinear estimation of the fundamental matrix with minimal parameters", TPAMI 2004, and homographies by a
tangent step of the unit-norm matrix.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

from kornia.geometry.epipolar._metrics import _sampson_from_quadratic_basis, _sampson_quadratic_basis
from kornia.geometry.epipolar.fundamental import (
    _eight_point_fundamental,
    _epipolar_design_rows,
    _refine_fundamental_lm,
    _rank2_projection,
    _seven_point_candidates,
)
from kornia.geometry.homography import (
    _four_point_homography,
    _refine_homography_lm,
    _transfer_basis,
    _transfer_from_basis,
)

__all__: list[str] = []


def normalize_correspondences(
    kp1: torch.Tensor, kp2: torch.Tensor, shared_scale: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    r"""Center each image's points and scale them so the mean distance to the centroid is :math:`\sqrt{2}`.

    Statistics use the correspondences that are finite in both images; the others stay non-finite and so are never
    counted as inliers. ``shared_scale`` uses one scale for both images, which keeps the Sampson distance a multiple
    of the pixel one.

    Returns:
        Homogeneous normalized points ``(N, 3)`` of each image, the ``(3, 3)`` transforms that map pixels to them, and
        the two scales (pixels per normalized unit).

    """
    finite = torch.isfinite(kp1).all(1) & torch.isfinite(kp2).all(1)
    count = finite.sum().clamp(min=1).to(kp1.dtype)
    weight = finite.to(kp1.dtype)[:, None]
    zero = torch.zeros_like(kp1)
    p1 = torch.where(finite[:, None], kp1, zero)
    p2 = torch.where(finite[:, None], kp2, zero)
    c1 = (p1 * weight).sum(0) / count
    c2 = (p2 * weight).sum(0) / count
    r1 = ((p1 - c1).norm(dim=1) * weight[:, 0]).sum() / count
    r2 = ((p2 - c2).norm(dim=1) * weight[:, 0]).sum() / count
    if shared_scale:
        r1 = r2 = (r1 + r2) / 2
    s1 = float(r1) / math.sqrt(2.0)
    s2 = float(r2) / math.sqrt(2.0)
    # Coincident points leave no scale; any positive one keeps the transform invertible.
    s1 = s1 if math.isfinite(s1) and s1 > 0 else 1.0
    s2 = s2 if math.isfinite(s2) and s2 > 0 else 1.0
    ones = torch.ones_like(kp1[:, :1])
    x1 = torch.cat([(kp1 - c1) / s1, ones], 1)
    x2 = torch.cat([(kp2 - c2) / s2, ones], 1)
    t1 = torch.eye(3, dtype=kp1.dtype, device=kp1.device)
    t2 = torch.eye(3, dtype=kp1.dtype, device=kp1.device)
    t1[0, 0] = t1[1, 1] = 1.0 / s1
    t2[0, 0] = t2[1, 1] = 1.0 / s2
    t1[:2, 2] = -c1 / s1
    t2[:2, 2] = -c2 / s2
    return x1, x2, t1, t2, s1, s2


rank2_projection = _rank2_projection


def fundamental_8pt(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Rank-2 fundamental matrices ``(B, 3, 3)`` from eight homogeneous normalized correspondences ``(B, 8, 3)``."""
    return _eight_point_fundamental(_epipolar_design_rows(x1, x2))


def fundamental_7pt(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Up to three fundamental matrices ``(B, 3, 3, 3)`` per seven correspondences ``(B, 7, 3)``; NaN where fewer."""
    candidates, valid = _seven_point_candidates(_epipolar_design_rows(x1, x2))
    return candidates.masked_fill(~valid[..., None, None], float("nan"))


def homography_4pt(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Homographies ``(B, 3, 3)``, of unit Frobenius norm, from four normalized correspondences ``(B, 4, 3)``."""
    return _four_point_homography(x1[..., :2], x2[..., :2])


sampson_basis = _sampson_quadratic_basis
sampson_errors = _sampson_from_quadratic_basis


transfer_basis = _transfer_basis
transfer_errors = _transfer_from_basis


refine_fundamental = _refine_fundamental_lm
refine_homography = _refine_homography_lm
