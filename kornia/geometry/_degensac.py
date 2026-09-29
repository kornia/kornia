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

"""DEGENSAC kernels: fundamental matrices unaffected by a dominant plane (Chum, Werner and Matas, CVPR 2005).

:class:`~kornia.geometry.ransac.RANSAC` runs them for seven-point fundamental matrices. They follow Chum's C
implementation in pydegensac (``DegUtils.c``, ``ranH.c``, ``rtools.h``, ``exp_ranF.c``): the H-degeneracy test of a
seven-point sample (``checksample``), the local optimization of the homography (``innerH``), and the plane-and-parallax
search (``rFtH``). Points are Hartley-normalized homogeneous points, and every threshold is a squared distance in that
frame.
"""

from __future__ import annotations

from typing import Optional

import torch

from kornia.geometry.epipolar.numeric import cross_product_matrix
from kornia.geometry.homography import find_homography_dlt, sampson_homography_distance

# Every five-element subset of seven correspondences contains one of these triplets (paper, section 3).
_TRIPLETS = ((0, 1, 2), (3, 4, 5), (0, 1, 6), (3, 4, 6), (2, 5, 6))


def _left_epipole(F: torch.Tensor) -> torch.Tensor:
    """Unit left null vectors ``(B, 3)`` of rank-2 matrices ``(B, 3, 3)``: ``F^T e' = 0``, the epipoles in image 2."""
    columns = F.mT
    crosses = torch.linalg.cross(columns[:, [0, 0, 1]], columns[:, [1, 2, 2]])
    best = crosses.square().sum(-1).argmax(1)
    e = crosses[torch.arange(F.shape[0], device=F.device), best]
    return e * e.square().sum(-1, keepdim=True).rsqrt()


def _homographies_from_fundamental(F: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Plane homographies ``(T, 3, 3)`` compatible with ``F`` ``(3, 3)`` through triplets ``x1``, ``x2`` ``(T, 3, 3)``.

    ``H = A - e' (M^{-1} b)^T`` with ``A = [e']_x F``, ``M`` the triplet's homogeneous first-image points as rows and
    ``b_i = (x'_i x A x_i)^T (x'_i x e') / |x'_i x e'|^2`` (Hartley and Zisserman, result 13.6; the paper's eq. 4).
    A collinear triplet gives a NaN homography.
    """
    e = _left_epipole(F[None])[0]
    A = cross_product_matrix(e) @ F
    c1 = torch.linalg.cross(x2, x1 @ A.T)
    c2 = torch.linalg.cross(x2, e.expand_as(x2))
    b = (c1 * c2).sum(-1) / c2.square().sum(-1)
    v, info = torch.linalg.solve_ex(x1, b)
    v = v.masked_fill((info != 0)[:, None], float("nan"))
    return A - e[None, :, None] * v[:, None, :]


def _h_degenerate_sample(
    F: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, threshold: float
) -> Optional[torch.Tensor]:
    """Chum's ``checksample``: a homography through at least five of the seven correspondences, if there is one.

    ``F`` ``(3, 3)`` is a seven-point model of the sample ``x1``, ``x2`` ``(7, 3)`` (homogeneous, ``w = 1``). Each
    triplet of :data:`_TRIPLETS` gives a homography compatible with ``F``; it is refitted by DLT to the five
    correspondences closest to it, and the sample is H-degenerate when five or more then have a squared Sampson
    distance below ``threshold``. Returns the first such homography ``(3, 3)``, or None.
    """
    index = torch.tensor(_TRIPLETS, device=x1.device)
    H = _homographies_from_fundamental(F, x1[index], x2[index])
    H = H[torch.isfinite(H).flatten(1).all(1)]
    if len(H) == 0:
        return None
    errors = sampson_homography_distance(x1[None, :, :2], x2[None, :, :2], H)
    closest = errors.argsort(1)[:, :5]
    refit = find_homography_dlt(x1[closest][..., :2], x2[closest][..., :2], solver="svd")
    errors = sampson_homography_distance(x1[None, :, :2], x2[None, :, :2], refit)
    degenerate = ((errors < threshold).sum(1) >= 5) & torch.isfinite(refit).flatten(1).all(1)
    if not bool(degenerate.any()):
        return None
    return refit[int(degenerate.nonzero()[0, 0])]


def _plane_parallax_fundamentals(
    H: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, first: torch.Tensor, second: torch.Tensor
) -> torch.Tensor:
    """Fundamental matrices ``(P, 3, 3)`` of unit Frobenius norm from a plane homography and pairs of correspondences.

    Plane and parallax (Hartley and Zisserman, section 13.3): the line through ``H x1`` and ``x2`` of an off-plane
    correspondence passes through the epipole, so two of them give ``e' = l_i x l_j`` and ``F = [e']_x H``. ``x1``,
    ``x2`` are homogeneous ``(n, 3)``; ``first`` and ``second`` ``(P,)`` index them. Parallel lines give NaN.
    """
    lines = torch.linalg.cross(x1 @ H.mT, x2)
    epipoles = torch.linalg.cross(lines[first], lines[second])
    # Parallel lines meet nowhere. Their cross product is exactly 0 only without fused multiply-adds; with them it is
    # rounding noise that would normalize into an arbitrary finite model, so a relative test makes both give NaN.
    scale = lines[first].norm(dim=-1) * lines[second].norm(dim=-1)
    parallel = epipoles.norm(dim=-1) <= 64 * torch.finfo(epipoles.dtype).eps * scale
    epipoles = epipoles.masked_fill(parallel[:, None], float("nan"))
    F = cross_product_matrix(epipoles) @ H
    return F * F.square().sum((-2, -1), keepdim=True).rsqrt()
