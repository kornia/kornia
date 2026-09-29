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

import math
import sys
from typing import List, Optional, Tuple

import torch

from kornia.geometry.epipolar._metrics import _sampson_from_quadratic_basis, _sampson_quadratic_basis
from kornia.geometry.epipolar.numeric import cross_product_matrix
from kornia.geometry.homography import find_homography_dlt, sampson_homography_distance

# Every five-element subset of seven correspondences contains one of these triplets (paper, section 3).
_TRIPLETS = ((0, 1, 2), (3, 4, 5), (0, 1, 6), (3, 4, 6), (2, 5, 6))
# innerH (ranH.c, rtools.h): repetitions, least-squares steps, initial selection multiple, points per fit, subset cap.
_RAN_REP = 10
_ILSQ_ITERS = 4
_TC = 4
_INL_LIMIT = 10
_SUBSET_CAP = 12
# SC_M scoring (rtools.h, the compiled __SCORE__): the MSAC gain truncates at 9/4 of the threshold.
_GAIN_SCALE = 9.0 / 4.0
# rFtH (DegUtils.c): stopping confidence, draw cap, support a kept model must exceed (sam_sizO), models kept.
_PP_CONFIDENCE = 0.999
_PP_MAX_DRAWS = 20000
_PP_MIN_SUPPORT = 4
_PP_KEEP = 8
# Two planes are the same when their inlier sets have a Jaccard index of at least this (VSAC's criterion for models).
_SAME_PLANE = 0.95


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


def _msac_gain(errors: torch.Tensor, threshold: float) -> torch.Tensor:
    """Chum's ``SC_M`` score ``(C,)`` of squared errors ``(C, N)``: ``sum(max(0, 1 - e / (9/4 threshold)))``."""
    return (1.0 - errors / (_GAIN_SCALE * threshold)).clamp_min(0.0).sum(1)


def _best_candidate(gains: torch.Tensor, alive: torch.Tensor) -> Optional[int]:
    """Index of the first largest finite gain among live candidates, or None.

    Candidates are in the order Chum's loops visit them; ``scoreLess`` is strict, so a tie keeps the earlier one, which
    is also the index ``argmax`` returns.
    """
    gains = torch.where(alive & torch.isfinite(gains), gains, torch.full_like(gains, -math.inf))
    if not bool(torch.isfinite(gains).any()):
        return None
    return int(gains.argmax())


def _inner_homography(
    H: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, threshold: float, generator: Optional[torch.Generator]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chum's ``innerH``: the local optimization of ``H`` ``(3, 3)``, scored with the homography Sampson distance.

    ``x1``, ``x2`` are ``(N, 2)`` float64 host points and ``threshold`` is ``16 t``. ``inHrani`` runs
    :data:`_RAN_REP` repetitions, batched here: each fits a DLT to a random subset of ``min(n / 2, 12)`` of the ``n``
    inliers of ``H`` and refines it with ``iterH``: a fit to that model's inliers, then :data:`_ILSQ_ITERS` refits with
    the selection threshold lowered from ``4 * threshold`` by ``3 * threshold / 4`` after each, every fit using at most
    :data:`_INL_LIMIT` selected points, a random subset when there are more. A repetition whose subset model has fewer
    than 4 inliers contributes nothing; one whose selection falls below 4 points stops. Every evaluated model is a
    candidate, ranked by :func:`_msac_gain`. ``innerH`` passes 10 as the argument it calls ``iters``, which
    ``inHrani`` receives as ``inlLimit``; the repetition count is the constant ``RAN_REP``, also 10.

    Returns:
        The best candidate, or ``H`` when no repetition produced one (``inHrani`` leaves it unchanged), and its squared
        Sampson distances ``(N,)``.
    """
    n_points = x1.shape[0]
    pts1, pts2 = x1[None], x2[None]

    def errors_of(models: torch.Tensor) -> torch.Tensor:
        return sampson_homography_distance(pts1, pts2, models)

    def fit(selected: torch.Tensor, cap: int) -> torch.Tensor:
        # A uniform random subset of at most ``cap`` selected points per row; zero weights pad the rest.
        keys = torch.rand(selected.shape, generator=generator, dtype=x1.dtype).masked_fill(~selected, -1.0)
        top = keys.topk(min(cap, n_points), dim=1)
        weights = (top.values >= 0).to(x1.dtype)
        return find_homography_dlt(x1[top.indices], x2[top.indices], weights, solver="svd")

    base = errors_of(H[None])[0]
    inliers = base <= threshold
    count = int(inliers.sum())
    if count < 8:
        return H, base
    models = fit(inliers.expand(_RAN_REP, -1), min(count // 2, _SUBSET_CAP))
    errors = errors_of(models)
    selected = errors <= threshold
    alive = selected.sum(1) >= 4
    candidates: List[torch.Tensor] = [models]
    gains: List[torch.Tensor] = [_msac_gain(errors, threshold)]
    live: List[torch.Tensor] = [alive]
    models = fit(selected, _INL_LIMIT)
    selection = _TC * threshold
    step = (selection - threshold) / _ILSQ_ITERS
    for _ in range(_ILSQ_ITERS):
        errors = errors_of(models)
        candidates.append(models)
        gains.append(_msac_gain(errors, threshold))
        live.append(alive)
        selected = errors <= selection
        alive = alive & (selected.sum(1) >= 4)
        models = fit(selected, _INL_LIMIT)
        selection -= step
    errors = errors_of(models)
    candidates.append(models)
    gains.append(_msac_gain(errors, threshold))
    live.append(alive)
    best = _best_candidate(torch.stack(gains, 1).flatten(), torch.stack(live, 1).flatten())
    if best is None:
        return H, base
    winner = torch.stack(candidates, 1).flatten(0, 1)[best]
    return winner, errors_of(winner[None])[0]


def _repeats_plane(plane: torch.Tensor, seen: List[torch.Tensor]) -> bool:
    """Whether the inlier mask ``plane`` matches one in ``seen``: a Jaccard index of at least 0.95.

    Every degenerate record setter of a dominant plane finds that plane again; VSAC likewise skips the local
    optimization of a model whose inliers repeat the best one's (Ivashechkin, Barath and Matas, ICCV 2021).
    """
    return any(float((plane & other).sum()) >= _SAME_PLANE * float((plane | other).sum()) for other in seen)


def _pair_draws(support: int, total: int, confidence: float) -> int:
    """Two-point samples needed for ``confidence`` with ``support`` inliers among ``total`` correspondences.

    :meth:`~kornia.geometry.ransac.RANSAC.max_samples_by_conf` for a sample of two, which this module cannot import.
    """
    if support >= total:
        return 1
    probability = support * (support - 1) / (total * (total - 1))
    if probability <= 0.0:
        return sys.maxsize
    return min(sys.maxsize, math.ceil(math.log1p(-confidence) / math.log1p(-probability)))


def _plane_parallax_search(
    H: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    threshold: float,
    batch: int,
    generator: Optional[torch.Generator],
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Chum's ``rFtH``: plane-and-parallax RANSAC over pairs of off-plane correspondences.

    ``x1``, ``x2`` ``(n, 3)`` are the off-plane correspondences, normalized, on their device; ``H`` is the plane
    homography there and ``threshold`` is ``2 t``. Pairs are drawn on the host from ``generator`` (a device's generator
    cannot draw on the host) in batches of ``batch`` and moved to the device. A model's support counts the off-plane
    correspondences with a squared Sampson distance below ``threshold``; drawing stops at twice the two-point bound for
    the best support at confidence 0.999, as ``rFtH``'s loop runs to ``2 * max_sam``, or after 20000 draws.

    Returns:
        Up to eight models with more than four off-plane inliers and their support, best first, or None.
    """
    n = x1.shape[0]
    basis = _sampson_quadratic_basis(x1, x2)
    kept = x1.new_zeros(0, 3, 3)
    kept_support = torch.zeros(0, dtype=torch.long, device=x1.device)
    best, drawn, limit = 0, 0, _PP_MAX_DRAWS
    while drawn < limit:
        size = min(batch, limit - drawn)
        first = (torch.rand(size, generator=generator, dtype=torch.float64) * n).long()
        second = (torch.rand(size, generator=generator, dtype=torch.float64) * (n - 1)).long()
        second = second + (second >= first).long()
        models = _plane_parallax_fundamentals(H, x1, x2, first.to(x1.device), second.to(x1.device))
        support = (_sampson_from_quadratic_basis(models, basis) < threshold).sum(1)
        kept_support, order = torch.cat([kept_support, support]).topk(min(_PP_KEEP, len(kept_support) + size))
        kept = torch.cat([kept, models])[order]
        drawn += size
        best = max(best, int(kept_support[0]))
        if best > _PP_MIN_SUPPORT:
            limit = min(limit, 2 * _pair_draws(best, n, _PP_CONFIDENCE))
    qualifying = kept_support > _PP_MIN_SUPPORT
    if not bool(qualifying.any()):
        return None
    return kept[qualifying], kept_support[qualifying]
