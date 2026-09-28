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

import warnings
from typing import Optional, Tuple

import torch

from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.core.utils import _extract_device_dtype, _torch_svd_cast, safe_inverse_with_mask, safe_solve_with_mask
from kornia.geometry.conversions import convert_points_from_homogeneous, convert_points_to_homogeneous
from kornia.geometry.epipolar import normalize_points, normalize_transformation
from kornia.geometry.epipolar._metrics import _shares_points
from kornia.geometry.epipolar.fundamental import _robust_loss
from kornia.geometry.linalg import transform_points
from kornia.geometry.solvers.homogeneous import _null_space_lu

__all__ = [
    "find_homography_dlt",
    "find_homography_dlt_iterated",
    "find_homography_lines_dlt",
    "find_homography_lines_dlt_iterated",
    "line_segment_transfer_error_one_way",
    "oneway_transfer_error",
    "sample_is_valid_for_homography",
    "symmetric_transfer_error",
]

TupleTensor = Tuple[torch.Tensor, torch.Tensor]


def oneway_transfer_error(
    pts1: torch.Tensor, pts2: torch.Tensor, H: torch.Tensor, squared: bool = True, eps: float = 1e-8
) -> torch.Tensor:
    r"""Return transfer error in image 2 for correspondences given the homography matrix.

    Convention:
        - ``oneway_transfer_error(pts1, pts2, H)`` measures in image 2, between ``H`` applied to ``pts1`` and
          ``pts2``.
        - One set of correspondences scored against several homographies (points with leading dimensions of
          size 1) is computed from one matrix product for all of them, in at least float32; the result matches the
          per-homography computation to roundoff.
        - ``squared=True``, the default here and in :func:`symmetric_transfer_error`, returns the squared
          distance; :func:`line_segment_transfer_error_one_way` defaults to ``squared=False``.
        - Known defects: ``eps`` is added to the projective denominator and inside the square root, so the error
          depends on the scale of ``H``, and an exact match scores ``sqrt(eps)``, not 0, with ``squared=False``
          (`#4881 <https://github.com/kornia/kornia/issues/4881>`_).

    Args:
        pts1: correspondences from the left images with shape
          (B, N, 2 or 3). If they are homogeneous, converted automatically.
        pts2: correspondences from the right images with shape
          (B, N, 2 or 3). If they are homogeneous, converted automatically.
        H: Homographies with shape :math:`(B, 3, 3)`.
        squared: if True (default), the squared distance is returned.
        eps: added to the projective denominator and, with ``squared=False``, inside the square root.

    Returns:
        the computed distance with shape :math:`(B, N)`.

    """
    KORNIA_CHECK_SHAPE(H, ["B", "3", "3"])
    if H.shape[0] >= 2 and _shares_points(pts1, pts2):
        return _oneway_transfer_error_shared_impl_(pts1, pts2, H, squared, eps)

    if pts1.shape[-1] == 3:
        x1y1 = convert_points_from_homogeneous(pts1)
        x1 = x1y1[..., 0]
        y1 = x1y1[..., 1]
    else:
        x1 = pts1[..., :, 0]
        y1 = pts1[..., :, 1]

    if pts2.shape[-1] == 3:
        u2v2 = convert_points_from_homogeneous(pts2)
        u2 = u2v2[..., 0]
        v2 = u2v2[..., 1]
    else:
        u2 = pts2[..., :, 0]
        v2 = pts2[..., :, 1]

    # ---- Grab H entries and broadcast across N ----
    h00 = H[..., 0, 0][..., None]
    h01 = H[..., 0, 1][..., None]
    h02 = H[..., 0, 2][..., None]
    h10 = H[..., 1, 0][..., None]
    h11 = H[..., 1, 1][..., None]
    h12 = H[..., 1, 2][..., None]
    h20 = H[..., 2, 0][..., None]
    h21 = H[..., 2, 1][..., None]
    h22 = H[..., 2, 2][..., None]

    # From Hartley and Zisserman, Error in one image (4.6)
    # dist = \sum_{i} ( d(x', Hx)**2)
    # ---- Apply homography to pts1 (Euclidean) and dehomogenize ----
    # [x'; y'; w']^T = H @ [x1, y1, 1]^T
    x_num = h00 * x1 + h01 * y1 + h02
    y_num = h10 * x1 + h11 * y1 + h12
    w_den = h20 * x1 + h21 * y1 + h22

    u1in2 = x_num / (w_den + eps)
    v1in2 = y_num / (w_den + eps)

    # ---- Squared transfer error in image 2 ----
    err2 = (u1in2 - u2).pow(2) + (v1in2 - v2).pow(2)
    if squared:
        return err2
    return (err2 + eps).sqrt()


def symmetric_transfer_error(
    pts1: torch.Tensor, pts2: torch.Tensor, H: torch.Tensor, squared: bool = True, eps: float = 1e-8
) -> torch.Tensor:
    r"""Return Symmetric transfer error for correspondences given the homography matrix.

    Convention:
        - Argument order as :func:`oneway_transfer_error`. The squared value is the image-2 error of ``H`` plus
          the image-1 error of ``H^-1``, and ``squared=False`` returns the square root of that sum.
        - Known defects: the ``eps`` defect of :func:`oneway_transfer_error` applies here too
          (`#4881 <https://github.com/kornia/kornia/issues/4881>`_).

    Args:
        pts1: correspondences from the left images with shape
          (B, N, 2 or 3). If they are homogeneous, converted automatically.
        pts2: correspondences from the right images with shape
          (B, N, 2 or 3). If they are homogeneous, converted automatically.
        H: Homographies with shape :math:`(B, 3, 3)`.
        squared: if True (default), the squared distance is returned.
        eps: added to the projective denominator and, with ``squared=False``, inside the square root.

    Returns:
        the computed distance with shape :math:`(B, N)`. Rows whose homography is not invertible
        score ``max_num``, i.e. ``torch.finfo(pts1.dtype).max``, for both values of ``squared``.

    """
    KORNIA_CHECK_SHAPE(H, ["B", "3", "3"])
    if pts1.size(-1) == 3:
        pts1 = convert_points_from_homogeneous(pts1)

    if pts2.size(-1) == 3:
        pts2 = convert_points_from_homogeneous(pts2)

    max_num = torch.finfo(pts1.dtype).max
    # From Hartley and Zisserman, Symmetric transfer error (4.7)
    # dist = \sum_{i} (d(x, H^-1 x')**2 + d(x', Hx)**2)
    # Shield the *input* of the inverse, not its output: ``inv_ex`` backward is
    # ``-H_inv^T @ grad @ H_inv^T`` over a saved ``H_inv`` full of non-finite values, so masking the
    # result afterwards still leaves ``0 * nan = nan`` in ``H.grad``. The mask is taken on a detached
    # copy so it carries no graph, and the differentiable inverse then only ever sees a valid matrix.
    _, good_H = safe_inverse_with_mask(H.detach())
    eye = torch.eye(3, device=H.device, dtype=H.dtype).expand_as(H)
    H_safe = torch.where(good_H.view(-1, 1, 1), H, eye)
    H_inv_safe, _ = safe_inverse_with_mask(H_safe)

    there: torch.Tensor = oneway_transfer_error(pts1, pts2, H_safe, True, eps)
    back: torch.Tensor = oneway_transfer_error(pts2, pts1, H_inv_safe, True, eps)
    good_H_reshape: torch.Tensor = good_H.view(-1, 1).expand_as(there)

    out = there + back
    if not squared:
        out = (out + eps).sqrt()
    max_tensor = torch.full_like(out, max_num)
    return torch.where(good_H_reshape, out, max_tensor)


def line_segment_transfer_error_one_way(
    ls1: torch.Tensor, ls2: torch.Tensor, H: torch.Tensor, squared: bool = False
) -> torch.Tensor:
    r"""Return transfer error in image 2 for line segment correspondences given the homography matrix.

    Both endpoints of each image-1 segment are mapped into image 2 by ``H`` and scored against the line through
    the matching image-2 segment. See :cite:`homolines2001` for details.

    Convention:
        - Argument order and direction as :func:`oneway_transfer_error`.
        - Known defects: the image-2 line is not normalised, so the error is the mean perpendicular distance of
          the two mapped endpoints multiplied by the length of the image-2 segment, not a pixel distance
          (`#4867 <https://github.com/kornia/kornia/issues/4867>`_).

    Args:
        ls1: line segment correspondences from the left images with shape
          (B, N, 2, 2).
        ls2: line segment correspondences from the right images with shape
          (B, N, 2, 2).
        H: Homographies with shape :math:`(B, 3, 3)`.
        squared: if True (default is False), the squared error is returned.

    Returns:
        the computed error with shape :math:`(B, N)`.

    """
    KORNIA_CHECK_SHAPE(H, ["B", "3", "3"])
    KORNIA_CHECK_SHAPE(ls1, ["B", "N", "2", "2"])
    KORNIA_CHECK_SHAPE(ls2, ["B", "N", "2", "2"])
    B, N = ls1.shape[:2]
    ps1, pe1 = torch.chunk(ls1, dim=2, chunks=2)
    ps2, pe2 = torch.chunk(ls2, dim=2, chunks=2)
    ps2_h = convert_points_to_homogeneous(ps2)
    pe2_h = convert_points_to_homogeneous(pe2)
    ln2 = torch.linalg.cross(ps2_h, pe2_h, dim=3)
    ps1_in2 = convert_points_to_homogeneous(transform_points(H, ps1))
    pe1_in2 = convert_points_to_homogeneous(transform_points(H, pe1))
    er_st1 = (ln2 @ ps1_in2.transpose(-2, -1)).view(B, N).abs()
    er_end1 = (ln2 @ pe1_in2.transpose(-2, -1)).view(B, N).abs()
    error = 0.5 * (er_st1 + er_end1)
    if squared:
        error = error**2
    return error


def _line_segment_squared_distance_one_way(ls1: torch.Tensor, ls2: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Squared perpendicular distance, in pixels, of the mapped image-1 endpoints from the image-2 line.

    :func:`line_segment_transfer_error_one_way` carries the image-2 segment length (#4867); dividing by it gives the
    pixel distance. A zero-length image-2 segment defines no line, so its distance is infinite.
    """
    residual = line_segment_transfer_error_one_way(ls1, ls2, H)
    length = (ls2[..., 1, :] - ls2[..., 0, :]).norm(dim=-1)
    distance = residual / torch.where(length > 0, length, torch.ones_like(length))
    return torch.where(length > 0, distance.square(), torch.full_like(distance, float("inf")))


def _homography_rows(p1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
    """DLT rows ``(..., 2N, 9)`` of homogeneous ``p1`` ``(..., N, 3)`` with unit last coordinate and ``points2``.

    Two rows per correspondence, ``[0, -p1, y2 p1]`` and ``[p1, 0, -x2 p1]``, so that ``row . vec(H) = 0`` for
    ``points2 ~ H p1`` with ``H`` row-major; only the first two coordinates of ``points2`` are read.
    """
    # DIAPO 11: https://www.uio.no/studier/emner/matnat/its/nedlagte-emner/UNIK4690/v16/forelesninger/lecture_4_3-estimating-homographies-from-feature-correspondences.pdf  # noqa: E501
    zeros = torch.zeros_like(p1)
    ax = torch.cat([zeros, -p1, points2[..., 1:2] * p1], dim=-1)
    ay = torch.cat([p1, zeros, -points2[..., 0:1] * p1], dim=-1)
    return torch.stack([ax, ay], dim=-2).flatten(-3, -2)


def _homography_design_rows(points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
    """DLT rows ``(..., 2N, 9)`` of correspondences ``(..., N, 2)``, as :func:`_homography_rows`.

    Each entry is the product the rows used to be written out with coordinate by coordinate, in fewer kernels.
    """
    return _homography_rows(torch.cat([points1, torch.ones_like(points1[..., :1])], dim=-1), points2)


def _four_point_homography(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Homographies ``(B, 3, 3)`` of unit Frobenius norm through four normalized correspondences.

    ``x1`` is homogeneous ``(B, 4, 3)`` with unit last coordinate and ``x2`` ``(B, 4, 2)`` or homogeneous. The null
    vector of :func:`_homography_rows` comes from :func:`~kornia.geometry.solvers.homogeneous._null_space_lu`, in at
    least float32. Unlike :func:`find_homography_dlt` there is no per-sample normalization, gauge solve or
    ``H[2, 2] = 1`` scaling: RANSAC's sampler normalizes once per call and scores unit-norm models.
    """
    A = _homography_rows(x1, x2)
    h = _null_space_lu(A.to(torch.promote_types(A.dtype, torch.float32)))[..., 0]
    return (h * h.square().sum(-1, keepdim=True).rsqrt()).reshape(-1, 3, 3).to(x1.dtype)


def _transfer_errors(H: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, eps: float) -> torch.Tensor:
    """Squared one-way transfer errors ``(M, N)`` of homographies ``(M, 3, 3)`` on one set of correspondences.

    ``x1`` is homogeneous ``(N, 3)`` with unit last coordinate and ``x2`` ``(N, 2)``. ``H x1`` of every model is one
    ``(3M, 3) @ (3, N)`` product; the rest is :func:`oneway_transfer_error`'s formula, ``eps`` in the projective
    denominator included.
    """
    m, n = H.shape[0], x1.shape[0]
    projected = (H.reshape(3 * m, 3) @ x1.T).view(m, 3, n)
    w = projected[:, 2] + eps
    return (projected[:, 0] / w - x2[:, 0]).square() + (projected[:, 1] / w - x2[:, 1]).square()


def _oneway_transfer_error_shared_impl_(
    pts1: torch.Tensor, pts2: torch.Tensor, H: torch.Tensor, squared: bool, eps: float
) -> torch.Tensor:
    """One-way transfer errors of many homographies on one set of correspondences, by :func:`_transfer_errors`.

    ``pts1`` and ``pts2`` have leading dimensions of size 1; homogeneous points are dehomogenized. Half precision is
    computed in float32 and returned in the input dtype.
    """
    num_points = pts1.shape[-2]
    dtype = torch.promote_types(torch.promote_types(pts1.dtype, pts2.dtype), H.dtype)
    work = torch.promote_types(dtype, torch.float32)
    x1 = pts1.reshape(num_points, pts1.shape[-1]).to(work)
    x2 = pts2.reshape(num_points, pts2.shape[-1]).to(work)
    if x1.shape[-1] == 3:
        x1 = convert_points_from_homogeneous(x1)
    if x2.shape[-1] == 3:
        x2 = convert_points_from_homogeneous(x2)
    err2 = _transfer_errors(H.to(work), convert_points_to_homogeneous(x1), x2, eps)
    out = err2 if squared else (err2 + eps).sqrt()
    shape = torch.broadcast_shapes(pts1.shape[:-2], pts2.shape[:-2], H.shape[:-2])
    return out.reshape(*shape, num_points).to(dtype)


def _transfer_basis(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Per-correspondence monomials ``(9, 3N)`` of the one-way transfer error, for RANSAC's sampling loop only.

    ``vec(H) @ basis`` is ``[P_0 - u P_2 | P_1 - v P_2 | P_2]`` for ``P = H x1`` and ``x2 = (u, v)``, with ``x1``
    homogeneous ``(N, 3)``. Folding the subtraction into the product is about 10% faster than
    :func:`_transfer_errors` in RANSAC's loop on CPU (2048 models x 500 points: 6.6 against 7.2 ms) and 20-35 us per
    CUDA batch of 512-8192 models; it is accurate on normalized points, but in pixel units its float32 error was 2.5x
    that of :func:`oneway_transfer_error`.
    """
    n = x1.shape[0]
    basis = x1.new_zeros(9, 3 * n)
    basis[0:3, :n] = x1.T
    basis[6:9, :n] = -(x2[:, 0:1] * x1).T
    basis[3:6, n : 2 * n] = x1.T
    basis[6:9, n : 2 * n] = -(x2[:, 1:2] * x1).T
    basis[6:9, 2 * n :] = x1.T
    return basis


def _transfer_from_basis(H: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Squared one-way transfer errors ``(M, N)`` of homographies ``(M, 3, 3)`` from :func:`_transfer_basis`."""
    n = basis.shape[1] // 3
    out = H.flatten(1) @ basis
    return (out[:, :n].square() + out[:, n : 2 * n].square()) / out[:, 2 * n :].square()


def _refine_homography_lm(
    H: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    mask: Optional[torch.Tensor],
    loss: str,
    scale2: float,
    iters: int,
) -> torch.Tensor:
    """Levenberg-Marquardt on the one-way transfer error, batched over homographies ``(K, 3, 3)``.

    In the spirit of PoseLib's ``refine_homography``: steps are taken in the eight-dimensional orthogonal complement of
    the unit-norm ``vec(H)`` and renormalized. ``x1`` is homogeneous ``(N, 3)`` and ``x2`` ``(N, 2)``, normalized by
    the caller; the loss, ``mask`` and step acceptance are those of
    :func:`~kornia.geometry.epipolar.fundamental._refine_fundamental_lm`. For RANSAC, under ``torch.no_grad``.
    """
    K = H.shape[0]
    dtype, device = H.dtype, H.device
    cpu = K > 0 and device.type == "cpu" and not torch.is_grad_enabled()
    if cpu and K == 1 and mask is not None and mask.dtype == torch.bool:
        x1, x2, mask = x1[mask[0]], x2[mask[0]], None
    eye8 = torch.eye(8, dtype=dtype, device=device)
    eye9 = torch.eye(9, dtype=dtype, device=device)
    last = eye9[8]
    target = torch.cat([x2[:, 0], x2[:, 1]])
    h = H.flatten(1)
    h = h * h.square().sum(1, keepdim=True).rsqrt()
    damping = torch.full((K, 1, 1), 1e-3, dtype=dtype, device=device)

    def normal_equations(
        h: torch.Tensor, weights: Optional[torch.Tensor] = mask
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Householder reflection of h onto the last axis: its other eight columns span the tangent space.
        v = h + torch.where(h[:, 8:9] >= 0, 1.0, -1.0) * last
        v = v * v.square().sum(1, keepdim=True).rsqrt()
        tangent = (eye9 - 2 * v[:, :, None] * v[:, None, :])[:, :, :8]  # (K, 9, 8)
        stacked = torch.cat([h[:, :, None], tangent], 2).mT.reshape(
            h.shape[0], 27, 3
        )  # rows of H, then of each direction
        P = (stacked @ x1.T).reshape(h.shape[0], 9, 3, -1)  # (K, 9, 3, N): P = H x1, then its directional derivatives
        iz = 1.0 / P[:, 0, 2]
        uv = P[:, 0, :2] * iz[:, None]  # (K, 2, N)
        r = uv.flatten(1) - target  # (K, 2N): u residuals, then v residuals
        J = ((P[:, 1:, :2] - uv[:, None] * P[:, 1:, 2:3]) * iz[:, None, None]).flatten(2)  # (K, 8, 2N)
        r2 = r[:, : x1.shape[0]].square() + r[:, x1.shape[0] :].square()
        w, rho = _robust_loss(r2, loss, scale2)
        if weights is not None:
            w, rho = w * weights, rho * weights
        Jw = J * torch.cat([w, w], 1)[:, None]
        return torch.cat([Jw @ J.mT, Jw @ r[..., None]], 2), rho.sum(1), tangent

    system, cost, tangent = normal_equations(h)
    for iteration in range(iters):
        delta = -torch.linalg.solve_ex(system[..., :8] + damping * eye8, system[..., 8:])[0]
        if cpu and delta.flatten(1).norm(dim=1).max() < 1e-10:
            break
        h_new = h + (tangent @ delta)[..., 0]
        h_new = h_new * h_new.square().sum(1, keepdim=True).rsqrt()
        if cpu:
            projection = h_new.reshape(K, 3, 3) @ x1.T
            residual = projection[:, :2] / projection[:, 2:3] - x2.T
            r2 = residual.square().sum(1)
            rho = torch.log1p(r2 / scale2) if loss == "cauchy" else torch.fmin(r2, torch.full_like(r2[:1, :1], scale2))
            cost_new = (rho if mask is None else rho * mask).sum(1)
            accepted = cost_new < cost
            if accepted.all():
                h, cost = h_new, cost_new
                damping *= 0.1
                if iteration + 1 < iters:
                    system, _, tangent = normal_equations(h)
            elif accepted.any():
                h[accepted], cost[accepted] = h_new[accepted], cost_new[accepted]
                damping *= torch.where(accepted[:, None, None], 0.1, 10.0)
                if iteration + 1 < iters:
                    weights = None if mask is None else mask[accepted]
                    system_new, _, tangent_new = normal_equations(h[accepted], weights)
                    system[accepted], tangent[accepted] = system_new, tangent_new
            else:
                damping *= 10.0
            continue
        system_new, cost_new, tangent_new = normal_equations(h_new)
        accept = (cost_new < cost)[:, None, None]
        h, cost = torch.where(accept[:, :, 0], h_new, h), torch.where(accept[:, 0, 0], cost_new, cost)
        system, tangent = torch.where(accept, system_new, system), torch.where(accept, tangent_new, tangent)
        damping = damping * torch.where(accept, 0.1, 10.0)
    return h.reshape(K, 3, 3)


def find_homography_dlt(
    points1: torch.Tensor, points2: torch.Tensor, weights: Optional[torch.Tensor] = None, solver: str = "lu"
) -> torch.Tensor:
    r"""Compute the homography matrix using the DLT formulation.

    The weighted DLT system of four or more correspondences is solved with ``solver``.

    Convention:
        - ``H`` maps ``points1`` to ``points2``, ``points2 ~ H @ points1``, and is scaled so that
          ``H[2, 2] = 1`` by :func:`~kornia.geometry.epipolar.normalize_transformation`, which leaves it at its
          unnormalised scale when ``|H[2, 2]|`` is at most ``1e-8``; :ref:`two-view-conventions` compares this
          with OpenCV.
        - ``weights`` multiply each correspondence's squared algebraic residual: a weight of 0 removes the
          correspondence from the equations, and only relative weights matter.
        - ``solver="lu"`` and ``"svd"`` give the same homography on exact data, to roundoff scaled by the
          conditioning of the system; on noisy data they solve different least-squares problems and differ.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2)`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2)`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.
          Zero-weight points are excluded from the DLT equations and Hartley normalization.
        solver: variants: svd, lu.


    Returns:
        the computed homography matrix with shape :math:`(B, 3, 3)`.

    """
    device, dtype = _extract_device_dtype([points1, points2])
    A, transform1, transform2 = _homography_dlt_system(points1, points2, weights)
    return _homography_from_dlt_system(
        A, weights, transform1, safe_inverse_with_mask(transform2)[0], solver, device, dtype
    )


def _homography_dlt_system(
    points1: torch.Tensor, points2: torch.Tensor, weights: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the weighted-normalized DLT design matrix and its two normalizing transforms."""
    if points1.shape != points2.shape:
        raise AssertionError(points1.shape)
    if points1.shape[1] < 4:
        raise AssertionError(points1.shape)
    KORNIA_CHECK_SHAPE(points1, ["B", "N", "2"])
    KORNIA_CHECK_SHAPE(points2, ["B", "N", "2"])

    if weights is not None and weights.shape != points1.shape[:2]:
        raise AssertionError(weights.shape)
    points1_norm, transform1 = normalize_points(points1, weights=weights)
    points2_norm, transform2 = normalize_points(points2, weights=weights)

    A = _homography_design_rows(points1_norm, points2_norm)
    return A, transform1, transform2


def _homography_from_dlt_system(
    A: torch.Tensor,
    weights: Optional[torch.Tensor],
    transform1: torch.Tensor,
    transform2_inv: torch.Tensor,
    solver: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Solve the (weighted) DLT system of :func:`_homography_dlt_system` and denormalize the homography.

    The operand order matches the original single-call implementation.
    """
    eps: float = 1e-8
    num_points = A.shape[1] // 2
    if weights is None:
        # All points are equally important
        w_full = None
    else:
        # We should use provided weights
        if not (len(weights.shape) == 2 and weights.shape == (A.shape[0], num_points)):
            raise AssertionError(weights.shape)
        w_full = weights.repeat_interleave(2, dim=1).unsqueeze(1)

    # Only the minimal four-point LU path works from the design matrix itself (see below).
    # Every other case forms the normal equations in the exact operand order the pre-gauge
    # implementation used, so weighted results stay bit-identical to it.
    minimal_lu = solver == "lu" and num_points == 4
    if not minimal_lu:
        A = A.transpose(-2, -1) @ A if w_full is None else (A.transpose(-2, -1) * w_full) @ A

    if solver == "svd":
        try:
            _, _, V = _torch_svd_cast(A)
        except RuntimeError:
            warnings.warn("SVD did not converge", RuntimeWarning, stacklevel=1)
            return torch.empty((A.shape[0], 3, 3), device=device, dtype=dtype)
        H = V[..., -1].view(-1, 3, 3)
    elif solver == "lu":
        if not minimal_lu:
            B = torch.ones(A.shape[0], A.shape[1], device=device, dtype=dtype)
            sol, _, _ = safe_solve_with_mask(B, A)
        else:
            # A four-point sample gives eight equations for nine unknowns, so the normal matrix
            # is singular and LU-factoring it is what produced all-NaN homographies. Work from
            # the design matrix instead: its null vector comes from a pivoted LU factorization of
            # it, the largest component of that vector fixes the homogeneous gauge, and the
            # retained 8x8 system is solved for the rest. A fixed h33=1 gauge is invalid whenever
            # the bottom-right entry is zero. Five or more points keep the normal-equation
            # formulation above.
            Aw = A if w_full is None else A * w_full.transpose(-2, -1)
            # torch.linalg.qr on CUDA can spin forever on a design matrix that mixes NaN with the
            # structured zeros of the DLT rows (#4770). Hand QR and the solve finite entries only,
            # and report the affected batch elements as NaN below, as the CPU path already does.
            finite_entries = Aw.isfinite()
            finite = finite_entries.flatten(1).all(-1)
            Aw = torch.where(finite_entries, Aw, torch.zeros_like(Aw))
            gauge_dtype = torch.float64 if dtype == torch.float64 else torch.float32
            design = Aw.detach().to(gauge_dtype)
            # One batched LU factorization on every backend: torch.linalg.qr has no batched CUDA kernel
            # (one cusolver call per matrix, ~250 ms for a 2048-sample RANSAC batch), and a batched SVD or
            # QR costs about three times the LU on CPU. A zero-weight correspondence leaves the null vector
            # finite, since the unit triangular factor the basis is solved from is never singular.
            null = _null_space_lu(design)[..., 0]
            null = null / null.norm(dim=-1, keepdim=True)
            gauge = null.abs().argmax(dim=-1)
            retained = torch.arange(8, device=device).expand(A.shape[0], -1)
            retained = retained + (retained >= gauge[:, None]).to(retained.dtype)
            selected = Aw.gather(-1, retained[:, None].expand(-1, 8, -1))
            B = -Aw.gather(-1, gauge[:, None, None].expand(-1, 8, 1)).squeeze(-1)
            sol, _, valid = safe_solve_with_mask(B, selected)
            sol = sol.squeeze(-1)
            # A fully de-weighted correspondence zeroes two rows and leaves the retained system
            # singular. The null vector is finite and still satisfies the surviving equations,
            # so fall back to it rather than handing the caller a NaN homography.
            null = (null / null.gather(-1, gauge[:, None])).to(sol.dtype)
            sol = torch.where(valid[:, None], sol, null.gather(-1, retained))
            positions = torch.arange(9, device=device).expand(A.shape[0], -1)
            source = (positions - (positions > gauge[:, None]).to(positions.dtype)).clamp(0, 7)
            sol = torch.where(
                positions == gauge[:, None], torch.ones_like(positions, dtype=sol.dtype), sol.gather(-1, source)
            )
            sol = torch.where(finite[:, None], sol, torch.full_like(sol, float("nan")))
        H = sol.reshape(-1, 3, 3)
    else:
        raise NotImplementedError
    H = transform2_inv @ (H @ transform1)
    return normalize_transformation(H, eps)


def find_homography_dlt_iterated(
    points1: torch.Tensor, points2: torch.Tensor, weights: torch.Tensor, soft_inl_th: float = 3.0, n_iter: int = 5
) -> torch.Tensor:
    r"""Compute the homography matrix using the iteratively-reweighted least squares (IRWLS).

    Convention:
        - Direction and ``H[2, 2] = 1`` as :func:`find_homography_dlt`. Each solve after the first re-weights
          with the Gaussian kernel ``exp(-e**2 / (2 * soft_inl_th**2))`` of the symmetric transfer error ``e``
          (the root of the summed squared forward and backward transfer errors), so ``soft_inl_th`` is a
          standard deviation in pixels: a correspondence with ``e = soft_inl_th`` keeps weight ``exp(-1/2)``.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2)`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2)`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.
          Used for the first iteration of the IRWLS.
        soft_inl_th: standard deviation, in pixels, of the Gaussian re-weighting kernel given above.
        n_iter: number of solves, including the initial one.

    Returns:
        the computed homography matrix with shape :math:`(B, 3, 3)`.

    """
    device, dtype = _extract_device_dtype([points1, points2])
    # Weighted Hartley normalization changes with each set of IRLS weights.
    A, transform1, transform2 = _homography_dlt_system(points1, points2, weights)
    transform2_inv = safe_inverse_with_mask(transform2)[0]
    H: torch.Tensor = _homography_from_dlt_system(A, weights, transform1, transform2_inv, "lu", device, dtype)
    for _ in range(n_iter - 1):
        squared_errors: torch.Tensor = symmetric_transfer_error(points1, points2, H, True)
        weights_new: torch.Tensor = torch.exp(-squared_errors / (2.0 * (soft_inl_th**2)))
        A, transform1, transform2 = _homography_dlt_system(points1, points2, weights_new)
        transform2_inv = safe_inverse_with_mask(transform2)[0]
        H = _homography_from_dlt_system(A, weights_new, transform1, transform2_inv, "lu", device, dtype)
    return H


def sample_is_valid_for_homography(points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
    """Implement oriented constraint check from :cite:`Marquez-Neila2015`.

    Analogous to https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/usac/degeneracy.cpp#L88

    Convention:
        - :class:`~kornia.geometry.ransac.RANSAC` uses it to discard minimal samples for ``"homography"``. It
          checks only that the four triples of the first four points keep their orientation across the two
          views, so a mirror-image sample is rejected. A triple that is collinear in both views, for instance
          through a repeated point, counts as kept, so collinearity alone does not reject a sample.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2)`; only the first four
          points are used.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2)`; only the first four
          points are used.

    Returns:
        Mask with the minimal sample is good for homography estimation :math:`(B)`.

    """
    if points1.shape != points2.shape:
        raise AssertionError(points1.shape)

    # Triples to test: (0,1,2), (0,1,3), (0,2,3), (1,2,3), gathered by slicing, shape (B, 4, 2). Index tensors
    # would be copied to the device on every call.
    def _triples(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p0, p1, p2, p3 = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
        return torch.stack([p0, p0, p0, p1], 1), torch.stack([p1, p1, p2, p2], 1), torch.stack([p2, p3, p3, p3], 1)

    p1_i, p1_j, p1_k = _triples(points1)
    p2_i, p2_j, p2_k = _triples(points2)

    # 2D orientation (signed area) for each triple:
    # orient(a,b,c) = cross2d(b-a, c-a) = (bx-ax)*(cy-ay) - (by-ay)*(cx-ax)
    def _orient(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        ab = b - a
        ac = c - a
        return ab[..., 0] * ac[..., 1] - ab[..., 1] * ac[..., 0]  # shape (B, 4)

    left_sign = torch.sign(_orient(p1_i, p1_j, p1_k))
    right_sign = torch.sign(_orient(p2_i, p2_j, p2_k))

    # Valid if all four orientation signs match across views
    return (left_sign == right_sign).all(dim=1)


def find_homography_lines_dlt(
    ls1: torch.Tensor, ls2: torch.Tensor, weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute the homography matrix from line segment correspondences with a DLT formulation.

    See :cite:`homolines2001` for details.

    Convention:
        - ``H`` maps image-1 points to image-2 points, as in :func:`find_homography_dlt`. Each segment is a
          ``[start, end]`` pair of ``(x, y)`` points, and ``weights`` has one entry per segment.
        - Both endpoints of image-1 segment ``i`` are constrained to lie, after mapping by ``H``, on the line
          through image-2 segment ``i``, so the endpoints need not be point correspondences.

    Args:
        ls1: A set of line segments in the first image with a tensor shape :math:`(B, N, 2, 2)`, or
          :math:`(N, 2, 2)`, which is treated as :math:`B = 1`.
        ls2: A set of line segments in the second image with a tensor shape :math:`(B, N, 2, 2)`, or
          :math:`(N, 2, 2)`, which is treated as :math:`B = 1`.
        weights: Tensor containing the weights per segment with a shape of :math:`(B, N)`.
          Zero-weight segments are excluded from Hartley normalization.

    Returns:
        the computed homography matrix with shape :math:`(B, 3, 3)`.

    """
    if len(ls1.shape) == 3:
        ls1 = ls1[None]
    if len(ls2.shape) == 3:
        ls2 = ls2[None]
    KORNIA_CHECK_SHAPE(ls1, ["B", "N", "2", "2"])
    KORNIA_CHECK_SHAPE(ls2, ["B", "N", "2", "2"])
    BS, N = ls1.shape[:2]
    device, dtype = _extract_device_dtype([ls1, ls2])

    points1 = ls1.reshape(BS, 2 * N, 2)
    points2 = ls2.reshape(BS, 2 * N, 2)

    if weights is not None and weights.shape != ls1.shape[:2]:
        raise AssertionError(weights.shape)
    endpoint_weights = weights.repeat_interleave(2, dim=1) if weights is not None else None
    points1_norm, transform1 = normalize_points(points1, weights=endpoint_weights)
    points2_norm, transform2 = normalize_points(points2, weights=endpoint_weights)
    # Pair each segment's own endpoints: the flattened points are [start_0, end_0, start_1, end_1, ...].
    segments1_norm = points1_norm.reshape(BS, N, 2, 2)
    segments2_norm = points2_norm.reshape(BS, N, 2, 2)
    lst1, le1 = segments1_norm[:, :, 0], segments1_norm[:, :, 1]
    lst2, le2 = segments2_norm[:, :, 0], segments2_norm[:, :, 1]

    xs1, ys1 = torch.chunk(lst1, dim=-1, chunks=2)  # BxNx1
    xs2, ys2 = torch.chunk(lst2, dim=-1, chunks=2)  # BxNx1
    xe1, ye1 = torch.chunk(le1, dim=-1, chunks=2)  # BxNx1
    xe2, ye2 = torch.chunk(le2, dim=-1, chunks=2)  # BxNx1

    A = ys2 - ye2
    B = xe2 - xs2
    C = xs2 * ye2 - xe2 * ys2

    eps: float = 1e-8

    # http://diis.unizar.es/biblioteca/00/09/000902.pdf
    ax = torch.cat([A * xs1, A * ys1, A, B * xs1, B * ys1, B, C * xs1, C * ys1, C], dim=-1)
    ay = torch.cat([A * xe1, A * ye1, A, B * xe1, B * ye1, B, C * xe1, C * ye1, C], dim=-1)
    A = torch.cat((ax, ay), dim=-1).reshape(ax.shape[0], -1, ax.shape[-1])

    if weights is None:
        # All points are equally important
        A = A.transpose(-2, -1) @ A
    else:
        # We should use provided weights
        if not ((len(weights.shape) == 2) and (weights.shape == ls1.shape[:2])):
            raise AssertionError(weights.shape)
        w_diag = torch.diag_embed(weights.unsqueeze(dim=-1).repeat(1, 1, 2).reshape(weights.shape[0], -1))
        A = A.transpose(-2, -1) @ w_diag @ A

    try:
        _, _, V = _torch_svd_cast(A)
    except RuntimeError:
        warnings.warn("SVD did not converge", RuntimeWarning, stacklevel=1)
        return torch.empty((points1_norm.size(0), 3, 3), device=device, dtype=dtype)

    H = V[..., -1].view(-1, 3, 3)
    H = safe_inverse_with_mask(transform2)[0] @ (H @ transform1)
    return normalize_transformation(H, eps)


def find_homography_lines_dlt_iterated(
    ls1: torch.Tensor, ls2: torch.Tensor, weights: torch.Tensor, soft_inl_th: float = 4.0, n_iter: int = 5
) -> torch.Tensor:
    r"""Compute the homography matrix using the iteratively-reweighted least squares (IRWLS) from line segments.

    Convention:
        - As :func:`find_homography_dlt_iterated`, with :func:`find_homography_lines_dlt` as the solver and, as
          ``e`` in the Gaussian kernel, the perpendicular distance in pixels of the mapped image-1 endpoints from
          the image-2 line: the residual of :func:`line_segment_transfer_error_one_way` divided by the image-2
          segment length it carries. A zero-length image-2 segment gets weight zero.
        - Known defect: the length-scaled residual of :func:`line_segment_transfer_error_one_way` still applies
          (`#4867 <https://github.com/kornia/kornia/issues/4867>`_).

    Args:
        ls1: A set of line segments in the first image with a tensor shape :math:`(B, N, 2, 2)`.
        ls2: A set of line segments in the second image with a tensor shape :math:`(B, N, 2, 2)`.
        weights: Tensor containing the weights per segment with a shape of :math:`(B, N)`.
          Used for the first iteration of the IRWLS.
        soft_inl_th: standard deviation, in pixels, of the Gaussian re-weighting kernel of
          :func:`find_homography_dlt_iterated`.
        n_iter: number of solves, including the initial one.

    Returns:
        the computed homography matrix with shape :math:`(B, 3, 3)`.

    """
    H: torch.Tensor = find_homography_lines_dlt(ls1, ls2, weights)
    for _ in range(n_iter - 1):
        squared_distances: torch.Tensor = _line_segment_squared_distance_one_way(ls1, ls2, H)
        weights_new: torch.Tensor = torch.exp(-squared_distances / (2.0 * (soft_inl_th**2)))
        H = find_homography_lines_dlt(ls1, ls2, weights_new)
    return H
