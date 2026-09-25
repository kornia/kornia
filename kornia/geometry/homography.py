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
from kornia.geometry.linalg import transform_points

TupleTensor = Tuple[torch.Tensor, torch.Tensor]


def oneway_transfer_error(
    pts1: torch.Tensor, pts2: torch.Tensor, H: torch.Tensor, squared: bool = True, eps: float = 1e-8
) -> torch.Tensor:
    r"""Return transfer error in image 2 for correspondences given the homography matrix.

    Convention:
        - ``oneway_transfer_error(pts1, pts2, H)`` measures in image 2, between ``H`` applied to ``pts1`` and
          ``pts2``.
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


def find_homography_dlt(
    points1: torch.Tensor, points2: torch.Tensor, weights: Optional[torch.Tensor] = None, solver: str = "lu"
) -> torch.Tensor:
    r"""Compute the homography matrix using the DLT formulation.

    The weighted DLT system of four or more correspondences is solved with ``solver``.

    Convention:
        - ``H`` maps ``points1`` to ``points2``, ``points2 ~ H @ points1``, and is scaled so that
          ``H[2, 2] = 1``; :ref:`two-view-conventions` compares this with OpenCV.
        - ``weights`` multiply each correspondence's squared algebraic residual: a weight of 0 removes the
          correspondence from the equations, and only relative weights matter.
        - ``solver="lu"`` and ``"svd"`` give the same homography on exact data, to roundoff scaled by the
          conditioning of the system; on noisy data they solve different least-squares problems and differ.
        - Known defects: a zero-weight correspondence still enters the point normalisation, so on noisy data it
          moves the result (`#4890 <https://github.com/kornia/kornia/issues/4890>`_); and ``H`` is divided by
          ``H[2, 2] + 1e-8``, so ``H[2, 2]`` is not exactly 1, and it can be far from 1 when the true ``H[2, 2]``
          is small (`#4874 <https://github.com/kornia/kornia/issues/4874>`_).

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2)`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2)`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.
        solver: variants: svd, lu.


    Returns:
        the computed homography matrix with shape :math:`(B, 3, 3)`.

    """
    if points1.shape != points2.shape:
        raise AssertionError(points1.shape)
    if points1.shape[1] < 4:
        raise AssertionError(points1.shape)
    KORNIA_CHECK_SHAPE(points1, ["B", "N", "2"])
    KORNIA_CHECK_SHAPE(points2, ["B", "N", "2"])

    device, dtype = _extract_device_dtype([points1, points2])

    eps: float = 1e-8
    points1_norm, transform1 = normalize_points(points1)
    points2_norm, transform2 = normalize_points(points2)

    x1, y1 = torch.chunk(points1_norm, dim=-1, chunks=2)  # BxNx1
    x2, y2 = torch.chunk(points2_norm, dim=-1, chunks=2)  # BxNx1
    ones, zeros = torch.ones_like(x1), torch.zeros_like(x1)

    # DIAPO 11: https://www.uio.no/studier/emner/matnat/its/nedlagte-emner/UNIK4690/v16/forelesninger/lecture_4_3-estimating-homographies-from-feature-correspondences.pdf  # noqa: E501
    ax = torch.cat([zeros, zeros, zeros, -x1, -y1, -ones, y2 * x1, y2 * y1, y2], dim=-1)
    ay = torch.cat([x1, y1, ones, zeros, zeros, zeros, -x2 * x1, -x2 * y1, -x2], dim=-1)
    A = torch.cat((ax, ay), dim=-1).reshape(ax.shape[0], -1, ax.shape[-1])

    if weights is None:
        # All points are equally important
        w_full = None
    else:
        # We should use provided weights
        if not (len(weights.shape) == 2 and weights.shape == points1.shape[:2]):
            raise AssertionError(weights.shape)
        w_full = weights.repeat_interleave(2, dim=1).unsqueeze(1)

    # Only the minimal four-point LU path works from the design matrix itself (see below).
    # Every other case forms the normal equations in the exact operand order the pre-gauge
    # implementation used, so weighted results stay bit-identical to it.
    minimal_lu = solver == "lu" and points1.shape[1] == 4
    if not minimal_lu:
        A = A.transpose(-2, -1) @ A if w_full is None else (A.transpose(-2, -1) * w_full) @ A

    if solver == "svd":
        try:
            _, _, V = _torch_svd_cast(A)
        except RuntimeError:
            warnings.warn("SVD did not converge", RuntimeWarning, stacklevel=1)
            return torch.empty((points1_norm.size(0), 3, 3), device=device, dtype=dtype)
        H = V[..., -1].view(-1, 3, 3)
    elif solver == "lu":
        if not minimal_lu:
            B = torch.ones(A.shape[0], A.shape[1], device=device, dtype=dtype)
            sol, _, _ = safe_solve_with_mask(B, A)
        else:
            # A four-point sample gives eight equations for nine unknowns, so the normal matrix
            # is singular and LU-factoring it is what produced all-NaN homographies. Work from
            # the design matrix instead: its null vector comes from a complete QR, the largest
            # component of that vector fixes the homogeneous gauge, and the retained 8x8 system
            # is solved for the rest. A fixed h33=1 gauge is invalid whenever the bottom-right
            # entry is zero. Five or more points keep the normal-equation formulation above.
            Aw = A if w_full is None else A * w_full.transpose(-2, -1)
            # torch.linalg.qr on CUDA can spin forever on a design matrix that mixes NaN with the
            # structured zeros of the DLT rows (#4770). Hand QR and the solve finite entries only,
            # and report the affected batch elements as NaN below, as the CPU path already does.
            finite_entries = Aw.isfinite()
            finite = finite_entries.flatten(1).all(-1)
            Aw = torch.where(finite_entries, Aw, torch.zeros_like(Aw))
            gauge_dtype = torch.float64 if dtype == torch.float64 else torch.float32
            Q, _ = torch.linalg.qr(Aw.detach().to(gauge_dtype).transpose(-2, -1), mode="complete")
            null = Q[..., -1]
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
    H = safe_inverse_with_mask(transform2)[0] @ (H @ transform1)
    return normalize_transformation(H, eps)


def find_homography_dlt_iterated(
    points1: torch.Tensor, points2: torch.Tensor, weights: torch.Tensor, soft_inl_th: float = 3.0, n_iter: int = 5
) -> torch.Tensor:
    r"""Compute the homography matrix using the iteratively-reweighted least squares (IRWLS).

    Convention:
        - Direction and ``H[2, 2] = 1`` as :func:`find_homography_dlt`. Each solve after the first re-weights
          with ``exp(-e / (2 * soft_inl_th**2))`` of the unsquared symmetric transfer error ``e``.
        - Known defects: the exponent is linear, not quadratic, in ``e``, so ``soft_inl_th`` is not a pixel
          standard deviation (`#4870 <https://github.com/kornia/kornia/issues/4870>`_); those of
          :func:`find_homography_dlt` apply (`#4874 <https://github.com/kornia/kornia/issues/4874>`_,
          `#4890 <https://github.com/kornia/kornia/issues/4890>`_).

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2)`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2)`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.
          Used for the first iteration of the IRWLS.
        soft_inl_th: scale in the re-weighting kernel given above.
        n_iter: number of solves, including the initial one.

    Returns:
        the computed homography matrix with shape :math:`(B, 3, 3)`.

    """
    H: torch.Tensor = find_homography_dlt(points1, points2, weights)
    for _ in range(n_iter - 1):
        errors: torch.Tensor = symmetric_transfer_error(points1, points2, H, False)
        weights_new: torch.Tensor = torch.exp(-errors / (2.0 * (soft_inl_th**2)))
        H = find_homography_dlt(points1, points2, weights_new)
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
    # Triples to test: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
    idx_i = torch.tensor([0, 0, 0, 1], device=points1.device)
    J = torch.tensor([1, 1, 2, 2], device=points1.device)
    K = torch.tensor([2, 3, 3, 3], device=points1.device)

    # Gather the triples for both sets: shape (B, 4, 2)
    p1_i, p1_j, p1_k = points1[:, idx_i], points1[:, J], points1[:, K]
    p2_i, p2_j, p2_k = points2[:, idx_i], points2[:, J], points2[:, K]

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
        - Known defects: each segment's equations are built from endpoints of two different segments, not from
          its own start and end, so the estimate is correct only when the endpoints are themselves point
          correspondences, and a zero weight does not remove its segment
          (`#4866 <https://github.com/kornia/kornia/issues/4866>`_); the endpoints of a zero-weight segment still
          enter the point normalisation (`#4890 <https://github.com/kornia/kornia/issues/4890>`_); and the
          ``H[2, 2]`` scaling of :func:`find_homography_dlt` applies
          (`#4874 <https://github.com/kornia/kornia/issues/4874>`_).

    Args:
        ls1: A set of line segments in the first image with a tensor shape :math:`(B, N, 2, 2)`, or
          :math:`(N, 2, 2)`, which is treated as :math:`B = 1`.
        ls2: A set of line segments in the second image with a tensor shape :math:`(B, N, 2, 2)`, or
          :math:`(N, 2, 2)`, which is treated as :math:`B = 1`.
        weights: Tensor containing the weights per segment with a shape of :math:`(B, N)`.

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

    points1_norm, transform1 = normalize_points(points1)
    points2_norm, transform2 = normalize_points(points2)
    lst1, le1 = torch.chunk(points1_norm, dim=1, chunks=2)
    lst2, le2 = torch.chunk(points2_norm, dim=1, chunks=2)

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
        - As :func:`find_homography_dlt_iterated`, with :func:`find_homography_lines_dlt` as the solver and the
          unsquared error of :func:`line_segment_transfer_error_one_way` as ``e``.
        - Known defects: those of the three functions apply
          (`#4866 <https://github.com/kornia/kornia/issues/4866>`_,
          `#4867 <https://github.com/kornia/kornia/issues/4867>`_,
          `#4870 <https://github.com/kornia/kornia/issues/4870>`_,
          `#4874 <https://github.com/kornia/kornia/issues/4874>`_,
          `#4890 <https://github.com/kornia/kornia/issues/4890>`_).

    Args:
        ls1: A set of line segments in the first image with a tensor shape :math:`(B, N, 2, 2)`.
        ls2: A set of line segments in the second image with a tensor shape :math:`(B, N, 2, 2)`.
        weights: Tensor containing the weights per segment with a shape of :math:`(B, N)`.
          Used for the first iteration of the IRWLS.
        soft_inl_th: scale in the re-weighting kernel of :func:`find_homography_dlt_iterated`.
        n_iter: number of solves, including the initial one.

    Returns:
        the computed homography matrix with shape :math:`(B, 3, 3)`.

    """
    H: torch.Tensor = find_homography_lines_dlt(ls1, ls2, weights)
    for _ in range(n_iter - 1):
        errors: torch.Tensor = line_segment_transfer_error_one_way(ls1, ls2, H, False)
        weights_new: torch.Tensor = torch.exp(-errors / (2.0 * (soft_inl_th**2)))
        H = find_homography_lines_dlt(ls1, ls2, weights_new)
    return H
