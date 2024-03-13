import warnings
from typing import Optional, Tuple

import torch

from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.utils import _extract_device_dtype, safe_inverse_with_mask, safe_solve_with_mask
from kornia.utils.helpers import _torch_svd_cast

from .conversions import convert_points_from_homogeneous, convert_points_to_homogeneous
from .epipolar import normalize_points
from .linalg import transform_points

TupleTensor = Tuple[Tensor, Tensor]


def oneway_transfer_error(pts1: Tensor, pts2: Tensor, H: Tensor, squared: bool = True, eps: float = 1e-8) -> Tensor:
    r"""Return transfer error in image 2 for correspondences given the homography matrix.

    Args:
        pts1: correspondences from the left images with shape
          (B, N, 2 or 3). If they are homogeneous, converted automatically.
        pts2: correspondences from the right images with shape
          (B, N, 2 or 3). If they are homogeneous, converted automatically.
        H: Homographies with shape :math:`(B, 3, 3)`.
        squared: if True (default), the squared distance is returned.
        eps: Small constant for safe sqrt.

    Returns:
        the computed distance with shape :math:`(B, N)`.
    """
    KORNIA_CHECK_SHAPE(H, ["B", "3", "3"])

    if pts1.size(-1) == 3:
        pts1 = convert_points_from_homogeneous(pts1)

    if pts2.size(-1) == 3:
        pts2 = convert_points_from_homogeneous(pts2)

    # From Hartley and Zisserman, Error in one image (4.6)
    # dist = \sum_{i} ( d(x', Hx)**2)
    pts1_in_2: Tensor = transform_points(H, pts1)
    error_squared: Tensor = (pts1_in_2 - pts2).pow(2).sum(dim=-1)
    if squared:
        return error_squared
    return (error_squared + eps).sqrt()


def symmetric_transfer_error(pts1: Tensor, pts2: Tensor, H: Tensor, squared: bool = True, eps: float = 1e-8) -> Tensor:
    r"""Return Symmetric transfer error for correspondences given the homography matrix.

    Args:
        pts1: correspondences from the left images with shape
          (B, N, 2 or 3). If they are homogeneous, converted automatically.
        pts2: correspondences from the right images with shape
          (B, N, 2 or 3). If they are homogeneous, converted automatically.
        H: Homographies with shape :math:`(B, 3, 3)`.
        squared: if True (default), the squared distance is returned.
        eps: Small constant for safe sqrt.

    Returns:
        the computed distance with shape :math:`(B, N)`.
    """
    KORNIA_CHECK_SHAPE(H, ["B", "3", "3"])
    if pts1.size(-1) == 3:
        pts1 = convert_points_from_homogeneous(pts1)

    if pts2.size(-1) == 3:
        pts2 = convert_points_from_homogeneous(pts2)

    max_num = torch.finfo(pts1.dtype).max
    # From Hartley and Zisserman, Symmetric transfer error (4.7)
    # dist = \sum_{i} (d(x, H^-1 x')**2 + d(x', Hx)**2)
    H_inv, good_H = safe_inverse_with_mask(H)

    there: Tensor = oneway_transfer_error(pts1, pts2, H, True, eps)
    back: Tensor = oneway_transfer_error(pts2, pts1, H_inv, True, eps)
    good_H_reshape: Tensor = good_H.view(-1, 1).expand_as(there)
    out = (there + back) * good_H_reshape.to(there.dtype) + max_num * (~good_H_reshape).to(there.dtype)
    if squared:
        return out
    return (out + eps).sqrt()


def line_segment_transfer_error_one_way(ls1: Tensor, ls2: Tensor, H: Tensor, squared: bool = False) -> Tensor:
    r"""Return transfer error in image 2 for line segment correspondences given the homography matrix.

    Line segment end points are reprojected into image 2, and point-to-line error is calculated w.r.t. line,
    induced by line segment in image 2. See :cite:`homolines2001` for details.

    Args:
        ls1: line segment correspondences from the left images with shape
          (B, N, 2, 2).
        ls2: line segment correspondences from the right images with shape
          (B, N, 2, 2).
        H: Homographies with shape :math:`(B, 3, 3)`.
        squared: if True (default is False), the squared distance is returned.

    Returns:
        the computed distance with shape :math:`(B, N)`.
    """
    KORNIA_CHECK_SHAPE(H, ["B", "3", "3"])
    KORNIA_CHECK_SHAPE(ls1, ["B", "N", "2", "2"])
    KORNIA_CHECK_SHAPE(ls2, ["B", "N", "2", "2"])
    B, N = ls1.shape[:2]
    ps1, pe1 = torch.chunk(ls1, dim=2, chunks=2)
    ps2, pe2 = torch.chunk(ls2, dim=2, chunks=2)
    ps2_h = convert_points_to_homogeneous(ps2)
    pe2_h = convert_points_to_homogeneous(pe2)
    ln2 = ps2_h.cross(pe2_h, dim=3)
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

    The linear system is solved by using the Weighted Least Squares Solution for the 4 Points algorithm.

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
        A = A.transpose(-2, -1) @ A
    else:
        # We should use provided weights
        if not (len(weights.shape) == 2 and weights.shape == points1.shape[:2]):
            raise AssertionError(weights.shape)
        w_diag = torch.diag_embed(weights.unsqueeze(dim=-1).repeat(1, 1, 2).reshape(weights.shape[0], -1))
        A = A.transpose(-2, -1) @ w_diag @ A

    if solver == "svd":
        try:
            _, _, V = _torch_svd_cast(A)
        except RuntimeError:
            warnings.warn("SVD did not converge", RuntimeWarning)
            return torch.empty((points1_norm.size(0), 3, 3), device=device, dtype=dtype)
        H = V[..., -1].view(-1, 3, 3)
    elif solver == "lu":
        B = torch.ones(A.shape[0], A.shape[1], device=device, dtype=dtype)
        sol, _, _ = safe_solve_with_mask(B, A)
        H = sol.reshape(-1, 3, 3)
    else:
        raise NotImplementedError
    H = safe_inverse_with_mask(transform2)[0] @ (H @ transform1)
    H_norm = H / (H[..., -1:, -1:] + eps)
    return H_norm


def find_homography_dlt_iterated(
    points1: Tensor, points2: Tensor, weights: Tensor, soft_inl_th: float = 3.0, n_iter: int = 5
) -> Tensor:
    r"""Compute the homography matrix using the iteratively-reweighted least squares (IRWLS).

    The linear system is solved by using the Reweighted Least Squares Solution for the 4 Points algorithm.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2)`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2)`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.
          Used for the first iteration of the IRWLS.
        soft_inl_th: Soft inlier threshold used for weight calculation.
        n_iter: number of iterations.

    Returns:
        the computed homography matrix with shape :math:`(B, 3, 3)`.
    """
    H: Tensor = find_homography_dlt(points1, points2, weights)
    for _ in range(n_iter - 1):
        errors: Tensor = symmetric_transfer_error(points1, points2, H, False)
        weights_new: Tensor = torch.exp(-errors / (2.0 * (soft_inl_th**2)))
        H = find_homography_dlt(points1, points2, weights_new)
    return H


def sample_is_valid_for_homography(points1: Tensor, points2: Tensor) -> Tensor:
    """Function, which implements oriented constraint check from :cite:`Marquez-Neila2015`.

    Analogous to https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/usac/degeneracy.cpp#L88

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, 4, 2)`.
        points2: A set of points in the second image with a tensor shape :math:`(B, 4, 2)`.

    Returns:
        Mask with the minimal sample is good for homography estimation:math:`(B, 3, 3)`.
    """
    if points1.shape != points2.shape:
        raise AssertionError(points1.shape)
    KORNIA_CHECK_SHAPE(points1, ["B", "4", "2"])
    KORNIA_CHECK_SHAPE(points2, ["B", "4", "2"])
    device = points1.device
    idx_perm = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=torch.long, device=device)
    points_src_h = convert_points_to_homogeneous(points1)
    points_dst_h = convert_points_to_homogeneous(points2)

    src_perm = points_src_h[:, idx_perm]
    dst_perm = points_dst_h[:, idx_perm]
    left_sign = (
        torch.cross(src_perm[..., 1:2, :], src_perm[..., 2:3, :]) @ src_perm[..., 0:1, :].permute(0, 1, 3, 2)
    ).sign()
    right_sign = (
        torch.cross(dst_perm[..., 1:2, :], dst_perm[..., 2:3, :]) @ dst_perm[..., 0:1, :].permute(0, 1, 3, 2)
    ).sign()
    sample_is_valid = (left_sign == right_sign).view(-1, 4).min(dim=1)[0]
    return sample_is_valid


def find_homography_lines_dlt(ls1: Tensor, ls2: Tensor, weights: Optional[Tensor] = None) -> Tensor:
    """Compute the homography matrix using the DLT formulation for line correspondences.

    See :cite:`homolines2001` for details.

    The linear system is solved by using the Weighted Least Squares Solution for the 4 Line correspondences algorithm.

    Args:
        ls1: A set of line segments in the first image with a tensor shape :math:`(B, N, 2, 2)`.
        ls2: A set of line segments in the second image with a tensor shape :math:`(B, N, 2, 2)`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.

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
        warnings.warn("SVD did not converge", RuntimeWarning)
        return torch.empty((points1_norm.size(0), 3, 3), device=device, dtype=dtype)

    H = V[..., -1].view(-1, 3, 3)
    H = safe_inverse_with_mask(transform2)[0] @ (H @ transform1)
    H_norm = H / (H[..., -1:, -1:] + eps)
    return H_norm


def find_homography_lines_dlt_iterated(
    ls1: Tensor, ls2: Tensor, weights: Tensor, soft_inl_th: float = 4.0, n_iter: int = 5
) -> Tensor:
    r"""Compute the homography matrix using the iteratively-reweighted least squares (IRWLS) from line segments. The
    linear system is solved by using the Reweighted Least Squares Solution for the 4 line segments algorithm.

    Args:
        ls1: A set of line segments in the first image with a tensor shape :math:`(B, N, 2, 2)`.
        ls2: A set of line segments in the second image with a tensor shape :math:`(B, N, 2, 2)`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.
          Used for the first iteration of the IRWLS.
        soft_inl_th: Soft inlier threshold used for weight calculation.
        n_iter: number of iterations.

    Returns:
        the computed homography matrix with shape :math:`(B, 3, 3)`.
    """
    H: Tensor = find_homography_lines_dlt(ls1, ls2, weights)
    for _ in range(n_iter - 1):
        errors: Tensor = line_segment_transfer_error_one_way(ls1, ls2, H, False)
        weights_new: Tensor = torch.exp(-errors / (2.0 * (soft_inl_th**2)))
        H = find_homography_lines_dlt(ls1, ls2, weights_new)
    return H
