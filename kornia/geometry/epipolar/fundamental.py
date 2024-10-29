"""Module containing the functionalities for computing the Fundamental Matrix."""

from typing import Literal, Optional, Tuple

import torch

from kornia.core import Tensor, concatenate, ones_like, stack, where, zeros
from kornia.core.check import KORNIA_CHECK_SAME_SHAPE, KORNIA_CHECK_SHAPE
from kornia.geometry.conversions import convert_points_from_homogeneous, convert_points_to_homogeneous
from kornia.geometry.linalg import transform_points
from kornia.geometry.solvers import solve_cubic
from kornia.utils.helpers import _torch_svd_cast, safe_inverse_with_mask


def normalize_points(points: Tensor, eps: float = 1e-8) -> Tuple[Tensor, Tensor]:
    r"""Normalizes points (isotropic).

    Computes the transformation matrix such that the two principal moments of the set of points
    are equal to unity, forming an approximately symmetric circular cloud of points of radius 1
    about the origin. Reference: Hartley/Zisserman 4.4.4 pag.107

    This operation is an essential step before applying the DLT algorithm in order to consider
    the result as optimal.

    Args:
       points: Tensor containing the points to be normalized with shape :math:`(B, N, 2)`.
       eps: epsilon value to avoid numerical instabilities.

    Returns:
       tuple containing the normalized points in the shape :math:`(B, N, 2)` and the transformation matrix
       in the shape :math:`(B, 3, 3)`.
    """
    if len(points.shape) != 3:
        raise AssertionError(points.shape)
    if points.shape[-1] != 2:
        raise AssertionError(points.shape)

    x_mean = torch.mean(points, dim=1, keepdim=True)  # Bx1x2

    scale = (points - x_mean).norm(dim=-1, p=2).mean(dim=-1)  # B
    scale = torch.sqrt(torch.tensor(2.0)) / (scale + eps)  # B

    ones, zeros = ones_like(scale), torch.zeros_like(scale)

    transform = stack(
        [scale, zeros, -scale * x_mean[..., 0, 0], zeros, scale, -scale * x_mean[..., 0, 1], zeros, zeros, ones], dim=-1
    )  # Bx9

    transform = transform.view(-1, 3, 3)  # Bx3x3
    points_norm = transform_points(transform, points)  # BxNx2

    return points_norm, transform


def normalize_transformation(M: Tensor, eps: float = 1e-8) -> Tensor:
    r"""Normalize a given transformation matrix.

    The function trakes the transformation matrix and normalize so that the value in
    the last row and column is one.

    Args:
        M: The transformation to be normalized of any shape with a minimum size of 2x2.
        eps: small value to avoid unstabilities during the backpropagation.

    Returns:
        the normalized transformation matrix with same shape as the input.
    """
    if len(M.shape) < 2:
        raise AssertionError(M.shape)
    norm_val: Tensor = M[..., -1:, -1:]
    return where(norm_val.abs() > eps, M / (norm_val + eps), M)


# Reference: Adapted from the 'run_7point' function in opencv
# https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/fundam.cpp
def run_7point(points1: Tensor, points2: Tensor) -> Tensor:
    r"""Compute the fundamental matrix using the 7-point algorithm.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2)`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2)`.

    Returns:
        the computed fundamental matrix with shape :math:`(B, 3*m, 3), Valid values of m are 1, 2 or 3`
    """
    KORNIA_CHECK_SHAPE(points1, ["B", "7", "2"])
    KORNIA_CHECK_SHAPE(points2, ["B", "7", "2"])

    batch_size = points1.shape[0]

    points1_norm, transform1 = normalize_points(points1)
    points2_norm, transform2 = normalize_points(points2)

    x1, y1 = torch.chunk(points1_norm, dim=-1, chunks=2)  # Bx1xN
    x2, y2 = torch.chunk(points2_norm, dim=-1, chunks=2)  # Bx1xN

    ones = ones_like(x1)
    # form a linear system: which represents
    # the equation (x2[i], 1)*F*(x1[i], 1) = 0
    X = concatenate([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, ones], -1)  # BxNx9

    # X * Fmat = 0 is singular (7 equations for 9 variables)
    # solving for nullspace of X to get two F
    ####### unstable failing gradcheck
    # _, _, v = torch.linalg.svd(X)
    _, _, v = _torch_svd_cast(X)

    # last two singular vector as a basic of the space
    f1 = v[..., 7].view(-1, 3, 3)
    f2 = v[..., 8].view(-1, 3, 3)

    # lambda*f1 + mu*f2 is an arbitrary fundamental matrix
    # f ~ lambda*f1 + (1 - lambda)*f2
    # det(f) = det(lambda*f1 + (1-lambda)*f2), find lambda
    # form a cubic equation
    # finding the coefficients of cubic polynomial (coeffs)

    coeffs = zeros((batch_size, 4), device=v.device, dtype=v.dtype)

    f1_det = torch.linalg.det(f1)
    f2_det = torch.linalg.det(f2)
    coeffs[:, 0] = f1_det
    coeffs[:, 1] = torch.einsum("bii->b", f2 @ safe_inverse_with_mask(f1)[0]) * f1_det
    coeffs[:, 2] = torch.einsum("bii->b", f1 @ safe_inverse_with_mask(f2)[0]) * f2_det
    coeffs[:, 3] = f2_det

    # solve the cubic equation, there can be 1 to 3 roots
    # roots = torch.tensor(np.roots(coeffs.numpy()))
    roots = solve_cubic(coeffs)

    fmatrix = zeros((batch_size, 3, 3, 3), device=v.device, dtype=v.dtype)
    valid_root_mask = (torch.count_nonzero(roots, dim=1) < 3) | (torch.count_nonzero(roots, dim=1) > 1)

    _lambda = roots
    _mu = torch.ones_like(_lambda)

    _s = f1[valid_root_mask, 2, 2].unsqueeze(dim=1) * roots[valid_root_mask] + f2[valid_root_mask, 2, 2].unsqueeze(
        dim=1
    )
    # _s_non_zero_mask = torch.abs(_s ) > 1e-16
    _s_non_zero_mask = ~torch.isclose(_s, torch.tensor(0.0, device=v.device, dtype=v.dtype))

    _mu[_s_non_zero_mask] = 1.0 / _s[_s_non_zero_mask]
    _lambda[_s_non_zero_mask] = _lambda[_s_non_zero_mask] * _mu[_s_non_zero_mask]

    f1_expanded = f1.unsqueeze(1).expand(batch_size, 3, 3, 3)
    f2_expanded = f2.unsqueeze(1).expand(batch_size, 3, 3, 3)

    fmatrix[valid_root_mask] = (
        f1_expanded[valid_root_mask] * _lambda[valid_root_mask, :, None, None]
        + f2_expanded[valid_root_mask] * _mu[valid_root_mask, :, None, None]
    )

    mat_ind = zeros(3, 3, dtype=torch.bool)
    mat_ind[2, 2] = True
    fmatrix[_s_non_zero_mask, mat_ind] = 1.0
    fmatrix[~_s_non_zero_mask, mat_ind] = 0.0

    trans1_exp = transform1[valid_root_mask].unsqueeze(1).expand(-1, fmatrix.shape[2], -1, -1)
    trans2_exp = transform2[valid_root_mask].unsqueeze(1).expand(-1, fmatrix.shape[2], -1, -1)

    fmatrix[valid_root_mask] = torch.matmul(
        trans2_exp.transpose(-2, -1), torch.matmul(fmatrix[valid_root_mask], trans1_exp)
    )

    return normalize_transformation(fmatrix)


def run_8point(points1: Tensor, points2: Tensor, weights: Optional[Tensor] = None) -> Tensor:
    r"""Compute the fundamental matrix using the DLT formulation.

    The linear system is solved by using the Weighted Least Squares Solution for the 8 Points algorithm.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2), N>=8`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2), N>=8`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.

    Returns:
        the computed fundamental matrix with shape :math:`(B, 3, 3)`.
    """
    KORNIA_CHECK_SHAPE(points1, ["B", "N", "2"])
    KORNIA_CHECK_SHAPE(points2, ["B", "N", "2"])
    KORNIA_CHECK_SAME_SHAPE(points1, points2)
    if points1.shape[1] < 8:
        raise AssertionError(points1.shape)
    if weights is not None:
        KORNIA_CHECK_SHAPE(weights, ["B", "N"])
        if not (weights.shape[1] == points1.shape[1]):
            raise AssertionError(weights.shape)

    points1_norm, transform1 = normalize_points(points1)
    points2_norm, transform2 = normalize_points(points2)

    x1, y1 = torch.chunk(points1_norm, dim=-1, chunks=2)  # Bx1xN
    x2, y2 = torch.chunk(points2_norm, dim=-1, chunks=2)  # Bx1xN

    ones = ones_like(x1)

    # build equations system and solve DLT
    # https://www.cc.gatech.edu/~afb/classes/CS4495-Fall2013/slides/CS4495-09-TwoViews-2.pdf
    # [x * x', x * y', x, y * x', y * y', y, x', y', 1]

    X = torch.cat([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, ones], dim=-1)  # BxNx9

    # apply the weights to the linear system
    if weights is None:
        X = X.transpose(-2, -1) @ X
    else:
        w_diag = torch.diag_embed(weights)
        X = X.transpose(-2, -1) @ w_diag @ X
    # compute eigevectors and retrieve the one with the smallest eigenvalue

    _, _, V = _torch_svd_cast(X)
    F_mat = V[..., -1].view(-1, 3, 3)

    # reconstruct and force the matrix to have rank2
    U, S, V = _torch_svd_cast(F_mat)
    rank_mask = torch.tensor([1.0, 1.0, 0.0], device=F_mat.device, dtype=F_mat.dtype)

    F_projected = U @ (torch.diag_embed(S * rank_mask) @ V.transpose(-2, -1))
    F_est = transform2.transpose(-2, -1) @ (F_projected @ transform1)

    return normalize_transformation(F_est)


def find_fundamental(
    points1: Tensor, points2: Tensor, weights: Optional[Tensor] = None, method: Literal["8POINT", "7POINT"] = "8POINT"
) -> Tensor:
    r"""
    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2), N>=8`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2), N>=8`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.
        method: The method to use for computing the fundamental matrix. Supported methods are "7POINT" and "8POINT".

    Returns:
        the computed fundamental matrix with shape :math:`(B, 3*m, 3)`, where `m` number of fundamental matrix.

    Raises:
        ValueError: If an invalid method is provided.

    """
    if method.upper() == "7POINT":
        result = run_7point(points1, points2)
    elif method.upper() == "8POINT":
        result = run_8point(points1, points2, weights)
    else:
        raise ValueError(f"Invalid method: {method}. Supported methods are '7POINT' and '8POINT'.")
    return result


def compute_correspond_epilines(points: Tensor, F_mat: Tensor) -> Tensor:
    r"""Compute the corresponding epipolar line for a given set of points.

    Args:
        points: tensor containing the set of points to project in the shape of :math:`(*, N, 2)` or :math:`(*, N, 3)`.
        F_mat: the fundamental to use for projection the points in the shape of :math:`(*, 3, 3)`.

    Returns:
        a tensor with shape :math:`(*, N, 3)` containing a vector of the epipolar
        lines corresponding to the points to the other image. Each line is described as
        :math:`ax + by + c = 0` and encoding the vectors as :math:`(a, b, c)`.
    """
    KORNIA_CHECK_SHAPE(points, ["*", "N", "DIM"])
    if points.shape[-1] == 2:
        points_h: Tensor = convert_points_to_homogeneous(points)
    elif points.shape[-1] == 3:
        points_h = points
    else:
        raise AssertionError(points.shape)
    KORNIA_CHECK_SHAPE(F_mat, ["*", "3", "3"])
    # project points and retrieve lines components
    points_h = torch.transpose(points_h, dim0=-2, dim1=-1)
    a, b, c = torch.chunk(F_mat @ points_h, dim=-2, chunks=3)

    # compute normal and compose equation line
    nu: Tensor = a * a + b * b
    nu = where(nu > 0.0, 1.0 / torch.sqrt(nu), torch.ones_like(nu))

    line = torch.cat([a * nu, b * nu, c * nu], dim=-2)  # *x3xN
    return torch.transpose(line, dim0=-2, dim1=-1)  # *xNx3


def get_perpendicular(lines: Tensor, points: Tensor) -> Tensor:
    r"""Compute the perpendicular to a line, through the point.

    Args:
        lines: tensor containing the set of lines :math:`(*, N, 3)`.
        points:  tensor containing the set of points :math:`(*, N, 2)`.

    Returns:
        a tensor with shape :math:`(*, N, 3)` containing a vector of the epipolar
        perpendicular lines. Each line is described as
        :math:`ax + by + c = 0` and encoding the vectors as :math:`(a, b, c)`.
    """
    KORNIA_CHECK_SHAPE(lines, ["*", "N", "3"])
    KORNIA_CHECK_SHAPE(points, ["*", "N", "two"])
    if points.shape[2] == 2:
        points_h: Tensor = convert_points_to_homogeneous(points)
    elif points.shape[2] == 3:
        points_h = points
    else:
        raise AssertionError(points.shape)
    infinity_point = lines * torch.tensor([1, 1, 0], dtype=lines.dtype, device=lines.device).view(1, 1, 3)
    perp: Tensor = points_h.cross(infinity_point, dim=2)
    return perp


def get_closest_point_on_epipolar_line(pts1: Tensor, pts2: Tensor, Fm: Tensor) -> Tensor:
    """Return closest point on the epipolar line to the correspondence, given the fundamental matrix.

    Args:
        pts1: correspondences from the left images with shape :math:`(*, N, (2|3))`. If they are not homogeneous,
              converted automatically.
        pts2: correspondences from the right images with shape :math:`(*, N, (2|3))`. If they are not homogeneous,
              converted automatically.
        Fm: Fundamental matrices with shape :math:`(*, 3, 3)`. Called Fm to avoid ambiguity with torch.nn.functional.

    Returns:
        point on epipolar line :math:`(*, N, 2)`.
    """
    if not isinstance(Fm, Tensor):
        raise TypeError(f"Fm type is not a torch.Tensor. Got {type(Fm)}")
    if (len(Fm.shape) < 3) or not Fm.shape[-2:] == (3, 3):
        raise ValueError(f"Fm must be a (*, 3, 3) tensor. Got {Fm.shape}")
    if pts1.shape[-1] == 2:
        pts1 = convert_points_to_homogeneous(pts1)
    if pts2.shape[-1] == 2:
        pts2 = convert_points_to_homogeneous(pts2)
    line1in2 = compute_correspond_epilines(pts1, Fm)
    perp = get_perpendicular(line1in2, pts2)
    points1_in_2 = convert_points_from_homogeneous(line1in2.cross(perp, dim=2))
    return points1_in_2


def fundamental_from_essential(E_mat: Tensor, K1: Tensor, K2: Tensor) -> Tensor:
    r"""Get the Fundamental matrix from Essential and camera matrices.

    Uses the method from Hartley/Zisserman 9.6 pag 257 (formula 9.12).

    Args:
        E_mat: The essential matrix with shape of :math:`(*, 3, 3)`.
        K1: The camera matrix from first camera with shape :math:`(*, 3, 3)`.
        K2: The camera matrix from second camera with shape :math:`(*, 3, 3)`.

    Returns:
        The fundamental matrix with shape :math:`(*, 3, 3)`.
    """
    KORNIA_CHECK_SHAPE(E_mat, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(K1, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(K2, ["*", "3", "3"])
    if not len(E_mat.shape[:-2]) == len(K1.shape[:-2]) == len(K2.shape[:-2]):
        raise AssertionError

    return (safe_inverse_with_mask(K2)[0]).transpose(-2, -1) @ E_mat @ (safe_inverse_with_mask(K1)[0])


# adapted from:
# https://github.com/opencv/opencv_contrib/blob/master/modules/sfm/src/fundamental.cpp#L109
# https://github.com/openMVG/openMVG/blob/160643be515007580086650f2ae7f1a42d32e9fb/src/openMVG/multiview/projection.cpp#L134


def fundamental_from_projections(P1: Tensor, P2: Tensor) -> Tensor:
    r"""Get the Fundamental matrix from Projection matrices.

    Args:
        P1: The projection matrix from first camera with shape :math:`(*, 3, 4)`.
        P2: The projection matrix from second camera with shape :math:`(*, 3, 4)`.

    Returns:
         The fundamental matrix with shape :math:`(*, 3, 3)`.
    """
    KORNIA_CHECK_SHAPE(P1, ["*", "3", "4"])
    KORNIA_CHECK_SHAPE(P2, ["*", "3", "4"])
    if P1.shape[:-2] != P2.shape[:-2]:
        raise AssertionError

    def vstack(x: Tensor, y: Tensor) -> Tensor:
        return concatenate([x, y], dim=-2)

    input_dtype = P1.dtype
    if input_dtype not in (torch.float32, torch.float64):
        P1 = P1.to(torch.float32)
        P2 = P2.to(torch.float32)

    X1 = P1[..., 1:, :]
    X2 = vstack(P1[..., 2:3, :], P1[..., 0:1, :])
    X3 = P1[..., :2, :]

    Y1 = P2[..., 1:, :]
    Y2 = vstack(P2[..., 2:3, :], P2[..., 0:1, :])
    Y3 = P2[..., :2, :]

    X1Y1, X2Y1, X3Y1 = vstack(X1, Y1), vstack(X2, Y1), vstack(X3, Y1)
    X1Y2, X2Y2, X3Y2 = vstack(X1, Y2), vstack(X2, Y2), vstack(X3, Y2)
    X1Y3, X2Y3, X3Y3 = vstack(X1, Y3), vstack(X2, Y3), vstack(X3, Y3)

    F_vec = torch.cat(
        [
            X1Y1.det().reshape(-1, 1),
            X2Y1.det().reshape(-1, 1),
            X3Y1.det().reshape(-1, 1),
            X1Y2.det().reshape(-1, 1),
            X2Y2.det().reshape(-1, 1),
            X3Y2.det().reshape(-1, 1),
            X1Y3.det().reshape(-1, 1),
            X2Y3.det().reshape(-1, 1),
            X3Y3.det().reshape(-1, 1),
        ],
        dim=1,
    )

    return F_vec.view(*P1.shape[:-2], 3, 3).to(input_dtype)
