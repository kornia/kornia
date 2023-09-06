"""Module containing functionalities for the Essential matrix."""
from typing import Optional, Tuple, Literal

import torch

from kornia.utils import eye_like, vec_like
from kornia.utils.helpers import _torch_svd_cast

from .numeric import cross_product_matrix
from .projection import depth_from_point, projection_from_KRt
from .triangulation import triangulate_points
import kornia.geometry.epipolar as epi

__all__ = [
    "essential_from_fundamental",
    "decompose_essential_matrix",
    "essential_from_Rt",
    "motion_from_essential",
    "motion_from_essential_choose_solution",
    "relative_camera_motion",
    "find_essential"
]

def o1(a, b):
        """
        a, b are first order polys [x,y,z,1]
          c is degree 2 poly with order
          [ x^2, x*y, x*z, x, y^2, y*z, y, z^2, z, 1]
        """
        # print(a[0] * b[2] + a[2] * b[0])
        return torch.stack([a[:, 0] * b[:, 0], a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0], a[:, 0] * b[:, 2] + a[:, 2] * b[:, 0],
                            a[:, 0] * b[:, 3] + a[:, 3] * b[:, 0], a[:, 1] * b[:, 1], a[:, 1] * b[:, 2] + a[:, 2] * b[:, 1],
                            a[:, 1] * b[:, 3] + a[:, 3] * b[:, 1], a[:, 2] * b[:, 2], a[:, 2] * b[:, 3] + a[:, 3] * b[:, 2],
                            a[:, 3] * b[:, 3]], dim=-1)

def o2(a, b):  # 10 4 20
    """
      a is second degree poly with order [ x^2, x*y, x*z, x, y^2, y*z, y, z^2, z, 1]
      b is first degree with order [x y z 1]
      c is third degree with order (same as nister's paper)
      [ x^3, y^3, x^2*y, x*y^2, x^2*z, x^2, y^2*z, y^2, x*y*z, x*y, x*z^2, x*z, x, y*z^2, y*z, y, z^3, z^2, z, 1]
    """
    return torch.stack(
        [a[:, 0] * b[:, 0], a[:, 4] * b[:, 1], a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0], a[:, 1] * b[:, 1] + a[:, 4] * b[:, 0], a[:, 0] * b[:, 2] + a[:, 2] * b[:, 0],
         a[:, 0] * b[:, 3] + a[:, 3] * b[:, 0], a[:, 4] * b[:, 2] + a[:, 5] * b[:, 1], a[:, 4] * b[:, 3] + a[:, 6] * b[:, 1],
         a[:, 1] * b[:, 2] + a[:, 2] * b[:, 1] + a[:, 5] * b[:, 0], a[:, 1] * b[:, 3] + a[:, 3] * b[:, 1] + a[:, 6] * b[:, 0],
         a[:, 2] * b[:, 2] + a[:, 7] * b[:, 0], a[:, 2] * b[:, 3] + a[:, 3] * b[:, 2] + a[:, 8] * b[:, 0], a[:, 3] * b[:, 3] + a[:, 9] * b[:, 0],
         a[:, 5] * b[:, 2] + a[:, 7] * b[:, 1], a[:, 5] * b[:, 3] + a[:, 6] * b[:, 2] + a[:, 8] * b[:, 1],
         a[:, 6] * b[:, 3] + a[:, 9] * b[:, 1], a[:, 7] * b[:, 2], a[:, 7] * b[:, 3] + a[:, 8] * b[:, 2],
         a[:, 8] * b[:, 3] + a[:, 9] * b[:, 2], a[:, 9] * b[:, 3]], dim=-1)

def run_5point(points1: torch.Tensor, points2: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""Compute the essential matrix using the 5-point algorithm

    The linear system is solved by Nister's 5-point algorithm.
    
    Args:
        points1: A set of carlibrated points in the first image with a tensor shape :math:`(B, N, 2), N>=8`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2), N>=8`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.

    Returns:
        the computed fundamental matrix with shape :math:`(B, 3, 3)`.
    """

    if points1.shape != points2.shape:
        raise AssertionError(points1.shape, points2.shape)
    if points1.shape[1] < 5:
        raise AssertionError(points1.shape)
    if weights is not None:
        if not (len(weights.shape) == 2 and weights.shape[1] == points1.shape[1]):
            raise AssertionError(weights.shape)

    batch_size, num, _ = points1.shape
    x1, y1 = torch.chunk(points1, dim=-1, chunks=2)  # Bx1xN
    x2, y2 = torch.chunk(points2, dim=-1, chunks=2)  # Bx1xN

    ones = torch.ones_like(x1)

    # build equations system and solve DLT
    # https://www.cc.gatech.edu/~afb/classes/CS4495-Fall2013/slides/CS4495-09-TwoViews-2.pdf
    # [x * x', x * y', x, y * x', y * y', y, x', y', 1]
    X = torch.cat([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, ones], dim=-1)  # BxNx9

    # apply the weights to the linear system
    if weights is None:
        X = X.transpose(-2, -1) @ X
    else:
        w_diag = torch.diag_embed(weights)
        X = X.transpose(-2, -1) @ w_diag @ X
    # compute eigevectors and retrieve the one with the smallest eigenvalue

    _, _, V = _torch_svd_cast(X)
    null_ = V[:, -4:, :].transpose(-1, -2) # the last four rows
    nullSpace = V[:, -4:, :]
    coeffs = torch.zeros(batch_size, 10, 20, device=null_.device, dtype=null_.dtype)
    d = torch.zeros(batch_size, 60, device=null_.device, dtype=null_.dtype)
    fun = lambda i, j : null_[:, 3 * j + i]

    # Determinant constraint
    coeffs[:, 9] = o2(o1(fun(0, 1), fun(1, 2)) - o1(fun(0, 2), fun(1, 1)), fun(2, 0)) +\
        o2(o1(fun(0, 2), fun(1, 0)) - o1(fun(0, 0), fun(1, 2)), fun(2, 1)) +\
        o2(o1(fun(0, 0), fun(1, 1)) - o1(fun(0, 1), fun(1, 0)), fun(2, 2))

    indices = torch.tensor([[0, 10, 20], [10, 40, 30], [20, 30, 50]])

    # Compute EE^T (equation 20 in paper)
    for i in range(3):
        for j in range(3):
            d[:, indices[i, j]: indices[i, j] + 10] = o1(fun(i, 0), fun(j, 0)) + \
                                                   o1(fun(i, 1), fun(j, 1)) + \
                                                   o1(fun(i, 2), fun(j, 2))

    for i in range(10):
        t = 0.5 * (d[:, indices[0, 0] + i] + d[:, indices[1, 1] + i] + d[:, indices[2, 2] + i])
        d[:, indices[0, 0] + i] -= t
        d[:, indices[1, 1] + i] -= t
        d[:, indices[2, 2] + i] -= t

    cnt = 0
    for i in range(3):
        for j in range(3):
            row = o2(d[:, indices[i, 0]: indices[i, 0] + 10], fun(0, j)) + \
                o2(d[:, indices[i, 1]: indices[i, 1] + 10], fun(1, j)) + \
                o2(d[:, indices[i, 2]: indices[i, 2] + 10], fun(2, j))
            coeffs[:, cnt] = row
            cnt += 1

    b = coeffs[:, :, 10:]
    singular_filter = torch.linalg.matrix_rank(coeffs[:, :, :10]) >= torch.max(
        torch.linalg.matrix_rank(coeffs),
        torch.ones_like(torch.linalg.matrix_rank(coeffs[:, :, :10]))*10)
    try:
        eliminated_mat = torch.linalg.solve(coeffs[singular_filter, :, :10], b[singular_filter])
    except Exception as e:
        print(e)

    coeffs_ = torch.concat((coeffs[singular_filter, :, :10], eliminated_mat), dim=-1)

    A = torch.zeros(coeffs_.shape[0], 3, 13, device=coeffs_.device, dtype=coeffs_.dtype)

    for i in range(3):
        A[:, i, 0] = 0.0
        A[:, i:i + 1, 1:4] = coeffs_[:, 4 + 2 * i: 5 + 2 * i, 10:13]
        A[:, i:i + 1, 0:3] -= coeffs_[:, 5 + 2 * i: 6 + 2 * i, 10:13]
        A[:, i, 4] = 0.0
        A[:, i:i + 1, 5:8] = coeffs_[:, 4 + 2 * i: 5 + 2 * i, 13:16]
        A[:, i:i + 1, 4:7] -= coeffs_[:, 5 + 2 * i: 6 + 2 * i, 13:16]
        A[:, i, 8] = 0.0
        A[:, i:i + 1, 9:13] = coeffs_[:, 4 + 2 * i: 5 + 2 * i, 16:20]
        A[:, i:i + 1, 8:12] -= coeffs_[:, 5 + 2 * i: 6 + 2 * i, 16:20]

    cs = torch.zeros(A.shape[0], 11, device=A.device, dtype=A.dtype)
    cs[:, 0] = A[:, 0, 12] * A[:, 1, 3] * A[:, 2, 7] - A[:, 0, 12] * A[:, 1, 7] * A[:, 2, 3] - A[:, 0, 3] * A[:, 2, 7] * A[:, 1, 12] + \
               A[:, 0, 7] * A[:, 2, 3] * A[:, 1, 12] + A[:, 0, 3] * A[:, 1, 7] * A[:, 2, 12] - A[:, 0, 7] * A[:, 1, 3] * A[:, 2, 12]

    cs[:, 1] = A[:, 0, 11] * A[:, 1, 3] * A[:, 2, 7] - A[:, 0, 11] * A[:, 1, 7] * A[:, 2, 3] + A[:, 0, 12] * A[:, 1, 2] * A[:, 2, 7] + \
           A[:, 0, 12] * A[:, 1, 3] * A[:, 2, 6] - A[:, 0, 12] * A[:, 1, 6] * A[:, 2, 3] - A[:, 0, 12] * A[:, 1, 7] * A[:, 2, 2] - \
           A[:, 0, 2] * A[:, 2, 7] * A[:, 1, 12] - A[:, 0, 3] * A[:, 2, 6] * A[:, 1, 12] - A[:, 0, 3] * A[:, 2, 7] * A[:, 1, 11] + \
           A[:, 0, 6] * A[:, 2, 3] * A[:, 1, 12] + A[:, 0, 7] * A[:, 2, 2] * A[:, 1, 12] + A[:, 0, 7] * A[:, 2, 3] * A[:, 1, 11] + \
           A[:, 0, 2] * A[:, 1, 7] * A[:, 2, 12] + A[:, 0, 3] * A[:, 1, 6] * A[:, 2, 12] + A[:, 0, 3] * A[:, 1, 7] * A[:, 2, 11] - \
           A[:, 0, 6] * A[:, 1, 3] * A[:, 2, 12] - A[:, 0, 7] * A[:, 1, 2] * A[:, 2, 12] - A[:, 0, 7] * A[:, 1, 3] * A[:, 2, 11]

    cs[:, 2] = A[:, 0, 10] * A[:, 1, 3] * A[:, 2, 7] - A[:, 0, 10] * A[:, 1, 7] * A[:, 2, 3] + A[:, 0, 11] * A[:, 1, 2] * A[:, 2, 7] + \
           A[:, 0, 11] * A[:, 1, 3] * A[:, 2, 6] - A[:, 0, 11] * A[:, 1, 6] * A[:, 2, 3] - A[:, 0, 11] * A[:, 1, 7] * A[:, 2, 2] + \
           A[:, 1, 1] * A[:, 0, 12] * A[:, 2, 7] + A[:, 0, 12] * A[:, 1, 2] * A[:, 2, 6] + A[:, 0, 12] * A[:, 1, 3] * A[:, 2, 5] - \
           A[:, 0, 12] * A[:, 1, 5] * A[:, 2, 3] - A[:, 0, 12] * A[:, 1, 6] * A[:, 2, 2] - A[:, 0, 12] * A[:, 1, 7] * A[:, 2, 1] - \
           A[:, 0, 1] * A[:, 2, 7] * A[:, 1, 12] - A[:, 0, 2] * A[:, 2, 6] * A[:, 1, 12] - A[:, 0, 2] * A[:, 2, 7] * A[:, 1, 11] - \
           A[:, 0, 3] * A[:, 2, 5] * A[:, 1, 12] - A[:, 0, 3] * A[:, 2, 6] * A[:, 1, 11] - A[:, 0, 3] * A[:, 2, 7] * A[:, 1, 10] + \
           A[:, 0, 5] * A[:, 2, 3] * A[:, 1, 12] + A[:, 0, 6] * A[:, 2, 2] * A[:, 1, 12] + A[:, 0, 6] * A[:, 2, 3] * A[:, 1, 11] + \
           A[:, 0, 7] * A[:, 2, 1] * A[:, 1, 12] + A[:, 0, 7] * A[:, 2, 2] * A[:, 1, 11] + A[:, 0, 7] * A[:, 2, 3] * A[:, 1, 10] + \
           A[:, 0, 1] * A[:, 1, 7] * A[:, 2, 12] + A[:, 0, 2] * A[:, 1, 6] * A[:, 2, 12] + A[:, 0, 2] * A[:, 1, 7] * A[:, 2, 11] + \
           A[:, 0, 3] * A[:, 1, 5] * A[:, 2, 12] + A[:, 0, 3] * A[:, 1, 6] * A[:, 2, 11] + A[:, 0, 3] * A[:, 1, 7] * A[:, 2, 10] - \
           A[:, 0, 5] * A[:, 1, 3] * A[:, 2, 12] - A[:, 0, 6] * A[:, 1, 2] * A[:, 2, 12] - A[:, 0, 6] * A[:, 1, 3] * A[:, 2, 11] - \
           A[:, 0, 7] * A[:, 1, 1] * A[:, 2, 12] - A[:, 0, 7] * A[:, 1, 2] * A[:, 2, 11] - A[:, 0, 7] * A[:, 1, 3] * A[:, 2, 10]

    cs[:, 3] = A[:, 0, 3] * A[:, 1, 7] * A[:, 2, 9] - A[:, 0, 3] * A[:, 1, 9] * A[:, 2, 7] - A[:, 0, 7] * A[:, 1, 3] * A[:, 2, 9] + \
           A[:, 0, 7] * A[:, 1, 9] * A[:, 2, 3] + A[:, 0, 9] * A[:, 1, 3] * A[:, 2, 7] - A[:, 0, 9] * A[:, 1, 7] * A[:, 2, 3] + \
           A[:, 0, 10] * A[:, 1, 2] * A[:, 2, 7] + A[:, 0, 10] * A[:, 1, 3] * A[:, 2, 6] - A[:, 0, 10] * A[:, 1, 6] * A[:, 2, 3] - \
           A[:, 0, 10] * A[:, 1, 7] * A[:, 2, 2] + A[:, 1, 0] * A[:, 0, 12] * A[:, 2, 7] + A[:, 0, 11] * A[:, 1, 1] * A[:, 2, 7] + \
           A[:, 0, 11] * A[:, 1, 2] * A[:, 2, 6] + A[:, 0, 11] * A[:, 1, 3] * A[:, 2, 5] - A[:, 0, 11] * A[:, 1, 5] * A[:, 2, 3] - \
           A[:, 0, 11] * A[:, 1, 6] * A[:, 2, 2] - A[:, 0, 11] * A[:, 1, 7] * A[:, 2, 1] + A[:, 1, 1] * A[:, 0, 12] * A[:, 2, 6] + \
           A[:, 0, 12] * A[:, 1, 2] * A[:, 2, 5] + A[:, 0, 12] * A[:, 1, 3] * A[:, 2, 4] - A[:, 0, 12] * A[:, 1, 4] * A[:, 2, 3] - \
           A[:, 0, 12] * A[:, 1, 5] * A[:, 2, 2] - A[:, 0, 12] * A[:, 1, 6] * A[:, 2, 1] - A[:, 0, 12] * A[:, 1, 7] * A[:, 2, 0] - \
           A[:, 0, 0] * A[:, 2, 7] * A[:, 1, 12] - A[:, 0, 1] * A[:, 2, 6] * A[:, 1, 12] - A[:, 0, 1] * A[:, 2, 7] * A[:, 1, 11] - \
           A[:, 0, 2] * A[:, 2, 5] * A[:, 1, 12] - A[:, 0, 2] * A[:, 2, 6] * A[:, 1, 11] - A[:, 0, 2] * A[:, 2, 7] * A[:, 1, 10] - \
           A[:, 0, 3] * A[:, 2, 4] * A[:, 1, 12] - A[:, 0, 3] * A[:, 2, 5] * A[:, 1, 11] - A[:, 0, 3] * A[:, 2, 6] * A[:, 1, 10] + \
           A[:, 0, 4] * A[:, 2, 3] * A[:, 1, 12] + A[:, 0, 5] * A[:, 2, 2] * A[:, 1, 12] + A[:, 0, 5] * A[:, 2, 3] * A[:, 1, 11] + \
           A[:, 0, 6] * A[:, 2, 1] * A[:, 1, 12] + A[:, 0, 6] * A[:, 2, 2] * A[:, 1, 11] + A[:, 0, 6] * A[:, 2, 3] * A[:, 1, 10] + \
           A[:, 0, 7] * A[:, 2, 0] * A[:, 1, 12] + A[:, 0, 7] * A[:, 2, 1] * A[:, 1, 11] + A[:, 0, 7] * A[:, 2, 2] * A[:, 1, 10] + \
           A[:, 0, 0] * A[:, 1, 7] * A[:, 2, 12] + A[:, 0, 1] * A[:, 1, 6] * A[:, 2, 12] + A[:, 0, 1] * A[:, 1, 7] * A[:, 2, 11] + \
           A[:, 0, 2] * A[:, 1, 5] * A[:, 2, 12] + A[:, 0, 2] * A[:, 1, 6] * A[:, 2, 11] + A[:, 0, 2] * A[:, 1, 7] * A[:, 2, 10] + \
           A[:, 0, 3] * A[:, 1, 4] * A[:, 2, 12] + A[:, 0, 3] * A[:, 1, 5] * A[:, 2, 11] + A[:, 0, 3] * A[:, 1, 6] * A[:, 2, 10] - \
           A[:, 0, 4] * A[:, 1, 3] * A[:, 2, 12] - A[:, 0, 5] * A[:, 1, 2] * A[:, 2, 12] - A[:, 0, 5] * A[:, 1, 3] * A[:, 2, 11] - \
           A[:, 0, 6] * A[:, 1, 1] * A[:, 2, 12] - A[:, 0, 6] * A[:, 1, 2] * A[:, 2, 11] - A[:, 0, 6] * A[:, 1, 3] * A[:, 2, 10] - \
           A[:, 0, 7] * A[:, 1, 0] * A[:, 2, 12] - A[:, 0, 7] * A[:, 1, 1] * A[:, 2, 11] - A[:, 0, 7] * A[:, 1, 2] * A[:, 2, 10]

    cs[:, 4] = A[:, 0, 2] * A[:, 1, 7] * A[:, 2, 9] - A[:, 0, 2] * A[:, 1, 9] * A[:, 2, 7] + A[:, 0, 3] * A[:, 1, 6] * A[:, 2, 9] + \
           A[:, 0, 3] * A[:, 1, 7] * A[:, 2, 8] - A[:, 0, 3] * A[:, 1, 8] * A[:, 2, 7] - A[:, 0, 3] * A[:, 1, 9] * A[:, 2, 6] - \
           A[:, 0, 6] * A[:, 1, 3] * A[:, 2, 9] + A[:, 0, 6] * A[:, 1, 9] * A[:, 2, 3] - A[:, 0, 7] * A[:, 1, 2] * A[:, 2, 9] - \
           A[:, 0, 7] * A[:, 1, 3] * A[:, 2, 8] + A[:, 0, 7] * A[:, 1, 8] * A[:, 2, 3] + A[:, 0, 7] * A[:, 1, 9] * A[:, 2, 2] + \
           A[:, 0, 8] * A[:, 1, 3] * A[:, 2, 7] - A[:, 0, 8] * A[:, 1, 7] * A[:, 2, 3] + A[:, 0, 9] * A[:, 1, 2] * A[:, 2, 7] + \
           A[:, 0, 9] * A[:, 1, 3] * A[:, 2, 6] - A[:, 0, 9] * A[:, 1, 6] * A[:, 2, 3] - A[:, 0, 9] * A[:, 1, 7] * A[:, 2, 2] + \
           A[:, 0, 10] * A[:, 1, 1] * A[:, 2, 7] + A[:, 0, 10] * A[:, 1, 2] * A[:, 2, 6] + A[:, 0, 10] * A[:, 1, 3] * A[:, 2, 5] - \
           A[:, 0, 10] * A[:, 1, 5] * A[:, 2, 3] - A[:, 0, 10] * A[:, 1, 6] * A[:, 2, 2] - A[:, 0, 10] * A[:, 1, 7] * A[:, 2, 1] + \
           A[:, 1, 0] * A[:, 0, 11] * A[:, 2, 7] + A[:, 1, 0] * A[:, 0, 12] * A[:, 2, 6] + A[:, 0, 11] * A[:, 1, 1] * A[:, 2, 6] + \
           A[:, 0, 11] * A[:, 1, 2] * A[:, 2, 5] + A[:, 0, 11] * A[:, 1, 3] * A[:, 2, 4] - A[:, 0, 11] * A[:, 1, 4] * A[:, 2, 3] - \
           A[:, 0, 11] * A[:, 1, 5] * A[:, 2, 2] - A[:, 0, 11] * A[:, 1, 6] * A[:, 2, 1] - A[:, 0, 11] * A[:, 1, 7] * A[:, 2, 0] + \
           A[:, 1, 1] * A[:, 0, 12] * A[:, 2, 5] + A[:, 0, 12] * A[:, 1, 2] * A[:, 2, 4] - A[:, 0, 12] * A[:, 1, 4] * A[:, 2, 2] - \
           A[:, 0, 12] * A[:, 1, 5] * A[:, 2, 1] - A[:, 0, 12] * A[:, 1, 6] * A[:, 2, 0] - A[:, 0, 0] * A[:, 2, 6] * A[:, 1, 12] - \
           A[:, 0, 0] * A[:, 2, 7] * A[:, 1, 11] - A[:, 0, 1] * A[:, 2, 5] * A[:, 1, 12] - A[:, 0, 1] * A[:, 2, 6] * A[:, 1, 11] - \
           A[:, 0, 1] * A[:, 2, 7] * A[:, 1, 10] - A[:, 0, 2] * A[:, 2, 4] * A[:, 1, 12] - A[:, 0, 2] * A[:, 2, 5] * A[:, 1, 11] - \
           A[:, 0, 2] * A[:, 2, 6] * A[:, 1, 10] - A[:, 0, 3] * A[:, 2, 4] * A[:, 1, 11] - A[:, 0, 3] * A[:, 2, 5] * A[:, 1, 10] + \
           A[:, 0, 4] * A[:, 2, 2] * A[:, 1, 12] + A[:, 0, 4] * A[:, 2, 3] * A[:, 1, 11] + A[:, 0, 5] * A[:, 2, 1] * A[:, 1, 12] + \
           A[:, 0, 5] * A[:, 2, 2] * A[:, 1, 11] + A[:, 0, 5] * A[:, 2, 3] * A[:, 1, 10] + A[:, 0, 6] * A[:, 2, 0] * A[:, 1, 12] + \
           A[:, 0, 6] * A[:, 2, 1] * A[:, 1, 11] + A[:, 0, 6] * A[:, 2, 2] * A[:, 1, 10] + A[:, 0, 7] * A[:, 2, 0] * A[:, 1, 11] + \
           A[:, 0, 7] * A[:, 2, 1] * A[:, 1, 10] + A[:, 0, 0] * A[:, 1, 6] * A[:, 2, 12] + A[:, 0, 0] * A[:, 1, 7] * A[:, 2, 11] + \
           A[:, 0, 1] * A[:, 1, 5] * A[:, 2, 12] + A[:, 0, 1] * A[:, 1, 6] * A[:, 2, 11] + A[:, 0, 1] * A[:, 1, 7] * A[:, 2, 10] + \
           A[:, 0, 2] * A[:, 1, 4] * A[:, 2, 12] + A[:, 0, 2] * A[:, 1, 5] * A[:, 2, 11] + A[:, 0, 2] * A[:, 1, 6] * A[:, 2, 10] + \
           A[:, 0, 3] * A[:, 1, 4] * A[:, 2, 11] + A[:, 0, 3] * A[:, 1, 5] * A[:, 2, 10] - A[:, 0, 4] * A[:, 1, 2] * A[:, 2, 12] - \
           A[:, 0, 4] * A[:, 1, 3] * A[:, 2, 11] - A[:, 0, 5] * A[:, 1, 1] * A[:, 2, 12] - A[:, 0, 5] * A[:, 1, 2] * A[:, 2, 11] - \
           A[:, 0, 5] * A[:, 1, 3] * A[:, 2, 10] - A[:, 0, 6] * A[:, 1, 0] * A[:, 2, 12] - A[:, 0, 6] * A[:, 1, 1] * A[:, 2, 11] - \
           A[:, 0, 6] * A[:, 1, 2] * A[:, 2, 10] - A[:, 0, 7] * A[:, 1, 0] * A[:, 2, 11] - A[:, 0, 7] * A[:, 1, 1] * A[:, 2, 10]

    cs[:, 5] = A[:, 0, 1] * A[:, 1, 7] * A[:, 2, 9] - A[:, 0, 1] * A[:, 1, 9] * A[:, 2, 7] + A[:, 0, 2] * A[:, 1, 6] * A[:, 2, 9] + \
           A[:, 0, 2] * A[:, 1, 7] * A[:, 2, 8] - A[:, 0, 2] * A[:, 1, 8] * A[:, 2, 7] - A[:, 0, 2] * A[:, 1, 9] * A[:, 2, 6] + \
           A[:, 0, 3] * A[:, 1, 5] * A[:, 2, 9] + A[:, 0, 3] * A[:, 1, 6] * A[:, 2, 8] - A[:, 0, 3] * A[:, 1, 8] * A[:, 2, 6] - \
           A[:, 0, 3] * A[:, 1, 9] * A[:, 2, 5] - A[:, 0, 5] * A[:, 1, 3] * A[:, 2, 9] + A[:, 0, 5] * A[:, 1, 9] * A[:, 2, 3] - \
           A[:, 0, 6] * A[:, 1, 2] * A[:, 2, 9] - A[:, 0, 6] * A[:, 1, 3] * A[:, 2, 8] + A[:, 0, 6] * A[:, 1, 8] * A[:, 2, 3] + \
           A[:, 0, 6] * A[:, 1, 9] * A[:, 2, 2] - A[:, 0, 7] * A[:, 1, 1] * A[:, 2, 9] - A[:, 0, 7] * A[:, 1, 2] * A[:, 2, 8] + \
           A[:, 0, 7] * A[:, 1, 8] * A[:, 2, 2] + A[:, 0, 7] * A[:, 1, 9] * A[:, 2, 1] + A[:, 0, 8] * A[:, 1, 2] * A[:, 2, 7] + \
           A[:, 0, 8] * A[:, 1, 3] * A[:, 2, 6] - A[:, 0, 8] * A[:, 1, 6] * A[:, 2, 3] - A[:, 0, 8] * A[:, 1, 7] * A[:, 2, 2] + \
           A[:, 0, 9] * A[:, 1, 1] * A[:, 2, 7] + A[:, 0, 9] * A[:, 1, 2] * A[:, 2, 6] + A[:, 0, 9] * A[:, 1, 3] * A[:, 2, 5] - \
           A[:, 0, 9] * A[:, 1, 5] * A[:, 2, 3] - A[:, 0, 9] * A[:, 1, 6] * A[:, 2, 2] - A[:, 0, 9] * A[:, 1, 7] * A[:, 2, 1] + \
           A[:, 0, 10] * A[:, 1, 0] * A[:, 2, 7] + A[:, 0, 10] * A[:, 1, 1] * A[:, 2, 6] + A[:, 0, 10] * A[:, 1, 2] * A[:, 2, 5] + \
           A[:, 0, 10] * A[:, 1, 3] * A[:, 2, 4] - A[:, 0, 10] * A[:, 1, 4] * A[:, 2, 3] - A[:, 0, 10] * A[:, 1, 5] * A[:, 2, 2] - \
           A[:, 0, 10] * A[:, 1, 6] * A[:, 2, 1] - A[:, 0, 10] * A[:, 1, 7] * A[:, 2, 0] + A[:, 1, 0] * A[:, 0, 11] * A[:, 2, 6] + \
           A[:, 1, 0] * A[:, 0, 12] * A[:, 2, 5] + A[:, 0, 11] * A[:, 1, 1] * A[:, 2, 5] + A[:, 0, 11] * A[:, 1, 2] * A[:, 2, 4] - \
           A[:, 0, 11] * A[:, 1, 4] * A[:, 2, 2] - A[:, 0, 11] * A[:, 1, 5] * A[:, 2, 1] - A[:, 0, 11] * A[:, 1, 6] * A[:, 2, 0] + \
           A[:, 1, 1] * A[:, 0, 12] * A[:, 2, 4] - A[:, 0, 12] * A[:, 1, 4] * A[:, 2, 1] - A[:, 0, 12] * A[:, 1, 5] * A[:, 2, 0] - \
           A[:, 0, 0] * A[:, 2, 5] * A[:, 1, 12] - A[:, 0, 0] * A[:, 2, 6] * A[:, 1, 11] - A[:, 0, 0] * A[:, 2, 7] * A[:, 1, 10] - \
           A[:, 0, 1] * A[:, 2, 4] * A[:, 1, 12] - A[:, 0, 1] * A[:, 2, 5] * A[:, 1, 11] - A[:, 0, 1] * A[:, 2, 6] * A[:, 1, 10] - \
           A[:, 0, 2] * A[:, 2, 4] * A[:, 1, 11] - A[:, 0, 2] * A[:, 2, 5] * A[:, 1, 10] - A[:, 0, 3] * A[:, 2, 4] * A[:, 1, 10] + \
           A[:, 0, 4] * A[:, 2, 1] * A[:, 1, 12] + A[:, 0, 4] * A[:, 2, 2] * A[:, 1, 11] + A[:, 0, 4] * A[:, 2, 3] * A[:, 1, 10] + \
           A[:, 0, 5] * A[:, 2, 0] * A[:, 1, 12] + A[:, 0, 5] * A[:, 2, 1] * A[:, 1, 11] + A[:, 0, 5] * A[:, 2, 2] * A[:, 1, 10] + \
           A[:, 0, 6] * A[:, 2, 0] * A[:, 1, 11] + A[:, 0, 6] * A[:, 2, 1] * A[:, 1, 10] + A[:, 0, 7] * A[:, 2, 0] * A[:, 1, 10] + \
           A[:, 0, 0] * A[:, 1, 5] * A[:, 2, 12] + A[:, 0, 0] * A[:, 1, 6] * A[:, 2, 11] + A[:, 0, 0] * A[:, 1, 7] * A[:, 2, 10] + \
           A[:, 0, 1] * A[:, 1, 4] * A[:, 2, 12] + A[:, 0, 1] * A[:, 1, 5] * A[:, 2, 11] + A[:, 0, 1] * A[:, 1, 6] * A[:, 2, 10] + \
           A[:, 0, 2] * A[:, 1, 4] * A[:, 2, 11] + A[:, 0, 2] * A[:, 1, 5] * A[:, 2, 10] + A[:, 0, 3] * A[:, 1, 4] * A[:, 2, 10] - \
           A[:, 0, 4] * A[:, 1, 1] * A[:, 2, 12] - A[:, 0, 4] * A[:, 1, 2] * A[:, 2, 11] - A[:, 0, 4] * A[:, 1, 3] * A[:, 2, 10] - \
           A[:, 0, 5] * A[:, 1, 0] * A[:, 2, 12] - A[:, 0, 5] * A[:, 1, 1] * A[:, 2, 11] - A[:, 0, 5] * A[:, 1, 2] * A[:, 2, 10] - \
           A[:, 0, 6] * A[:, 1, 0] * A[:, 2, 11] - A[:, 0, 6] * A[:, 1, 1] * A[:, 2, 10] - A[:, 0, 7] * A[:, 1, 0] * A[:, 2, 10]

    cs[:, 6] = A[:, 0, 0] * A[:, 1, 7] * A[:, 2, 9] - A[:, 0, 0] * A[:, 1, 9] * A[:, 2, 7] + A[:, 0, 1] * A[:, 1, 6] * A[:, 2, 9] + \
           A[:, 0, 1] * A[:, 1, 7] * A[:, 2, 8] - A[:, 0, 1] * A[:, 1, 8] * A[:, 2, 7] - A[:, 0, 1] * A[:, 1, 9] * A[:, 2, 6] + \
           A[:, 0, 2] * A[:, 1, 5] * A[:, 2, 9] + A[:, 0, 2] * A[:, 1, 6] * A[:, 2, 8] - A[:, 0, 2] * A[:, 1, 8] * A[:, 2, 6] - \
           A[:, 0, 2] * A[:, 1, 9] * A[:, 2, 5] + A[:, 0, 3] * A[:, 1, 4] * A[:, 2, 9] + A[:, 0, 3] * A[:, 1, 5] * A[:, 2, 8] - \
           A[:, 0, 3] * A[:, 1, 8] * A[:, 2, 5] - A[:, 0, 3] * A[:, 1, 9] * A[:, 2, 4] - A[:, 0, 4] * A[:, 1, 3] * A[:, 2, 9] + \
           A[:, 0, 4] * A[:, 1, 9] * A[:, 2, 3] - A[:, 0, 5] * A[:, 1, 2] * A[:, 2, 9] - A[:, 0, 5] * A[:, 1, 3] * A[:, 2, 8] + \
           A[:, 0, 5] * A[:, 1, 8] * A[:, 2, 3] + A[:, 0, 5] * A[:, 1, 9] * A[:, 2, 2] - A[:, 0, 6] * A[:, 1, 1] * A[:, 2, 9] - \
           A[:, 0, 6] * A[:, 1, 2] * A[:, 2, 8] + A[:, 0, 6] * A[:, 1, 8] * A[:, 2, 2] + A[:, 0, 6] * A[:, 1, 9] * A[:, 2, 1] - \
           A[:, 0, 7] * A[:, 1, 0] * A[:, 2, 9] - A[:, 0, 7] * A[:, 1, 1] * A[:, 2, 8] + A[:, 0, 7] * A[:, 1, 8] * A[:, 2, 1] + \
           A[:, 0, 7] * A[:, 1, 9] * A[:, 2, 0] + A[:, 0, 8] * A[:, 1, 1] * A[:, 2, 7] + A[:, 0, 8] * A[:, 1, 2] * A[:, 2, 6] + \
           A[:, 0, 8] * A[:, 1, 3] * A[:, 2, 5] - A[:, 0, 8] * A[:, 1, 5] * A[:, 2, 3] - A[:, 0, 8] * A[:, 1, 6] * A[:, 2, 2] - \
           A[:, 0, 8] * A[:, 1, 7] * A[:, 2, 1] + A[:, 0, 9] * A[:, 1, 0] * A[:, 2, 7] + A[:, 0, 9] * A[:, 1, 1] * A[:, 2, 6] + \
           A[:, 0, 9] * A[:, 1, 2] * A[:, 2, 5] + A[:, 0, 9] * A[:, 1, 3] * A[:, 2, 4] - A[:, 0, 9] * A[:, 1, 4] * A[:, 2, 3] - \
           A[:, 0, 9] * A[:, 1, 5] * A[:, 2, 2] - A[:, 0, 9] * A[:, 1, 6] * A[:, 2, 1] - A[:, 0, 9] * A[:, 1, 7] * A[:, 2, 0] + \
           A[:, 0, 10] * A[:, 1, 0] * A[:, 2, 6] + A[:, 0, 10] * A[:, 1, 1] * A[:, 2, 5] + A[:, 0, 10] * A[:, 1, 2] * A[:, 2, 4] - \
           A[:, 0, 10] * A[:, 1, 4] * A[:, 2, 2] - A[:, 0, 10] * A[:, 1, 5] * A[:, 2, 1] - A[:, 0, 10] * A[:, 1, 6] * A[:, 2, 0] + \
           A[:, 1, 0] * A[:, 0, 11] * A[:, 2, 5] + A[:, 1, 0] * A[:, 0, 12] * A[:, 2, 4] + A[:, 0, 11] * A[:, 1, 1] * A[:, 2, 4] - \
           A[:, 0, 11] * A[:, 1, 4] * A[:, 2, 1] - A[:, 0, 11] * A[:, 1, 5] * A[:, 2, 0] - A[:, 0, 12] * A[:, 1, 4] * A[:, 2, 0] - \
           A[:, 0, 0] * A[:, 2, 4] * A[:, 1, 12] - A[:, 0, 0] * A[:, 2, 5] * A[:, 1, 11] - A[:, 0, 0] * A[:, 2, 6] * A[:, 1, 10] - \
           A[:, 0, 1] * A[:, 2, 4] * A[:, 1, 11] - A[:, 0, 1] * A[:, 2, 5] * A[:, 1, 10] - A[:, 0, 2] * A[:, 2, 4] * A[:, 1, 10] + \
           A[:, 0, 4] * A[:, 2, 0] * A[:, 1, 12] + A[:, 0, 4] * A[:, 2, 1] * A[:, 1, 11] + A[:, 0, 4] * A[:, 2, 2] * A[:, 1, 10] + \
           A[:, 0, 5] * A[:, 2, 0] * A[:, 1, 11] + A[:, 0, 5] * A[:, 2, 1] * A[:, 1, 10] + A[:, 0, 6] * A[:, 2, 0] * A[:, 1, 10] + \
           A[:, 0, 0] * A[:, 1, 4] * A[:, 2, 12] + A[:, 0, 0] * A[:, 1, 5] * A[:, 2, 11] + A[:, 0, 0] * A[:, 1, 6] * A[:, 2, 10] + \
           A[:, 0, 1] * A[:, 1, 4] * A[:, 2, 11] + A[:, 0, 1] * A[:, 1, 5] * A[:, 2, 10] + A[:, 0, 2] * A[:, 1, 4] * A[:, 2, 10] - \
           A[:, 0, 4] * A[:, 1, 0] * A[:, 2, 12] - A[:, 0, 4] * A[:, 1, 1] * A[:, 2, 11] - A[:, 0, 4] * A[:, 1, 2] * A[:, 2, 10] - \
           A[:, 0, 5] * A[:, 1, 0] * A[:, 2, 11] - A[:, 0, 5] * A[:, 1, 1] * A[:, 2, 10] - A[:, 0, 6] * A[:, 1, 0] * A[:, 2, 10]

    cs[:, 7] = A[:, 0, 0] * A[:, 1, 6] * A[:, 2, 9] + A[:, 0, 0] * A[:, 1, 7] * A[:, 2, 8] - A[:, 0, 0] * A[:, 1, 8] * A[:, 2, 7] - \
           A[:, 0, 0] * A[:, 1, 9] * A[:, 2, 6] + A[:, 0, 1] * A[:, 1, 5] * A[:, 2, 9] + A[:, 0, 1] * A[:, 1, 6] * A[:, 2, 8] - \
           A[:, 0, 1] * A[:, 1, 8] * A[:, 2, 6] - A[:, 0, 1] * A[:, 1, 9] * A[:, 2, 5] + A[:, 0, 2] * A[:, 1, 4] * A[:, 2, 9] + \
           A[:, 0, 2] * A[:, 1, 5] * A[:, 2, 8] - A[:, 0, 2] * A[:, 1, 8] * A[:, 2, 5] - A[:, 0, 2] * A[:, 1, 9] * A[:, 2, 4] + \
           A[:, 0, 3] * A[:, 1, 4] * A[:, 2, 8] - A[:, 0, 3] * A[:, 1, 8] * A[:, 2, 4] - A[:, 0, 4] * A[:, 1, 2] * A[:, 2, 9] - \
           A[:, 0, 4] * A[:, 1, 3] * A[:, 2, 8] + A[:, 0, 4] * A[:, 1, 8] * A[:, 2, 3] + A[:, 0, 4] * A[:, 1, 9] * A[:, 2, 2] - \
           A[:, 0, 5] * A[:, 1, 1] * A[:, 2, 9] - A[:, 0, 5] * A[:, 1, 2] * A[:, 2, 8] + A[:, 0, 5] * A[:, 1, 8] * A[:, 2, 2] + \
           A[:, 0, 5] * A[:, 1, 9] * A[:, 2, 1] - A[:, 0, 6] * A[:, 1, 0] * A[:, 2, 9] - A[:, 0, 6] * A[:, 1, 1] * A[:, 2, 8] + \
           A[:, 0, 6] * A[:, 1, 8] * A[:, 2, 1] + A[:, 0, 6] * A[:, 1, 9] * A[:, 2, 0] - A[:, 0, 7] * A[:, 1, 0] * A[:, 2, 8] + \
           A[:, 0, 7] * A[:, 1, 8] * A[:, 2, 0] + A[:, 0, 8] * A[:, 1, 0] * A[:, 2, 7] + A[:, 0, 8] * A[:, 1, 1] * A[:, 2, 6] + \
           A[:, 0, 8] * A[:, 1, 2] * A[:, 2, 5] + A[:, 0, 8] * A[:, 1, 3] * A[:, 2, 4] - A[:, 0, 8] * A[:, 1, 4] * A[:, 2, 3] - \
           A[:, 0, 8] * A[:, 1, 5] * A[:, 2, 2] - A[:, 0, 8] * A[:, 1, 6] * A[:, 2, 1] - A[:, 0, 8] * A[:, 1, 7] * A[:, 2, 0] + \
           A[:, 0, 9] * A[:, 1, 0] * A[:, 2, 6] + A[:, 0, 9] * A[:, 1, 1] * A[:, 2, 5] + A[:, 0, 9] * A[:, 1, 2] * A[:, 2, 4] - \
           A[:, 0, 9] * A[:, 1, 4] * A[:, 2, 2] - A[:, 0, 9] * A[:, 1, 5] * A[:, 2, 1] - A[:, 0, 9] * A[:, 1, 6] * A[:, 2, 0] + \
           A[:, 0, 10] * A[:, 1, 0] * A[:, 2, 5] + A[:, 0, 10] * A[:, 1, 1] * A[:, 2, 4] - A[:, 0, 10] * A[:, 1, 4] * A[:, 2, 1] - \
           A[:, 0, 10] * A[:, 1, 5] * A[:, 2, 0] + A[:, 1, 0] * A[:, 0, 11] * A[:, 2, 4] - A[:, 0, 11] * A[:, 1, 4] * A[:, 2, 0] - \
           A[:, 0, 0] * A[:, 2, 4] * A[:, 1, 11] - A[:, 0, 0] * A[:, 2, 5] * A[:, 1, 10] - A[:, 0, 1] * A[:, 2, 4] * A[:, 1, 10] + \
           A[:, 0, 4] * A[:, 2, 0] * A[:, 1, 11] + A[:, 0, 4] * A[:, 2, 1] * A[:, 1, 10] + A[:, 0, 5] * A[:, 2, 0] * A[:, 1, 10] + \
           A[:, 0, 0] * A[:, 1, 4] * A[:, 2, 11] + A[:, 0, 0] * A[:, 1, 5] * A[:, 2, 10] + A[:, 0, 1] * A[:, 1, 4] * A[:, 2, 10] - \
           A[:, 0, 4] * A[:, 1, 0] * A[:, 2, 11] - A[:, 0, 4] * A[:, 1, 1] * A[:, 2, 10] - A[:, 0, 5] * A[:, 1, 0] * A[:, 2, 10]

    cs[:, 8] = A[:, 0, 0] * A[:, 1, 5] * A[:, 2, 9] + A[:, 0, 0] * A[:, 1, 6] * A[:, 2, 8] - A[:, 0, 0] * A[:, 1, 8] * A[:, 2, 6] - \
           A[:, 0, 0] * A[:, 1, 9] * A[:, 2, 5] + A[:, 0, 1] * A[:, 1, 4] * A[:, 2, 9] + A[:, 0, 1] * A[:, 1, 5] * A[:, 2, 8] - \
           A[:, 0, 1] * A[:, 1, 8] * A[:, 2, 5] - A[:, 0, 1] * A[:, 1, 9] * A[:, 2, 4] + A[:, 0, 2] * A[:, 1, 4] * A[:, 2, 8] - \
           A[:, 0, 2] * A[:, 1, 8] * A[:, 2, 4] - A[:, 0, 4] * A[:, 1, 1] * A[:, 2, 9] - A[:, 0, 4] * A[:, 1, 2] * A[:, 2, 8] + \
           A[:, 0, 4] * A[:, 1, 8] * A[:, 2, 2] + A[:, 0, 4] * A[:, 1, 9] * A[:, 2, 1] - A[:, 0, 5] * A[:, 1, 0] * A[:, 2, 9] - \
           A[:, 0, 5] * A[:, 1, 1] * A[:, 2, 8] + A[:, 0, 5] * A[:, 1, 8] * A[:, 2, 1] + A[:, 0, 5] * A[:, 1, 9] * A[:, 2, 0] - \
           A[:, 0, 6] * A[:, 1, 0] * A[:, 2, 8] + A[:, 0, 6] * A[:, 1, 8] * A[:, 2, 0] + A[:, 0, 8] * A[:, 1, 0] * A[:, 2, 6] + \
           A[:, 0, 8] * A[:, 1, 1] * A[:, 2, 5] + A[:, 0, 8] * A[:, 1, 2] * A[:, 2, 4] - A[:, 0, 8] * A[:, 1, 4] * A[:, 2, 2] - \
           A[:, 0, 8] * A[:, 1, 5] * A[:, 2, 1] - A[:, 0, 8] * A[:, 1, 6] * A[:, 2, 0] + A[:, 0, 9] * A[:, 1, 0] * A[:, 2, 5] + \
           A[:, 0, 9] * A[:, 1, 1] * A[:, 2, 4] - A[:, 0, 9] * A[:, 1, 4] * A[:, 2, 1] - A[:, 0, 9] * A[:, 1, 5] * A[:, 2, 0] + \
           A[:, 0, 10] * A[:, 1, 0] * A[:, 2, 4] - A[:, 0, 10] * A[:, 1, 4] * A[:, 2, 0] - A[:, 0, 0] * A[:, 2, 4] * A[:, 1, 10] + \
           A[:, 0, 4] * A[:, 2, 0] * A[:, 1, 10] + A[:, 0, 0] * A[:, 1, 4] * A[:, 2, 10] - A[:, 0, 4] * A[:, 1, 0] * A[:, 2, 10]

    cs[:, 9] = A[:, 0, 0] * A[:, 1, 4] * A[:, 2, 9] + A[:, 0, 0] * A[:, 1, 5] * A[:, 2, 8] - A[:, 0, 0] * A[:, 1, 8] * A[:, 2, 5] - \
           A[:, 0, 0] * A[:, 1, 9] * A[:, 2, 4] + A[:, 0, 1] * A[:, 1, 4] * A[:, 2, 8] - A[:, 0, 1] * A[:, 1, 8] * A[:, 2, 4] - \
           A[:, 0, 4] * A[:, 1, 0] * A[:, 2, 9] - A[:, 0, 4] * A[:, 1, 1] * A[:, 2, 8] + A[:, 0, 4] * A[:, 1, 8] * A[:, 2, 1] + \
           A[:, 0, 4] * A[:, 1, 9] * A[:, 2, 0] - A[:, 0, 5] * A[:, 1, 0] * A[:, 2, 8] + A[:, 0, 5] * A[:, 1, 8] * A[:, 2, 0] + \
           A[:, 0, 8] * A[:, 1, 0] * A[:, 2, 5] + A[:, 0, 8] * A[:, 1, 1] * A[:, 2, 4] - A[:, 0, 8] * A[:, 1, 4] * A[:, 2, 1] - \
           A[:, 0, 8] * A[:, 1, 5] * A[:, 2, 0] + A[:, 0, 9] * A[:, 1, 0] * A[:, 2, 4] - A[:, 0, 9] * A[:, 1, 4] * A[:, 2, 0]

    cs[:, 10] = A[:, 0, 0] * A[:, 1, 4] * A[:, 2, 8] - A[:, 0, 0] * A[:, 1, 8] * A[:, 2, 4] - A[:, 0, 4] * A[:, 1, 0] * A[:, 2, 8] + \
            A[:, 0, 4] * A[:, 1, 8] * A[:, 2, 0] + A[:, 0, 8] * A[:, 1, 0] * A[:, 2, 4] - A[:, 0, 8] * A[:, 1, 4] * A[:, 2, 0]
    
    E_models = []
    #s = StrumPolynomialSolver(10)
    #n_solss, rootss = s.bisect_sturm(cs)

    # for loop because of different numbers of solutions
    for bi in range(A.shape[0]):
        A_i = A[bi]
        null_i = nullSpace[bi]

        # companion matrix solver
        # try:
        C = torch.zeros((10, 10), device=cs.device, dtype=cs.dtype)
        C[0:-1, 1:] = torch.eye(C[0:-1, 0:-1].shape[0], device=cs.device, dtype=cs.dtype)
        C[-1, :] = -cs[bi][:-1]/cs[bi][-1]
        # check if the companion matrix contains nans or infs
        if torch.isnan(C).any() or torch.isinf(C).any():
            continue
            #n_sols, roots = s.bisect_sturm(cs[bi])
            #print("nan in C")
        else:
            roots = torch.real(torch.linalg.eigvals(C))
        # except ValueError:
        #n_sols, roots = s.bisect_sturm(cs[bi])

        if roots is None:
            continue
        n_sols = roots.size()
        if n_sols == 0:
            continue
        Bs = torch.stack((A_i[:3, :1] * (roots ** 3) + A_i[:3, 1:2] * roots.square() + A_i[0:3, 2:3] * (roots) + A_i[0:3, 3:4],
                        A_i[0:3, 4:5] * (roots ** 3) + A_i[0:3, 5:6] * roots.square() + A_i[0:3, 6:7] * (roots) + A_i[0:3, 7:8]), dim=0).transpose(0 ,-1)

        bs = (A_i[0:3, 8:9] * (roots ** 4) + A_i[0:3, 9:10] * (roots ** 3) + A_i[0:3, 10:11] * roots.square() + A_i[0:3, 11:12] * roots + A_i[0:3, 12:13]).T.unsqueeze(-1)

        # We try to solve using top two rows, if fails, will use matrix decomposition to solve Ax=b.
        try:
            xzs = Bs[:, 0:2, 0:2].inverse() @ (bs[:, 0:2])
        except:
            continue
        mask = (abs(Bs[:, 2].unsqueeze(1) @ xzs - bs[:, 2].unsqueeze(1)) > 1e-3).flatten()
        if torch.sum(mask) != 0:
            q, r = torch.linalg.qr(Bs[mask].clone())#
            xzs[mask] = torch.linalg.solve(r, q.transpose(-1, -2) @ bs[mask])#[mask]

        # models
        Es = null_i[0] * (-xzs[:, 0]) + null_i[1] * (-xzs[:, 1]) + null_i[2] * roots.unsqueeze(-1) + null_i[3]

        # Since the rows of N are orthogonal unit vectors, we can normalize the coefficients instead
        inv = 1.0 / torch.sqrt((-xzs[:, 0]) ** 2 + (-xzs[:, 1]) ** 2 + roots.unsqueeze(-1) ** 2 + 1.0)
        Es *= inv
        if Es.shape[0] < 10:
            Es = torch.concat((Es.clone(), torch.eye(3, device=Es.device, dtype=Es.dtype).repeat(10-Es.shape[0], 1).reshape(-1, 9)))
        E_models.append(Es)

    if not E_models:
        return torch.eye(3, device=cs.device, dtype=cs.dtype).unsqueeze(0)
    else:
        return torch.concat(E_models).view(-1,  3,  3).transpose(-1, -2)
        # be careful of the differences between c++ and python, transpose

   
    # return normalize_transformation(F_est)

def essential_from_fundamental(F_mat: torch.Tensor, K1: torch.Tensor, K2: torch.Tensor) -> torch.Tensor:
    r"""Get Essential matrix from Fundamental and Camera matrices.

    Uses the method from Hartley/Zisserman 9.6 pag 257 (formula 9.12).

    Args:
        F_mat: The fundamental matrix with shape of :math:`(*, 3, 3)`.
        K1: The camera matrix from first camera with shape :math:`(*, 3, 3)`.
        K2: The camera matrix from second camera with shape :math:`(*, 3, 3)`.

    Returns:
        The essential matrix with shape :math:`(*, 3, 3)`.
    """
    if not (len(F_mat.shape) >= 2 and F_mat.shape[-2:] == (3, 3)):
        raise AssertionError(F_mat.shape)
    if not (len(K1.shape) >= 2 and K1.shape[-2:] == (3, 3)):
        raise AssertionError(K1.shape)
    if not (len(K2.shape) >= 2 and K2.shape[-2:] == (3, 3)):
        raise AssertionError(K2.shape)
    if not len(F_mat.shape[:-2]) == len(K1.shape[:-2]) == len(K2.shape[:-2]):
        raise AssertionError

    return K2.transpose(-2, -1) @ F_mat @ K1


def decompose_essential_matrix(E_mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Decompose an essential matrix to possible rotations and translation.

    This function decomposes the essential matrix E using svd decomposition [96]
    and give the possible solutions: :math:`R1, R2, t`.

    Args:
       E_mat: The essential matrix in the form of :math:`(*, 3, 3)`.

    Returns:
       A tuple containing the first and second possible rotation matrices and the translation vector.
       The shape of the tensors with be same input :math:`[(*, 3, 3), (*, 3, 3), (*, 3, 1)]`.
    """
    if not (len(E_mat.shape) >= 2 and E_mat.shape[-2:]):
        raise AssertionError(E_mat.shape)

    # decompose matrix by its singular values
    U, _, V = _torch_svd_cast(E_mat)
    Vt = V.transpose(-2, -1)

    mask = torch.ones_like(E_mat)
    mask[..., -1:] *= -1.0  # fill last column with negative values

    maskt = mask.transpose(-2, -1)

    # avoid singularities
    U = torch.where((torch.det(U) < 0.0)[..., None, None], U * mask, U)
    Vt = torch.where((torch.det(Vt) < 0.0)[..., None, None], Vt * maskt, Vt)

    W = cross_product_matrix(torch.tensor([[0.0, 0.0, 1.0]]).type_as(E_mat))
    W[..., 2, 2] += 1.0

    # reconstruct rotations and retrieve translation vector
    U_W_Vt = U @ W @ Vt
    U_Wt_Vt = U @ W.transpose(-2, -1) @ Vt

    # return values
    R1 = U_W_Vt
    R2 = U_Wt_Vt
    T = U[..., -1:]
    return (R1, R2, T)


def essential_from_Rt(R1: torch.Tensor, t1: torch.Tensor, R2: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    r"""Get the Essential matrix from Camera motion (Rs and ts).

    Reference: Hartley/Zisserman 9.6 pag 257 (formula 9.12)

    Args:
        R1: The first camera rotation matrix with shape :math:`(*, 3, 3)`.
        t1: The first camera translation vector with shape :math:`(*, 3, 1)`.
        R2: The second camera rotation matrix with shape :math:`(*, 3, 3)`.
        t2: The second camera translation vector with shape :math:`(*, 3, 1)`.

    Returns:
        The Essential matrix with the shape :math:`(*, 3, 3)`.
    """
    if not (len(R1.shape) >= 2 and R1.shape[-2:] == (3, 3)):
        raise AssertionError(R1.shape)
    if not (len(t1.shape) >= 2 and t1.shape[-2:] == (3, 1)):
        raise AssertionError(t1.shape)
    if not (len(R2.shape) >= 2 and R2.shape[-2:] == (3, 3)):
        raise AssertionError(R2.shape)
    if not (len(t2.shape) >= 2 and t2.shape[-2:] == (3, 1)):
        raise AssertionError(t2.shape)

    # first compute the camera relative motion
    R, t = relative_camera_motion(R1, t1, R2, t2)

    # get the cross product from relative translation vector
    Tx = cross_product_matrix(t[..., 0])

    return Tx @ R


def motion_from_essential(E_mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Get Motion (R's and t's ) from Essential matrix.

    Computes and return four possible poses exist for the decomposition of the Essential
    matrix. The possible solutions are :math:`[R1,t], [R1,-t], [R2,t], [R2,-t]`.

    Args:
        E_mat: The essential matrix in the form of :math:`(*, 3, 3)`.

    Returns:
        The rotation and translation containing the four possible combination for the retrieved motion.
        The tuple is as following :math:`[(*, 4, 3, 3), (*, 4, 3, 1)]`.
    """
    if not (len(E_mat.shape) >= 2 and E_mat.shape[-2:] == (3, 3)):
        raise AssertionError(E_mat.shape)

    # decompose the essential matrix by its possible poses
    R1, R2, t = decompose_essential_matrix(E_mat)

    # compbine and returns the four possible solutions
    Rs = torch.stack([R1, R1, R2, R2], dim=-3)
    Ts = torch.stack([t, -t, t, -t], dim=-3)

    return (Rs, Ts)


def motion_from_essential_choose_solution(
    E_mat: torch.Tensor,
    K1: torch.Tensor,
    K2: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Recover the relative camera rotation and the translation from an estimated essential matrix.

    The method checks the corresponding points in two images and also returns the triangulated
    3d points. Internally uses :py:meth:`~kornia.geometry.epipolar.decompose_essential_matrix` and then chooses
    the best solution based on the combination that gives more 3d points in front of the camera plane from
    :py:meth:`~kornia.geometry.epipolar.triangulate_points`.

    Args:
        E_mat: The essential matrix in the form of :math:`(*, 3, 3)`.
        K1: The camera matrix from first camera with shape :math:`(*, 3, 3)`.
        K2: The camera matrix from second camera with shape :math:`(*, 3, 3)`.
        x1: The set of points seen from the first camera frame in the camera plane
          coordinates with shape :math:`(*, N, 2)`.
        x2: The set of points seen from the first camera frame in the camera plane
          coordinates with shape :math:`(*, N, 2)`.
        mask: A boolean mask which can be used to exclude some points from choosing
          the best solution. This is useful for using this function with sets of points of
          different cardinality (for instance after filtering with RANSAC) while keeping batch
          semantics. Mask is of shape :math:`(*, N)`.

    Returns:
        The rotation and translation plus the 3d triangulated points.
        The tuple is as following :math:`[(*, 3, 3), (*, 3, 1), (*, N, 3)]`.
    """
    if not (len(E_mat.shape) >= 2 and E_mat.shape[-2:] == (3, 3)):
        raise AssertionError(E_mat.shape)
    if not (len(K1.shape) >= 2 and K1.shape[-2:] == (3, 3)):
        raise AssertionError(K1.shape)
    if not (len(K2.shape) >= 2 and K2.shape[-2:] == (3, 3)):
        raise AssertionError(K2.shape)
    if not (len(x1.shape) >= 2 and x1.shape[-1] == 2):
        raise AssertionError(x1.shape)
    if not (len(x2.shape) >= 2 and x2.shape[-1] == 2):
        raise AssertionError(x2.shape)
    if not len(E_mat.shape[:-2]) == len(K1.shape[:-2]) == len(K2.shape[:-2]):
        raise AssertionError
    if mask is not None:
        if len(mask.shape) < 1:
            raise AssertionError(mask.shape)
        if mask.shape != x1.shape[:-1]:
            raise AssertionError(mask.shape)

    unbatched = len(E_mat.shape) == 2

    if unbatched:
        # add a leading batch dimension. We will remove it at the end, before
        # returning the results
        E_mat = E_mat[None]
        K1 = K1[None]
        K2 = K2[None]
        x1 = x1[None]
        x2 = x2[None]
        if mask is not None:
            mask = mask[None]

    # compute four possible pose solutions
    Rs, ts = motion_from_essential(E_mat)

    # set reference view pose and compute projection matrix
    R1 = eye_like(3, E_mat)  # Bx3x3
    t1 = vec_like(3, E_mat)  # Bx3x1

    # compute the projection matrices for first camera
    R1 = R1[:, None].expand(-1, 4, -1, -1)
    t1 = t1[:, None].expand(-1, 4, -1, -1)
    K1 = K1[:, None].expand(-1, 4, -1, -1)
    P1 = projection_from_KRt(K1, R1, t1)  # 1x4x4x4

    # compute the projection matrices for second camera
    R2 = Rs
    t2 = ts
    K2 = K2[:, None].expand(-1, 4, -1, -1)
    P2 = projection_from_KRt(K2, R2, t2)  # Bx4x4x4

    # triangulate the points
    x1 = x1[:, None].expand(-1, 4, -1, -1)
    x2 = x2[:, None].expand(-1, 4, -1, -1)
    X = triangulate_points(P1, P2, x1, x2)  # Bx4xNx3

    # project points and compute their depth values
    d1 = depth_from_point(R1, t1, X)
    d2 = depth_from_point(R2, t2, X)

    # verify the point values that have a positive depth value
    depth_mask = (d1 > 0.0) & (d2 > 0.0)
    if mask is not None:
        depth_mask &= mask.unsqueeze(1)

    mask_indices = torch.max(depth_mask.sum(-1), dim=-1, keepdim=True)[1]

    # get pose and points 3d and return
    R_out = Rs[:, mask_indices][:, 0, 0]
    t_out = ts[:, mask_indices][:, 0, 0]
    points3d_out = X[:, mask_indices][:, 0, 0]

    if unbatched:
        R_out = R_out[0]
        t_out = t_out[0]
        points3d_out = points3d_out[0]

    return R_out, t_out, points3d_out


def relative_camera_motion(
    R1: torch.Tensor, t1: torch.Tensor, R2: torch.Tensor, t2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the relative camera motion between two cameras.

    Given the motion parameters of two cameras, computes the motion parameters of the second
    one assuming the first one to be at the origin. If :math:`T1` and :math:`T2` are the camera motions,
    the computed relative motion is :math:`T = T_{2}T^{-1}_{1}`.

    Args:
        R1: The first camera rotation matrix with shape :math:`(*, 3, 3)`.
        t1: The first camera translation vector with shape :math:`(*, 3, 1)`.
        R2: The second camera rotation matrix with shape :math:`(*, 3, 3)`.
        t2: The second camera translation vector with shape :math:`(*, 3, 1)`.

    Returns:
        A tuple with the relative rotation matrix and
        translation vector with the shape of :math:`[(*, 3, 3), (*, 3, 1)]`.
    """
    if not (len(R1.shape) >= 2 and R1.shape[-2:] == (3, 3)):
        raise AssertionError(R1.shape)
    if not (len(t1.shape) >= 2 and t1.shape[-2:] == (3, 1)):
        raise AssertionError(t1.shape)
    if not (len(R2.shape) >= 2 and R2.shape[-2:] == (3, 3)):
        raise AssertionError(R2.shape)
    if not (len(t2.shape) >= 2 and t2.shape[-2:] == (3, 1)):
        raise AssertionError(t2.shape)

    # compute first the relative rotation
    R = R2 @ R1.transpose(-2, -1)

    # compute the relative translation vector
    t = t2 - R @ t1

    return (R, t)


def find_essential(
    points1: torch.Tensor, points2: torch.Tensor, weights: Optional[torch.Tensor] = None, gt_model: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""
    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2), N>=5`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2), N>=5`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(5, N)`.

    Returns:
        the computed essential matrix with shape :math:`(B, 3*m, 3)`, where `m` number of essential matrix.

    Raises:
        ValueError: If an invalid method is provided.

    """
    E = run_5point(points1, points2, weights)

    # select one out of 10 possible solutions from 5PC Nister solver.
    solution_num = 10
    batch_size = points1.shape[0]

    error = torch.zeros((batch_size, solution_num))
    if gt_model is None:
        for b in range(batch_size):
            error[b] = epi.sampson_epipolar_distance(points1[b], points2[b], E.view(batch_size, solution_num, 3, 3)[b]).sum(-1)
    else:
        for b in range(batch_size):
            error[b] = torch.norm(E.view(batch_size, solution_num, 3, 3)[b] - gt_model[b], dim=(1, 2)).view(solution_num, -1)
    
    assert batch_size == error.shape[0]
    assert solution_num == error.shape[1]

    chosen_indices = torch.argmin(error, dim=-1)
    result = torch.stack([(E.view(-1, solution_num, 3, 3))[i, chosen_indices[i], :] for i in range(batch_size)])#int(E.shape[0] / solution_num)

    return result
