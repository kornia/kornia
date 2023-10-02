"""Module containing functionalities for the Essential matrix."""
from typing import Optional, Tuple

import torch

import kornia.geometry.epipolar as epi
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SAME_SHAPE, KORNIA_CHECK_SHAPE
from kornia.utils import eye_like, vec_like
from kornia.utils.helpers import _torch_svd_cast

from .numeric import cross_product_matrix
from .projection import depth_from_point, projection_from_KRt
from .triangulation import triangulate_points

__all__ = [
    "find_essential",
    "essential_from_fundamental",
    "decompose_essential_matrix",
    "essential_from_Rt",
    "motion_from_essential",
    "motion_from_essential_choose_solution",
    "relative_camera_motion",
]

# Reference
# Nistér, David. An efficient solution to the five-point relative pose problem. 2004.
# https://github.com/danini/graph-cut-ransac/blob/master/src/pygcransac/include/estimators/solver_essential_matrix_five_point_nister.h
# Wei T, Patel Y, Matas J, Barath D. Generalized differentiable RANSAC[J]. arXiv preprint arXiv:2212.13185, 2023.
# https://github.com/weitong8591/differentiable_ransac/blob/main/estimators/essential_matrix_estimator_nister.py


def run_5point(points1: torch.Tensor, points2: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""Compute the essential matrix using the 5-point algorithm from Nister.

    The linear system is solved by Nister's 5-point algorithm.

    Args:
        points1: A set of carlibrated points in the first image with a tensor shape :math:`(B, N, 2), N>=8`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2), N>=8`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.

    Returns:
        the computed fundamental matrix with shape :math:`(B, 3, 3)`.
    """
    KORNIA_CHECK_SHAPE(points1, ['B', 'N', '2'])
    KORNIA_CHECK_SAME_SHAPE(points1, points2)
    KORNIA_CHECK(points1.shape[1] >= 5, "Number of points should be >=5")
    if weights is not None:
        KORNIA_CHECK_SAME_SHAPE(points1[:, :, 0], weights)

    batch_size, _, _ = points1.shape
    x1, y1 = torch.chunk(points1, dim=-1, chunks=2)  # Bx1xN
    x2, y2 = torch.chunk(points2, dim=-1, chunks=2)  # Bx1xN
    ones = torch.ones_like(x1)

    # build equations system and solve DLT
    # https://www.cc.gatech.edu/~afb/classes/CS4495-Fall2013/slides/CS4495-09-TwoViews-2.pdf
    # [x * x', x * y', x, y * x', y * y', y, x', y', 1]
    # BxNx9
    X = torch.cat([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, ones], dim=-1)

    # apply the weights to the linear system
    if weights is None:
        X = X.transpose(-2, -1) @ X
    else:
        w_diag = torch.diag_embed(weights)
        X = X.transpose(-2, -1) @ w_diag @ X
    # compute eigevectors and retrieve the one with the smallest eigenvalue, using SVD
    # turn off the grad check due to the unstable gradients from SVD.
    # several close to zero values of eigenvalues.
    _, _, V = _torch_svd_cast(X)  # torch.svd
    null_ = V[:, :, -4:]  # the last four rows
    nullSpace = V.transpose(-1, -2)[:, -4:, :]

    coeffs = torch.zeros(batch_size, 10, 20, device=null_.device, dtype=null_.dtype)
    d = torch.zeros(batch_size, 60, device=null_.device, dtype=null_.dtype)

    # fun = lambda i, j : null_[:, 3 * j + i]
    def fun(i: int, j: int) -> torch.Tensor:
        return null_[:, 3 * j + i]

    # Determinant constraint
    coeffs[:, 9] = (
        epi.numeric.o2(epi.numeric.o1(fun(0, 1), fun(1, 2)) - epi.numeric.o1(fun(0, 2), fun(1, 1)), fun(2, 0))
        + epi.numeric.o2(epi.numeric.o1(fun(0, 2), fun(1, 0)) - epi.numeric.o1(fun(0, 0), fun(1, 2)), fun(2, 1))
        + epi.numeric.o2(epi.numeric.o1(fun(0, 0), fun(1, 1)) - epi.numeric.o1(fun(0, 1), fun(1, 0)), fun(2, 2))
    )

    indices = torch.tensor([[0, 10, 20], [10, 40, 30], [20, 30, 50]])

    # Compute EE^T (Eqn. 20 in the paper)
    for i in range(3):
        for j in range(3):
            d[:, indices[i, j] : indices[i, j] + 10] = (
                epi.numeric.o1(fun(i, 0), fun(j, 0))
                + epi.numeric.o1(fun(i, 1), fun(j, 1))
                + epi.numeric.o1(fun(i, 2), fun(j, 2))
            )

    for i in range(10):
        t = 0.5 * (d[:, indices[0, 0] + i] + d[:, indices[1, 1] + i] + d[:, indices[2, 2] + i])
        d[:, indices[0, 0] + i] -= t
        d[:, indices[1, 1] + i] -= t
        d[:, indices[2, 2] + i] -= t

    cnt = 0
    for i in range(3):
        for j in range(3):
            row = (
                epi.numeric.o2(d[:, indices[i, 0] : indices[i, 0] + 10], fun(0, j))
                + epi.numeric.o2(d[:, indices[i, 1] : indices[i, 1] + 10], fun(1, j))
                + epi.numeric.o2(d[:, indices[i, 2] : indices[i, 2] + 10], fun(2, j))
            )
            coeffs[:, cnt] = row
            cnt += 1

    b = coeffs[:, :, 10:]
    singular_filter = torch.linalg.matrix_rank(coeffs[:, :, :10]) >= torch.max(
        torch.linalg.matrix_rank(coeffs), torch.ones_like(torch.linalg.matrix_rank(coeffs[:, :, :10])) * 10
    )

    eliminated_mat = torch.linalg.solve(coeffs[singular_filter, :, :10], b[singular_filter])

    coeffs_ = torch.cat((coeffs[singular_filter, :, :10], eliminated_mat), dim=-1)

    A = torch.zeros(coeffs_.shape[0], 3, 13, device=coeffs_.device, dtype=coeffs_.dtype)

    for i in range(3):
        A[:, i, 0] = 0.0
        A[:, i : i + 1, 1:4] = coeffs_[:, 4 + 2 * i : 5 + 2 * i, 10:13]
        A[:, i : i + 1, 0:3] -= coeffs_[:, 5 + 2 * i : 6 + 2 * i, 10:13]
        A[:, i, 4] = 0.0
        A[:, i : i + 1, 5:8] = coeffs_[:, 4 + 2 * i : 5 + 2 * i, 13:16]
        A[:, i : i + 1, 4:7] -= coeffs_[:, 5 + 2 * i : 6 + 2 * i, 13:16]
        A[:, i, 8] = 0.0
        A[:, i : i + 1, 9:13] = coeffs_[:, 4 + 2 * i : 5 + 2 * i, 16:20]
        A[:, i : i + 1, 8:12] -= coeffs_[:, 5 + 2 * i : 6 + 2 * i, 16:20]

    cs = epi.numeric.det_to_poly(A)
    E_models = []

    # for loop because of different numbers of solutions
    for bi in range(A.shape[0]):
        A_i = A[bi]
        null_i = nullSpace[bi]

        # companion matrix solver for polynomial
        C = torch.zeros((10, 10), device=cs.device, dtype=cs.dtype)
        C[0:-1, 1:] = torch.eye(C[0:-1, 0:-1].shape[0], device=cs.device, dtype=cs.dtype)
        C[-1, :] = -cs[bi][:-1] / cs[bi][-1]

        roots = torch.real(torch.linalg.eigvals(C))

        if roots is None:
            continue
        n_sols = roots.size()
        if n_sols == 0:
            continue
        Bs = torch.stack(
            (
                A_i[:3, :1] * (roots**3) + A_i[:3, 1:2] * roots.square() + A_i[0:3, 2:3] * (roots) + A_i[0:3, 3:4],
                A_i[0:3, 4:5] * (roots**3) + A_i[0:3, 5:6] * roots.square() + A_i[0:3, 6:7] * (roots) + A_i[0:3, 7:8],
            ),
            dim=0,
        ).transpose(0, -1)

        bs = (
            A_i[0:3, 8:9] * (roots**4)
            + A_i[0:3, 9:10] * (roots**3)
            + A_i[0:3, 10:11] * roots.square()
            + A_i[0:3, 11:12] * roots
            + A_i[0:3, 12:13]
        ).T.unsqueeze(-1)

        # We try to solve using top two rows,
        xzs = Bs[:, 0:2, 0:2].inverse() @ (bs[:, 0:2])

        mask = (abs(Bs[:, 2].unsqueeze(1) @ xzs - bs[:, 2].unsqueeze(1)) > 1e-3).flatten()
        if torch.sum(mask) != 0:
            q, r = torch.linalg.qr(Bs[mask].clone())  #
            xzs[mask] = torch.linalg.solve(r, q.transpose(-1, -2) @ bs[mask])  # [mask]

        # models
        Es = null_i[0] * (-xzs[:, 0]) + null_i[1] * (-xzs[:, 1]) + null_i[2] * roots.unsqueeze(-1) + null_i[3]

        # Since the rows of N are orthogonal unit vectors, we can normalize the coefficients instead
        inv = 1.0 / torch.sqrt((-xzs[:, 0]) ** 2 + (-xzs[:, 1]) ** 2 + roots.unsqueeze(-1) ** 2 + 1.0)
        Es *= inv
        if Es.shape[0] < 10:
            Es = torch.cat(
                (Es.clone(), torch.eye(3, device=Es.device, dtype=Es.dtype).repeat(10 - Es.shape[0], 1).reshape(-1, 9))
            )
        E_models.append(Es)

    # if not E_models:
    #     return torch.eye(3, device=cs.device, dtype=cs.dtype).unsqueeze(0)
    # else:
    return torch.cat(E_models).view(-1, 3, 3).transpose(-1, -2)


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
    points1: torch.Tensor, points2: torch.Tensor, weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""
    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2), N>=5`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2), N>=5`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(5, N)`.

    Returns:
        the computed essential matrix with shape :math:`(B, 3, 3)`,
        one model for each batch selceted out of ten solutions by Sampson distances.

    """
    E = run_5point(points1, points2, weights).to(points1.dtype)

    # select one out of 10 possible solutions from 5PC Nister solver.
    solution_num = 10
    batch_size = points1.shape[0]

    error = torch.zeros((batch_size, solution_num))

    for b in range(batch_size):
        error[b] = epi.sampson_epipolar_distance(points1[b], points2[b], E.view(batch_size, solution_num, 3, 3)[b]).sum(
            -1
        )

    KORNIA_CHECK_SHAPE(error, ['f{batch_size}', '10'])

    chosen_indices = torch.argmin(error, dim=-1)
    result = torch.stack([(E.view(-1, solution_num, 3, 3))[i, chosen_indices[i], :] for i in range(batch_size)])

    return result
