"""Module containing functionalities for the Essential matrix."""

from typing import Optional, Tuple

import torch

from kornia.core import eye, ones_like, stack, where, zeros
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SAME_SHAPE, KORNIA_CHECK_SHAPE
from kornia.geometry import solvers
from kornia.utils import eye_like, vec_like
from kornia.utils.helpers import _torch_solve_cast, _torch_svd_cast

from .numeric import cross_product_matrix, matrix_cofactor_tensor
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
    "decompose_essential_matrix_no_svd",
]


def run_5point(points1: torch.Tensor, points2: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""Compute the essential matrix using the 5-point algorithm from Nister.

    The linear system is solved by Nister's 5-point algorithm [@nister2004efficient],
    and the solver implemented referred to [@barath2020magsac++][@wei2023generalized][@wang2023vggsfm].

    Args:
        points1: A set of carlibrated points in the first image with a tensor shape :math:`(B, N, 2), N>=8`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2), N>=8`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.

    Returns:
        the computed essential matrix with shape :math:`(B, 3, 3)`.
    """
    KORNIA_CHECK_SHAPE(points1, ["B", "N", "2"])
    KORNIA_CHECK_SAME_SHAPE(points1, points2)
    KORNIA_CHECK(points1.shape[1] >= 5, "Number of points should be >=5")
    if weights is not None:
        KORNIA_CHECK_SAME_SHAPE(points1[:, :, 0], weights)

    batch_size, _, _ = points1.shape
    x1, y1 = torch.chunk(points1, dim=-1, chunks=2)  # Bx1xN
    x2, y2 = torch.chunk(points2, dim=-1, chunks=2)  # Bx1xN
    ones = ones_like(x1)

    # build the equation system and find the null space.
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

    # use Nister's 5PC to solve essential matrix
    E_Nister = null_to_Nister_solution(X, batch_size)

    return E_Nister


def fun_select(null_mat: torch.Tensor, i: int, j: int, ratio: int = 3) -> torch.Tensor:
    return null_mat[:, ratio * j + i]


def null_to_Nister_solution(X: torch.Tensor, batch_size: int) -> torch.Tensor:
    r"""Use Nister's 5PC to solve essential matrix.

    The linear system is solved by Nister's 5-point algorithm [@nister2004efficient],
    and the solver implemented referred to [@barath2020magsac++][@wei2023generalized][@wang2023vggsfm].

    Args:
        X: Coefficients for the null space :math:`(B, N, 2), N>=8`.
        batch_size: batcs size of the input, the number of image pairs :math:`B`.

    Returns:
        the computed essential matrix with shape :math:`(B, 3, 3)`.
        Note that the returned E matrices should be the same batch size with the input.
    """

    # compute eigenvectors and retrieve the one with the smallest eigenvalue, using SVD
    # turn off the grad check due to the unstable gradients from SVD.
    # several close to zero values of eigenvalues.
    _, _, V = _torch_svd_cast(X)  # torch.svd

    null_ = V[:, :, -4:]  # the last four rows
    nullSpace = V.transpose(-1, -2)[:, -4:, :]

    coeffs = zeros(batch_size, 10, 20, device=null_.device, dtype=null_.dtype)
    d = zeros(batch_size, 60, device=null_.device, dtype=null_.dtype)

    # Determinant constraint
    coeffs[:, 9] = (
        solvers.multiply_deg_two_one_poly(
            solvers.multiply_deg_one_poly(fun_select(null_, 0, 1), fun_select(null_, 1, 2))
            - solvers.multiply_deg_one_poly(fun_select(null_, 0, 2), fun_select(null_, 1, 1)),
            fun_select(null_, 2, 0),
        )
        + solvers.multiply_deg_two_one_poly(
            solvers.multiply_deg_one_poly(fun_select(null_, 0, 2), fun_select(null_, 1, 0))
            - solvers.multiply_deg_one_poly(fun_select(null_, 0, 0), fun_select(null_, 1, 2)),
            fun_select(null_, 2, 1),
        )
        + solvers.multiply_deg_two_one_poly(
            solvers.multiply_deg_one_poly(fun_select(null_, 0, 0), fun_select(null_, 1, 1))
            - solvers.multiply_deg_one_poly(fun_select(null_, 0, 1), fun_select(null_, 1, 0)),
            fun_select(null_, 2, 2),
        )
    )

    indices = torch.tensor([[0, 10, 20], [10, 40, 30], [20, 30, 50]])

    # Compute EE^T (Eqn. 20 in the paper)
    for i in range(3):
        for j in range(3):
            d[:, indices[i, j] : indices[i, j] + 10] = (
                solvers.multiply_deg_one_poly(fun_select(null_, i, 0), fun_select(null_, j, 0))
                + solvers.multiply_deg_one_poly(fun_select(null_, i, 1), fun_select(null_, j, 1))
                + solvers.multiply_deg_one_poly(fun_select(null_, i, 2), fun_select(null_, j, 2))
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
                solvers.multiply_deg_two_one_poly(d[:, indices[i, 0] : indices[i, 0] + 10], fun_select(null_, 0, j))
                + solvers.multiply_deg_two_one_poly(d[:, indices[i, 1] : indices[i, 1] + 10], fun_select(null_, 1, j))
                + solvers.multiply_deg_two_one_poly(d[:, indices[i, 2] : indices[i, 2] + 10], fun_select(null_, 2, j))
            )
            coeffs[:, cnt] = row
            cnt += 1

    b = coeffs[:, :, 10:]
    singular_filter = torch.linalg.matrix_rank(coeffs[:, :, :10]) >= torch.max(
        torch.linalg.matrix_rank(coeffs), ones_like(torch.linalg.matrix_rank(coeffs[:, :, :10])) * 10
    )

    # check if there is no solution
    if singular_filter.sum() == 0:
        return torch.eye(3, dtype=coeffs.dtype, device=coeffs.device)[None].expand(batch_size, 10, -1, -1).clone()

    eliminated_mat = _torch_solve_cast(coeffs[singular_filter, :, :10], b[singular_filter])

    coeffs_ = torch.cat((coeffs[singular_filter, :, :10], eliminated_mat), dim=-1)

    # check the batch size after singular filter, for batch operation afterwards
    batch_size_filtered = coeffs_.shape[0]

    A = zeros(coeffs_.shape[0], 3, 13, device=coeffs_.device, dtype=coeffs_.dtype)

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

    # Bx11
    cs = solvers.determinant_to_polynomial(A)

    # A: Bx3x13
    # nullSpace: Bx4x9
    # companion matrices to solve the polynomial, in batch
    C = zeros((batch_size_filtered, 10, 10), device=cs.device, dtype=cs.dtype)
    eye_mat = eye(C[0, 0:-1, 0:-1].shape[0], device=cs.device, dtype=cs.dtype)
    C[:, 0:-1, 1:] = eye_mat

    cs_de = cs[:, -1].unsqueeze(-1)
    cs_de = torch.where(cs_de == 0, torch.tensor(1e-8, dtype=cs_de.dtype), cs_de)
    C[:, -1, :] = -cs[:, :-1] / cs_de

    roots = torch.real(torch.linalg.eigvals(C))

    roots_unsqu = roots.unsqueeze(1)

    Bs = stack(
        (
            A[:, :3, :1] * (roots_unsqu**3)
            + A[:, :3, 1:2] * roots_unsqu.square()
            + A[:, 0:3, 2:3] * roots_unsqu
            + A[:, 0:3, 3:4],
            A[:, 0:3, 4:5] * (roots_unsqu**3)
            + A[:, 0:3, 5:6] * roots_unsqu.square()
            + A[:, 0:3, 6:7] * roots_unsqu
            + A[:, 0:3, 7:8],
        ),
        dim=1,
    )

    Bs = Bs.transpose(1, -1)

    bs = (
        (
            A[:, 0:3, 8:9] * (roots_unsqu**4)
            + A[:, 0:3, 9:10] * (roots_unsqu**3)
            + A[:, 0:3, 10:11] * roots_unsqu.square()
            + A[:, 0:3, 11:12] * roots_unsqu
            + A[:, 0:3, 12:13]
        )
        .transpose(1, 2)
        .unsqueeze(-1)
    )

    xzs = torch.matmul(torch.inverse(Bs[:, :, 0:2, 0:2]), bs[:, :, 0:2])

    mask = (abs(Bs[:, 2].unsqueeze(1) @ xzs - bs[:, 2].unsqueeze(1)) > 1e-3).flatten()

    # mask: bx10x1x1
    mask = (
        abs(torch.matmul(Bs[:, :, 2, :].unsqueeze(2), xzs) - bs[:, :, 2, :].unsqueeze(2)) > 1e-3
    )  # .flatten(start_dim=1)

    # bx10
    mask = mask.squeeze(3).squeeze(2)

    if torch.any(mask):
        q_batch, r_batch = torch.linalg.qr(Bs[mask])
        xyz_to_feed = torch.linalg.solve(r_batch, torch.matmul(q_batch.transpose(-1, -2), bs[mask]))
        xzs[mask] = xyz_to_feed

    nullSpace_filtered = nullSpace[singular_filter]

    Es = (
        nullSpace_filtered[:, 0:1] * (-xzs[:, :, 0])
        + nullSpace_filtered[:, 1:2] * (-xzs[:, :, 1])
        + nullSpace_filtered[:, 2:3] * roots.unsqueeze(-1)
        + nullSpace_filtered[:, 3:4]
    )

    inv = 1.0 / torch.sqrt((-xzs[:, :, 0]) ** 2 + (-xzs[:, :, 1]) ** 2 + roots.unsqueeze(-1) ** 2 + 1.0)
    Es *= inv

    Es = Es.view(batch_size_filtered, -1, 3, 3).transpose(-1, -2)

    # make sure the returned batch size equals to that of inputs
    E_return = torch.eye(3, dtype=Es.dtype, device=Es.device)[None].expand(batch_size, 10, -1, -1).clone()
    E_return[singular_filter] = Es

    return E_return


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
    KORNIA_CHECK_SHAPE(F_mat, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(K1, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(K2, ["*", "3", "3"])
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
    KORNIA_CHECK_SHAPE(E_mat, ["*", "3", "3"])

    # decompose matrix by its singular values
    U, _, V = _torch_svd_cast(E_mat)
    Vt = V.transpose(-2, -1)

    mask = ones_like(E_mat)
    mask[..., -1:] *= -1.0  # fill last column with negative values

    maskt = mask.transpose(-2, -1)

    # avoid singularities
    U = where((torch.det(U) < 0.0)[..., None, None], U * mask, U)
    Vt = where((torch.det(Vt) < 0.0)[..., None, None], Vt * maskt, Vt)

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


def decompose_essential_matrix_no_svd(E_mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Decompose the essential matrix to rotation and translation.

       Recover rotation and translation from essential matrices without SVD
      reference: Horn, Berthold KP. Recovering baseline and orientation from essential matrix[J].
      J. Opt. Soc. Am, 1990, 110.

      Args:
       E_mat: The essential matrix in the form of :math:`(*, 3, 3)`.

    Returns:
       A tuple containing the first and second possible rotation matrices and the translation vector.
       The shape of the tensors with be same input :math:`[(*, 3, 3), (*, 3, 3), (*, 3, 1)]`.
    """
    KORNIA_CHECK_SHAPE(E_mat, ["*", "3", "3"])
    if len(E_mat.shape) != 3:
        E_mat = E_mat.view(-1, 3, 3)

    B = E_mat.shape[0]

    # Eq.18, choose the largest of the three possible pairwise cross-products
    e1, e2, e3 = E_mat[..., 0], E_mat[..., 1], E_mat[..., 2]

    # sqrt(1/2 trace(EE^T)), B
    scale_factor = torch.sqrt(0.5 * torch.diagonal(E_mat @ E_mat.transpose(-1, -2), dim1=-1, dim2=-2).sum(-1))

    # B, 3, 3
    cross_products = torch.stack([torch.cross(e1, e2), torch.cross(e2, e3), torch.cross(e3, e1)], dim=1)
    # B, 3, 1
    norms = torch.norm(cross_products, dim=-1, keepdim=True)

    # B, to select which b1
    largest = torch.argmax(norms, dim=-2)

    # B, 3, 3
    e_cross_products = scale_factor[:, None, None] * cross_products / norms

    # broadcast the index
    index_expanded = largest.unsqueeze(-1).expand(-1, -1, e_cross_products.size(-1))

    # slice at dim=1, select for each batch one b (e1*e2 or e2*e3 or e3*e1), B, 1, 3
    b1 = torch.gather(e_cross_products, dim=1, index=index_expanded).squeeze(1)
    # normalization
    b1_ = b1 / torch.norm(b1, dim=-1, keepdim=True)

    # skew-symmetric matrix
    B1 = torch.zeros((B, 3, 3), device=E_mat.device, dtype=E_mat.dtype)
    t0, t1, t2 = b1[:, 0], b1[:, 1], b1[:, 2]
    B1[:, 0, 1], B1[:, 1, 0] = -t2, t2
    B1[:, 0, 2], B1[:, 2, 0] = t1, -t1
    B1[:, 1, 2], B1[:, 2, 1] = -t0, t0

    # the second translation and rotation
    B2 = -B1
    b2 = -b1

    # Eq.24, recover R
    # (bb)R = Cofactors(E)^T - BE
    R1 = (matrix_cofactor_tensor(E_mat) - B1 @ E_mat) / (b1 * b1).sum().unsqueeze(-1)
    R2 = (matrix_cofactor_tensor(E_mat) - B2 @ E_mat) / (b2 * b2).sum().unsqueeze(-1)

    return (R1, R2, b1_.unsqueeze(-1))


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
    KORNIA_CHECK_SHAPE(R1, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(R2, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(t1, ["*", "3", "1"])
    KORNIA_CHECK_SHAPE(t2, ["*", "3", "1"])

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
    KORNIA_CHECK_SHAPE(E_mat, ["*", "3", "3"])

    # decompose the essential matrix by its possible poses
    R1, R2, t = decompose_essential_matrix(E_mat)

    # compbine and returns the four possible solutions
    Rs = stack([R1, R1, R2, R2], dim=-3)
    Ts = stack([t, -t, t, -t], dim=-3)

    return Rs, Ts


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
    KORNIA_CHECK_SHAPE(E_mat, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(K1, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(K2, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(x1, ["*", "N", "2"])
    KORNIA_CHECK_SHAPE(x2, ["*", "N", "2"])
    KORNIA_CHECK(len(E_mat.shape[:-2]) == len(K1.shape[:-2]) == len(K2.shape[:-2]))

    if mask is not None:
        KORNIA_CHECK_SHAPE(mask, ["*", "N"])
        KORNIA_CHECK(mask.shape == x1.shape[:-1])

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
    KORNIA_CHECK_SHAPE(R1, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(R2, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(t1, ["*", "3", "1"])
    KORNIA_CHECK_SHAPE(t2, ["*", "3", "1"])

    # compute first the relative rotation
    R = R2 @ R1.transpose(-2, -1)

    # compute the relative translation vector
    t = t2 - R @ t1

    return R, t


def find_essential(
    points1: torch.Tensor, points2: torch.Tensor, weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""
     Args:
         points1: A set of points in the first image with a tensor shape :math:`(B, N, 2), N>=5`.
         points2: A set of points in the second image with a tensor shape :math:`(B, N, 2), N>=5`.
         weights: Tensor containing the weights per point correspondence with a shape of :math:`(5, N)`.

    Returns:
         the computed essential matrices with shape :math:`(B, 10, 3, 3)`.
         Note that all possible solutions are returned, i.e., 10 essential matrices for each image pair.
         To choose the best one out of 10, try to check the one with the lowest Sampson distance.

    """
    E = run_5point(points1, points2, weights).to(points1.dtype)
    return E
