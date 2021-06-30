"""Module containing functionalities for the Essential matrix."""
from typing import Optional, Tuple

import torch

from kornia.geometry.epipolar import numeric, projection, triangulation


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
    assert len(F_mat.shape) >= 2 and F_mat.shape[-2:] == (3, 3), F_mat.shape
    assert len(K1.shape) >= 2 and K1.shape[-2:] == (3, 3), K1.shape
    assert len(K2.shape) >= 2 and K2.shape[-2:] == (3, 3), K2.shape
    assert len(F_mat.shape[:-2]) == len(K1.shape[:-2]) == len(K2.shape[:-2])

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
    assert len(E_mat.shape) >= 2 and E_mat.shape[-2:], E_mat.shape

    # decompose matrix by its singular values
    U, S, V = torch.svd(E_mat)
    Vt = V.transpose(-2, -1)

    mask = torch.ones_like(E_mat)
    mask[..., -1:] *= -1.0  # fill last column with negative values

    maskt = mask.transpose(-2, -1)

    # avoid singularities
    U = torch.where((torch.det(U) < 0.0)[..., None, None], U * mask, U)
    Vt = torch.where((torch.det(Vt) < 0.0)[..., None, None], Vt * maskt, Vt)

    W = numeric.cross_product_matrix(torch.tensor([[0.0, 0.0, 1.0]]).type_as(E_mat))
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
    assert len(R1.shape) >= 2 and R1.shape[-2:] == (3, 3), R1.shape
    assert len(t1.shape) >= 2 and t1.shape[-2:] == (3, 1), t1.shape
    assert len(R2.shape) >= 2 and R2.shape[-2:] == (3, 3), R2.shape
    assert len(t2.shape) >= 2 and t2.shape[-2:] == (3, 1), t2.shape

    # first compute the camera relative motion
    R, t = relative_camera_motion(R1, t1, R2, t2)

    # get the cross product from relative translation vector
    Tx = numeric.cross_product_matrix(t[..., 0])

    return Tx @ R


def motion_from_essential(E_mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Get Motion (R's and t's ) from Essential matrix.

    Computes and return four possible poses exist for the decomposition of the Essential
    matrix. The posible solutions are :math:`[R1,t], [R1,−t], [R2,t], [R2,−t]`.

    Args:
        E_mat: The essential matrix in the form of :math:`(*, 3, 3)`.

    Returns:
        The rotation and translation containing the four possible combination for the retrieved motion.
        The tuple is as following :math:`[(*, 4, 3, 3), (*, 4, 3, 1)]`.

    """
    assert len(E_mat.shape) >= 2 and E_mat.shape[-2:], E_mat.shape

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
    assert len(E_mat.shape) >= 2 and E_mat.shape[-2:], E_mat.shape
    assert len(K1.shape) >= 2 and K1.shape[-2:] == (3, 3), K1.shape
    assert len(K2.shape) >= 2 and K2.shape[-2:] == (3, 3), K2.shape
    assert len(x1.shape) >= 2 and x1.shape[-1] == 2, x1.shape
    assert len(x2.shape) >= 2 and x2.shape[-1] == 2, x2.shape
    assert len(E_mat.shape[:-2]) == len(K1.shape[:-2]) == len(K2.shape[:-2])
    if mask is not None:
        assert len(mask.shape) >= 1, mask.shape
        assert mask.shape == x1.shape[:-1], mask.shape

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
    R1 = numeric.eye_like(3, E_mat)  # Bx3x3
    t1 = numeric.vec_like(3, E_mat)  # Bx3x1

    # compute the projection matrices for first camera
    R1 = R1[:, None].expand(-1, 4, -1, -1)
    t1 = t1[:, None].expand(-1, 4, -1, -1)
    K1 = K1[:, None].expand(-1, 4, -1, -1)
    P1 = projection.projection_from_KRt(K1, R1, t1)  # 1x4x4x4

    # compute the projection matrices for second camera
    R2 = Rs
    t2 = ts
    K2 = K2[:, None].expand(-1, 4, -1, -1)
    P2 = projection.projection_from_KRt(K2, R2, t2)  # Bx4x4x4

    # triangulate the points
    x1 = x1[:, None].expand(-1, 4, -1, -1)
    x2 = x2[:, None].expand(-1, 4, -1, -1)
    X = triangulation.triangulate_points(P1, P2, x1, x2)  # Bx4xNx3

    # project points and compute their depth values
    d1 = projection.depth(R1, t1, X)
    d2 = projection.depth(R2, t2, X)

    # verify the point values that have a postive depth value
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
    r"""Computes the relative camera motion between two cameras.

    Given the motion parameters of two cameras, computes the motion parameters of the second
    one assuming the first one to be at the origin. If :math:`T1` and :math:`T2` are the camera motions,
    the computed relative motion is :math:`T = T_{2}T^{−1}_{1}`.

    Args:
        R1: The first camera rotation matrix with shape :math:`(*, 3, 3)`.
        t1: The first camera translation vector with shape :math:`(*, 3, 1)`.
        R2: The second camera rotation matrix with shape :math:`(*, 3, 3)`.
        t2: The second camera translation vector with shape :math:`(*, 3, 1)`.

    Returns:
        A tuple with the relative rotation matrix and
        translation vector with the shape of :math:`[(*, 3, 3), (*, 3, 1)]`.

    """
    assert len(R1.shape) >= 2 and R1.shape[-2:] == (3, 3), R1.shape
    assert len(t1.shape) >= 2 and t1.shape[-2:] == (3, 1), t1.shape
    assert len(R2.shape) >= 2 and R2.shape[-2:] == (3, 3), R2.shape
    assert len(t2.shape) >= 2 and t2.shape[-2:] == (3, 1), t2.shape

    # compute first the relative rotation
    R = R2 @ R1.transpose(-2, -1)

    # compute the relative translation vector
    t = t2 - R @ t1

    return (R, t)
