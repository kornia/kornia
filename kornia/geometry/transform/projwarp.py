"""Module to perform projective transformations to tensors."""
from typing import Tuple, List

import torch
import kornia as K
from kornia.geometry.conversions import convert_affinematrix_to_homography3d
from kornia.geometry.warp import normalize_homography3d

__all__ = [
    "warp_projective",
    "get_projective_transform",
    "projection_from_Rt",
    "get_3d_perspective_transform"
]


def warp_projective(src: torch.Tensor,
                    M: torch.Tensor,
                    dsize: Tuple[int, int, int],
                    flags: str = 'bilinear',
                    padding_mode: str = 'zeros',
                    align_corners: bool = True) -> torch.Tensor:
    r"""Applies a projective transformation a to 3d tensor.

    .. warning::
        This API signature it is experimental and might suffer some changes in the future.

    Args:
        src (torch.Tensor): input tensor of shape :math:`(B, C, D, H, W)`.
        M (torch.Tensor): projective transformation matrix of shape :math:`(B, 3, 4)`.
        dsize (Tuple[int, int, int]): size of the output image (depth, height, width).
        mode (str): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (str): padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.
        align_corners (bool): mode for grid_generation. Default: True.

    Returns:
        torch.Tensor: the warped 3d tensor with shape :math:`(B, C, D, H, W)`.

    """
    assert len(src.shape) == 5, src.shape
    assert len(M.shape) == 3 and M.shape[-2:] == (3, 4), M.shape
    assert len(dsize) == 3, dsize
    B, C, D, H, W = src.size()

    size_src: Tuple[int, int, int] = (D, H, W)
    size_out: Tuple[int, int, int] = dsize

    M_4x4 = convert_affinematrix_to_homography3d(M)  # Bx4x4

    # we need to normalize the transformation since grid sample needs -1/1 coordinates
    dst_norm_trans_src_norm: torch.Tensor = normalize_homography3d(
        M_4x4, size_src, size_out)    # Bx4x4

    src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm)
    P_norm: torch.Tensor = src_norm_trans_dst_norm[:, :3]  # Bx3x4

    # compute meshgrid and apply to input
    dsize_out: List[int] = [B, C] + list(size_out)
    grid = torch.nn.functional.affine_grid(P_norm, dsize_out, align_corners=align_corners)
    return torch.nn.functional.grid_sample(
        src, grid, align_corners=align_corners, mode=flags, padding_mode=padding_mode)


def projection_from_Rt(rmat: torch.Tensor, tvec: torch.Tensor) -> torch.Tensor:
    r"""Compute the projection matrix from Rotation and translation.

    .. warning::
        This API signature it is experimental and might suffer some changes in the future.

    Concatenates the batch of rotations and translations such that :math:`P = [R | t]`.

    Args:
       rmat (torch.Tensor): the rotation matrix with shape :math:`(*, 3, 3)`.
       tvec (torch.Tensor): the translation vector with shape :math:`(*, 3, 1)`.

    Returns:
       torch.Tensor: the projection matrix with shape :math:`(*, 3, 4)`.

    """
    assert len(rmat.shape) >= 2 and rmat.shape[-2:] == (3, 3), rmat.shape
    assert len(tvec.shape) >= 2 and tvec.shape[-2:] == (3, 1), tvec.shape

    return torch.cat([rmat, tvec], dim=-1)  # Bx3x4


def get_projective_transform(center: torch.Tensor, angles: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    r"""Calculates the projection matrix for a 3D rotation.

    .. warning::
        This API signature it is experimental and might suffer some changes in the future.

    The function computes the projection matrix given the center and angles per axis.

    Args:
        center (torch.Tensor): center of the rotation (x,y,z) in the source with shape :math:`(B, 3)`.
        angles (torch.Tensor): angle axis vector containing the rotation angles in degrees in the form
            of (rx, ry, rz) with shape :math:`(B, 3)`. Internally it calls Rodrigues to compute
            the rotation matrix from axis-angle.
        scales (torch.Tensor): scale factor for x-y-z-directions with shape :math:`(B, 3)`.

    Returns:
        torch.Tensor: the projection matrix of 3D rotation with shape :math:`(B, 3, 4)`.

    """
    assert len(center.shape) == 2 and center.shape[-1] == 3, center.shape
    assert len(angles.shape) == 2 and angles.shape[-1] == 3, angles.shape
    assert center.device == angles.device, (center.device, angles.device)
    assert center.dtype == angles.dtype, (center.dtype, angles.dtype)

    # create rotation matrix
    angle_axis_rad: torch.Tensor = K.deg2rad(angles)
    rmat: torch.Tensor = K.angle_axis_to_rotation_matrix(angle_axis_rad)  # Bx3x3
    scaling_matrix: torch.Tensor = K.eye_like(3, rmat)
    scaling_matrix = scaling_matrix * scales.unsqueeze(dim=1)
    rmat = rmat @ scaling_matrix.to(rmat)

    # define matrix to move forth and back to origin
    from_origin_mat = torch.eye(4)[None].repeat(rmat.shape[0], 1, 1).type_as(center)  # Bx4x4
    from_origin_mat[..., :3, -1] += center

    to_origin_mat = from_origin_mat.clone()
    to_origin_mat = from_origin_mat.inverse()

    # append tranlation with zeros
    proj_mat = projection_from_Rt(rmat, torch.zeros_like(center)[..., None])  # Bx3x4

    # chain 4x4 transforms
    proj_mat = convert_affinematrix_to_homography3d(proj_mat)  # Bx4x4
    proj_mat = (from_origin_mat @ proj_mat @ to_origin_mat)

    return proj_mat[..., :3, :]  # Bx3x4


def get_3d_perspective_transform(src, dst):
    r"""Calculate a 3d perspective transform from four pairs of the corresponding points.

    The function calculates the matrix of a perspective transform so that:

    .. math ::

        \begin{bmatrix}
        t_{i}x_{i}^{'} \\
        t_{i}y_{i}^{'} \\
        t_{i}z_{i}^{'} \\
        t_{i} \\
        \end{bmatrix}
        =
        \textbf{map_matrix} \cdot
        \begin{bmatrix}
        x_{i} \\
        y_{i} \\
        z_{i} \\
        1 \\
        \end{bmatrix}

    where

    .. math ::
        dst(i) = (x_{i}^{'},y_{i}^{'},z_{i}^{'}), src(i) = (x_{i}, y_{i}, z_{i}), i = 0,1,2,5,7

    Concrete math is as below:

    .. math ::

        \[ u_i =\frac{c_{00} * x_i + c_{01} * y_i + c_{02} * z_i + c_{03}}{c_{30} * x_i + c_{31} * y_i + c_{32} * z_i + c_{33}} \]
        \[ v_i =\frac{c_{10} * x_i + c_{11} * y_i + c_{12} * z_i + c_{13}}{c_{30} * x_i + c_{31} * y_i + c_{32} * z_i + c_{33}} \]
        \[ w_i =\frac{c_{20} * x_i + c_{21} * y_i + c_{22} * z_i + c_{23}}{c_{30} * x_i + c_{31} * y_i + c_{32} * z_i + c_{33}} \]

    .. math ::

        \begin{pmatrix}
        x_0 & y_0 & z_0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -x_0*u_0 & -y_0*u_0 & -z_0 * u_0 \\
        x_1 & y_1 & z_1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -x_1*u_1 & -y_1*u_1 & -z_1 * u_1 \\
        x_2 & y_2 & z_2 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -x_2*u_2 & -y_2*u_2 & -z_2 * u_2 \\
        x_5 & y_5 & z_5 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -x_5*u_5 & -y_5*u_5 & -z_5 * u_5 \\
        x_7 & y_7 & z_7 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -x_7*u_7 & -y_7*u_7 & -z_7 * u_7 \\
        0 & 0 & 0 & 0 & x_0 & y_0 & z_0 & 1 & 0 & 0 & 0 & 0 & -x_0*v_0 & -y_0*v_0 & -z_0 * v_0 \\
        0 & 0 & 0 & 0 & x_1 & y_1 & z_1 & 1 & 0 & 0 & 0 & 0 & -x_1*v_1 & -y_1*v_1 & -z_1 * v_1 \\
        0 & 0 & 0 & 0 & x_2 & y_2 & z_2 & 1 & 0 & 0 & 0 & 0 & -x_2*v_2 & -y_2*v_2 & -z_2 * v_2 \\
        0 & 0 & 0 & 0 & x_5 & y_5 & z_5 & 1 & 0 & 0 & 0 & 0 & -x_5*v_5 & -y_5*v_5 & -z_5 * v_5 \\
        0 & 0 & 0 & 0 & x_7 & y_7 & z_7 & 1 & 0 & 0 & 0 & 0 & -x_7*v_7 & -y_7*v_7 & -z_7 * v_7 \\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & x_0 & y_0 & z_0 & 1 & -x_0*w_0 & -y_0*w_0 & -z_0 * w_0 \\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & x_1 & y_1 & z_1 & 1 & -x_1*w_1 & -y_1*w_1 & -z_1 * w_1 \\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & x_2 & y_2 & z_2 & 1 & -x_2*w_2 & -y_2*w_2 & -z_2 * w_2 \\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & x_5 & y_5 & z_5 & 1 & -x_5*w_5 & -y_5*w_5 & -z_5 * w_5 \\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & x_7 & y_7 & z_7 & 1 & -x_7*w_7 & -y_7*w_7 & -z_7 * w_7 \\
        \end{pmatrix}

    Args:
        src (Tensor): coordinates of quadrangle vertices in the source image.
        dst (Tensor): coordinates of the corresponding quadrangle vertices in
            the destination image.

    Returns:
        Tensor: the perspective transformation.

    Shape:
        - Input: :math:`(B, 8, 3)` and :math:`(B, 8, 3)`
        - Output: :math:`(B, 4, 4)`
    """
    if not torch.is_tensor(src):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(src)))
    if not torch.is_tensor(dst):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(dst)))
    if not src.shape[-2:] == (8, 3):
        raise ValueError("Inputs must be a Bx8x3 tensor. Got {}"
                         .format(src.shape))
    if not src.shape == dst.shape:
        raise ValueError("Inputs must have the same shape. Got {}"
                         .format(dst.shape))
    if not (src.shape[0] == dst.shape[0]):
        raise ValueError("Inputs must have same batch size dimension. Expect {} but got {}"
                         .format(src.shape, dst.shape))

    def ax(p, q):
        ones = torch.ones_like(p)[..., 0:1]
        zeros = torch.zeros_like(p)[..., 0:1]
        return torch.cat([
            p[:, 0:1], p[:, 1:2], p[:, 2:3], ones,
            zeros, zeros, zeros, zeros,
            zeros, zeros, zeros, zeros,
            -p[:, 0:1] * q[:, 0:1], -p[:, 1:2] * q[:, 0:1], -p[:, 2:3] * q[:, 0:1]
        ], dim=1)

    def ay(p, q):
        ones = torch.ones_like(p)[..., 0:1]
        zeros = torch.zeros_like(p)[..., 0:1]
        return torch.cat([
            zeros, zeros, zeros, zeros,
            p[:, 0:1], p[:, 1:2], p[:, 2:3], ones,
            zeros, zeros, zeros, zeros,
            -p[:, 0:1] * q[:, 1:2], -p[:, 1:2] * q[:, 1:2], -p[:, 2:3] * q[:, 1:2]
        ], dim=1)

    def az(p, q):
        ones = torch.ones_like(p)[..., 0:1]
        zeros = torch.zeros_like(p)[..., 0:1]
        return torch.cat([
            zeros, zeros, zeros, zeros,
            zeros, zeros, zeros, zeros,
            p[:, 0:1], p[:, 1:2], p[:, 2:3], ones,
            -p[:, 0:1] * q[:, 2:3], -p[:, 1:2] * q[:, 2:3], -p[:, 2:3] * q[:, 2:3]
        ], dim=1)

    # we build matrix A by using only 4 point correspondence. The linear
    # system is solved with the least square method, so here
    # we could even pass more correspondence
    p = []

    # 000, 100, 110, 101, 011
    for i in [0, 1, 2, 5, 7]:
        p.append(ax(src[:, i], dst[:, i]))
        p.append(ay(src[:, i], dst[:, i]))
        p.append(az(src[:, i], dst[:, i]))

    # A is Bx15x15
    A = torch.stack(p, dim=1)

    # b is a Bx15x1
    b = torch.stack([
        dst[:, 0:1, 0], dst[:, 0:1, 1], dst[:, 0:1, 2],
        dst[:, 1:2, 0], dst[:, 1:2, 1], dst[:, 1:2, 2],
        dst[:, 2:3, 0], dst[:, 2:3, 1], dst[:, 2:3, 2],
        # dst[:, 3:4, 0], dst[:, 3:4, 1], dst[:, 3:4, 2],
        # dst[:, 4:5, 0], dst[:, 4:5, 1], dst[:, 4:5, 2],
        dst[:, 5:6, 0], dst[:, 5:6, 1], dst[:, 5:6, 2],
        # dst[:, 6:7, 0], dst[:, 6:7, 1], dst[:, 6:7, 2],
        dst[:, 7:8, 0], dst[:, 7:8, 1], dst[:, 7:8, 2],
    ], dim=1)

    # solve the system Ax = b
    X, LU = torch.solve(b, A)

    # create variable to return
    batch_size = src.shape[0]
    M = torch.ones(batch_size, 16, device=src.device, dtype=src.dtype)
    M[..., :15] = torch.squeeze(X, dim=-1)
    return M.view(-1, 4, 4)  # Bx4x4
