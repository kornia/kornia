"""Module to perform projective transformations to tensors."""
from typing import Tuple, List

import torch
import kornia as K

__all__ = [
    "warp_projective",
    "get_projective_transform",
    "normal_transform3d_pixel",
    "matrix_to_homogeneous",
    "projection_from_Rt",
]


def warp_projective(src: torch.Tensor,
                    M: torch.Tensor,
                    dsize: Tuple[int, int, int],
                    flags: str = 'bilinear',
                    padding_mode: str = 'zeros',
                    align_corners: bool = True) -> torch.Tensor:
    r"""Applies a projective transformation a to 3d tensor.

        src (torch.Tensor): input tensor of shape :math:`(B, C, D, H, W)`.
        M (torch.Tensor): projective transformation matrix of shape :math:`(B, 3, 4)`.
        dsize (Tuple[int, int, int]): size of the output image (depth, height, width).
        mode (str): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (str): padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.
        align_corners (bool): mode for grid_generation. Default: True. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for details

    Returns:
        torch.Tensor: the warped 3d tensor with shape :math:`(B, C, D, H, W)`.

    """
    assert len(src.shape) == 5, src.shape
    assert len(M.shape) == 3 and M.shape[-2:] == (3, 4), M.shape
    assert len(dsize) == 3, dsize
    B, C, D, H, W = src.size()

    size_src: Tuple[int, int, int] = (D, H, W)
    size_out: Tuple[int, int, int] = dsize

    M_4x4 = matrix_to_homogeneous(M)  # Bx4x4

    # we need to normalize the transformation since grid sample needs -1/1 coordinates
    dst_norm_trans_src_norm: torch.Tensor = _normalize_projection_matrix(
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


def matrix_to_homogeneous(M: torch.Tensor) -> torch.Tensor:
    r"""Converts a generic transformation matrix to make it usable for homogeneous coordinates.

    Appends to the last matrix row an extra row filled with zeros with a one to the end.

    Args:
        M (torch.Tensor): the transformation matrix with at least two dimensions :math:`(*, M, N)`.

    Returns:
        torch.Tensor: the transformation matrix with shape :math:`(*, M, N + 1)`.

    """
    M_homo = torch.nn.functional.pad(M, [0, 0, 0, 1], "constant", value=0.)
    M_homo[..., -1, -1] += 1.0

    return M_homo


def _normalize_projection_matrix(
        dst_pix_trans_src_pix: torch.Tensor,
        dsize_src: Tuple[int, int, int],
        dsize_dst: Tuple[int, int, int]) -> torch.Tensor:
    r"""Computes the transformation matrix to normalize points before applying sampling."""
    # source and destination sizes
    src_d, src_h, src_w = dsize_src
    src_d, dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: torch.Tensor = normal_transform3d_pixel(
        src_d, src_h, src_w).to(dst_pix_trans_src_pix)

    src_pix_trans_src_norm = torch.inverse(src_norm_trans_src_pix)

    dst_norm_trans_dst_pix: torch.Tensor = normal_transform3d_pixel(
        src_d, dst_h, dst_w).to(dst_pix_trans_src_pix)

    # compute chain transformations
    dst_norm_trans_src_norm: torch.Tensor = (
        dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm)
    )
    return dst_norm_trans_src_norm


def normal_transform3d_pixel(depth: int, height: int, width: int) -> torch.Tensor:
    r"""Compute the normalization matrix from image size in pixels to [-1, 1].

    Args:
        depth (int): tensor depht.
        height (int): tensor height.
        width (int): tensor width.

    Returns:
        torch.Tensor: normalized transform with shape :math:`(1, 4, 4)`.

    """
    tr_mat = torch.tensor([[1.0, 0.0, 0.0, -1.0],
                           [0.0, 1.0, 0.0, -1.0],
                           [0.0, 0.0, 1.0, -1.0],
                           [0.0, 0.0, 0.0, 1.0]])  # 4x4

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / (width - 1.0)
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / (height - 1.0)
    tr_mat[2, 2] = tr_mat[2, 2] * 2.0 / (depth - 1.0)

    return tr_mat[None]  # 1x4x4


def get_projective_transform(center: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    r"""Calculates the projection matrix for a 3D rotation.

    The function computes the projection matrix given the center, scales and angles per axis.

    Args:
        center (torch.Tensor): center of the rotation in the source with shape :math:`(B, 3)`.
        angles (Tensor): angle axis vector containing the rotation angles in degrees in the form
            of (rx, ry, rz) with shape :math:`(B, 3)`. Internally it calls Rodrigues to compute
            the rotation matrix from axis-angle.

    Returns:
        Tensor: the projection matrix of 3D rotation with shape :math:`(B, 3, 4)`.

    """
    assert len(center.shape) == 2 and center.shape[-1] == 3, center.shape
    assert len(angles.shape) == 2 and angles.shape[-1] == 3, angles.shape
    assert center.device == angles.device, (center.device, angles.device)
    assert center.dtype == angles.dtype, (center.dtype, angles.dtype)

    # create rotation matrix
    angle_axis_rad: torch.Tensor = K.deg2rad(angles)
    rmat: torch.Tensor = K.angle_axis_to_rotation_matrix(angle_axis_rad)  # Bx3x3

    # define matrix to move forth and back to origin
    from_origin_mat = torch.eye(4)[None].repeat(rmat.shape[0], 1, 1).type_as(center)  # Bx4x4
    from_origin_mat[..., :3, -1] += center

    to_origin_mat = from_origin_mat.clone()
    to_origin_mat = from_origin_mat.inverse()

    # append tranlation with zeros
    proj_mat = projection_from_Rt(rmat, torch.zeros_like(center)[..., None])  # Bx3x4

    # chain 4x4 transforms
    proj_mat = matrix_to_homogeneous(proj_mat)  # Bx4x4
    proj_mat = (from_origin_mat @ proj_mat @ to_origin_mat)

    return proj_mat[..., :3, :]  # Bx3x4
