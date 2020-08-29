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
        center (torch.Tensor): center of the rotation in the source with shape :math:`(B, 3)`.
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
