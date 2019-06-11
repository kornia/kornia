from typing import Optional

import torch
import torch.nn.functional as F

from kornia.geometry.conversions import convert_points_to_homogeneous


def unproject_points(
        point_2d: torch.Tensor,
        depth: torch.Tensor,
        camera_matrix: torch.Tensor,
        normalize: Optional[bool] = False) -> torch.Tensor:
    r"""Unprojects a 2d point in 3d.

    Transform coordinates in the pixel frame to the camera frame.

    Args:
        point2d (torch.Tensor): the 2d to be projected to world coordinates.
            The shape of the tensor can be (*, 2).
        depth (torch.Tensor): the depth value of each 2d points. The tensor
            shape must be equal to point2d (*, 1).
        camera_matrix (torch.Tensor): the intrinsics camera matrix. The tensor
            shape must be Bx4x4.
        normalize (Optional[bool]): wether to normalize the pointcloud. This
            must be set to `True` when the depth is represented as the Euclidean
            ray length from the camera position. Default is `False`.

    Returns:
        torch.Tensor: array of (u, v, 1) cam coordinates with shape (*, 3).
    """
    if not torch.is_tensor(point_2d):
        raise TypeError("Input point_2d type is not a torch.Tensor. Got {}"
                        .format(type(points_2d)))
    if not torch.is_tensor(depth):
        raise TypeError("Input depth type is not a torch.Tensor. Got {}"
                        .format(type(depth)))
    if not torch.is_tensor(camera_matrix):
        raise TypeError("Input camera_matrix type is not a torch.Tensor. Got {}"
                        .format(type(camera_matrix)))
    if not (point_2d.device == depth.device == camera_matrix.device):
        raise ValueError("Input tensors must be all in the same device.")
    if not point_2d.shape[-1] == 2:
        raise ValueError("Input points_2d must be in the shape of (*, 2)."
                         " Got {}".format(point_2d.shape))
    if not depth.shape[-1] == 1:
        raise ValueError("Input depth must be in the shape of (*, 1)."
                         " Got {}".format(depth.shape))
    if not camera_matrix.shape[-2:] == (3, 3):
        raise ValueError("Input camera_matrix must be in the shape of (*, 3, 3).")
    # projection eq. K_inv * [u v 1]'
    # inverse the camera matrix
    camera_matrix_inv: torch.Tensor = torch.inverse(camera_matrix)

    # compute ray from center to camera
    uvw: torch.Tensor = convert_points_to_homogeneous(point_2d)

    # apply inverse intrinsics to points
    xyz: torch.Tensor = torch.matmul(
        camera_matrix.view(-1, 3, 3), uvw.view(-1, 3, 1))

    # back to input shape and normalize if specified
    xyz_norm: torch.Tensor = xyz.view((*point_2d.shape[:-1], 3))

    if normalize:
        xyz_norm = F.normalize(xyz_norm, dim=-1, p=2)

    # apply depth
    return xyz_norm * depth
