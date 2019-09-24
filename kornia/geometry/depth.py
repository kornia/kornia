"""Module containing operators to work on RGB-Depth images."""

import torch
import torch.nn.functional as F

from kornia.utils import create_meshgrid
from kornia.geometry import unproject_points
from kornia.filters import spatial_gradient


def depth_to_3d(depth: torch.Tensor, camera_matrix: torch.Tensor) -> torch.Tensor:
    """Compute a 3d point per pixel given its depth value and the camera intrinsics.

    Args:
        depth (torch.Tensor): image tensor containing a depth value per pixel.
        camera_matrix (torch.Tensor): tensor containing the camera intrinsics.

    Shape:
        - Input: :math:`(B, 1, H, W)` and :math:`(B, 3, 3)`
        - Output: :math:`(B, 3, H, W)`

    Return:
        torch.Tensor: tensor with a 3d point per pixel of the same resolution as the input.

    """
    if not isinstance(depth, torch.Tensor):
        raise TypeError(f"Input depht type is not a torch.Tensor. Got {type(depth)}.")

    if not len(depth.shape) == 4 and depth.shape[-3] == 1:
        raise ValueError(f"Input depth musth have a shape (B, 1, H, W). Got: {depth.shape}")

    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError(f"Input camera_matrix type is not a torch.Tensor. "
                        f"Got {type(camera_matrix)}.")

    if not len(camera_matrix.shape) == 3 and camera_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input camera_matrix must have a shape (B, 3, 3). "
                         f"Got: {camera_matrix.shape}.")

    # create base coordinates grid
    batch_size, _, height, width = depth.shape
    points_2d: torch.Tensor = create_meshgrid(
        height, width, normalized_coordinates=False)  # 1xHxWx2
    points_2d = points_2d.to(depth.device).to(depth.dtype)

    # depth should come in Bx1xHxW
    points_depth: torch.Tensor = depth.permute(0, 2, 3, 1)  # 1xHxWx1

    # project pixels to camera frame
    camera_matrix_tmp: torch.Tensor = camera_matrix[:, None, None]  # Bx1x1x3x3
    points_3d: torch.Tensor = unproject_points(
        points_2d, points_depth, camera_matrix_tmp, normalize=True)  # BxHxWx3

    return points_3d.permute(0, 3, 1, 2)  # Bx3xHxW


def depth_to_normals(depth: torch.Tensor, camera_matrix: torch.Tensor) -> torch.Tensor:
    """Compute the normal surface per pixel.

    Args:
        depth (torch.Tensor): image tensor containing a depth value per pixel.
        camera_matrix (torch.Tensor): tensor containing the camera intrinsics.

    Shape:
        - Input: :math:`(B, 1, H, W)` and :math:`(B, 3, 3)`
        - Output: :math:`(B, 3, H, W)`

    Return:
        torch.Tensor: tensor with a normal surface vector per pixel of the same resolution as the input.

    """
    if not isinstance(depth, torch.Tensor):
        raise TypeError(f"Input depht type is not a torch.Tensor. Got {type(depth)}.")

    if not len(depth.shape) == 4 and depth.shape[-3] == 1:
        raise ValueError(f"Input depth musth have a shape (B, 1, H, W). Got: {depth.shape}")

    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError(f"Input camera_matrix type is not a torch.Tensor. "
                        f"Got {type(camera_matrix)}.")

    if not len(camera_matrix.shape) == 3 and camera_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input camera_matrix must have a shape (B, 3, 3). "
                         f"Got: {camera_matrix.shape}.")

    # compute the 3d points from depth
    xyz: torch.Tensor = depth_to_3d(depth, camera_matrix)  # Bx3xHxW

    # compute the pointcloud spatial gradients
    gradients: torch.Tensor = spatial_gradient(xyz)  # Bx3x2xHxW

    # compute normals
    a, b = gradients[:, :, 0], gradients[:, :, 1]  # Bx3xHxW

    normals: torch.Tensor = torch.cross(a, b, dim=1)  # Bx3xHxW
    return F.normalize(normals, dim=1, p=2)
