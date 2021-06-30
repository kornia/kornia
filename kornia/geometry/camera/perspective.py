import torch
import torch.nn.functional as F

from kornia.geometry.conversions import convert_points_from_homogeneous, convert_points_to_homogeneous


def project_points(point_3d: torch.Tensor, camera_matrix: torch.Tensor) -> torch.Tensor:
    r"""Projects a 3d point onto the 2d camera plane.

    Args:
        point3d: tensor containing the 3d points to be projected
            to the camera plane. The shape of the tensor can be :math:`(*, 3)`.
        camera_matrix: tensor containing the intrinsics camera
            matrix. The tensor shape must be :math:`(*, 3, 3)`.

    Returns:
        tensor of (u, v) cam coordinates with shape :math:`(*, 2)`.
    """
    if not isinstance(point_3d, torch.Tensor):
        raise TypeError("Input point_3d type is not a torch.Tensor. Got {}".format(type(point_3d)))

    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError("Input camera_matrix type is not a torch.Tensor. Got {}".format(type(camera_matrix)))

    if not (point_3d.device == camera_matrix.device):
        raise ValueError("Input tensors must be all in the same device.")

    if not point_3d.shape[-1] == 3:
        raise ValueError("Input points_3d must be in the shape of (*, 3)." " Got {}".format(point_3d.shape))

    if not camera_matrix.shape[-2:] == (3, 3):
        raise ValueError("Input camera_matrix must be in the shape of (*, 3, 3).")

    # projection eq. [u, v, w]' = K * [x y z 1]'
    # u = fx * X / Z + cx
    # v = fy * Y / Z + cy

    # project back using depth dividing in a safe way
    xy_coords: torch.Tensor = convert_points_from_homogeneous(point_3d)
    x_coord: torch.Tensor = xy_coords[..., 0]
    y_coord: torch.Tensor = xy_coords[..., 1]

    # unpack intrinsics
    fx: torch.Tensor = camera_matrix[..., 0, 0]
    fy: torch.Tensor = camera_matrix[..., 1, 1]
    cx: torch.Tensor = camera_matrix[..., 0, 2]
    cy: torch.Tensor = camera_matrix[..., 1, 2]

    # apply intrinsics ans return
    u_coord: torch.Tensor = x_coord * fx + cx
    v_coord: torch.Tensor = y_coord * fy + cy

    return torch.stack([u_coord, v_coord], dim=-1)


def unproject_points(
    point_2d: torch.Tensor, depth: torch.Tensor, camera_matrix: torch.Tensor, normalize: bool = False
) -> torch.Tensor:
    r"""Unprojects a 2d point in 3d.

    Transform coordinates in the pixel frame to the camera frame.

    Args:
        point2d: tensor containing the 2d to be projected to
            world coordinates. The shape of the tensor can be :math:`(*, 2)`.
        depth: tensor containing the depth value of each 2d
            points. The tensor shape must be equal to point2d :math:`(*, 1)`.
        camera_matrix: tensor containing the intrinsics camera
            matrix. The tensor shape must be :math:`(*, 3, 3)`.
        normalize: whether to normalize the pointcloud. This
            must be set to `True` when the depth is represented as the Euclidean
            ray length from the camera position.

    Returns:
        tensor of (x, y, z) world coordinates with shape :math:`(*, 3)`.
    """
    if not isinstance(point_2d, torch.Tensor):
        raise TypeError("Input point_2d type is not a torch.Tensor. Got {}".format(type(point_2d)))

    if not isinstance(depth, torch.Tensor):
        raise TypeError("Input depth type is not a torch.Tensor. Got {}".format(type(depth)))

    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError("Input camera_matrix type is not a torch.Tensor. Got {}".format(type(camera_matrix)))

    if not (point_2d.device == depth.device == camera_matrix.device):
        raise ValueError("Input tensors must be all in the same device.")

    if not point_2d.shape[-1] == 2:
        raise ValueError("Input points_2d must be in the shape of (*, 2)." " Got {}".format(point_2d.shape))

    if not depth.shape[-1] == 1:
        raise ValueError("Input depth must be in the shape of (*, 1)." " Got {}".format(depth.shape))

    if not camera_matrix.shape[-2:] == (3, 3):
        raise ValueError("Input camera_matrix must be in the shape of (*, 3, 3).")

    # projection eq. K_inv * [u v 1]'
    # x = (u - cx) * Z / fx
    # y = (v - cy) * Z / fy

    # unpack coordinates
    u_coord: torch.Tensor = point_2d[..., 0]
    v_coord: torch.Tensor = point_2d[..., 1]

    # unpack intrinsics
    fx: torch.Tensor = camera_matrix[..., 0, 0]
    fy: torch.Tensor = camera_matrix[..., 1, 1]
    cx: torch.Tensor = camera_matrix[..., 0, 2]
    cy: torch.Tensor = camera_matrix[..., 1, 2]

    # projective
    x_coord: torch.Tensor = (u_coord - cx) / fx
    y_coord: torch.Tensor = (v_coord - cy) / fy

    xyz: torch.Tensor = torch.stack([x_coord, y_coord], dim=-1)
    xyz = convert_points_to_homogeneous(xyz)

    if normalize:
        xyz = F.normalize(xyz, dim=-1, p=2.0)

    return xyz * depth
