import torch
import torch.nn.functional as F

from kornia.geometry.conversions import (
    convert_points_from_homogeneous,
    convert_points_to_homogeneous,
    denormalize_points_with_intrinsics,
    normalize_points_with_intrinsics,
)


def project_points(point_3d: torch.Tensor, camera_matrix: torch.Tensor) -> torch.Tensor:
    r"""Project a 3d point onto the 2d camera plane.

    Args:
        point3d: tensor containing the 3d points to be projected
            to the camera plane. The shape of the tensor can be :math:`(*, 3)`.
        camera_matrix: tensor containing the intrinsics camera
            matrix. The tensor shape must be :math:`(*, 3, 3)`.

    Returns:
        tensor of (u, v) cam coordinates with shape :math:`(*, 2)`.

    Example:
        >>> _ = torch.manual_seed(0)
        >>> X = torch.rand(1, 3)
        >>> K = torch.eye(3)[None]
        >>> project_points(X, K)
        tensor([[5.6088, 8.6827]])
    """
    # projection eq. [u, v, w]' = K * [x y z 1]'
    # u = fx * X / Z + cx
    # v = fy * Y / Z + cy
    # project back using depth dividing in a safe way
    xy_coords: torch.Tensor = convert_points_from_homogeneous(point_3d)
    return denormalize_points_with_intrinsics(xy_coords, camera_matrix)


def unproject_points(
    point_2d: torch.Tensor, depth: torch.Tensor, camera_matrix: torch.Tensor, normalize: bool = False
) -> torch.Tensor:
    r"""Unproject a 2d point in 3d.

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

    Example:
        >>> _ = torch.manual_seed(0)
        >>> x = torch.rand(1, 2)
        >>> depth = torch.ones(1, 1)
        >>> K = torch.eye(3)[None]
        >>> unproject_points(x, depth, K)
        tensor([[0.4963, 0.7682, 1.0000]])
    """
    if not isinstance(depth, torch.Tensor):
        raise TypeError(f"Input depth type is not a torch.Tensor. Got {type(depth)}")

    if not depth.shape[-1] == 1:
        raise ValueError(f"Input depth must be in the shape of (*, 1). Got {depth.shape}")

    xy: torch.Tensor = normalize_points_with_intrinsics(point_2d, camera_matrix)
    xyz: torch.Tensor = convert_points_to_homogeneous(xy)
    if normalize:
        xyz = F.normalize(xyz, dim=-1, p=2.0)

    return xyz * depth
