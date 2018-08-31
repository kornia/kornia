import torch


__all__ = [
    "pi",
    "rad2deg",
    "deg2rad",
    "convert_points_from_homogeneous",
    "convert_points_to_homogeneous",
    "transform_points",
    "inverse",
    "inverse_pose",
]


"""Constant with number pi
"""
pi = torch.Tensor([3.141592653589793])


def rad2deg(x):
    """Converts angles from radians to degrees.

    Args:
        x (Tensor): tensor of unspecified size.

    Returns:
        Tensor: tensor with same size as input.
    """
    if not torch.is_tensor(x):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(x)))

    return 180. * x / pi.to(x.device).type(x.dtype)

    
def deg2rad(x):
    """Converts angles from degrees to radians.

    Args:
        x (Tensor): tensor of unspecified size.

    Returns:
        Tensor: tensor with same size as input.
    """
    if not torch.is_tensor(x):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(x)))

    return x * pi.to(x.device).type(x.dtype) / 180.


def convert_points_from_homogeneous(points, eps=1e-6):
    """Converts points from homogeneous to Euclidean space.

    Args:
        points (Tensor): tensor of N-dimensional points of size (B, D, N).

    Returns:
        Tensor: tensor of N-1-dimensional points of size (B, D, N-1).
    """
    if not torch.is_tensor(points):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(points)))

    if not len(points.shape) == 3:
        raise ValueError("Input size must be a three dimensional tensor. Got {}"
                         .format(points.shape))

    return points[..., :-1] / (points[..., -1:] + eps)


def convert_points_to_homogeneous(points):
    """Converts points from Euclidean to homogeneous space.

    Args:
        points (Tensor): tensor of N-dimensional points of size (B, D, N).

    Returns:
        Tensor: tensor of N+1-dimensional points of size (B, D, N+1).
    """
    if not torch.is_tensor(points):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(points)))

    if not len(points.shape) == 3:
        raise ValueError("Input size must be a three dimensional tensor. Got {}"
                         .format(points.shape))

    return torch.cat([points, torch.ones_like(points)[..., :1]], dim=-1)


def transform_points(dst_homo_src, points_src):
    # TODO: add documentation
    """Applies Transformation to points.
    """
    if not torch.is_tensor(dst_homo_src) or not torch.is_tensor(points_src):
        raise TypeError("Input type is not a torch.Tensor")
    if not dst_homo_src.device == points_src.device:
        raise TypeError("Tensor must be in the same device")
    if not len(dst_homo_src.shape) == 3 or not len(points_src.shape) == 3:
        raise ValueError("Input size must be a three dimensional tensor")
    if not dst_homo_src.shape[0] == points_src.shape[0]:
        raise ValueError("Input batch size must be the same for both tensors")
    if not dst_homo_src.shape[1] == (points_src.shape[1] + 1):
        raise ValueError("Input dimensions must differe by one unit")
    # to homogeneous
    points_src_h = convert_points_to_homogeneous(points_src)  # BxNx3
    # transform coordinates
    points_dst_h = torch.matmul(dst_homo_src, points_src_h.transpose(1, 2))  # Bx3xN
    points_dst_h = points_dst_h.permute(0, 2, 1)  # BxNx3
    # to euclidean
    points_dst = convert_points_from_homogeneous(points_dst_h)  # BxNx2
    return points_dst


def inverse(homography):
    # TODO: add documentation
    # NOTE: we expect in the future to have a native Pytorch function
    """Batched version of torch.inverse(...)
    """
    if not len(homography.shape) == 3:
        raise ValueError("Input size must be a three dimensional tensor. Got {}"
                         .format(points.shape))
    # iterate, compute inverse and stack tensors
    return torch.stack([torch.inverse(homo) for homo in homography])


def inverse_pose(pose):
    """Inverts a 4x4 pose.

    Args:
        points (Tensor): tensor of either size (4, 4) or (B, 4, 4).

    Returns:
        Tensor: tensor with same size as input.
    """
    if not torch.is_tensor(pose):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(pose)))
    if not pose.shape[-2:] == (4, 4):
        raise ValueError("Input size must be a 4x4 tensor. Got {}"
                         .format(pose.shape))
    pose_shape = pose.shape
    if len(pose_shape) == 2:
        pose = torch.unsqueeze(pose, dim=0)

    pose_inv = pose.clone()
    pose_inv[..., :3, 0:3] = torch.transpose(pose[..., :3, :3], 1, 2)
    pose_inv[..., :3, 3:4] = torch.matmul(
        -1.0 * pose_inv[..., :3, :3], pose[..., :3, 3:4])

    if len(pose_shape) == 2:
        pose_inv = torch.squeeze(pose_inv, dim=0)

    return pose_inv
