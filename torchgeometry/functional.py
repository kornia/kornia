import torch

from .transforms import rtvec_to_pose


__all__ = [
    "pi",
    "rad2deg",
    "deg2rad",
    "convert_points_from_homogeneous",
    "convert_points_to_homogeneous",
    "transform_points",
    "inverse",
    "inverse_pose",
    "pinhole_matrix",
    "inv_pinhole_matrix",
    "scale_pinhole",
    "homography_i_H_ref",
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
    if not dst_homo_src.shape[2] == (points_src.shape[2] + 1):
        raise ValueError("Input dimensions must differe by one unit")
    # to homogeneous
    points_src_h = convert_points_to_homogeneous(points_src)  # BxNx3
    # transform coordinates
    points_dst_h = torch.matmul(dst_homo_src, points_src_h.transpose(1, 2))
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
                         .format(homography.shape))
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


def pinhole_matrix(pinhole):
    assert len(pinhole.shape) == 2 and pinhole.shape[1] == 12, pinhole.shape
    # unpack pinhole values
    fx, fy, cx, cy = torch.chunk(pinhole[..., :4], 4, dim=1)  # Nx1
    # create output container
    k = torch.eye(4, device=pinhole.device, dtype=pinhole.dtype)
    k = k.view(1, 4, 4).repeat(pinhole.shape[0], 1, 1)   # Nx4x4
    # fill output with pinhole values
    k[..., 0, 0] = fx
    k[..., 0, 2] = cx
    k[..., 1, 1] = fy
    k[..., 1, 2] = cy
    return k


def inv_pinhole_matrix(pinhole, eps=1e-6):
    assert len(pinhole.shape) == 2 and pinhole.shape[1] == 12, pinhole.shape
    # unpack pinhole values
    fx, fy, cx, cy = torch.chunk(pinhole[..., :4], 4, dim=1)  # Nx1
    # create output container
    k = torch.eye(4, device=pinhole.device, dtype=pinhole.dtype)
    k = k.view(1, 4, 4).repeat(pinhole.shape[0], 1, 1)   # Nx4x4
    # fill output with inverse values
    k[..., 0, 0] = 1. / (fx + eps)
    k[..., 1, 1] = 1. / (fy + eps)
    k[..., 0, 2] = -1. * cx / (fx + eps)
    k[..., 1, 2] = -1. * cy / (fy + eps)
    return k


def scale_pinhole(pinhole, scale):
    """Scales the pinhole matrix from a pinhole model.
    """
    assert len(pinhole.shape) == 2 and pinhole.shape[1] == 12, pinhole.shape
    assert len(scale.shape) == 2 and scale.shape[1] == 1, scale.shape
    pinhole_scaled = pinhole.clone()
    pinhole_scaled[..., :6] = pinhole[..., :6] * scale
    return pinhole_scaled


def get_optical_pose_base(pinhole):
    assert len(pinhole.shape) == 2 and pinhole.shape[1] == 12, pinhole.shape
    optical_pose_parent = pinhole[..., 6:]
    return rtvec_to_pose(optical_pose_parent)


def homography_i_H_ref(pinhole_i, pinhole_ref):
    """
    Homography from pinhole_ref to pinhole_i
    :returns matrix (4x4) that converts depth points from pinhole_ref to
        pinhole_i
    """
    assert len(pinhole_i.shape) == 2 and pinhole_i.shape[1] == 12, pinhole.shape
    assert pinhole_i.shape == pinhole_ref.shape, pinhole_ref.shape
    i_pose_base = get_optical_pose_base(pinhole_i)
    ref_pose_base = get_optical_pose_base(pinhole_ref)
    i_pose_ref = torch.matmul(i_pose_base, inverse_pose(ref_pose_base))
    return torch.matmul(
        pinhole_matrix(pinhole_i),
        torch.matmul(
            i_pose_ref,
            inv_pinhole_matrix(pinhole_ref)))
