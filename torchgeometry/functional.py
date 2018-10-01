import torch

from .conversions import rtvec_to_pose

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


def rad2deg(tensor):
    """Converts angles from radians to degrees.

    Args:
        tensor (Tensor): Tensor to be converted of unspecified shape.

    Returns:
        Tensor: Converted tensor with same shape as input.

    Example:
        >>> input = tgm.pi * torch.rand(1, 3, 3)
        >>> output = tgm.rad2deg(input)
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(tensor)))

    return 180. * tensor / pi.to(tensor.device).type(tensor.dtype)


def deg2rad(tensor):
    """Converts angles from degrees to radians.

    Args:
        tensor (Tensor): Tensor to be converted of unspecified shape.

    Returns:
        Tensor: Converted tensor with same shape as input.

    Example:
        >>> input = 360. * torch.rand(1, 3, 3)
        >>> output = tgm.deg2rad(input)
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(tensor)))

    return tensor * pi.to(tensor.device).type(tensor.dtype) / 180.


def convert_points_from_homogeneous(points, eps=1e-6):
    """Converts points from homogeneous to Euclidean space.

    Args:
        points (Tensor): tensor of N-dimensional points of size (B, D, N).

    Returns:
        Tensor: tensor of N-1-dimensional points of size (B, D, N-1).
    """
    if not torch.is_tensor(points):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(points)))

    if not len(points.shape) == 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                points.shape))

    return points[..., :-1] / (points[..., -1:] + eps)


def convert_points_to_homogeneous(points):
    """Converts points from Euclidean to homogeneous space.

    Args:
        points (Tensor): tensor of N-dimensional points of size (B, D, N).

    Returns:
        Tensor: tensor of N+1-dimensional points of size (B, D, N+1).
    """
    if not torch.is_tensor(points):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(points)))

    if not len(points.shape) == 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                points.shape))

    return torch.cat([points, torch.ones_like(points)[..., :1]], dim=-1)


def transform_points(dst_pose_src, points_src):
    """Applies batch of transformations to batch of sets of points.

    Args: 
        dst_pose_src (Tensor): tensor for transformations of size (B, D+1, D+1).
        points_src (Tensor): tensor of points of size (B, N, D).

    Returns:
        Tensor: tensor of N-dimensional points of size (B, D, N).

    """
    if not torch.is_tensor(dst_pose_src) or not torch.is_tensor(points_src):
        raise TypeError("Input type is not a torch.Tensor")
    if not dst_pose_src.device == points_src.device:
        raise TypeError("Tensor must be in the same device")
    if not len(dst_pose_src.shape) == 3 or not len(points_src.shape) == 3:
        raise ValueError("Input size must be a three dimensional tensor")
    if not dst_pose_src.shape[0] == points_src.shape[0]:
        raise ValueError("Input batch size must be the same for both tensors")
    if not dst_pose_src.shape[2] == (points_src.shape[2] + 1):
        raise ValueError("Input dimensions must differe by one unit")
    # to homogeneous
    points_src_h = convert_points_to_homogeneous(points_src)  # BxNxD+1
    # transform coordinates
    points_dst_h = torch.matmul(dst_pose_src, points_src_h.transpose(1, 2))
    points_dst_h = points_dst_h.permute(0, 2, 1)  # BxNxD+1
    # to euclidean
    points_dst = convert_points_from_homogeneous(points_dst_h)  # BxNxD
    return points_dst


def inverse(transforms):
    """Batched version of torch.inverse(...)

    Args:
        transforms (Tensor): tensor of transformations of size (B, D, D).

    Returns:
        Tensor: tensor of inverted transformations of size (B, D, D).

    """
    if not len(transforms.shape) == 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                transforms.shape))
    # iterate, compute inverse and stack tensors
    return torch.stack([torch.inverse(transform) for transform in transforms])


def inverse_pose(pose):
    """Inverts a 4x4 pose.

    Args:
        points (Tensor): tensor of either size (4, 4) or (B, 4, 4).

    Returns:
        Tensor: tensor with same size as input.
    """
    if not torch.is_tensor(pose):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(pose)))
    if not pose.shape[-2:] == (4, 4):
        raise ValueError("Input size must be a 4x4 tensor. Got {}"
                         .format(pose.shape))
    pose_shape = pose.shape
    if len(pose_shape) == 2:
        pose = torch.unsqueeze(pose, dim=0)

    pose_inv = pose.clone()
    pose_inv[..., :3, 0:3] = torch.transpose(pose[..., :3, :3], 1, 2)
    pose_inv[..., :3, 3:4] = torch.matmul(-1.0 * pose_inv[..., :3, :3],
                                          pose[..., :3, 3:4])

    if len(pose_shape) == 2:
        pose_inv = torch.squeeze(pose_inv, dim=0)

    return pose_inv


def pinhole_matrix(pinholes):
    """Converts vector pinhole representation to 4x4 Tensor

    Args:
        pinholes (Tensor): tensor of form [fx fy cx cy h w rx ry rz tx ty tz] of size (N, 12).

    Returns:
        Tensor: tensor of pinhole matrices of size (N, 4, 4).

    """
    assert len(pinholes.shape) == 2 and pinholes.shape[1] == 12, pinholes.shape
    # unpack pinhole values
    fx, fy, cx, cy = torch.chunk(pinholes[..., :4], 4, dim=1)  # Nx1
    # create output container
    k = torch.eye(4, device=pinholes.device, dtype=pinholes.dtype)
    k = k.view(1, 4, 4).repeat(pinholes.shape[0], 1, 1)  # Nx4x4
    # fill output with pinhole values
    k[..., 0, 0] = fx
    k[..., 0, 2] = cx
    k[..., 1, 1] = fy
    k[..., 1, 2] = cy
    return k


def inv_pinhole_matrix(pinhole, eps=1e-6):
    """Invert a set of pinholes

    Args:
        pinholes (Tensor): tensor of form [fx fy cx cy h w rx ry rz tx ty tz] of size (N, 12).

    Returns:
        Tensor: tensor of inverted pinhole matrices of size (N, 4, 4).

    """
    assert len(pinhole.shape) == 2 and pinhole.shape[1] == 12, pinhole.shape
    # unpack pinhole values
    fx, fy, cx, cy = torch.chunk(pinhole[..., :4], 4, dim=1)  # Nx1
    # create output container
    k = torch.eye(4, device=pinhole.device, dtype=pinhole.dtype)
    k = k.view(1, 4, 4).repeat(pinhole.shape[0], 1, 1)  # Nx4x4
    # fill output with inverse values
    k[..., 0, 0] = 1. / (fx + eps)
    k[..., 1, 1] = 1. / (fy + eps)
    k[..., 0, 2] = -1. * cx / (fx + eps)
    k[..., 1, 2] = -1. * cy / (fy + eps)
    return k


def scale_pinhole(pinholes, scale):
    """Scales the pinhole matrix for each pinhole model.

    Args:
        pinholes (Tensor): tensor of form [fx fy cx cy h w rx ry rz tx ty tz] of size (N, 12).
        scale (Tensor): tensor of scales of form [N, 1]

    Returns:
        Tensor: tensor of scaled pinholes of form [fx fy cx cy h w rx ry rz tx ty tz] of size (N, 12).

    """
    assert len(pinholes.shape) == 2 and pinholes.shape[1] == 12, pinholes.shape
    assert len(scale.shape) == 2 and scale.shape[1] == 1, scale.shape
    pinholes_scaled = pinholes.clone()
    pinholes_scaled[..., :6] = pinholes[..., :6] * scale
    return pinholes_scaled


def get_optical_pose_base(pinholes):
    """Get extrinsic transformation matrices for pinholes

    Args:
        pinholes (Tensor): tensor of form [fx fy cx cy h w rx ry rz tx ty tz] of size (N, 12).

    Returns:
        Tensor: tensor of extrinsic transformation matrices of size (N, 4, 4).

    """
    assert len(pinholes.shape) == 2 and pinholes.shape[1] == 12, pinholes.shape
    optical_pose_parent = pinholes[..., 6:]
    return rtvec_to_pose(optical_pose_parent)


def homography_i_H_ref(pinhole_i, pinhole_ref):
    """
    Homography from pinhole_ref to pinhole_i

    Args: 
        pinhole_i (Tensor): tensor of form [fx fy cx cy h w rx ry rz tx ty tz] of size (N, 12).
        pinhole_ref (Tensor): tensor of form [fx fy cx cy h w rx ry rz tx ty tz] of size (N, 12).

    Returns:
        Tensor: tensors that convert depth points (u, v, d) from pinhole_ref to pinhole_i (N, 4, 4).
        
    """
    assert len(
        pinhole_i.shape) == 2 and pinhole_i.shape[1] == 12, pinhole.shape
    assert pinhole_i.shape == pinhole_ref.shape, pinhole_ref.shape
    i_pose_base = get_optical_pose_base(pinhole_i)
    ref_pose_base = get_optical_pose_base(pinhole_ref)
    i_pose_ref = torch.matmul(i_pose_base, inverse_pose(ref_pose_base))
    return torch.matmul(
        pinhole_matrix(pinhole_i),
        torch.matmul(i_pose_ref, inv_pinhole_matrix(pinhole_ref)))
