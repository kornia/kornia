import torch
import torch.nn as nn
import numpy as np

__all__ = [
    # functional api
    "pi",
    "rad2deg",
    "deg2rad",
    "convert_points_from_homogeneous",
    "convert_points_to_homogeneous",
    "transform_points",
    "angle_axis_to_rotation_matrix",
    "rotation_matrix_to_angle_axis",
    "rotation_matrix_to_quaternion",
    "quaternion_to_angle_axis",
    "rtvec_to_pose",
    # layer api
    "RadToDeg",
    "DegToRad",
    "ConvertPointsFromHomogeneous",
    "ConvertPointsToHomogeneous",
    "TransformPoints",
    "AngleAxisToRotationMatrix",
    "RotationMatrixToAngleAxis",
    "RotationMatrixToQuaternion",
    "QuaternionToAngleAxis",
    "RtvecToPose",
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
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))

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
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))

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


def angle_axis_to_rotation_matrix(angle_axis):
    """Convert 3d vector of axis-angle rotation to 4x4 rotation matrix

    Args:
        angle_axis (Tensor): tensor of 3d vector of axis-angle rotations
                             of size (N, 3).

    Returns:
        Tensor: tensor of 4x4 rotation matrices of size (N, 4, 4).

    """
    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat(
            [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat(
            [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(4).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = \
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx4x4


def rtvec_to_pose(rtvec):
    """
    Convert axis-angle rotation and translation vector to 4x4 pose matrix

    Args:
        rtvec (Tensor): Rodrigues vector transformations of size (N, 6).

    Returns:
        Tensor: transformation matrices of size (N, 4, 4).

    """
    assert rtvec.shape[-1] == 6, 'rtvec=[rx, ry, rz, tx, ty, tz]'
    pose = angle_axis_to_rotation_matrix(rtvec[..., :3])
    pose[..., :3, 3] = rtvec[..., 3:]
    return pose


def rotation_matrix_to_angle_axis(rotation_matrix):
    """Convert 4x4 rotation matrix to 4d quaternion vector

    Args:
        rotation_matrix (Tensor): rotation matrix of size (4, 4).

    Returns:
        Tensor: Rodrigues vector transformation of size (1, 6).

    """
    # todo add check that matrix is a valid rotation matrix
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(quaternion)


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """Convert 4x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion
    
    """
    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (1 - mask_d0_d1)
    mask_c2 = (1 - mask_d2) * mask_d0_nd1
    mask_c3 = (1 - mask_d2) * (1 - mask_d0_nd1)
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)
    q *= 0.5
    return q


def quaternion_to_angle_axis(quaternion, eps=1e-6):
    """Convert quaternion vector to angle axis of rotation
    
    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (Tensor): vector of length 4

    Return:
        Tensor: angle axis of rotation (vector of length 3)
    """
    assert quaternion.size(1) == 4, 'Input must be a vector of length 4'
    normalizer = 1 / torch.norm(quaternion, dim=1)
    q1 = quaternion[:, 1] * normalizer
    q2 = quaternion[:, 2] * normalizer
    q3 = quaternion[:, 3] * normalizer

    sin_squared = q1 * q1 + q2 * q2 + q3 * q3
    mask = (sin_squared > eps).to(sin_squared.device)
    mask_pos = (mask).type_as(sin_squared)
    mask_neg = (mask == False).type_as(sin_squared)  # noqa
    batch_size = quaternion.size(0)
    angle_axis = torch.zeros(
        batch_size, 3, dtype=quaternion.dtype).to(
        quaternion.device)

    sin_theta = torch.sqrt(sin_squared)
    cos_theta = quaternion[:, 0] * normalizer
    mask_theta = (cos_theta < eps).view(1, -1)
    mask_theta_neg = (mask_theta).type_as(cos_theta)
    mask_theta_pos = (mask_theta == False).type_as(cos_theta)  # noqa

    theta = torch.atan2(-sin_theta, -cos_theta) * mask_theta_neg \
        + torch.atan2(sin_theta, cos_theta) * mask_theta_pos

    two_theta = 2 * theta
    k_pos = two_theta / sin_theta
    k_neg = 2.0
    k = k_neg * mask_neg + k_pos * mask_pos

    angle_axis[:, 0] = q1 * k
    angle_axis[:, 1] = q2 * k
    angle_axis[:, 2] = q3 * k
    return angle_axis

# TODO: add below funtionalities
#  - rotation_matrix_to_angle_axis
#  - pose_to_rtvec


# layer api


class RadToDeg(nn.Module):
    def __init__(self):
        super(RadToDeg, self).__init__()

    def forward(self, input):
        return rad2deg(input)


class DegToRad(nn.Module):
    def __init__(self):
        super(DegToRad, self).__init__()

    def forward(self, input):
        return deg2rad(input)


class ConvertPointsFromHomogeneous(nn.Module):
    def __init__(self):
        super(ConvertPointsFromHomogeneous, self).__init__()

    def forward(self, input):
        return convert_points_from_homogeneous(input)


class ConvertPointsToHomogeneous(nn.Module):
    def __init__(self):
        super(ConvertPointsToHomogeneous, self).__init__()

    def forward(self, input):
        return convert_points_to_homogeneous(input)


class TransformPoints(nn.Module):
    def __init__(self):
        super(TransformPoints, self).__init__()

    def forward(self, dst_homo_src, points_src):
        return transform_points(dst_homo_src, points_src)


class AngleAxisToRotationMatrix(nn.Module):
    def __init__(self):
        super(AngleAxisToRotationMatrix, self).__init__()

    def forward(self, input):
        return angle_axis_to_rotation_matrix(input)


class RotationMatrixToAngleAxis(nn.Module):
    def __init__(self):
        super(RotationMatrixToAngleAxis, self).__init__()

    def forward(self, input):
        return rotation_matrix_to_angle_axis(input)


class RotationMatrixToQuaternion(nn.Module):
    def __init__(self):
        super(RotationMatrixToQuaterion, self).__init__()

    def forward(self, input):
        return rotation_matrix_to_quaterion(input)


class QuaternionToAngleAxis(nn.Module):
    def __init__(self):
        super(QuaternionToAngleAxis, self).__init__()

    def forward(self, input):
        return quaterion_to_angle_axis(input)


class RtvecToPose(nn.Module):
    def __init__(self):
        super(RtvecToPose, self).__init__()

    def forward(self, input):
        return rtvec_to_pose(input)
