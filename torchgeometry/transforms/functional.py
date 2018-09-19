import torch
import numpy as np


def angle_axis_to_rotation_matrix_numpy(angle_axis):
    """
    Convert 3d vector of axis-angle rotation to 4x4 rotation matrix
    """
    # stolen from ceres/rotation.h
    k_one = 1.0
    theta2 = angle_axis.dot(angle_axis)
    rotation_matrix = np.eye(4, dtype=angle_axis.dtype)
    if theta2 > np.finfo(np.float32).eps:
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        theta = np.sqrt(theta2)
        wx = angle_axis[0] / theta
        wy = angle_axis[1] / theta
        wz = angle_axis[2] / theta
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        rotation_matrix[0, 0] = cos_theta + wx * wx * (k_one - cos_theta)
        rotation_matrix[1, 0] = wz * sin_theta + wx * wy * (k_one - cos_theta)
        rotation_matrix[2, 0] = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        rotation_matrix[0, 1] = wx * wy * (k_one - cos_theta) - wz * sin_theta
        rotation_matrix[1, 1] = cos_theta + wy * wy * (k_one - cos_theta)
        rotation_matrix[2, 1] = wx * sin_theta + wy * wz * (k_one - cos_theta)
        rotation_matrix[0, 2] = wy * sin_theta + wx * wz * (k_one - cos_theta)
        rotation_matrix[1, 2] = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        rotation_matrix[2, 2] = cos_theta + wz * wz * (k_one - cos_theta)
    else:
        # Near zero, we switch to using the first order Taylor expansion.
        rotation_matrix[0, 0] = k_one
        rotation_matrix[1, 0] = angle_axis[2]
        rotation_matrix[2, 0] = -angle_axis[1]
        rotation_matrix[0, 1] = -angle_axis[2]
        rotation_matrix[1, 1] = k_one
        rotation_matrix[2, 1] = angle_axis[0]
        rotation_matrix[0, 2] = angle_axis[1]
        rotation_matrix[1, 2] = -angle_axis[0]
        rotation_matrix[2, 2] = k_one
    return rotation_matrix


def angle_axis_to_rotation_matrix_torch(angle_axis, eps=1e-6):
    """
    Convert 3d vector of axis-angle rotation to 4x4 rotation matrix
    """
    def _compute_rotation_matrix(angle_axis, theta2):
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
        rotation_matrix = torch.cat([
            r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)
    
    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat([
            k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask == True).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(4).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = \
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx4x4


def angle_axis_to_rotation_matrix(angle_axis):
    if isinstance(angle_axis, np.ndarray):
       return angle_axis_to_rotation_matrix_numpy(angle_axis)
    elif isinstance(angle_axis, torch.Tensor):
       if not (len(angle_axis.shape) == 2 and angle_axis.shape[1] == 3):
           raise ValueError("Input must be a two dimensional torch.Tensor.")
       return angle_axis_to_rotation_matrix_torch(angle_axis)
    else:
       raise NotImplementedError('Not suported type {}'.format(type(angle_axis)))


def rtvec_to_pose(rtvec):
    """
    Convert axis-angle rotation and translation vector to 4x4 pose matrix
    """
    assert rtvec.shape[-1] == 6, 'rtvec=[rx, ry, rz, tx, ty, tz]'
    pose = angle_axis_to_rotation_matrix(rtvec[..., :3])
    pose[..., :3, 3] = rtvec[..., 3:]
    return pose


# TODO: add below funtionalities
#  - rotation_matrix_to_angle_axis
#  - pose_to_rtvec
