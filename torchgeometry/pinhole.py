import torch
import torch.nn as nn

from .conversions import rtvec_to_pose


__all__ = [
    # functional api
    "inverse_pose",
    "pinhole_matrix",
    "inverse_pinhole_matrix",
    "scale_pinhole",
    "homography_i_H_ref",
    # layer api
    "InversePose",
    "PinholeMatrix",
    "InversePinholeMatrix",
    "ScalePinhole",
    "Homography_i_H_ref",
]


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
        torch.matmul(i_pose_ref, inverse_pinhole_matrix(pinhole_ref)))


# layer api


class InversePose(nn.Module):
    def __init__(self):
        super(InversePose, self).__init__()

    def forward(self, input):
        return inverse_pose(input)


class PinholeMatrix(nn.Module):
    def __init__(self):
        super(PinholeMatrix, self).__init__()

    def forward(self, input):
        return pinhole_matrix(input)


class InversePinholeMatrix(nn.Module):
    def __init__(self):
        super(InversePinholeMatrix, self).__init__()

    def forward(self, input):
        return inverse_pinhole_matrix(input)


class ScalePinhole(nn.Module):
    def __init__(self):
        super(ScalePinhole, self).__init__()

    def forward(self, input, scale):
        return scale_pinhole(input, scale)


class Homography_i_H_ref(nn.Module):
    def __init__(self):
        super(Homography_i_H_ref, self).__init__()

    def forward(self, pinhole_i, pinhole_ref):
        return homography_i_H_ref(pinhole_i, pinhole_ref)
