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
]


def inverse_pose(pose, eps=1e-6):
    r"""Function that inverts a 4x4 pose.

    Args:
        points (Tensor): tensor with poses.

    Returns:
        Tensor: tensor with inverted poses.

    Shape:
        - Input: :math:`(N, 4, 4)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> pose = torch.rand(1, 4, 4)         # Nx4x4
        >>> pose_inv = tgm.inverse_pose(pose)  # Nx4x4
    """
    if not torch.is_tensor(pose):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(pose)))
    if not len(pose.shape) == 3 and pose.shape[-2:] == (4, 4):
        raise ValueError("Input size must be a Nx4x4 tensor. Got {}"
                         .format(pose.shape))

    r_mat = pose[..., :3, 0:3]  # Nx3x3
    t_vec = pose[..., :3, 3:4]  # Nx3x1
    r_mat_trans = torch.transpose(r_mat, 1, 2)

    pose_inv = pose.new_zeros(pose.shape) + eps
    pose_inv[..., :3, 0:3] = r_mat_trans
    pose_inv[..., :3, 3:4] = torch.matmul(-1.0 * r_mat_trans, t_vec)
    pose_inv[..., 3, 3] = 1.0
    return pose_inv


def pinhole_matrix(pinholes, eps=1e-6):
    r"""Function that returns the pinhole matrix from a pinhole model

    Args:
        pinholes (Tensor): tensor of pinhole models.

    Returns:
        Tensor: tensor of pinhole matrices.

    Shape:
        - Input: :math:`(N, 12)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> pinhole = torch.rand(1, 12)    # Nx12
        >>> pinhole_matrix = tgm.pinhole_matrix(pinhole)  # Nx4x4
    """
    assert len(pinholes.shape) == 2 and pinholes.shape[1] == 12, pinholes.shape
    # unpack pinhole values
    fx, fy, cx, cy = torch.chunk(pinholes[..., :4], 4, dim=1)  # Nx1
    # create output container
    k = torch.eye(4, device=pinholes.device, dtype=pinholes.dtype) + eps
    k = k.view(1, 4, 4).repeat(pinholes.shape[0], 1, 1)  # Nx4x4
    # fill output with pinhole values
    k[..., 0, 0:1] = fx
    k[..., 0, 2:3] = cx
    k[..., 1, 1:2] = fy
    k[..., 1, 2:3] = cy
    return k


def inverse_pinhole_matrix(pinhole, eps=1e-6):
    r"""Returns the inverted pinhole matrix from a pinhole model

    Args:
        pinholes (Tensor): tensor with pinhole models.

    Returns:
        Tensor: tensor of inverted pinhole matrices.

    Shape:
        - Input: :math:`(N, 12)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> pinhole = torch.rand(1, 12)    # Nx12
        >>> pinhole_matrix_inv = tgm.inverse_pinhole_matrix(pinhole)  # Nx4x4
    """
    assert len(pinhole.shape) == 2 and pinhole.shape[1] == 12, pinhole.shape
    # unpack pinhole values
    fx, fy, cx, cy = torch.chunk(pinhole[..., :4], 4, dim=1)  # Nx1
    # create output container
    k = torch.eye(4, device=pinhole.device, dtype=pinhole.dtype)
    k = k.view(1, 4, 4).repeat(pinhole.shape[0], 1, 1)  # Nx4x4
    # fill output with inverse values
    k[..., 0, 0:1] = 1. / (fx + eps)
    k[..., 1, 1:2] = 1. / (fy + eps)
    k[..., 0, 2:3] = -1. * cx / (fx + eps)
    k[..., 1, 2:3] = -1. * cy / (fy + eps)
    return k


def scale_pinhole(pinholes, scale):
    r"""Scales the pinhole matrix for each pinhole model.

    Args:
        pinholes (Tensor): tensor with the pinhole model.
        scale (Tensor): tensor of scales.

    Returns:
        Tensor: tensor of scaled pinholes.

    Shape:
        - Input: :math:`(N, 12)` and :math:`(N, 1)`
        - Output: :math:`(N, 12)`

    Example:
        >>> pinhole_i = torch.rand(1, 12)  # Nx12
        >>> scales = 2.0 * torch.ones(1)   # N
        >>> pinhole_i_scaled = tgm.scale_pinhole(pinhole_i)  # Nx12
    """
    assert len(pinholes.shape) == 2 and pinholes.shape[1] == 12, pinholes.shape
    assert len(scale.shape) == 1, scale.shape
    pinholes_scaled = pinholes.clone()
    pinholes_scaled[..., :6] = pinholes[..., :6] * scale.unsqueeze(-1)
    return pinholes_scaled


def get_optical_pose_base(pinholes):
    """Get extrinsic transformation matrices for pinholes

    Args:
        pinholes (Tensor): tensor of form [fx fy cx cy h w rx ry rz tx ty tz]
                           of size (N, 12).

    Returns:
        Tensor: tensor of extrinsic transformation matrices of size (N, 4, 4).

    """
    assert len(pinholes.shape) == 2 and pinholes.shape[1] == 12, pinholes.shape
    optical_pose_parent = pinholes[..., 6:]
    return rtvec_to_pose(optical_pose_parent)


def homography_i_H_ref(pinhole_i, pinhole_ref):
    r"""Homography from reference to ith pinhole

    .. math::

        H_{ref}^{i} = K_{i} * T_{ref}^{i} * K_{ref}^{-1}

    Args:
        pinhole_i (Tensor): tensor with pinhole model for ith frame.
        pinhole_ref (Tensor): tensor with pinhole model for reference frame.

    Returns:
        Tensor: tensors that convert depth points (u, v, d) from
        pinhole_ref to pinhole_i.

    Shape:
        - Input: :math:`(N, 12)` and :math:`(N, 12)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> pinhole_i = torch.rand(1, 12)    # Nx12
        >>> pinhole_ref = torch.rand(1, 12)  # Nx12
        >>> i_H_ref = tgm.homography_i_H_ref(pinhole_i, pinhole_ref)  # Nx4x4
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
    r"""Creates a transformation that inverts a 4x4 pose.

    Args:
        points (Tensor): tensor with poses.

    Returns:
        Tensor: tensor with inverted poses.

    Shape:
        - Input: :math:`(N, 4, 4)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> pose = torch.rand(1, 4, 4)  # Nx4x4
        >>> transform = tgm.InversePose()
        >>> pose_inv = transform(pose)  # Nx4x4
    """

    def __init__(self):
        super(InversePose, self).__init__()

    def forward(self, input):
        return inverse_pose(input)


class PinholeMatrix(nn.Module):
    r"""Creates an object that returns the pinhole matrix from a pinhole model

    Args:
        pinholes (Tensor): tensor of pinhole models.

    Returns:
        Tensor: tensor of pinhole matrices.

    Shape:
        - Input: :math:`(N, 12)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> pinhole = torch.rand(1, 12)          # Nx12
        >>> transform = tgm.PinholeMatrix()
        >>> pinhole_matrix = transform(pinhole)  # Nx4x4
    """

    def __init__(self):
        super(PinholeMatrix, self).__init__()

    def forward(self, input):
        return pinhole_matrix(input)


class InversePinholeMatrix(nn.Module):
    r"""Returns and object that inverts a pinhole matrix from a pinhole model

    Args:
        pinholes (Tensor): tensor with pinhole models.

    Returns:
        Tensor: tensor of inverted pinhole matrices.

    Shape:
        - Input: :math:`(N, 12)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> pinhole = torch.rand(1, 12)              # Nx12
        >>> transform = tgm.InversePinholeMatrix()
        >>> pinhole_matrix_inv = transform(pinhole)  # Nx4x4
    """

    def __init__(self):
        super(InversePinholeMatrix, self).__init__()

    def forward(self, input):
        return inverse_pinhole_matrix(input)
