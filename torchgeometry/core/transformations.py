from typing import Optional

import torch
import torch.nn as nn


__all__ = [
    "inverse_pose",
    "relative_pose",
    "InversePose",
]


def inverse_pose(pose, eps=1e-6):
    r"""Function that inverts a 4x4 pose :math:`P =
    \begin{bmatrix} R & t \\ \mathbf{0} & 1 \end{bmatrix}`

    The inverse pose is computed as follows:

    .. math::

        P^{-1} = \begin{bmatrix} R^T & -R^T t \\ \mathbf{0} &
        1\end{bmatrix}

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


def relative_pose(pose_1: torch.Tensor, pose_2: torch.Tensor,
                  eps: Optional[float] = 1e-6) -> torch.Tensor:
    r"""Function that computes the relative transformation from a reference
    pose :math:`P_1^{\{W\}} = \begin{bmatrix} R_1 & t_1 \\ \mathbf{0} & 1
    \end{bmatrix}` to destination :math:`P_2^{\{W\}} = \begin{bmatrix} R_2 &
    t_2 \\ \mathbf{0} & 1 \end{bmatrix}`.

    The relative transformation is computed as follows:

    .. math::

        P_1^{2} = \begin{bmatrix} R_2 R_1^T & R_1^T (t_2 - t_1) \\ \mathbf{0} &
        1\end{bmatrix}

    Arguments:
        pose_1 (torch.Tensor): reference pose tensor of shape
         :math:`(N, 4, 4)`.
        pose_2 (torch.Tensor): destination pose tensor of shape
         :math:`(N, 4, 4)`.

    Shape:
        - Output: :math:`(N, 4, 4)`

    Returns:
        torch.Tensor: the relative transformation between the poses.

    Example::

        >>> pose_1 = torch.eye(4).unsqueeze(0)  # 1x4x4
        >>> pose_2 = torch.eye(4).unsqueeze(0)  # 1x4x4
        >>> pose_21 = tgm.relative_pose(pose_1, pose_2)  # 1x4x4
    """
    if not torch.is_tensor(pose_1):
        raise TypeError("Input pose_1 type is not a torch.Tensor. Got {}"
                        .format(type(pose_1)))
    if not torch.is_tensor(pose_2):
        raise TypeError("Input pose_2 type is not a torch.Tensor. Got {}"
                        .format(type(pose_2)))
    if not (len(pose_1.shape) == 3 and pose_1.shape[-2:] == (4, 4)):
        raise ValueError("Input must be a of the shape Nx4x4."
                         " Got {}".format(pose_1.shape, pose_2.shape))
    if not pose_1.shape == pose_2.shape:
        raise ValueError("Input pose_1 and pose_2 must be a of the same shape."
                         " Got {}".format(pose_1.shape, pose_2.shape))
    # unpack input data
    r_mat_1 = pose_1[..., :3, :3]  # Nx3x3
    r_mat_2 = pose_2[..., :3, :3]  # Nx3x3
    t_vec_1 = pose_1[..., :3, -1:]  # Nx3x1
    t_vec_2 = pose_2[..., :3, -1:]  # Nx3x1

    # compute relative pose
    r_mat_1_trans = r_mat_1.transpose(1, 2)
    r_mat_21 = torch.matmul(r_mat_2, r_mat_1_trans)
    t_vec_21 = torch.matmul(r_mat_1_trans, t_vec_2 - t_vec_1)

    # pack output data
    pose_21 = torch.zeros_like(pose_1)
    pose_21[..., :3, :3] = r_mat_21
    pose_21[..., :3, -1:] = t_vec_21
    pose_21[..., -1, -1] += 1.0
    return pose_21 + eps


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
