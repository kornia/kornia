from typing import Optional

import torch
import torch.nn as nn


__all__ = [
    "compose_transformations",
    "inverse_transformation",
    "relative_transformation",
]


def compose_transformations(
        trans_1: torch.Tensor, trans_2: torch.Tensor,
        eps: Optional[float] = 1e-6) -> torch.Tensor:
    r"""Functions that composes two homogeneous transformations."""
    if not torch.is_tensor(trans_1):
        raise TypeError("Input trans_1 type is not a torch.Tensor. Got {}"
                        .format(type(trans_1)))
    if not torch.is_tensor(trans_2):
        raise TypeError("Input trans_2 type is not a torch.Tensor. Got {}"
                        .format(type(trans_2)))
    if not (len(trans_1.shape) == 3 and trans_1.shape[-2:] == (4, 4)):
        raise ValueError("Input trans_1 must be a of the shape Nx4x4."
                         " Got {}".format(trans_1.shape))
    if not (len(trans_2.shape) == 3 and trans_2.shape[-2:] == (4, 4)):
        raise ValueError("Input trans_2 must be a of the shape Nx4x4."
                         " Got {}".format(trans_2.shape))
    # unpack input data
    rmat_1: torch.Tensor = trans_1[..., :3, :3]  # Nx3x3
    rmat_2: torch.Tensor = trans_2[..., :3, :3]  # Nx3x3
    tvec_1: torch.Tensor = trans_1[..., :3, -1:]  # Nx3x1
    tvec_2: torch.Tensor = trans_2[..., :3, -1:]  # Nx3x1

    # compute the actual transforms composition, return the transform
    # from frame two to one. The variable should be trans_2_to_1, but for
    # simplicity it's just trans_21.
    rmat_21: torch.Tensor = torch.matmul(rmat_1, rmat_2)
    tvec_21: torch.Tensor = torch.matmul(rmat_1, tvec_2) + tvec_1

    # pack output tensor
    trans_21: torch.Tensor = torch.zeros_like(trans_1)
    trans_21[..., :3, 0:3] += rmat_21
    trans_21[..., :3, -1:] += tvec_21
    trans_21[..., -1, -1:] += 1.0
    return trans_21 + eps


def inverse_transformation(trans_12, eps=1e-6):
    r"""Function that inverts a 4x4 homogeneous transformation
    :math:`P = \begin{bmatrix} R & t \\ \mathbf{0} & 1 \end{bmatrix}`

    The inverse transformation is computed as follows:

    .. math::

        P^{-1} = \begin{bmatrix} R^T & -R^T t \\ \mathbf{0} &
        1\end{bmatrix}

    Args:
        points (torch.Tensor): tensor with transformations.

    Returns:
        torch.Tensor: tensor with inverted transformations.

    Shape:
        - Input: :math:`(N, 4, 4)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> trans_12 = torch.rand(1, 4, 4)  # Nx4x4
        >>> trans_21 = tgm.inverse_transformation(trans_12)  # Nx4x4
    """
    if not torch.is_tensor(trans_12):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(trans_12)))
    if not len(trans_12.shape) == 3 and trans_12.shape[-2:] == (4, 4):
        raise ValueError("Input size must be a Nx4x4 tensor. Got {}"
                         .format(trans_12.shape))
    # unpack input tensor
    rmat_12: torch.Tensor = trans_12[..., :3, 0:3]  # Nx3x3
    tvec_12: torch.Tensor = trans_12[..., :3, 3:4]  # Nx3x1

    # compute the actual inverse
    rmat_21: torch.Tensor = torch.transpose(rmat_12, 1, 2)
    tvec_21: torch.Tensor = torch.matmul(-rmat_21, tvec_12)

    # pack to output tensor
    trans_21: torch.Tensor = torch.zeros_like(trans_12)
    trans_21[..., :3, 0:3] += rmat_21
    trans_21[..., :3, -1:] += tvec_21
    trans_21[..., -1, -1:] += 1.0
    return trans_21 + eps


def relative_transformation(
        trans_1: torch.Tensor, trans_2: torch.Tensor,
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
    if not torch.is_tensor(trans_1):
        raise TypeError("Input trans_1 type is not a torch.Tensor. Got {}"
                        .format(type(pose_1)))
    if not torch.is_tensor(trans_2):
        raise TypeError("Input trans_2 type is not a torch.Tensor. Got {}"
                        .format(type(trans_2)))
    if not (len(trans_1.shape) == 3 and trans_1.shape[-2:] == (4, 4)):
        raise ValueError("Input must be a of the shape Nx4x4."
                         " Got {}".format(trans_1.shape))
    if not (len(trans_2.shape) == 3 and trans_2.shape[-2:] == (4, 4)):
        raise ValueError("Input must be a of the shape Nx4x4."
                         " Got {}".format(trans_2.shape))
    # unpack input data
    rmat_1: torch.Tensor = trans_1[..., :3, :3]  # Nx3x3
    rmat_2: torch.Tensor = trans_2[..., :3, :3]  # Nx3x3
    tvec_1: torch.Tensor = trans_1[..., :3, -1:]  # Nx3x1
    tvec_2: torch.Tensor = trans_2[..., :3, -1:]  # Nx3x1

    # compute relative transformation
    rmat_2_trans: torch.Tensor = rmat_2.transpose(1, 2)
    rmat_12: torch.Tensor = torch.matmul(rmat_2_trans, rmat_1)
    tvec_12: torch.Tensor = torch.matmul(rmat_2_trans, tvec_1 - tvec_2)

    # pack output data
    trans_12: torch.Tensor = torch.zeros_like(trans_1)
    trans_12[..., :3, 0:3] += rmat_12
    trans_12[..., :3, -1:] += tvec_12
    trans_12[..., -1, -1:] += 1.0
    return trans_12 + eps


# layer api


'''class InversePose(nn.Module):
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
        return inverse_pose(input)'''
