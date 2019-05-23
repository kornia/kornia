from typing import Optional

import torch
import torch.nn as nn

from kornia.geometry.conversions import convert_points_to_homogeneous
from kornia.geometry.conversions import convert_points_from_homogeneous


__all__ = [
    "compose_transformations",
    "relative_transformation",
    "inverse_transformation",
    "transform_points",
]


def compose_transformations(
        trans_01: torch.Tensor, trans_12: torch.Tensor) -> torch.Tensor:
    r"""Functions that composes two homogeneous transformations.

    .. math::

        T_0^{2} = \begin{bmatrix} R_0^1 R_1^{2} & R_0^{1} t_1^{2} + t_0^{1} \\
        \mathbf{0} & 1\end{bmatrix}

    Args:
        trans_01 (torch.Tensor): tensor with the homogenous transformation from
          a reference frame 1 respect to a frame 0. The tensor has must have a
          shape of :math:`(B, 4, 4)` or :math:`(4, 4)`.
        trans_12 (torch.Tensor): tensor with the homogenous transformation from
          a reference frame 2 respect to a frame 1. The tensor has must have a
          shape of :math:`(B, 4, 4)` or :math:`(4, 4)`.

    Shape:
        - Output: :math:`(N, 4, 4)` or :math:`(4, 4)`

    Returns:
        torch.Tensor: the transformation between the two frames.

    Example::
        >>> trans_01 = torch.eye(4)  # 4x4
        >>> trans_12 = torch.eye(4)  # 4x4
        >>> trans_02 = kornia.compose_transformations(trans_01, trans_12)  # 4x4

    """
    if not torch.is_tensor(trans_01):
        raise TypeError("Input trans_01 type is not a torch.Tensor. Got {}"
                        .format(type(trans_01)))
    if not torch.is_tensor(trans_12):
        raise TypeError("Input trans_12 type is not a torch.Tensor. Got {}"
                        .format(type(trans_12)))
    if not trans_01.dim() in (2, 3) and trans_01.shape[-2:] == (4, 4):
        raise ValueError("Input trans_01 must be a of the shape Nx4x4 or 4x4."
                         " Got {}".format(trans_01.shape))
    if not trans_12.dim() in (2, 3) and trans_12.shape[-2:] == (4, 4):
        raise ValueError("Input trans_12 must be a of the shape Nx4x4 or 4x4."
                         " Got {}".format(trans_12.shape))
    if not trans_01.dim() == trans_12.dim():
        raise ValueError("Input number of dims must match. Got {} and {}"
                         .format(trans_01.dim(), trans_12.dim()))
    # unpack input data
    rmat_01: torch.Tensor = trans_01[..., :3, :3]  # Nx3x3
    rmat_12: torch.Tensor = trans_12[..., :3, :3]  # Nx3x3
    tvec_01: torch.Tensor = trans_01[..., :3, -1:]  # Nx3x1
    tvec_12: torch.Tensor = trans_12[..., :3, -1:]  # Nx3x1

    # compute the actual transforms composition
    rmat_02: torch.Tensor = torch.matmul(rmat_01, rmat_12)
    tvec_02: torch.Tensor = torch.matmul(rmat_01, tvec_12) + tvec_01

    # pack output tensor
    trans_02: torch.Tensor = torch.zeros_like(trans_01)
    trans_02[..., :3, 0:3] += rmat_02
    trans_02[..., :3, -1:] += tvec_02
    trans_02[..., -1, -1:] += 1.0
    return trans_02


def inverse_transformation(trans_12):
    r"""Function that inverts a 4x4 homogeneous transformation
    :math:`T_1^{2} = \begin{bmatrix} R_1 & t_1 \\ \mathbf{0} & 1 \end{bmatrix}`

    The inverse transformation is computed as follows:

    .. math::

        T_2^{1} = (T_1^{2})^{-1} = \begin{bmatrix} R_1^T & -R_1^T t_1 \\
        \mathbf{0} & 1\end{bmatrix}

    Args:
        trans_12 (torch.Tensor): transformation tensor of shape
          :math:`(N, 4, 4)` or :math:`(4, 4)`.

    Returns:
        torch.Tensor: tensor with inverted transformations.

    Shape:
        - Output: :math:`(N, 4, 4)` or :math:`(4, 4)`

    Example:
        >>> trans_12 = torch.rand(1, 4, 4)  # Nx4x4
        >>> trans_21 = kornia.inverse_transformation(trans_12)  # Nx4x4
    """
    if not torch.is_tensor(trans_12):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(trans_12)))
    if not trans_12.dim() in (2, 3) and trans_12.shape[-2:] == (4, 4):
        raise ValueError("Input size must be a Nx4x4 or 4x4. Got {}"
                         .format(trans_12.shape))
    # unpack input tensor
    rmat_12: torch.Tensor = trans_12[..., :3, 0:3]  # Nx3x3
    tvec_12: torch.Tensor = trans_12[..., :3, 3:4]  # Nx3x1

    # compute the actual inverse
    rmat_21: torch.Tensor = torch.transpose(rmat_12, -1, -2)
    tvec_21: torch.Tensor = torch.matmul(-rmat_21, tvec_12)

    # pack to output tensor
    trans_21: torch.Tensor = torch.zeros_like(trans_12)
    trans_21[..., :3, 0:3] += rmat_21
    trans_21[..., :3, -1:] += tvec_21
    trans_21[..., -1, -1:] += 1.0
    return trans_21


def relative_transformation(
        trans_01: torch.Tensor, trans_02: torch.Tensor) -> torch.Tensor:
    r"""Function that computes the relative homogenous transformation from a
    reference transformation :math:`T_1^{0} = \begin{bmatrix} R_1 & t_1 \\
    \mathbf{0} & 1 \end{bmatrix}` to destination :math:`T_2^{0} =
    \begin{bmatrix} R_2 & t_2 \\ \mathbf{0} & 1 \end{bmatrix}`.

    The relative transformation is computed as follows:

    .. math::

        T_1^{2} = (T_0^{1})^{-1} \cdot T_0^{2}

    Arguments:
        trans_01 (torch.Tensor): reference transformation tensor of shape
         :math:`(N, 4, 4)` or :math:`(4, 4)`.
        trans_02 (torch.Tensor): destination transformation tensor of shape
         :math:`(N, 4, 4)` or :math:`(4, 4)`.

    Shape:
        - Output: :math:`(N, 4, 4)` or :math:`(4, 4)`.

    Returns:
        torch.Tensor: the relative transformation between the transformations.

    Example::
        >>> trans_01 = torch.eye(4)  # 4x4
        >>> trans_02 = torch.eye(4)  # 4x4
        >>> trans_12 = kornia.relative_transformation(trans_01, trans_02)  # 4x4
    """
    if not torch.is_tensor(trans_01):
        raise TypeError("Input trans_01 type is not a torch.Tensor. Got {}"
                        .format(type(trans_01)))
    if not torch.is_tensor(trans_02):
        raise TypeError("Input trans_02 type is not a torch.Tensor. Got {}"
                        .format(type(trans_02)))
    if not trans_01.dim() in (2, 3) and trans_01.shape[-2:] == (4, 4):
        raise ValueError("Input must be a of the shape Nx4x4 or 4x4."
                         " Got {}".format(trans_01.shape))
    if not trans_02.dim() in (2, 3) and trans_02.shape[-2:] == (4, 4):
        raise ValueError("Input must be a of the shape Nx4x4 or 4x4."
                         " Got {}".format(trans_02.shape))
    if not trans_01.dim() == trans_02.dim():
        raise ValueError("Input number of dims must match. Got {} and {}"
                         .format(trans_01.dim(), trans_02.dim()))
    trans_10: torch.Tensor = inverse_transformation(trans_01)
    trans_12: torch.Tensor = compose_transformations(trans_10, trans_02)
    return trans_12


def transform_points(trans_01: torch.Tensor,
                     points_1: torch.Tensor) -> torch.Tensor:
    r"""Function that applies transformations to a set of points.

    Args:
        trans_01 (torch.Tensor): tensor for transformations of shape
          :math:`(B, D+1, D+1)`.
        points_1 (torch.Tensor): tensor of points of shape :math:`(B, N, D)`.
    Returns:
        torch.Tensor: tensor of N-dimensional points.

    Shape:
        - Output: :math:`(B, N, D)`

    Examples:

        >>> points_1 = torch.rand(2, 4, 3)  # BxNx3
        >>> trans_01 = torch.eye(4).view(1, 4, 4)  # Bx4x4
        >>> points_0 = kornia.transform_points(trans_01, points_1)  # BxNx3
    """
    if not torch.is_tensor(trans_01) or not torch.is_tensor(points_1):
        raise TypeError("Input type is not a torch.Tensor")
    if not trans_01.device == points_1.device:
        raise TypeError("Tensor must be in the same device")
    if not trans_01.shape[0] == points_1.shape[0]:
        raise ValueError("Input batch size must be the same for both tensors")
    if not trans_01.shape[-1] == (points_1.shape[-1] + 1):
        raise ValueError("Last input dimensions must differe by one unit")
    # to homogeneous
    points_1_h = convert_points_to_homogeneous(points_1)  # BxNxD+1
    # transform coordinates
    points_0_h = torch.matmul(
        trans_01.unsqueeze(1), points_1_h.unsqueeze(-1))
    points_0_h = torch.squeeze(points_0_h, dim=-1)
    # to euclidean
    points_0 = convert_points_from_homogeneous(points_0_h)  # BxNxD
    return points_0

# TODO:
# - project_points: from opencv


# layer api

# NOTE: is it needed ?
'''class TransformPoints(nn.Module):
    r"""Creates an object to transform a set of points.

    Args:
        dst_pose_src (torhc.Tensor): tensor for transformations of
          shape :math:`(B, D+1, D+1)`.

    Returns:
        torch.Tensor: tensor of N-dimensional points.

    Shape:
        - Input: :math:`(B, D, N)`
        - Output: :math:`(B, N, D)`

    Examples:
        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> transform = torch.eye(4).view(1, 4, 4)   # Bx4x4
        >>> transform_op = kornia.TransformPoints(transform)
        >>> output = transform_op(input)  # BxNx3
    """

    def __init__(self, dst_homo_src: torch.Tensor) -> None:
        super(TransformPoints, self).__init__()
        self.dst_homo_src: torch.Tensor = dst_homo_src

    def forward(self, points_src: torch.Tensor) -> torch.Tensor:  # type: ignore
        return transform_points(self.dst_homo_src, points_src)


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
        >>> transform = kornia.InversePose()
        >>> pose_inv = transform(pose)  # Nx4x4
    """

    def __init__(self):
        super(InversePose, self).__init__()

    def forward(self, input):
        return inverse_pose(input)'''
