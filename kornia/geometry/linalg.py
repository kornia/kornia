import torch

from kornia.testing import KORNIA_CHECK_IS_TENSOR, check_is_tensor
from .conversions import convert_points_from_homogeneous, convert_points_to_homogeneous
from torch import Tensor


__all__ = [
    "compose_transformations",
    "relative_transformation",
    "inverse_transformation",
    "transform_points",
    "point_line_distance",
]


def compose_transformations(trans_01: torch.Tensor, trans_12: torch.Tensor) -> torch.Tensor:
    r"""Function that composes two homogeneous transformations.

    .. math::
        T_0^{2} = \begin{bmatrix} R_0^1 R_1^{2} & R_0^{1} t_1^{2} + t_0^{1} \\
        \mathbf{0} & 1\end{bmatrix}

    Args:
        trans_01: tensor with the homogeneous transformation from
          a reference frame 1 respect to a frame 0. The tensor has must have a
          shape of :math:`(N, 4, 4)` or :math:`(4, 4)`.
        trans_12: tensor with the homogeneous transformation from
          a reference frame 2 respect to a frame 1. The tensor has must have a
          shape of :math:`(N, 4, 4)` or :math:`(4, 4)`.

    Returns:
        the transformation between the two frames with shape :math:`(N, 4, 4)` or :math:`(4, 4)`.

    Example::
        >>> trans_01 = torch.eye(4)  # 4x4
        >>> trans_12 = torch.eye(4)  # 4x4
        >>> trans_02 = compose_transformations(trans_01, trans_12)  # 4x4

    """
    if not torch.is_tensor(trans_01):
        raise TypeError(f"Input trans_01 type is not a torch.Tensor. Got {type(trans_01)}")

    if not torch.is_tensor(trans_12):
        raise TypeError(f"Input trans_12 type is not a torch.Tensor. Got {type(trans_12)}")

    if not ((trans_01.dim() in (2, 3)) and (trans_01.shape[-2:] == (4, 4))):
        raise ValueError("Input trans_01 must be a of the shape Nx4x4 or 4x4." " Got {}".format(trans_01.shape))

    if not ((trans_12.dim() in (2, 3)) and (trans_12.shape[-2:] == (4, 4))):
        raise ValueError("Input trans_12 must be a of the shape Nx4x4 or 4x4." " Got {}".format(trans_12.shape))

    if not trans_01.dim() == trans_12.dim():
        raise ValueError(f"Input number of dims must match. Got {trans_01.dim()} and {trans_12.dim()}")

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
        trans_12: transformation tensor of shape :math:`(N, 4, 4)` or :math:`(4, 4)`.

    Returns:
        tensor with inverted transformations with shape :math:`(N, 4, 4)` or :math:`(4, 4)`.

    Example:
        >>> trans_12 = torch.rand(1, 4, 4)  # Nx4x4
        >>> trans_21 = inverse_transformation(trans_12)  # Nx4x4
    """
    if not torch.is_tensor(trans_12):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(trans_12)}")
    if not ((trans_12.dim() in (2, 3)) and (trans_12.shape[-2:] == (4, 4))):
        raise ValueError(f"Input size must be a Nx4x4 or 4x4. Got {trans_12.shape}")
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


def relative_transformation(trans_01: torch.Tensor, trans_02: torch.Tensor) -> torch.Tensor:
    r"""Function that computes the relative homogeneous transformation from a
    reference transformation :math:`T_1^{0} = \begin{bmatrix} R_1 & t_1 \\
    \mathbf{0} & 1 \end{bmatrix}` to destination :math:`T_2^{0} =
    \begin{bmatrix} R_2 & t_2 \\ \mathbf{0} & 1 \end{bmatrix}`.

    The relative transformation is computed as follows:

    .. math::

        T_1^{2} = (T_0^{1})^{-1} \cdot T_0^{2}

    Args:
        trans_01: reference transformation tensor of shape :math:`(N, 4, 4)` or :math:`(4, 4)`.
        trans_02: destination transformation tensor of shape :math:`(N, 4, 4)` or :math:`(4, 4)`.

    Returns:
        the relative transformation between the transformations with shape :math:`(N, 4, 4)` or :math:`(4, 4)`.

    Example::
        >>> trans_01 = torch.eye(4)  # 4x4
        >>> trans_02 = torch.eye(4)  # 4x4
        >>> trans_12 = relative_transformation(trans_01, trans_02)  # 4x4
    """
    if not torch.is_tensor(trans_01):
        raise TypeError(f"Input trans_01 type is not a torch.Tensor. Got {type(trans_01)}")
    if not torch.is_tensor(trans_02):
        raise TypeError(f"Input trans_02 type is not a torch.Tensor. Got {type(trans_02)}")
    if not ((trans_01.dim() in (2, 3)) and (trans_01.shape[-2:] == (4, 4))):
        raise ValueError("Input must be a of the shape Nx4x4 or 4x4." " Got {}".format(trans_01.shape))
    if not ((trans_02.dim() in (2, 3)) and (trans_02.shape[-2:] == (4, 4))):
        raise ValueError("Input must be a of the shape Nx4x4 or 4x4." " Got {}".format(trans_02.shape))
    if not trans_01.dim() == trans_02.dim():
        raise ValueError(f"Input number of dims must match. Got {trans_01.dim()} and {trans_02.dim()}")
    trans_10: torch.Tensor = inverse_transformation(trans_01)
    trans_12: torch.Tensor = compose_transformations(trans_10, trans_02)
    return trans_12


def transform_points(trans_01: torch.Tensor, points_1: torch.Tensor) -> torch.Tensor:
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
        >>> points_0 = transform_points(trans_01, points_1)  # BxNx3
    """
    check_is_tensor(trans_01)
    check_is_tensor(points_1)
    if not trans_01.shape[0] == points_1.shape[0] and trans_01.shape[0] != 1:
        raise ValueError(
            "Input batch size must be the same for both tensors or 1." f"Got {trans_01.shape} and {points_1.shape}"
        )
    if not trans_01.shape[-1] == (points_1.shape[-1] + 1):
        raise ValueError("Last input dimensions must differ by one unit" f"Got{trans_01} and {points_1}")

    # We reshape to BxNxD in case we get more dimensions, e.g., MxBxNxD
    shape_inp = list(points_1.shape)
    points_1 = points_1.reshape(-1, points_1.shape[-2], points_1.shape[-1])
    trans_01 = trans_01.reshape(-1, trans_01.shape[-2], trans_01.shape[-1])
    # We expand trans_01 to match the dimensions needed for bmm
    trans_01 = torch.repeat_interleave(trans_01, repeats=points_1.shape[0] // trans_01.shape[0], dim=0)
    # to homogeneous
    points_1_h = convert_points_to_homogeneous(points_1)  # BxNxD+1
    # transform coordinates
    points_0_h = torch.bmm(points_1_h, trans_01.permute(0, 2, 1))
    points_0_h = torch.squeeze(points_0_h, dim=-1)
    # to euclidean
    points_0 = convert_points_from_homogeneous(points_0_h)  # BxNxD
    # reshape to the input shape
    shape_inp[-2] = points_0.shape[-2]
    shape_inp[-1] = points_0.shape[-1]
    points_0 = points_0.reshape(shape_inp)
    return points_0


def point_line_distance(point: Tensor, line: Tensor, eps: float = 1e-8) -> Tensor:
    r"""Return the distance from points to lines.

    Args:
       point: (possibly homogeneous) points :math:`(*, N, 2 or 3)`.
       line: lines coefficients :math:`(a, b, c)` with shape :math:`(*, N, 3)`, where :math:`ax + by + c = 0`.
       eps: Small constant for safe sqrt.

    Returns:
        the computed distance with shape :math:`(*, N)`.
    """
    KORNIA_CHECK_IS_TENSOR(point)
    KORNIA_CHECK_IS_TENSOR(line)

    if not point.shape[-1] in (2, 3):
        raise ValueError(f"pts must be a (*, 2 or 3) tensor. Got {point.shape}")

    if not line.shape[-1] == 3:
        raise ValueError(f"lines must be a (*, 3) tensor. Got {line.shape}")

    numerator = (line[..., 0] * point[..., 0] + line[..., 1] * point[..., 1] + line[..., 2]).abs()
    denominator = line[..., :2].norm(-1)

    return numerator / (denominator + eps)


# TODO:
# - project_points: from opencv
