"""Module with the functionalities for triangulation."""

import torch

from kornia.core import zeros
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.geometry.conversions import convert_points_from_homogeneous
from kornia.utils.helpers import _torch_svd_cast

# https://github.com/opencv/opencv_contrib/blob/master/modules/sfm/src/triangulation.cpp#L68


def triangulate_points(
    P1: torch.Tensor, P2: torch.Tensor, points1: torch.Tensor, points2: torch.Tensor
) -> torch.Tensor:
    r"""Reconstructs a bunch of points by triangulation.

    Triangulates the 3d position of 2d correspondences between several images.
    Reference: Internally it uses DLT method from Hartley/Zisserman 12.2 pag.312

    The input points are assumed to be in homogeneous coordinate system and being inliers
    correspondences. The method does not perform any robust estimation.

    Args:
        P1: The projection matrix for the first camera with shape :math:`(*, 3, 4)`.
        P2: The projection matrix for the second camera with shape :math:`(*, 3, 4)`.
        points1: The set of points seen from the first camera frame in the camera plane
          coordinates with shape :math:`(*, N, 2)`.
        points2: The set of points seen from the second camera frame in the camera plane
          coordinates with shape :math:`(*, N, 2)`.

    Returns:
        The reconstructed 3d points in the world frame with shape :math:`(*, N, 3)`.
    """
    KORNIA_CHECK_SHAPE(P1, ["*", "3", "4"])
    KORNIA_CHECK_SHAPE(P2, ["*", "3", "4"])
    KORNIA_CHECK_SHAPE(points1, ["*", "N", "2"])
    KORNIA_CHECK_SHAPE(points2, ["*", "N", "2"])

    # allocate and construct the equations matrix with shape (*, 4, 4)
    points_shape = max(points1.shape, points2.shape)  # this allows broadcasting
    X = zeros(points_shape[:-1] + (4, 4)).type_as(points1)

    for i in range(4):
        X[..., 0, i] = points1[..., 0] * P1[..., 2:3, i] - P1[..., 0:1, i]
        X[..., 1, i] = points1[..., 1] * P1[..., 2:3, i] - P1[..., 1:2, i]
        X[..., 2, i] = points2[..., 0] * P2[..., 2:3, i] - P2[..., 0:1, i]
        X[..., 3, i] = points2[..., 1] * P2[..., 2:3, i] - P2[..., 1:2, i]

    # 1. Solve the system Ax=0 with smallest eigenvalue
    # 2. Return homogeneous coordinates

    _, _, V = _torch_svd_cast(X)

    points3d_h = V[..., -1]
    points3d: torch.Tensor = convert_points_from_homogeneous(points3d_h)
    return points3d
