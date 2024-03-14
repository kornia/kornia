"""Module containing functions for orthographic projection."""

# inspired by: https://github.com/farm-ng/sophus-rs/blob/main/src/sensor/ortho_camera.rs
import kornia.core as ops
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK_SHAPE


def project_points_orthographic(points_in_camera: Tensor) -> Tensor:
    r"""Project one or more points from the camera frame into the canonical z=1 plane through orthographic
    projection.

    .. math::
        \begin{bmatrix} u \\ v \end{bmatrix} =
        \begin{bmatrix} x \\ y \\ z \end{bmatrix}


    Args:
        points_in_camera: Tensor representing the points to project.

    Returns:
        Tensor representing the projected points.

    Example:
        >>> points = torch.tensor([1., 2., 3.])
        >>> project_points_orthographic(points)
        tensor([1., 2.])
    """
    KORNIA_CHECK_SHAPE(points_in_camera, ["*", "3"])
    return points_in_camera[..., :2]


def unproject_points_orthographic(points_in_camera: Tensor, extension: Tensor) -> Tensor:
    r"""Unproject one or more points from the canonical z=1 plane into the camera frame.

    .. math::
        \begin{bmatrix} x \\ y \\ z \end{bmatrix} =
        \begin{bmatrix} u \\ v \\ w \end{bmatrix}

    Args:
        points_in_camera: Tensor representing the points to unproject with shape (..., 2).
        extension: Tensor representing the extension of the points to unproject with shape (..., 1).

    Returns:
        Tensor representing the unprojected points with shape (..., 3).

    Example:
        >>> points = torch.tensor([1., 2.])
        >>> extension = torch.tensor([3.])
        >>> unproject_points_orthographic(points, extension)
        tensor([1., 2., 3.])
    """
    KORNIA_CHECK_SHAPE(points_in_camera, ["*", "2"])

    if len(points_in_camera.shape) != len(extension.shape):
        extension = extension[..., None]

    return ops.concatenate([points_in_camera, extension], dim=-1)


def dx_project_points_orthographic(points_in_camera: Tensor) -> Tensor:
    r"""Compute the derivative of the x projection with respect to the x coordinate.

    .. math::
        \frac{\partial u}{\partial x} = 1

    Args:
        points_in_camera: Tensor representing the points to project.

    Returns:
        Tensor representing the derivative of the x projection with respect to the x coordinate.

    Example:
        >>> points = torch.tensor([1., 2., 3.])
        >>> dx_project_points_orthographic(points)
        tensor([1.])
    """
    KORNIA_CHECK_SHAPE(points_in_camera, ["*", "3"])
    return ops.ones_like(points_in_camera[..., 0:1])
