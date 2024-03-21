"""Module containing the affine distortion model."""

# inspired by: https://github.com/farm-ng/sophus-rs/blob/main/src/sensor/affine.rs
import kornia.core as ops
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK_SHAPE


def distort_points_affine(projected_points_in_camera_z1_plane: Tensor, params: Tensor) -> Tensor:
    r"""Distort one or more points from the canonical z=1 plane into the camera frame.

    .. math::
        \begin{bmatrix} u \\ v \end{bmatrix} =
        \begin{bmatrix} f_x & 0 \\ 0 & f_y \end{bmatrix}
        \begin{bmatrix} x \\ y \end{bmatrix} +
        \begin{bmatrix} c_x \\ c_y \end{bmatrix}

    Args:
        projected_points_in_camera_z1_plane: Tensor representing the points to distort with shape (..., 2).
        params: Tensor representing the parameters of the affine distortion model with shape (..., 4).

    Returns:
        Tensor representing the distorted points with shape (..., 2).

    Example:
        >>> points = torch.tensor([319.5, 239.5])  # center of a 640x480 image
        >>> params = torch.tensor([600., 600., 319.5, 239.5])
        >>> distort_points_affine(points, params)
        tensor([192019.5000, 143939.5000])
    """
    KORNIA_CHECK_SHAPE(projected_points_in_camera_z1_plane, ["*", "2"])
    KORNIA_CHECK_SHAPE(params, ["*", "4"])

    x = projected_points_in_camera_z1_plane[..., 0]
    y = projected_points_in_camera_z1_plane[..., 1]

    fx, fy = params[..., 0], params[..., 1]
    cx, cy = params[..., 2], params[..., 3]

    u = fx * x + cx
    v = fy * y + cy

    return ops.stack([u, v], dim=-1)


def undistort_points_affine(distorted_points_in_camera: Tensor, params: Tensor) -> Tensor:
    r"""Undistort one or more points from the camera frame into the canonical z=1 plane.

    .. math::
        \begin{bmatrix} x \\ y \end{bmatrix} =
        \begin{bmatrix} u \\ v \end{bmatrix} -
        \begin{bmatrix} c_x \\ c_y \end{bmatrix}
        \begin{bmatrix} f_x & 0 \\ 0 & f_y \end{bmatrix}^{-1}

    Args:
        distorted_points_in_camera: Tensor representing the points to undistort with shape (..., 2).
        params: Tensor representing the parameters of the affine distortion model with shape (..., 4).

    Returns:
        Tensor representing the undistorted points with shape (..., 2).

    Example:
        >>> points = torch.tensor([319.5, 239.5])  # center of a 640x480 image
        >>> params = torch.tensor([600., 600., 319.5, 239.5])
        >>> undistort_points_affine(points, params)
        tensor([0., 0.])
    """
    KORNIA_CHECK_SHAPE(distorted_points_in_camera, ["*", "2"])
    KORNIA_CHECK_SHAPE(params, ["*", "4"])

    u = distorted_points_in_camera[..., 0]
    v = distorted_points_in_camera[..., 1]

    fx, fy = params[..., 0], params[..., 1]
    cx, cy = params[..., 2], params[..., 3]

    x = (u - cx) / fx
    y = (v - cy) / fy

    return ops.stack([x, y], dim=-1)


def dx_distort_points_affine(projected_points_in_camera_z1_plane: Tensor, params: Tensor) -> Tensor:
    r"""Compute the derivative of the x distortion with respect to the x coordinate.

    .. math::
        \frac{\partial u}{\partial x} =
        \begin{bmatrix} f_x & 0 \\ 0 & f_y \end{bmatrix}

    Args:
        projected_points_in_camera_z1_plane: Tensor representing the points to distort with shape (..., 2).
        params: Tensor representing the parameters of the affine distortion model with shape (..., 4).

    Returns:
        Tensor representing the derivative of the x distortion with respect to the x coordinate with shape (..., 2).

    Example:
        >>> points = torch.tensor([319.5, 239.5])  # center of a 640x480 image
        >>> params = torch.tensor([600., 600., 319.5, 239.5])
        >>> dx_distort_points_affine(points, params)
        tensor([[600.,   0.],
                [  0., 600.]])
    """
    KORNIA_CHECK_SHAPE(projected_points_in_camera_z1_plane, ["*", "2"])
    KORNIA_CHECK_SHAPE(params, ["*", "4"])

    fx, fy = params[..., 0], params[..., 1]

    zeros = ops.zeros_like(fx)

    return ops.stack([ops.stack([fx, zeros], dim=-1), ops.stack([zeros, fy], dim=-1)], dim=-2)
