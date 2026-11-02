# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""The affine camera model: flat [fx, fy, cx, cy] intrinsics between normalized z = 1 points and pixels."""

# inspired by: https://github.com/farm-ng/sophus-rs/blob/main/src/sensor/affine.rs
import torch

from kornia.core.check import KORNIA_CHECK_SHAPE


def distort_points_affine(projected_points_in_camera_z1_plane: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    r"""Apply flat [fx, fy, cx, cy] intrinsics to normalized z = 1 points, giving pixels.

    ``uv = distort_points_affine(xy, params)`` takes points on the normalized ``z = 1`` plane ``(..., 2)`` and the
    vector ``params = [fx, fy, cx, cy]``, ``(..., 4)``, and returns the pixels ``(fx x + cx, fy y + cy)`` as
    ``(..., 2)``. Despite its name it adds no lens distortion: it is the intrinsics step of a camera model, the
    one :func:`distort_points_kannala_brandt` applies after its fisheye distortion. For a ``(*, 3, 3)`` ``K``, use
    :func:`~kornia.geometry.conversions.denormalize_points_with_intrinsics` instead.

    .. math::
        \begin{bmatrix} u \\ v \end{bmatrix} =
        \begin{bmatrix} f_x & 0 \\ 0 & f_y \end{bmatrix}
        \begin{bmatrix} x \\ y \end{bmatrix} +
        \begin{bmatrix} c_x \\ c_y \end{bmatrix}

    Convention:
        - ``projected_points_in_camera_z1_plane`` is a point on the **normalized** :math:`z = 1` plane, not a
          pixel, and ``params`` is the flat vector ``[fx, fy, cx, cy]``; the result is in pixels (integer pixel
          centres; see :class:`~kornia.geometry.camera.pinhole.PinholeCamera`).
        - :func:`~kornia.geometry.calibration.distort_points` is the same camera parametrized by a
          ``(*, 3, 3)`` ``K`` and an OpenCV coefficient vector; it takes a **pixel** input instead.
        - :func:`undistort_points_affine` is the closed-form inverse.
        - the leading dimensions ``...`` of the points and of ``params`` broadcast by torch's rules: one ``(4,)``
          vector serves every point, and per-camera ``params`` for ``(B, N, 2)`` points are ``(B, 1, 4)``, since
          a ``(B, 4)`` vector lines up with the ``N`` axis.

    Args:
        projected_points_in_camera_z1_plane: torch.Tensor representing the points to distort with shape
            ``(..., 2)``.
        params: torch.Tensor representing the parameters of the affine distortion model with shape ``(..., 4)``.

    Returns:
        torch.Tensor representing the distorted points with shape ``(..., 2)``.

    See Also:
        - :func:`undistort_points_affine`: the closed-form inverse, pixels to normalized points.
        - :func:`dx_distort_points_affine`: its ``(..., 2, 2)`` Jacobian with respect to the point.
        - :func:`distort_points_kannala_brandt`: the same intrinsics after a fisheye distortion of the ray angle.
        - :func:`~kornia.geometry.calibration.distort_points`: OpenCV's Brown-Conrady distortion with a ``3x3``
          ``K``, from pixels to pixels.
        - :func:`~kornia.geometry.conversions.denormalize_points_with_intrinsics`: the same map from a
          ``(*, 3, 3)`` ``K``.

    Example:
        >>> points = torch.tensor([0.1, -0.2])  # on the normalized z = 1 plane
        >>> params = torch.tensor([600., 600., 319.5, 239.5])
        >>> distort_points_affine(points, params)
        tensor([379.5000, 119.5000])

    """
    KORNIA_CHECK_SHAPE(projected_points_in_camera_z1_plane, ["*", "2"])
    KORNIA_CHECK_SHAPE(params, ["*", "4"])

    x = projected_points_in_camera_z1_plane[..., 0]
    y = projected_points_in_camera_z1_plane[..., 1]

    fx, fy = params[..., 0], params[..., 1]
    cx, cy = params[..., 2], params[..., 3]

    u = fx * x + cx
    v = fy * y + cy

    return torch.stack([u, v], dim=-1)


def undistort_points_affine(distorted_points_in_camera: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    r"""Map pixels back to normalized z = 1 points with flat [fx, fy, cx, cy] intrinsics.

    ``xy = undistort_points_affine(uv, params)`` takes pixels ``(..., 2)`` and the ``(..., 4)`` vector
    ``[fx, fy, cx, cy]`` and returns ``((u - cx) / fx, (v - cy) / fy)`` as ``(..., 2)``, on the normalized
    ``z = 1`` plane; it removes no lens distortion. For a ``(*, 3, 3)`` ``K``, use
    :func:`~kornia.geometry.conversions.normalize_points_with_intrinsics` instead.

    .. math::
        \begin{bmatrix} x \\ y \end{bmatrix} =
        \begin{bmatrix} f_x & 0 \\ 0 & f_y \end{bmatrix}^{-1}
        \left( \begin{bmatrix} u \\ v \end{bmatrix} -
        \begin{bmatrix} c_x \\ c_y \end{bmatrix} \right)

    Convention:
        - ``distorted_points_in_camera`` is a **pixel** coordinate and the result is a point on the normalized
          :math:`z = 1` plane; ``params`` is the same ``[fx, fy, cx, cy]`` vector that
          :func:`distort_points_affine` takes, and this function is its closed-form inverse.
        - ``params`` values are not validated (a design choice shared with the other intrinsics consumers): a
          zero ``fx`` or ``fy`` gives ``inf`` or ``nan`` on that axis.
        - the leading dimensions of the points and of ``params`` broadcast as in :func:`distort_points_affine`.

    Args:
        distorted_points_in_camera: torch.Tensor representing the points to undistort with shape ``(..., 2)``.
        params: torch.Tensor representing the parameters of the affine distortion model with shape ``(..., 4)``.

    Returns:
        torch.Tensor representing the undistorted points with shape ``(..., 2)``.

    See Also:
        - :func:`distort_points_affine`: the forward map, normalized points to pixels.
        - :func:`dx_distort_points_affine`: the Jacobian of the forward map.
        - :func:`undistort_points_kannala_brandt`: the inverse of the fisheye model, found iteratively.
        - :func:`~kornia.geometry.calibration.undistort_points`: removes OpenCV's Brown-Conrady distortion, from
          pixels to pixels.
        - :func:`~kornia.geometry.conversions.normalize_points_with_intrinsics`: the same map from a
          ``(*, 3, 3)`` ``K``.

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

    return torch.stack([x, y], dim=-1)


def dx_distort_points_affine(projected_points_in_camera_z1_plane: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    r"""Compute the 2x2 Jacobian of the affine [fx, fy, cx, cy] model with respect to the point.

    ``J = dx_distort_points_affine(xy, params)`` takes normalized points ``(..., 2)`` and the ``(..., 4)`` vector
    ``[fx, fy, cx, cy]`` and returns the Jacobian of :func:`distort_points_affine` at each point, ``diag(fx, fy)``,
    as ``(..., 2, 2)``. Chain it after :func:`dx_project_points_z1` for the Jacobian of a pinhole camera model.

    .. math::
        \frac{\partial (u, v)}{\partial (x, y)} =
        \begin{bmatrix} f_x & 0 \\ 0 & f_y \end{bmatrix}

    Convention:
        - the result is the ``(..., 2, 2)`` Jacobian of :func:`distort_points_affine` with respect to the
          **point**: rows are the output components ``(u, v)``, columns the input components ``(x, y)``.
        - the map is affine, so that Jacobian is ``diag(fx, fy)`` and does not depend on the point.
        - ``...`` is the broadcast of the points' and the params' leading shapes, as in
          :func:`distort_points_affine`: ``N`` points with one set of params give ``N`` Jacobians.

    Args:
        projected_points_in_camera_z1_plane: torch.Tensor representing the points to distort with shape
            ``(..., 2)``.
        params: torch.Tensor representing the parameters of the affine distortion model with shape ``(..., 4)``.

    Returns:
        torch.Tensor representing the Jacobian of the distortion with respect to the point
        with shape ``(..., 2, 2)``.

    See Also:
        - :func:`distort_points_affine`: the map this differentiates.
        - :func:`undistort_points_affine`: the inverse map; its Jacobian is ``diag(1 / fx, 1 / fy)``.
        - :func:`dx_distort_points_kannala_brandt`: the Jacobian of the fisheye model, equal to this one at the
          optical axis.
        - :func:`dx_project_points_z1`: the Jacobian of the projection step before it.

    Example:
        >>> points = torch.tensor([0.1, -0.2])  # on the normalized z = 1 plane
        >>> params = torch.tensor([600., 600., 319.5, 239.5])
        >>> dx_distort_points_affine(points, params)
        tensor([[600.,   0.],
                [  0., 600.]])

    """
    KORNIA_CHECK_SHAPE(projected_points_in_camera_z1_plane, ["*", "2"])
    KORNIA_CHECK_SHAPE(params, ["*", "4"])

    # Broadcast against the points as well, so a batch of points with shared params gets one Jacobian per point.
    fx, fy, _ = torch.broadcast_tensors(params[..., 0], params[..., 1], projected_points_in_camera_z1_plane[..., 0])

    zeros = torch.zeros_like(fx)

    return torch.stack([torch.stack([fx, zeros], dim=-1), torch.stack([zeros, fy], dim=-1)], dim=-2)
