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

# inspired by: shttps://github.com/farm-ng/sophus-rs/blob/main/src/sensor/kannala_brandt.rs

import torch

from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.geometry.camera.distortion_affine import distort_points_affine


def _distort_points_kannala_brandt_impl(
    projected_points_in_camera_z1_plane: torch.Tensor,
    params: torch.Tensor,
    radius_sq: torch.Tensor,
) -> torch.Tensor:
    # https://github.com/farm-ng/sophus-rs/blob/20f6cac68f17fe1ac41d0aa8a27489e2b886806f/
    # src/sensor/kannala_brandt.rs#L51-L67
    x = projected_points_in_camera_z1_plane[..., 0]
    y = projected_points_in_camera_z1_plane[..., 1]

    fx, fy = params[..., 0], params[..., 1]
    cx, cy = params[..., 2], params[..., 3]

    k0 = params[..., 4]
    k1 = params[..., 5]
    k2 = params[..., 6]
    k3 = params[..., 7]

    radius = radius_sq.sqrt()
    radius_inverse = 1.0 / radius
    theta = radius.atan2(torch.ones_like(radius))
    theta2 = theta**2
    theta4 = theta2**2
    theta6 = theta2 * theta4
    theta8 = theta4**2

    r_distorted = theta * (1.0 + k0 * theta2 + k1 * theta4 + k2 * theta6 + k3 * theta8)

    scaling = r_distorted * radius_inverse

    u = fx * scaling * x + cx
    v = fy * scaling * y + cy

    return torch.stack([u, v], dim=-1)


def distort_points_kannala_brandt(
    projected_points_in_camera_z1_plane: torch.Tensor, params: torch.Tensor
) -> torch.Tensor:
    r"""Distort points from the canonical z=1 plane into the camera frame using the Kannala-Brandt model.

    Args:
        projected_points_in_camera_z1_plane: torch.Tensor representing the points to distort with shape (..., 2).
        params: torch.Tensor representing the parameters of the Kannala-Brandt distortion model with shape (..., 8).

    Returns:
        torch.Tensor representing the distorted points with shape (..., 2).

    Example:
        >>> points = torch.tensor([319.5, 239.5])  # center of a 640x480 image
        >>> params = torch.tensor([1000.0, 1000.0, 320.0, 280.0, 0.1, 0.01, 0.001, 0.0001])
        >>> distort_points_kannala_brandt(points, params)
        tensor([1982.6832, 1526.3619])

    """
    KORNIA_CHECK_SHAPE(projected_points_in_camera_z1_plane, ["*", "2"])
    KORNIA_CHECK_SHAPE(params, ["*", "8"])

    x = projected_points_in_camera_z1_plane[..., 0]
    y = projected_points_in_camera_z1_plane[..., 1]

    radius_sq = x**2 + y**2

    distorted_points = torch.where(
        radius_sq[..., None] > 1e-8,
        _distort_points_kannala_brandt_impl(
            projected_points_in_camera_z1_plane,
            params,
            radius_sq,
        ),
        distort_points_affine(projected_points_in_camera_z1_plane, params[..., :4]),
    )

    return distorted_points


def undistort_points_kannala_brandt(distorted_points_in_camera: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    r"""Undistort points from the camera frame into the canonical z=1 plane using the Kannala-Brandt model.

    Args:
        distorted_points_in_camera: torch.Tensor representing the points to undistort with shape (..., 2).
        params: torch.Tensor representing the parameters of the Kannala-Brandt distortion model with shape (..., 8).

    Returns:
        torch.Tensor representing the undistorted points with shape (..., 2).

    Example:
        >>> points = torch.tensor([319.5, 239.5])  # center of a 640x480 image
        >>> params = torch.tensor([1000.0, 1000.0, 320.0, 280.0, 0.1, 0.01, 0.001, 0.0001])
        >>> undistort_points_kannala_brandt(points, params).shape
        torch.Size([2])

    """
    KORNIA_CHECK_SHAPE(distorted_points_in_camera, ["*", "2"])
    KORNIA_CHECK_SHAPE(params, ["*", "8"])

    iters = 10
    eps = 1e-8
    device = distorted_points_in_camera.device
    out_dtype = distorted_points_in_camera.dtype

    pts = distorted_points_in_camera.to(device=device, dtype=params.dtype)
    p = params.to(device=device, dtype=params.dtype)

    x = pts[..., 0]
    y = pts[..., 1]

    fx = p[..., 0]
    fy = p[..., 1]
    cx = p[..., 2]
    cy = p[..., 3]
    k0 = p[..., 4]
    k1 = p[..., 5]
    k2 = p[..., 6]
    k3 = p[..., 7]

    un = (x - cx) / fx
    vn = (y - cy) / fy

    rth2 = un * un + vn * vn
    rth = rth2.sqrt()

    th = rth.clamp(min=1e-16).sqrt()

    # gauss-newton
    for _ in range(iters):
        th2 = th * th
        inner = k0 + th2 * (k1 + th2 * (k2 + th2 * k3))
        thd = th * (1.0 + th2 * inner)
        d_thd = 1.0 + th2 * (3.0 * k0 + th2 * (5.0 * k1 + th2 * (7.0 * k2 + 9.0 * k3 * th2)))
        step = (thd - rth) / (d_thd + 1e-12)
        th = th - step

    radius_undistorted = th.tan()
    denom = rth + eps
    mag = radius_undistorted.abs() / denom
    undistorted = torch.stack([mag * un, mag * vn], dim=-1)

    return undistorted.to(device=device, dtype=out_dtype)


def dx_distort_points_kannala_brandt(
    projected_points_in_camera_z1_plane: torch.Tensor, params: torch.Tensor
) -> torch.Tensor:
    r"""Compute the derivative of the x distortion with respect to the x coordinate.

    .. math::
        \frac{\partial u}{\partial x} =
        \begin{bmatrix} f_x & 0 \\ 0 & f_y \end{bmatrix}

    Args:
        projected_points_in_camera_z1_plane: torch.Tensor representing the points to distort with shape (..., 2).
        params: torch.Tensor representing the parameters of the Kannala-Brandt distortion model with shape (..., 8).

    Returns:
        torch.Tensor representing the derivative of the x distortion with respect to the x coordinate
        with shape (..., 2).

    Example:
        >>> points = torch.tensor([1., 2.])
        >>> params = torch.tensor([1000.0, 1000.0, 320.0, 280.0, 0.1, 0.01, 0.001, 0.0001])
        >>> dx_distort_points_kannala_brandt(points, params)
        tensor([[ 486.0507, -213.5573],
                [-213.5573,  165.7147]])

    """
    KORNIA_CHECK_SHAPE(projected_points_in_camera_z1_plane, ["*", "2"])
    KORNIA_CHECK_SHAPE(params, ["*", "8"])

    a = projected_points_in_camera_z1_plane[..., 0]
    b = projected_points_in_camera_z1_plane[..., 1]

    fx, fy = params[..., 0], params[..., 1]

    k0 = params[..., 4]
    k1 = params[..., 5]
    k2 = params[..., 6]
    k3 = params[..., 7]

    # TODO: return identity matrix if a and b are zero
    # radius_sq = a ** 2 + b ** 2

    c0 = a.pow(2.0)
    c1 = b.pow(2.0)
    c2 = c0 + c1
    c3 = c2.pow(5.0 / 2.0)
    c4 = c2 + 1.0
    c5 = c2.sqrt().atan()
    c6 = c5.pow(2.0)
    c7 = c6 * k0
    c8 = c5.pow(4.0)
    c9 = c8 * k1
    c10 = c5.pow(6.0)
    c11 = c10 * k2
    c12 = c5.pow(8.0) * k3
    c13 = 1.0 * c4 * c5 * (c11 + c12 + c7 + c9 + 1.0)
    c14 = c13 * c3
    c15 = c2.pow(3.0 / 2.0)
    c16 = c13 * c15
    c17 = 1.0 * c11 + 1.0 * c12 + 2.0 * c6 * (4.0 * c10 * k3 + 2.0 * c6 * k1 + 3.0 * c8 * k2 + k0)
    c18 = c17 * c2.pow(2.0)
    c19 = 1.0 / c4
    c20 = c19 / c2.pow(3.0)
    c21 = a * b * c19 * (-c13 * c2 + c15 * c17) / c3

    return torch.stack(
        [
            torch.stack([c20 * fx * (-c0 * c16 + c0 * c18 + c14), c21 * fx], dim=-1),
            torch.stack([c21 * fy, c20 * fy * (-c1 * c16 + c1 * c18 + c14)], dim=-1),
        ],
        dim=-2,
    )
