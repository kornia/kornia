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

"""Project camera-frame points and unproject pixels with the Double Sphere model."""

import torch

from kornia.core.check import KORNIA_CHECK_SHAPE


def project_points_double_sphere(points: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    r"""Project 3D camera-frame points to pixels with the Double Sphere model.

    ``pixels = project_points_double_sphere(points, params)`` takes points ``(..., 3)`` in the camera frame and
    ``params = [fx, fy, cx, cy, xi, alpha]`` with shape ``(..., 6)``, and maps the valid points to pixels ``(..., 2)``.
    Unlike a pinhole distortion, this is a complete 3D projection: it can represent valid rays with ``z <= 0``.

    Convention:
        - Camera coordinates are ``+X`` right, ``+Y`` down and ``+Z`` forward. Pixels are ``(u, v)`` in pixel units,
          with integer pixel centres.
        - The projection follows Eqs. 40--45 of Usenko et al. (2018). It is defined on the model's valid domain
          ``z > -w2 * ||points||``; points outside that domain have no projection. This function does not mask or
          validate that domain.
        - Leading dimensions of ``points`` and ``params`` follow PyTorch broadcasting rules. For per-camera
          parameters and a point cloud, use ``params`` shaped ``(B, 1, 6)`` with ``points`` shaped ``(B, N, 3)``.

    Args:
        points: Camera-frame 3D points with shape ``(..., 3)``.
        params: Camera parameters ``[fx, fy, cx, cy, xi, alpha]`` with shape ``(..., 6)``.

    Returns:
        Pixels with shape ``(..., 2)``.

    See Also:
        - :func:`unproject_points_double_sphere`: unproject pixels to unit-length camera-frame rays.
        - :class:`~kornia.sensors.camera.DoubleSphereModel`: the sensor-model wrapper for the same mapping.

    Example:
        >>> points = torch.tensor([[0.3, -0.2, 2.0]])
        >>> params = torch.tensor([300.0, 320.0, 160.0, 120.0, 0.5, 0.4])
        >>> project_points_double_sphere(points, params).round(decimals=3)
        tensor([[189.7550,  98.8410]])

    """
    KORNIA_CHECK_SHAPE(points, ["*", "3"])
    KORNIA_CHECK_SHAPE(params, ["*", "6"])

    x, y, z = points.unbind(dim=-1)
    fx, fy, cx, cy, xi, alpha = params.unbind(dim=-1)

    d1 = torch.sqrt(x**2 + y**2 + z**2)
    xi_d1_plus_z = xi * d1 + z
    d2 = torch.sqrt(x**2 + y**2 + xi_d1_plus_z**2)
    denominator = alpha * d2 + (1.0 - alpha) * xi_d1_plus_z

    u = fx * x / denominator + cx
    v = fy * y / denominator + cy
    return torch.stack((u, v), dim=-1)


def unproject_points_double_sphere(pixels: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    r"""Unproject Double Sphere pixels to unit-length camera-frame rays.

    ``rays = unproject_points_double_sphere(pixels, params)`` takes pixels ``(..., 2)`` and the matching
    ``params = [fx, fy, cx, cy, xi, alpha]`` with shape ``(..., 6)``, and returns unit-length bearing vectors
    ``(..., 3)``. The inverse is closed form and does not require iteration.

    Convention:
        - Camera coordinates are ``+X`` right, ``+Y`` down and ``+Z`` forward. A returned vector is a bearing, not a
          point at a camera-frame ``z`` depth.
        - The inverse follows Eqs. 46--51 of Usenko et al. (2018). For ``alpha <= 0.5`` all finite pixels are in the
          algebraic domain; for ``alpha > 0.5`` the normalized squared radius must be at most ``1 / (2 * alpha - 1)``.
          This function does not mask or validate that domain.
        - Leading dimensions of ``pixels`` and ``params`` follow PyTorch broadcasting rules. For per-camera
          parameters and a pixel grid, use ``params`` shaped ``(B, 1, 6)`` with ``pixels`` shaped ``(B, N, 2)``.

    Args:
        pixels: Double Sphere pixel coordinates with shape ``(..., 2)``.
        params: Camera parameters ``[fx, fy, cx, cy, xi, alpha]`` with shape ``(..., 6)``.

    Returns:
        Unit-length camera-frame bearing vectors with shape ``(..., 3)``.

    See Also:
        - :func:`project_points_double_sphere`: project valid camera-frame points to pixels.
        - :class:`~kornia.sensors.camera.DoubleSphereModel`: the sensor-model wrapper, including range scaling.

    Example:
        >>> pixels = torch.tensor([[189.755, 98.841]])
        >>> params = torch.tensor([300.0, 320.0, 160.0, 120.0, 0.5, 0.4])
        >>> unproject_points_double_sphere(pixels, params).round(decimals=3)
        tensor([[ 0.1480, -0.0980,  0.9840]])

    """
    KORNIA_CHECK_SHAPE(pixels, ["*", "2"])
    KORNIA_CHECK_SHAPE(params, ["*", "6"])

    fx, fy, cx, cy, xi, alpha = params.unbind(dim=-1)
    mx = (pixels[..., 0] - cx) / fx
    my = (pixels[..., 1] - cy) / fy
    radius_sq = mx**2 + my**2

    mz = (1.0 - alpha**2 * radius_sq) / (alpha * torch.sqrt(1.0 - (2.0 * alpha - 1.0) * radius_sq) + 1.0 - alpha)
    scale = (mz * xi + torch.sqrt(mz**2 + (1.0 - xi**2) * radius_sq)) / (mz**2 + radius_sq)

    ray_x = scale * mx
    ray_y = scale * my
    ray_z = scale * mz - xi
    return torch.stack((ray_x, ray_y, ray_z), dim=-1)
