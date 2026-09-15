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

"""nn.Module for the projection of points in the canonical z=1 plane."""

# inspired by: https://github.com/farm-ng/sophus-rs/blob/main/src/sensor/perspective_camera.rs
from __future__ import annotations

from typing import Optional

import torch

from kornia.core.check import KORNIA_CHECK_SHAPE


def project_points_z1(points_in_camera: torch.Tensor) -> torch.Tensor:
    r"""Project one or more points from the camera frame into the canonical z=1 plane through perspective division.

    .. math::

        \begin{bmatrix} u \\ v \\ w \end{bmatrix} =
        \begin{bmatrix} x \\ y \\ z \end{bmatrix} / z

    Convention:
        - the input is a **camera-frame** point and the output its position on the canonical ``z = 1`` plane,
          which is a normalized coordinate rather than a pixel. Applying a ``K`` to it with
          :func:`~kornia.geometry.conversions.denormalize_points_with_intrinsics` approximately gives the pixel
          that :func:`~kornia.geometry.camera.perspective.project_points` returns.
        - when ``abs(z) > 1e-8`` the first two components are divided by exactly ``z``. When
          ``abs(z) <= 1e-8`` they are returned unchanged, matching the homogeneous conversion used by
          :func:`~kornia.geometry.camera.perspective.project_points`. Points behind the camera keep the signed
          division and are not rejected.

    Args:
        points_in_camera: torch.Tensor representing the points to project with shape (..., 3).

    Returns:
        torch.Tensor representing the projected points with shape (..., 2).

    Example:
        >>> points = torch.tensor([1., 2., 3.])
        >>> project_points_z1(points)
        tensor([0.3333, 0.6667])

    """
    KORNIA_CHECK_SHAPE(points_in_camera, ["*", "3"])
    z = points_in_camera[..., 2:3]
    mask = torch.abs(z) > 1e-8
    safe_z = torch.where(mask, z, torch.ones_like(z))
    scale = torch.where(mask, 1.0 / safe_z, torch.ones_like(z))
    return scale * points_in_camera[..., :2]


def unproject_points_z1(
    points_in_cam_canonical: torch.Tensor, extension: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""Unproject one or more points from the canonical z=1 plane into the camera frame.

    .. math::
        \begin{bmatrix} x \\ y \\ z \end{bmatrix} =
        \begin{bmatrix} u \\ v \end{bmatrix} \cdot w

    Convention:
        - ``extension`` is the camera-frame ``z`` of the unprojected point: the canonical point is multiplied
          by it and it becomes the third component.
          :meth:`~kornia.sensors.camera.projection_model.Z1Projection.unproject` names the same argument
          ``depth``.
        - the guard compares the rank of ``extension`` with the rank of the points. Both ``(...,)`` and
          ``(..., 1)`` extensions are accepted when their leading dimensions match those of the points.

    Args:
        points_in_cam_canonical: torch.Tensor representing the points to unproject with shape (..., 2).
        extension: torch.Tensor representing the extension (depth) of the points to unproject with shape
            (..., 1) or (...), matching the points' leading dimensions. Defaults to unit depth.

    Returns:
        torch.Tensor representing the unprojected points with shape (..., 3).

    Example:
        >>> points = torch.tensor([1., 2.])
        >>> extension = torch.tensor([3.])
        >>> unproject_points_z1(points, extension)
        tensor([3., 6., 3.])

    """
    KORNIA_CHECK_SHAPE(points_in_cam_canonical, ["*", "2"])

    if extension is None:
        extension = torch.ones(
            points_in_cam_canonical.shape[:-1] + (1,),
            device=points_in_cam_canonical.device,
            dtype=points_in_cam_canonical.dtype,
        )  # (..., 1)
    elif extension.ndim == points_in_cam_canonical.ndim - 1:
        extension = extension[..., None]  # (..., 1)

    return torch.cat([points_in_cam_canonical * extension, extension], dim=-1)


def dx_project_points_z1(points_in_camera: torch.Tensor) -> torch.Tensor:
    r"""Compute the derivative of the x projection with respect to the x coordinate.

    Returns point derivative of inverse depth point projection with respect to the x coordinate.

    .. math::
        \frac{\partial \pi}{\partial x} =
        \begin{bmatrix}
            \frac{1}{z} & 0 & -\frac{x}{z^2} \\
            0 & \frac{1}{z} & -\frac{y}{z^2}
        \end{bmatrix}

    Convention:
        - the result is the full Jacobian of :func:`~kornia.geometry.camera.project_points_z1` with shape
          ``(..., 2, 3)``, laid out row-major in the output index: the ``u`` row then the ``v`` row, each
          holding the derivatives with respect to ``x``, ``y`` and ``z``. It agrees with
          :func:`torch.autograd.functional.jacobian`.
        - it uses the same strict ``abs(z) > 1e-8`` branch as the projection. The masked branch is the identity
          on ``(x, y)``, so its Jacobian is ``[[1, 0, 0], [0, 1, 0]]``.

    Args:
        points_in_camera: torch.Tensor representing the points to project with shape (..., 3).

    Returns:
        torch.Tensor representing the derivative of the x projection with respect to the x coordinate
        with shape (..., 2, 3).

    Example:
        >>> points = torch.tensor([1., 2., 3.])
        >>> dx_project_points_z1(points)
        tensor([[ 0.3333,  0.0000, -0.1111],
                [ 0.0000,  0.3333, -0.2222]])

    """
    KORNIA_CHECK_SHAPE(points_in_camera, ["*", "3"])

    x = points_in_camera[..., 0]
    y = points_in_camera[..., 1]
    z = points_in_camera[..., 2]

    mask = torch.abs(z) > 1e-8
    safe_z = torch.where(mask, z, torch.ones_like(z))
    z_inv = 1.0 / safe_z
    z_sq = z_inv * z_inv
    zeros = torch.zeros_like(z_inv)
    perspective = torch.stack(
        [
            torch.stack([z_inv, zeros, -x * z_sq], dim=-1),
            torch.stack([zeros, z_inv, -y * z_sq], dim=-1),
        ],
        dim=-2,
    )
    ones = torch.ones_like(z_inv)
    passthrough = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
        ],
        dim=-2,
    )
    return torch.where(mask[..., None, None], perspective, passthrough)
