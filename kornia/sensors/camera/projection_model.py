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

from __future__ import annotations

import torch

from kornia.geometry.camera.projection_orthographic import (
    project_points_orthographic,
    unproject_points_orthographic,
)
from kornia.geometry.vector import Vector2, Vector3


class Z1Projection:
    """Project ``Vector3`` camera-frame points onto the normalized ``z = 1`` plane by dividing by ``z``, and back.

    ``xy = Z1Projection().project(points)`` takes a :class:`~kornia.geometry.vector.Vector3` of camera-frame points and
    returns the :class:`~kornia.geometry.vector.Vector2` ``(x / z, y / z)``, not pixels; ``unproject(xy, depth)``
    returns ``(x * depth, y * depth, depth)``. It is the projection of the pinhole, Brown-Conrady and Kannala-Brandt
    models.

    See Also:
        - :class:`OrthographicProjection`: drops ``z`` instead of dividing by it.
        - :func:`~kornia.geometry.camera.project_points_z1`: the same projection on plain tensors.
        - :func:`~kornia.geometry.camera.unproject_points_z1`: the same unprojection on plain tensors.

    """

    def project(self, points: Vector3) -> Vector2:
        """Project one or more Vector3 from the camera frame into the canonical z=1 plane through perspective division.

        Convention:
            - ``points`` is in the **camera frame** and the result is on the normalized :math:`z = 1` plane,
              not in pixels: the map is ``xy / z``, with no epsilon and no validation.

        .. warning::
            A point with :math:`z = 0` projects to an infinity (``nan`` for a zero numerator) instead of raising:
            `#4267 <https://github.com/kornia/kornia/issues/4267>`_. A point behind the camera projects to a
            finite coordinate and is not flagged: `#4555 <https://github.com/kornia/kornia/issues/4555>`_.

        Args:
            points: Vector3 representing the points to project.

        Returns:
            Vector2 representing the projected points.

        Example:
            >>> points = Vector3.from_coords(1., 2., 3.)
            >>> Z1Projection().project(points)
            x: 0.3333333432674408
            y: 0.6666666865348816

        """
        xy = points.data[..., :2]
        z = points.z
        # For scalar z, xy is 1-D, so no unsqueeze is needed
        uv = xy / z.unsqueeze(-1) if len(z.shape) else xy * 1 / z
        return Vector2(uv)

    def unproject(self, points: Vector2, depth: torch.Tensor | float) -> Vector3:
        """Unproject one or more Vector2 from the canonical z=1 plane into the camera frame.

        Convention:
            - ``depth`` is the camera-frame ``z``: the :math:`z = 1` point is multiplied by it, so the third
              coordinate of the result is the ``depth`` that was passed in, and not a Euclidean ray length.
            - a python ``float`` or ``int`` ``depth`` is converted to the device and dtype of ``points``.

        Args:
            points: Vector2 representing the points to unproject.
            depth: a :class:`torch.Tensor` of shape ``(B,)``, or a python scalar for a single point.

        Returns:
            Vector3 representing the unprojected points.

        Example:
            >>> points = Vector2.from_coords(1., 2.)
            >>> Z1Projection().unproject(points, 3)
            x: tensor([3.])
            y: tensor([6.])
            z: tensor([3.])

        """
        if isinstance(depth, (float, int)):
            depth = torch.as_tensor([depth], device=points.data.device, dtype=points.data.dtype)
        return Vector3.from_coords(points.x * depth, points.y * depth, depth)


class OrthographicProjection:
    """Project ``Vector3`` camera-frame points orthographically by dropping ``z``, and back by appending a depth.

    ``xy = OrthographicProjection().project(points)`` takes a :class:`~kornia.geometry.vector.Vector3` of camera-frame
    points and returns the :class:`~kornia.geometry.vector.Vector2` ``(x, y)``; ``unproject(xy, depth)`` returns
    ``(x, y, depth)``. It is the projection of :class:`~kornia.sensors.camera.Orthographic`.

    See Also:
        - :class:`Z1Projection`: divides by ``z`` instead of dropping it, so distant points move to the center.
        - :func:`~kornia.geometry.camera.project_points_orthographic`: the same projection on plain tensors.
        - :func:`~kornia.geometry.camera.unproject_points_orthographic`: the same unprojection on plain tensors.

    """

    def project(self, points: Vector3) -> Vector2:
        """Project camera-frame points orthographically, keeping ``x`` and ``y`` and dropping ``z``.

        Args:
            points: ``Vector3`` of camera-frame points, with any leading dimensions.

        Returns:
            ``Vector2`` of the ``x`` and ``y`` coordinates.

        """
        return Vector2(project_points_orthographic(points.data))

    def unproject(self, points: Vector2, depth: torch.Tensor) -> Vector3:
        """Unproject orthographic image-plane points to camera-frame points at the given depths.

        Args:
            points: ``Vector2`` of image-plane points ``(x, y)``.
            depth: tensor of the ``z`` coordinate of each point, ``(*,)`` or ``(*, 1)``.

        Returns:
            ``Vector3`` with ``x`` and ``y`` copied from ``points`` and ``z`` from ``depth``.

        """
        return Vector3(unproject_points_orthographic(points.data, depth))
