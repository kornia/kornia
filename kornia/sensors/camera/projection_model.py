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

from kornia.geometry.vector import Vector2, Vector3


class Z1Projection:
    """Project 3D points from the camera frame into the canonical $z=1$ plane.

    This performs perspective division by dividing the $x$ and $y$ coordinates
    by the depth $z$.
    """

    def project(self, points: Vector3) -> Vector2:
        """Project one or more Vector3 from the camera frame into the canonical z=1 plane through perspective division.

        Convention:
            - ``points`` is in the **camera frame** and the result is on the normalized :math:`z = 1` plane,
              not in pixels: the map is ``xy / z``, with no epsilon and no validation. A point on the camera
              plane (:math:`z = 0`) therefore projects to an infinity instead of raising -- to ``nan`` on an
              axis whose numerator is zero as well -- and a point behind the camera to a finite coordinate.

        .. warning::
            That :math:`z = 0` answer is one of several that the projection entry points of kornia give for
            the same input; they are collected in `#4267 <https://github.com/kornia/kornia/issues/4267>`_.
            The behaviour above is documented as it is.

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
        if len(z.shape):
            uv = xy / z.unsqueeze(-1)
        else:
            # For scalar z, xy is 1-D, so no transpose needed
            uv = xy * 1 / z
        return Vector2(uv)

    def unproject(self, points: Vector2, depth: torch.Tensor | float) -> Vector3:
        """Unproject one or more Vector2 from the canonical z=1 plane into the camera frame.

        Convention:
            - ``depth`` is the camera-frame ``z``: the :math:`z = 1` point is multiplied by it, so the third
              coordinate of the result is the ``depth`` that was passed in, and not a Euclidean ray length.
            - a python ``float`` or ``int`` ``depth`` is promoted to a one-element tensor on the device and in
              the dtype of ``points``, so it gives the same result as the tensor spelling of the same value.

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
    """Project 3D points using an orthographic projection model.

    This model assumes parallel projection where the $z$ coordinate is
    discarded and no perspective scaling is applied.

    .. warning::
        Both methods are placeholders: :meth:`project` and :meth:`unproject` raise ``NotImplementedError``
        with an empty message, which is what makes :class:`~kornia.sensors.camera.Orthographic` unusable in
        either direction. Tracked in `#4284 <https://github.com/kornia/kornia/issues/4284>`_.
        :func:`~kornia.geometry.camera.project_points_orthographic` is the implemented equivalent.
    """

    def project(self, points: Vector3) -> Vector2:
        """Project 3D camera-frame points with an orthographic camera model.

        Orthographic projection keeps the horizontal and vertical coordinates
        unchanged and discards depth. Unlike perspective projection, objects do
        not shrink as their ``z`` value increases.

        Args:
            points: Three-dimensional point container with coordinates
                ``x``, ``y``, and ``z``. Leading dimensions may represent a
                batch of points.

        Returns:
            Two-dimensional point container containing the projected ``x`` and
            ``y`` coordinates.

        Raises:
            NotImplementedError: This projection model is declared as an
                interface placeholder and is not implemented yet.
        """
        raise NotImplementedError

    def unproject(self, points: Vector2, depth: torch.Tensor) -> Vector3:
        """Lift orthographic image-plane points back into 3D using depth.

        Args:
            points: Two-dimensional point container with image-plane
                coordinates ``x`` and ``y``.
            depth: Tensor containing the target ``z`` coordinate for each
                unprojected point.

        Returns:
            Three-dimensional point container with ``x`` and ``y`` copied from
            ``points`` and ``z`` supplied by ``depth``.

        Raises:
            NotImplementedError: This projection model is declared as an
                interface placeholder and is not implemented yet.
        """
        raise NotImplementedError
