from __future__ import annotations

from kornia.core import Tensor
from kornia.geometry.vector import Vector2, Vector3


class Z1Projection:
    def project(self, points: Vector3) -> Vector2:
        """Project one or more Vector3 from the camera frame into the canonical z=1 plane through perspective
        division.

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
        uv = (xy.T @ z.diag().inverse()).T if len(z.shape) else xy.T * 1 / z
        return Vector2(uv)

    def unproject(self, points: Vector2, depth: Tensor | float) -> Vector3:
        """Unproject one or more Vector2 from the canonical z=1 plane into the camera frame.

        Args:
            points: Vector2 representing the points to unproject.
            depth: Tensor representing the depth of the points to unproject.

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
            depth = Tensor([depth])
        return Vector3.from_coords(points.x * depth, points.y * depth, depth)


class OrthographicProjection:
    def project(self, points: Vector3) -> Vector2:
        raise NotImplementedError

    def unproject(self, points: Vector2, depth: Tensor) -> Vector3:
        raise NotImplementedError
