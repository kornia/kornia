from __future__ import annotations

from kornia.core import Tensor, diag, stack
from kornia.geometry.vector import Vector2, Vector3


class Z1Projection:
    def project(self, points: Vector3 | Tensor) -> Vector2 | Tensor:
        """Project one or more Vector3 or Tensor from the camera frame into the canonical z=1 plane through
        perspective division.

        Args:
            points: Vector3 or Tensor representing the points to project.

        Returns:
            Vector2 or Tensor representing the projected points.

        Example:
            >>> points = Vector3.from_coords(1., 2., 3.)
            >>> Z1Projection().project(points)
            x: 0.3333333432674408
            y: 0.6666666865348816
        """
        is_vector = isinstance(points, Vector3)
        if is_vector:
            points = points.data
        elif isinstance(points, Tensor):
            if len(points.shape) > 2 or points.shape[-1] != 3:
                raise ValueError(f"Expected points to be of shape (B, 3) or (3,), got {points.shape}")
        else:
            raise TypeError(f"Expected points to be Vector3 or Tensor, got {type(points)}")
        xy = points[..., :2]
        z = points[..., 2]
        uv = (xy.T @ diag(z).inverse()).T if len(z.shape) else xy.T * 1 / z
        if is_vector:
            return Vector2(uv)
        else:
            return uv

    def unproject(self, points: Vector2 | Tensor, depth: Tensor | float) -> Vector3 | Tensor:
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
        is_vector = isinstance(points, Vector2)
        if is_vector:
            points = points.data
        elif isinstance(points, Tensor):
            if len(points.shape) > 2 or points.shape[-1] != 2:
                raise ValueError(f"Expected points to be of shape (B, 2) or (2,), got {points.shape}")
        else:
            raise TypeError(f"Expected points to be Vector2 or Tensor, got {type(points)}")
        if isinstance(depth, (float, int)):
            depth = Tensor([depth])
        elif isinstance(depth, Tensor):
            if len(depth.shape) > 1:
                raise ValueError(f"Expected depth to be of shape (B,), got {depth.shape}")
        else:
            raise TypeError(f"Expected depth to be Tensor or float, got {type(depth)}")
        xyz = stack([points[..., 0] * depth, points[..., 1] * depth, depth], -1)
        if is_vector:
            return Vector3(xyz)
        else:
            return xyz


class OrthographicProjection:
    def project(self, points: Vector3 | Tensor) -> Vector2 | Tensor:
        raise NotImplementedError

    def unproject(self, points: Vector2 | Tensor, depth: Tensor) -> Vector3 | Tensor:
        raise NotImplementedError
