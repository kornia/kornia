from kornia.core import Tensor
from kornia.geometry.vector import Vector2


class AffineTransform:
    def distort(self, params: Tensor, points: Vector2) -> Vector2:
        """Distort one or more Vector2 points using the affine transform.

        Args:
            params: Tensor representing the affine transform parameters.
            points: Vector2 representing the points to distort.

        Returns:
            Vector2 representing the distorted points.

        Example:
            >>> params = Tensor([1., 2., 3., 4.])
            >>> points = Vector2.from_coords(1., 2.)
            >>> AffineTransform().distort(params, points)
            x: 4.0
            y: 8.0
        """
        fx, fy, cx, cy = params[..., 0], params[..., 1], params[..., 2], params[..., 3]
        u = points.x * fx + cx
        v = points.y * fy + cy
        return Vector2.from_coords(u, v)

    def undistort(self, params: Tensor, points: Vector2) -> Vector2:
        """Undistort one or more Vector2 points using the affine transform.

        Args:
            params: Tensor representing the affine transform parameters.
            points: Vector2 representing the points to undistort.

        Returns:
            Vector2 representing the undistorted points.

        Example:
            >>> params = Tensor([1., 2., 3., 4.])
            >>> points = Vector2.from_coords(1., 2.)
            >>> AffineTransform().undistort(params, points)
            x: -2.0
            y: -1.0
        """
        fx, fy, cx, cy = params[..., 0], params[..., 1], params[..., 2], params[..., 3]
        x = (points.x - cx) / fx
        y = (points.y - cy) / fy
        return Vector2.from_coords(x, y)


class BrownConradyTransform:
    def distort(self, params: Tensor, points: Vector2) -> Vector2:
        raise NotImplementedError

    def undistort(self, params: Tensor, points: Vector2) -> Vector2:
        raise NotImplementedError


class KannalaBrandtK3Transform:
    def distort(self, params: Tensor, points: Vector2) -> Vector2:
        raise NotImplementedError

    def undistort(self, params: Tensor, points: Vector2) -> Vector2:
        raise NotImplementedError
