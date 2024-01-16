from kornia.core import Tensor, stack
from kornia.geometry.vector import Vector2


class AffineTransform:
    def distort(self, params: Tensor, points: Vector2 | Tensor) -> Vector2 | Tensor:
        """Distort one or more points using the affine transform.

        Args:
            params: Tensor of shape (N,) representing the affine transform parameters.
            points: Vector2 or Tensor representing the points to distort.

        Returns:
            Vector2 or Tensor representing the distorted points.

        Example:
            >>> params = Tensor([1., 2., 3., 4.])
            >>> points = Vector2.from_coords(1., 2.)
            >>> AffineTransform().distort(params, points)
            x: 4.0
            y: 8.0
        """
        params = params.squeeze()
        if len(params.shape) != 1:
            raise ValueError(f"Expected params to be of shape (N,), got {params.shape}")
        is_vector = isinstance(points, Vector2)
        if is_vector:
            points = points.data
        elif isinstance(points, Tensor):
            if len(points.shape) > 2 or points.shape[-1] != 2:
                raise ValueError(f"Expected points to be of shape (B, 2) or (2,), got {points.shape}")
        else:
            raise TypeError(f"Expected points to be Vector2 or Tensor, got {type(points)}")

        fx, fy, cx, cy = params[..., 0], params[..., 1], params[..., 2], params[..., 3]
        u = fx * points[..., 0] + cx
        v = fy * points[..., 1] + cy
        if is_vector:
            return Vector2.from_coords(u, v)
        elif len(points.shape) == 1:
            return Tensor([u, v]).to(device=points.device, dtype=points.dtype)
        else:
            return stack([u, v], dim=-1)

    def undistort(self, params: Tensor, points: Vector2 | Tensor) -> Vector2 | Tensor:
        """Undistort one or more points using the affine transform.

        Args:
            params: Tensor representing the affine transform parameters.
            points: Vector2 or Tensor representing the points to undistort.

        Returns:
            Vector2 or Tensor representing the undistorted points.

        Example:
            >>> params = Tensor([1., 2., 3., 4.])
            >>> points = Vector2.from_coords(1., 2.)
            >>> AffineTransform().undistort(params, points)
            x: -2.0
            y: -1.0
        """
        params = params.squeeze()
        if len(params.shape) != 1:
            raise ValueError(f"Expected params to be of shape (N,), got {params.shape}")
        is_vector = isinstance(points, Vector2)
        if is_vector:
            points = points.data
        elif isinstance(points, Tensor):
            if len(points.shape) > 2 or points.shape[-1] != 2:
                raise ValueError(f"Expected points to be of shape (B, 2) or (2,), got {points.shape}")
        else:
            raise TypeError(f"Expected points to be Vector2 or Tensor, got {type(points)}")
        fx, fy, cx, cy = params[..., 0], params[..., 1], params[..., 2], params[..., 3]
        u = (points[..., 0] - cx) / fx
        v = (points[..., 1] - cy) / fy
        if is_vector:
            return Vector2.from_coords(u, v)
        else:
            return stack([u, v], -1)


class BrownConradyTransform:
    def distort(self, params: Tensor, points: Vector2 | Tensor) -> Vector2 | Tensor:
        raise NotImplementedError

    def undistort(self, params: Tensor, points: Vector2 | Tensor) -> Vector2 | Tensor:
        raise NotImplementedError


class KannalaBrandtK3Transform:
    def distort(self, params: Tensor, points: Vector2 | Tensor) -> Vector2 | Tensor:
        raise NotImplementedError

    def undistort(self, params: Tensor, points: Vector2 | Tensor) -> Vector2 | Tensor:
        raise NotImplementedError
