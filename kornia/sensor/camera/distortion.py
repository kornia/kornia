from kornia.core import Tensor
from kornia.geometry.vector import Vector2


class AffineTransform:
    def distort(self, params: Tensor, points: Vector2) -> Vector2:
        fx, fy, cx, cy = params[..., 0], params[..., 1], params[..., 2], params[..., 3]
        u = points.x * fx + cx
        v = points.y * fy + cy
        return Vector2.from_coords(u, v)

    def undistort(self, params: Tensor, points: Vector2) -> Vector2:
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
