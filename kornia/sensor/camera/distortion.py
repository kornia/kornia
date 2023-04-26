from enum import Enum

from kornia.core import Module, Tensor
from kornia.geometry.vector import Vector2


class CameraDistortionType(Enum):
    AFFINE = 0
    BROWN_CONRADY = 1
    KANNALA_BRANDT = 2


class DistortionModel(Module):
    def __init__(self, params: Tensor):
        super().__init__()
        self.params = params

    def distort(self, points: Vector2):
        raise NotImplementedError

    def undistort(self, points):
        raise NotImplementedError


class AffineTransform(DistortionModel):
    def __init__(self, params: Tensor):
        # params of the form: [fx, fy, cx, cy]
        super().__init__(params)

    def distort(self, points: Vector2) -> Vector2:
        u = points.x * self.params[..., 0] + self.params[..., 2]
        v = points.y * self.params[..., 1] + self.params[..., 3]
        return Vector2.from_coords(u, v)

    def undistort(self, points: Vector2) -> Vector2:
        x = (points.x - self.params[..., 2]) / self.params[..., 0]
        y = (points.y - self.params[..., 3]) / self.params[..., 1]
        return Vector2.from_coords(x, y)
