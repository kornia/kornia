from enum import Enum

from kornia.core import Module, Tensor, zeros_like, stack
from kornia.geometry.vector import Vector2


class DistortionModel(Module):
    def __init__(self, params: Tensor):
        super().__init__()
        self.params = params
        self._fx = params[..., 0]
        self._fy = params[..., 1]
        self._cx = params[..., 2]
        self._cy = params[..., 3]

    @property
    def fx(self) -> Tensor:
        return self._fx

    @property
    def fy(self) -> Tensor:
        return self._fy

    @property
    def cx(self) -> Tensor:
        return self._cx

    @property
    def cy(self) -> Tensor:
        return self._cy
    
    def matrix(self) -> Tensor:
        raise NotImplementedError

    def distort(self, points: Vector2):
        raise NotImplementedError

    def undistort(self, points):
        raise NotImplementedError


class AffineTransform(DistortionModel):
    def __init__(self, params: Tensor):
        super().__init__(params)
        
    
    def matrix(self) -> Tensor:
        z = zeros_like(self.fx)
        row1 = stack((self.fx, z, self.cx), -1)
        row2 = stack((z, self.fy, self.cy), -1)
        row3 = stack((z, z, z), -1)
        K = stack((row1, row2, row3), -2)
        K[..., -1, -1] = 1.0
        return K

    def distort(self, points: Vector2) -> Vector2:
        u = points.x * self.fx + self.cx
        v = points.y * self.fy + self.cy
        return Vector2.from_coords(u, v)

    def undistort(self, points: Vector2) -> Vector2:
        x = (points.x - self.cx) / self.fx
        y = (points.y - self.cy) / self.fy
        return Vector2.from_coords(x, y)

class KannalaBrandtK3Transform(DistortionModel):
    def __init__(self, params: Tensor):
        # params of the form: [fx, fy, cx, cy, kb0, kb1, kb2, kb3]
        super().__init__(params)