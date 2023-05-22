import torch  # TODO remove this import

from kornia.core import Tensor
from kornia.geometry.vector import Vector2, Vector3


class Z1Projection:
    def project(self, points: Vector3) -> Vector2:
        xy = points.data[..., :2]
        z = points.z
        uv = (xy.T @ torch.diag(z).invers).T if len(z.shape) else xy.T * 1 / z
        return Vector2(uv)

    def unproject(self, points: Vector2, depth: Tensor) -> Vector3:
        return Vector3.from_coords(points.x * depth, points.y * depth, depth)


class OrthographicProjection:
    def project(self, points):
        raise NotImplementedError

    def unproject(self, points):
        raise NotImplementedError
