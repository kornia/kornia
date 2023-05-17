# todo remove
import torch

from kornia.core import Module, Tensor
from kornia.geometry.vector import Vector2, Vector3


class ProjectionModel(Module):
    def __init__(self):
        super().__init__()

    def project(self, points):
        raise NotImplementedError

    def unproject(self, points):
        raise NotImplementedError


class Z1Projection(ProjectionModel):
    def __init__(self):
        super().__init__()

    def project(self, points: Vector3) -> Vector2:
        xy = points.data[..., :2]
        z = points.z
        uv = (xy.T @ torch.diag(z).inverse()).T if len(z.shape) else xy.T * 1 / z
        return Vector2(uv)

    def unproject(self, points: Vector2, depth: Tensor) -> Vector3:
        return Vector3.from_coords(points.x * depth, points.y * depth, depth)


class OrthographicProjection(ProjectionModel):
    def __init__(self):
        super().__init__()

    def project(self, points):
        raise NotImplementedError

    def unproject(self, points):
        raise NotImplementedError
