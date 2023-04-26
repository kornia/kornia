from enum import Enum

import torch  # todo remove

from kornia.core import Module, Tensor
from kornia.geometry.vector import Vector2, Vector3


class CameraProjectionType(Enum):
    Z1 = 0
    ORTHOGRAPHIC = 1


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
        # add something like vector3.xy?
        return Vector2(points.data[..., :2] @ torch.diag(points.z).inverse())

    def unproject(self, points: Vector2, depth: Tensor) -> Vector3:
        return Vector3.from_coords(points.x * depth, points.y * depth, depth)


class OrthographicProjection(ProjectionModel):
    def __init__(self):
        super().__init__()

    def project(self, points):
        raise NotImplementedError

    def unproject(self, points):
        raise NotImplementedError
