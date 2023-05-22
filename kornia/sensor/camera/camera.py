from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK_TYPE
from kornia.geometry.vector import Vector2, Vector3
from kornia.sensor.camera.distortion import AffineTransform, BrownConradyTransform
from kornia.sensor.camera.projection import Z1Projection, OrthographicProjection


class CameraDistortionType(Enum):
    PINHOLE = 0
    BROWN_CONRADY = 1
    KANNALA_BRANDT_K3 = 2
    ORTHOGRAPHIC = 3


@dataclass
class ImageSize:
    height: int
    width: int


def getModelFromType(type: CameraDistortionType, params: Tensor):
    if type == CameraDistortionType.PINHOLE:
        return PinholeModel(params)
    elif type == CameraDistortionType.BROWN_CONRADY:
        return BrownConradyModel(params)
    else:
        raise ValueError("Invalid distortion type")


CameraDistortionTypeUnion = Union[AffineTransform, BrownConradyTransform]
CameraProjectionTypeUnion = Union[Z1Projection, OrthographicProjection]


class CameraModelT:
    def __init__(
        self, Distortion: CameraDistortionTypeUnion, Projection: CameraProjectionTypeUnion, params: Optional[Tensor]
    ) -> None:
        self.params = params
        self.Distortion = Distortion
        self.Projection = Projection

    def cam_project(self, points: Vector3) -> Vector2:
        return self.Distortion().distort(self.params, self.Projection().project(points))

    def cam_unproject(self, points: Vector2, depth: Tensor) -> Vector3:
        return self.Projection().unproject(self.Distortion().undistort(self.params, points), depth)


class PinholeModel(CameraModelT):
    def __init__(self, params: Tensor | None) -> None:
        super().__init__(AffineTransform, Z1Projection, params)


class BrownConradyModel(CameraModelT):
    def __init__(self, params: Tensor | None) -> None:
        super().__init__(BrownConradyTransform, Z1Projection, params)


class CameraModel:
    def __init__(self, image_size: ImageSize, type: CameraDistortionType, params: Tensor) -> None:
        self.image_size = image_size
        self.type = type
        self.params = params
        self.model = getModelFromType(type, params)

    def cam_project(self, points: Vector3) -> Vector2:
        return self.model.cam_project(points)

    def cam_unproject(self, points: Vector2, depth: Tensor) -> Vector3:
        return self.model.cam_unproject(points, depth)
