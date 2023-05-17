from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


from kornia.core import Module, Tensor
from kornia.core.check import KORNIA_CHECK_TYPE
from kornia.geometry.liegroup.se3 import Se3
from kornia.geometry.vector import Vector2, Vector3
from kornia.sensor.camera.distortion import AffineTransform, DistortionModel
from kornia.sensor.camera.projection import OrthographicProjection, ProjectionModel, Z1Projection


@dataclass
class ImageSize:
    height: int
    width: int


class CameraDistortionType(Enum):
    AFFINE = 0
    BROWN_CONRADY = 1
    KANNALA_BRANDT = 2


class CameraProjectionType(Enum):
    Z1 = 0
    ORTHOGRAPHIC = 1


class CameraModel(Module):
    def __init__(
        self,
        image_size: ImageSize | tuple[int, int],
        distortion_type: CameraDistortionType,
        projection_type: CameraProjectionType,
        params: Tensor,
    ):
        super().__init__()
        KORNIA_CHECK_TYPE(distortion_type, CameraDistortionType)
        KORNIA_CHECK_TYPE(projection_type, CameraProjectionType)
        if isinstance(image_size, ImageSize):
            self._height = image_size.height
            self._width = image_size.width
        else:
            self._height = image_size[0]
            self._width = image_size[1]
        #check params according to distortion type
        check_params(params, distortion_type)
        self.distortion_model = get_distortion_model(distortion_type, params)
        self.projection_model = get_projection_model(projection_type)
        self._fx = params[..., 0]
        self._fy = params[..., 1]
        self._cx = params[..., 2]
        self._cy = params[..., 3]

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

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
        return self.distortion_model.matrix()

    @property
    def K(self) -> Tensor:
        return self.matrix()

    def project(self, points: Vector3) -> Vector2:
        return self.distortion_model.distort(self.projection_model.project(points))

    def unproject(self, points: Vector2, depth: Tensor) -> Vector3:
        return self.projection_model.unproject(self.distortion_model.undistort(points), depth)


def check_params(params: Tensor, distortion_type: CameraDistortionType) -> None:
    if distortion_type == CameraDistortionType.AFFINE:
        if params.shape[-1] != 4:
            raise ValueError("Invalid number of parameters for affine distortion")
    elif distortion_type == CameraDistortionType.BROWN_CONRADY:
        if params.shape[-1] != 12:
            raise ValueError("Invalid number of parameters for Brown-Conrady distortion")
    elif distortion_type == CameraDistortionType.KANNALA_BRANDT:
        if params.shape[-1] != 8:
            raise ValueError("Invalid number of parameters for Kannala-Brandt distortion")
    else:
        raise ValueError("Invalid distortion type")

def get_projection_model(projection_type: CameraProjectionType) -> ProjectionModel:
    if projection_type == CameraProjectionType.Z1:
        return Z1Projection()
    elif projection_type == CameraProjectionType.ORTHOGRAPHIC:
        return OrthographicProjection()
    else:
        raise ValueError("Invalid projection type")

def get_distortion_model(distortion_type: CameraDistortionType, params: Tensor) -> DistortionModel:
    if distortion_type == CameraDistortionType.AFFINE:
        return AffineTransform(params)
    else:
        raise ValueError("Invalid distortion type")
