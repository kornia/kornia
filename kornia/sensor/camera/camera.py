from __future__ import annotations

from enum import Enum
from typing import Any, Union

from kornia.core import Tensor, stack, zeros_like
from kornia.geometry.vector import Vector2, Vector3
from kornia.image import ImageSize
from kornia.sensor.camera.distortion import AffineTransform, BrownConradyTransform, KannalaBrandtK3Transform
from kornia.sensor.camera.projection import OrthographicProjection, Z1Projection


class CameraModelType(Enum):
    PINHOLE = 0
    BROWN_CONRADY = 1
    KANNALA_BRANDT_K3 = 2
    ORTHOGRAPHIC = 3


def get_model_from_type(model_type: CameraModelType, image_size: ImageSize, params: Tensor) -> CameraModelVariants:
    if model_type == CameraModelType.PINHOLE:
        if params.shape[-1] != 4:
            raise ValueError("params must be of shape B, 4 for PINHOLE Camera")
        return PinholeModel(image_size, params)
    elif model_type == CameraModelType.BROWN_CONRADY:
        if params.shape[-1] != 12:
            raise ValueError("params must be of shape B, 12 for BROWN_CONRADY Camera")
        return BrownConradyModel(image_size, params)
    elif model_type == CameraModelType.KANNALA_BRANDT_K3:
        if params.shape[-1] != 8:
            raise ValueError("params must be of shape B, 8 for KANNALA_BRANDT_K3 Camera")
        return KannalaBrandtK3(image_size, params)
    elif model_type == CameraModelType.ORTHOGRAPHIC:
        if params.shape[-1] != 4:
            raise ValueError("params must be of shape B, 4 for ORTHOGRAPHIC Camera")
        return Orthographic(image_size, params)
    else:
        raise ValueError("Invalid Camera Model Type")


# TO DO change to |
CameraDistortionType = Union[AffineTransform, BrownConradyTransform, KannalaBrandtK3Transform]
CameraProjectionType = Union[Z1Projection, OrthographicProjection]


class CameraModelT:
    def __init__(
        self, distortion: CameraDistortionType, projection: CameraProjectionType, image_size: ImageSize, params: Tensor
    ) -> None:
        self.distortion = distortion
        self.projection = projection
        self._image_size = image_size
        self._height = image_size.height
        self._width = image_size.width
        self._params = params
        self._fx = params[..., 0]
        self._fy = params[..., 1]
        self._cx = params[..., 2]
        self._cy = params[..., 3]

    @property
    def image_size(self) -> ImageSize:
        return self._image_size

    @property
    def height(self) -> int | Tensor:
        return self._height

    @property
    def width(self) -> int | Tensor:
        return self._width

    @property
    def params(self) -> Tensor:
        return self._params

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

    def K(self) -> Tensor:
        return self.matrix()

    def project(self, points: Vector3) -> Vector2:
        return self.distortion.distort(self.params, self.projection.project(points))

    def unproject(self, points: Vector2, depth: Tensor) -> Vector3:
        return self.projection.unproject(self.distortion.undistort(self.params, points), depth)


class PinholeModel(CameraModelT):
    def __init__(self, image_size: ImageSize, params: Tensor) -> None:
        super().__init__(AffineTransform(), Z1Projection(), image_size, params)
        self.model_type_string = "PINHOLE"

    def matrix(self) -> Tensor:
        z = zeros_like(self.fx)
        row1 = stack((self.fx, z, self.cx), -1)
        row2 = stack((z, self.fy, self.cy), -1)
        row3 = stack((z, z, z), -1)
        K = stack((row1, row2, row3), -2)
        K[..., -1, -1] = 1.0
        return K

    def scale(self, scale_factor: Tensor) -> PinholeModel:
        fx = self.fx * scale_factor
        fy = self.fy * scale_factor
        cx = self.cx * scale_factor
        cy = self.cy * scale_factor
        params = stack((fx, fy, cx, cy), -1)
        image_size = ImageSize(self.image_size.height * scale_factor, self.image_size.width * scale_factor)
        return PinholeModel(image_size, params)


class BrownConradyModel(CameraModelT):
    def __init__(self, image_size: ImageSize, params: Tensor) -> None:
        super().__init__(BrownConradyTransform(), Z1Projection(), image_size, params)
        self.model_type_string = "BROWN_CONRADY"


class KannalaBrandtK3(CameraModelT):
    def __init__(self, image_size: ImageSize, params: Tensor) -> None:
        super().__init__(KannalaBrandtK3Transform(), Z1Projection(), image_size, params)
        self.model_type_string = "KANNALA_BRANDT_K3"


class Orthographic(CameraModelT):
    def __init__(self, image_size: ImageSize, params: Tensor) -> None:
        super().__init__(AffineTransform(), OrthographicProjection(), image_size, params)
        self.model_type_string = "ORTHOGRAPHIC"


CameraModelVariants = Union[PinholeModel, BrownConradyModel, KannalaBrandtK3, Orthographic]


class CameraModel:
    def __init__(self, image_size: ImageSize, model_type: CameraModelType, params: Tensor) -> None:
        self._model = get_model_from_type(model_type, image_size, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def __repr__(self) -> str:
        return f"CameraModel(image_size={self.image_size}, model_type={self.model_type_string}, params={self.params})"
