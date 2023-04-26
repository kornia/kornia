from enum import Enum
from typing import Optional, Tuple
from dataclasses import dataclass

from kornia.core import Module, Tensor, zeros_like, stack
from kornia.core.check import KORNIA_CHECK_TYPE
from kornia.geometry.vector import Vector3, Vector2


class CameraProjectionType(Enum):
    UNKNOWN = 0
    PINHOLE = 1
    ORTHOGRAPHIC = 2


@dataclass
class ImageSize:
    height: int
    width: int


class CameraModel(Module):
    def __init__(self, image_size: ImageSize, projection_type: CameraProjectionType, params: Tensor):
        # KORNIA_CHECK_SHAPE(params, ["B", 4])
        KORNIA_CHECK_TYPE(projection_type, CameraProjectionType)
        self._fx = params[..., 0]
        self._fy = params[..., 1]
        self._cx = params[..., 2]
        self._cy = params[..., 3]
        self.image_size = image_size
        self.projection_type = projection_type

    def __repr__(self):
        return f"\n{self.projection_type.name} camera model \n{self.width}x{self.height} image \nfx={self.fx}\nfy={self.fy}\ncx={self.cx}\ncy={self.cy}"

    @property
    def params(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self._fx, self._fy, self._cx, self._cy

    @property
    def height(self) -> int:
        return self.image_size.height

    @property
    def width(self) -> int:
        return self.image_size.width

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

    def matrix(self):
        z = zeros_like(self._fx)
        row1 = stack((self._fx, z, self._cx), -1)
        row2 = stack((z, self._fy, self._cy), -1)
        row3 = stack((z, z, z), -1)
        K = stack((row1, row2, row3), -2)
        K[..., -1, -1] = 1.0
        return K

    @property
    def K(self):
        return self.matrix()

    def project(self, points: Vector3) -> Vector2:
        raise NotImplementedError

    def unproject(self, points: Vector2, depth: Tensor) -> Vector3:
        raise NotImplementedError


class PinholeCameraModel(CameraModel):
    def __init__(self, image_size: ImageSize, params: Optional[Tensor] = None):
        if params is None:
            params = Tensor(
                [
                    image_size.width * 0.5,
                    image_size.width * 0.5,
                    (image_size.width - 1) * 0.5,
                    (image_size.height - 1) * 0.5,
                ]
            )
        super().__init__(image_size, CameraProjectionType.PINHOLE, params)

    def project(self, points: Vector3) -> Vector2:
        u = points.x / points.z * self.fx + self.cx
        v = points.y / points.z * self.fy + self.cy
        return Vector2.from_coords(u, v)

    def unproject(self, points: Vector2, depth: Tensor) -> Vector3:
        x = (points.x - self.cx) * depth / self.fx
        y = (points.y - self.cy) * depth / self.fy
        return Vector3.from_coords(x, y, depth)
