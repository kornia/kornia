from __future__ import annotations

from dataclasses import dataclass

from kornia.core import Module, Tensor, stack, zeros_like
from kornia.core.check import KORNIA_CHECK_TYPE
from kornia.geometry.liegroup.se3 import Se3
from kornia.geometry.vector import Vector2, Vector3
from kornia.sensor.camera.distortion import AffineTransform, CameraDistortionType
from kornia.sensor.camera.projection import CameraProjectionType, Z1Projection


@dataclass
class ImageSize:
    height: int
    width: int


class CameraModel(Module):
    def __init__(
        self,
        image_size: ImageSize,
        distortion_type: CameraDistortionType,
        projection_type: CameraProjectionType,
        params: Tensor,
    ):
        super().__init__()
        KORNIA_CHECK_TYPE(distortion_type, CameraDistortionType)
        KORNIA_CHECK_TYPE(projection_type, CameraProjectionType)
        self.image_size = image_size
        # TO DO: check params according to distortion type
        self._fx = params[..., 0]
        self._fy = params[..., 1]
        self._cx = params[..., 2]
        self._cy = params[..., 3]
        self.distortion_model = None
        self.projection_model = None
        self.projection_type = projection_type
        self.distortion_type = distortion_type
        if distortion_type == CameraDistortionType.AFFINE:
            self.distortion_model = AffineTransform(params)
        # elif distortion_type == CameraDistortionType.BROWN_CONRADY:
        #     self.distortion_model = BrownConradyTransform()
        if projection_type == CameraProjectionType.Z1:
            self.projection_model = Z1Projection()
        # elif projection_type == CameraProjectionType.ORTHOGRAPHIC:
        #     self.projection_model = OrthographicProjection()

    @property
    def params(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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

    def matrix(self) -> Tensor:
        raise NotImplementedError

    @property
    def K(self) -> Tensor:
        return self.matrix()

    def project(self, points: Vector3) -> Vector2:
        return self.distortion_model.distort(self.projection_model.project(points))

    def unproject(self, points: Vector2, depth: Tensor) -> Vector3:
        return self.projection_model.unproject(self.distortion_model.undistort(points), depth)


class PinholeCameraModel(CameraModel):
    def __init__(self, image_size: ImageSize, params: Tensor | None = None):
        if params is None:
            params = Tensor(
                [
                    image_size.width * 0.5,
                    image_size.width * 0.5,
                    (image_size.width - 1) * 0.5,
                    (image_size.height - 1) * 0.5,
                ]
            )
        super().__init__(image_size, CameraDistortionType.AFFINE, CameraProjectionType.Z1, params)

    def matrix(self) -> Tensor:
        z = zeros_like(self.fx)
        row1 = stack((self.fx, z, self.cx), -1)
        row2 = stack((z, self.fy, self.cy), -1)
        row3 = stack((z, z, z), -1)
        K = stack((row1, row2, row3), -2)
        K[..., -1, -1] = 1.0
        return K


class NamedPose(Module):
    def __init__(self, pose: Se3, source: str | list[str], destination: str | list[str]):
        self.dst_pose_src = pose
        self.source = source
        self.destination = destination

    def __mul__(self, right: NamedPose):
        pass


class PosedCameraModel:
    def __init__(self, camera: CameraModel, pose: NamedPose):
        pass

    def transform_to_camera_view(self):
        pass
