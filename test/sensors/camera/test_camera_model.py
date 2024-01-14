import pytest
import torch

from kornia.geometry.vector import Vector3
from kornia.image import ImageSize
from kornia.sensors.camera import CameraModel, CameraModelType
from kornia.testing import BaseTester


class TestPinholeCamera(BaseTester):
    def _make_rand_data(self, device, dtype):
        image_sizes = torch.randint(1, 100, (2,)).to(dtype).to(device)
        params = (
            torch.rand(
                4,
            )
            .to(dtype)
            .to(device)
        )
        return params, ImageSize(image_sizes[0], image_sizes[1])

    def test_smoke(self, device, dtype):
        params, image_size = self._make_rand_data(device, dtype)
        cam = CameraModel(image_size, CameraModelType.PINHOLE, params)
        assert isinstance(cam, CameraModel)
        self.assert_close(cam.params, params)
        self.assert_close(image_size.height, cam.height)
        self.assert_close(image_size.width, cam.width)

    @pytest.mark.skip(reason="Unnecessary test")
    def test_cardinality(self, device, dtype):
        pass

    def test_exception(self, device, dtype):
        # test for invalid params for different camera models
        params = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        image_size = ImageSize(100, 100)
        with pytest.raises(ValueError):
            CameraModel(image_size, CameraModelType.PINHOLE, params)
        with pytest.raises(ValueError):
            CameraModel(image_size, CameraModelType.BROWN_CONRADY, params)
        with pytest.raises(ValueError):
            CameraModel(image_size, CameraModelType.KANNALA_BRANDT_K3, params)
        with pytest.raises(ValueError):
            CameraModel(image_size, CameraModelType.ORTHOGRAPHIC, params)

    @pytest.mark.skip(reason="Unnecessary test")
    def test_gradcheck(self, device):
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_module(self, device, dtype):
        pass

    @pytest.mark.parametrize("batch_size", [None, 1, 2, 5])
    def test_project_unproject(self, device, dtype, batch_size):
        params, image_size = self._make_rand_data(device, dtype)
        cam = CameraModel(image_size, CameraModelType.PINHOLE, params)
        if batch_size is None:
            points = torch.rand((3,), device=device, dtype=dtype)
        else:
            points = torch.rand((batch_size, 3), device=device, dtype=dtype)
        points_vector = Vector3(points)
        projected_points = cam.project(points)
        projected_points_vector = cam.project(points_vector)
        unprojected_points = cam.unproject(projected_points, points[..., 2])
        unprojected_points_vector = cam.unproject(projected_points_vector, points_vector.z)
        self.assert_close(points, unprojected_points)
        self.assert_close(points, unprojected_points_vector.data)

    def test_matrix(self, device, dtype):
        params, image_size = self._make_rand_data(device, dtype)
        cam = CameraModel(image_size, CameraModelType.PINHOLE, params)
        K = torch.tensor(
            [
                [params[0], 0, params[2]],
                [0, params[1], params[3]],
                [0, 0, 1],
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(cam.matrix(), K)

    def test_properties(self, device, dtype):
        params, image_size = self._make_rand_data(device, dtype)
        cam = CameraModel(image_size, CameraModelType.PINHOLE, params)
        self.assert_close(cam.fx, params[0])
        self.assert_close(cam.fy, params[1])
        self.assert_close(cam.cx, params[2])
        self.assert_close(cam.cy, params[3])
        self.assert_close(cam.width, image_size.width)
        self.assert_close(cam.height, image_size.height)

    def test_scale(self, device, dtype):
        params, image_size = self._make_rand_data(device, dtype)
        cam = CameraModel(image_size, CameraModelType.PINHOLE, params)
        scale = torch.rand(1, device=device, dtype=dtype).squeeze()
        scaled_cam = cam.scale(scale)
        self.assert_close(cam.fx * scale, scaled_cam.fx)
        self.assert_close(cam.fy * scale, scaled_cam.fy)
        self.assert_close(cam.cx * scale, scaled_cam.cx)
        self.assert_close(cam.cy * scale, scaled_cam.cy)
        self.assert_close(cam.width * scale, scaled_cam.width)
        self.assert_close(cam.height * scale, scaled_cam.height)
