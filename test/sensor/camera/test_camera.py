import pytest
import torch

from kornia.geometry.vector import Vector3
from kornia.image import ImageSize
from kornia.sensor.camera import CameraModel, CameraModelType
from kornia.testing import BaseTester


class TestPinholeCamera(BaseTester):
    def _make_rand_data(self, batch_size, device, dtype):
        params = torch.rand(batch_size, 4).to(dtype).to(device)
        image_sizes = torch.randint(1, 100, (batch_size, 2)).to(dtype).to(device)
        return params, ImageSize(image_sizes[:, 0], image_sizes[:, 1])

    def test_smoke(self, device, dtype):
        params, image_size = self._make_rand_data(1, device, dtype)
        cam = CameraModel(image_size, CameraModelType.PINHOLE, params)
        assert isinstance(cam, CameraModel)
        self.assert_close(cam.params, params)
        self.assert_close(image_size.height, cam.height)
        self.assert_close(image_size.width, cam.width)

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

    def test_gradcheck(self, device):
        pass

    def test_jit(self, device, dtype):
        pass

    def test_module(self, device, dtype):
        pass

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_project_unproject(self, device, dtype, batch_size):
        params, image_size = self._make_rand_data(batch_size, device, dtype)
        cam = CameraModel(image_size, CameraModelType.PINHOLE, params)
        points = torch.rand((batch_size, 3), device=device, dtype=dtype)
        projected = cam.project(Vector3(points))
        unprojected = cam.unproject(projected, points[..., 2])
        self.assert_close(points, unprojected.data)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_matrix(self, device, dtype, batch_size):
        params, image_size = self._make_rand_data(batch_size, device, dtype)
        cam = CameraModel(image_size, CameraModelType.PINHOLE, params)
        z = torch.zeros(batch_size, dtype=dtype, device=device)
        o = torch.ones(batch_size, dtype=dtype, device=device)
        K = torch.stack([params[:, 0], z, params[:, 2], z, params[:, 1], params[:, 3], z, z, o], dim=1).reshape(
            batch_size, 3, 3
        )
        self.assert_close(cam.matrix(), K)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_properties(self, device, dtype, batch_size):
        params, image_size = self._make_rand_data(batch_size, device, dtype)
        cam = CameraModel(image_size, CameraModelType.PINHOLE, params)
        self.assert_close(cam.fx, params[:, 0])
        self.assert_close(cam.fy, params[:, 1])
        self.assert_close(cam.cx, params[:, 2])
        self.assert_close(cam.cy, params[:, 3])
        self.assert_close(cam.width, image_size.width)
        self.assert_close(cam.height, image_size.height)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_scale(self, device, dtype, batch_size):
        params, image_size = self._make_rand_data(batch_size, device, dtype)
        cam = CameraModel(image_size, CameraModelType.PINHOLE, params)
        scale = torch.rand(batch_size, device=device, dtype=dtype)
        scaled_cam = cam.scale(scale)
        self.assert_close(cam.fx * scale, scaled_cam.fx)
        self.assert_close(cam.fy * scale, scaled_cam.fy)
        self.assert_close(cam.cx * scale, scaled_cam.cx)
        self.assert_close(cam.cy * scale, scaled_cam.cy)
        self.assert_close(cam.width * scale, scaled_cam.width)
        self.assert_close(cam.height * scale, scaled_cam.height)
