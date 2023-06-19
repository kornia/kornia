import pytest
import torch

from kornia.geometry.vector import Vector2
from kornia.sensors.camera.distortion_model import AffineTransform
from kornia.testing import BaseTester


class TestAffineTransform(BaseTester):
    @pytest.mark.skip(reason='Unnecessary test')
    def test_smoke(self, device, dtype):
        pass

    @pytest.mark.skip(reason='Unnecessary test')
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason='Unnecessary test')
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason='Unnecessary test')
    def test_gradcheck(self, device):
        pass

    @pytest.mark.skip(reason='Unnecessary test')
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason='Unnecessary test')
    def test_module(self, device, dtype):
        pass

    def test_distort(self, device, dtype):
        distortion = AffineTransform()
        points = torch.tensor([[1.0, 1.0], [1.0, 5.0], [2.0, 4.0], [3.0, 9.0]], device=device, dtype=dtype)
        params = torch.tensor([[328.0, 328.0, 150.0, 150.0]], device=device, dtype=dtype)
        expected = torch.tensor(
            [[478.0, 478.0], [478.0, 1790.0], [806.0, 1462.0], [1134.0, 3102.0]], device=device, dtype=dtype
        )
        self.assert_close(distortion.distort(params, Vector2(points)).data, expected)

    def test_undistort(self, device, dtype):
        distortion = AffineTransform()
        points = torch.tensor(
            [[478.0, 478.0], [478.0, 1790.0], [806.0, 1462.0], [1134.0, 3102.0]], device=device, dtype=dtype
        )
        params = torch.tensor([[328.0, 328.0, 150.0, 150.0]], device=device, dtype=dtype)
        expected = torch.tensor([[1.0, 1.0], [1.0, 5.0], [2.0, 4.0], [3.0, 9.0]], device=device, dtype=dtype)
        self.assert_close(distortion.undistort(params, Vector2(points)).data, expected)
