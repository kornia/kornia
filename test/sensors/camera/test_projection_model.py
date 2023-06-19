import torch

from kornia.geometry.vector import Vector2, Vector3
from kornia.sensors.camera.projection_model import Z1Projection
from kornia.testing import BaseTester


class TestZ1Projection(BaseTester):
    def test_smoke(self, device, dtype):
        pass

    def test_cardinality(self, device, dtype):
        pass

    def test_exception(self, device, dtype):
        pass

    def test_gradcheck(self, device):
        pass

    def test_jit(self, device, dtype):
        pass

    def test_module(self, device, dtype):
        pass

    def test_project(self, device, dtype):
        projection = Z1Projection()
        points = torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [6.0, 6.0, 2.0], [9.0, 9.0, 3.0]], device=device, dtype=dtype
        )
        expected = torch.tensor([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0], [3.0, 3.0]], device=device, dtype=dtype)
        self.assert_close(projection.project(Vector3(points)).data, expected)

    def test_unproject(self, device, dtype):
        projection = Z1Projection()
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0], [3.0, 3.0]], device=device, dtype=dtype)
        expected = torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [6.0, 6.0, 2.0], [9.0, 9.0, 3.0]], device=device, dtype=dtype
        )
        self.assert_close(
            projection.unproject(Vector2(points), torch.tensor([1.0, 1.0, 2.0, 3.0], device=device, dtype=dtype)).data,
            expected,
        )
