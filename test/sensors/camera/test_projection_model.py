import pytest
import torch

from kornia.geometry.vector import Vector2, Vector3
from kornia.sensors.camera.projection_model import Z1Projection
from kornia.testing import BaseTester


class TestZ1Projection(BaseTester):
    @pytest.mark.skip(reason="Unnecessary test")
    def test_smoke(self, device, dtype):
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_gradcheck(self, device):
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_module(self, device, dtype):
        pass

    def test_project(self, device, dtype):
        projection = Z1Projection()
        # batched points
        points = torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [6.0, 6.0, 2.0], [9.0, 9.0, 3.0]], device=device, dtype=dtype
        )
        expected = torch.tensor([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0], [3.0, 3.0]], device=device, dtype=dtype)
        self.assert_close(projection.project(Vector3(points)).data, expected)
        self.assert_close(projection.project(points), expected)
        # unbatched points
        points = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        expected = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
        self.assert_close(projection.project(Vector3(points)).data, expected)
        self.assert_close(projection.project(points), expected)

    def test_unproject(self, device, dtype):
        projection = Z1Projection()
        # batched points
        depth = torch.tensor([1.0, 1.0, 2.0, 3.0], device=device, dtype=dtype)
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0], [3.0, 3.0]], device=device, dtype=dtype)
        expected = torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [6.0, 6.0, 2.0], [9.0, 9.0, 3.0]], device=device, dtype=dtype
        )
        self.assert_close(projection.unproject(Vector2(points), depth).data, expected)
        self.assert_close(projection.unproject(points, depth), expected)
        # unbatched points
        depth = torch.tensor(1.0, device=device, dtype=dtype)
        points = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
        expected = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        self.assert_close(projection.unproject(Vector2(points), depth).data, expected)
        self.assert_close(projection.unproject(points, depth), expected)
