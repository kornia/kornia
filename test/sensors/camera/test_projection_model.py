import pytest
import torch

from kornia.core import Device, Dtype
from kornia.geometry.vector import Vector2, Vector3
from kornia.sensors.camera.projection_model import Z1Projection
from kornia.sensors.camera.projection_z1 import dx_proj_x, project_points_z1, unproject_points_z1
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


class TestProjectionZ1(BaseTester):
    def test_project_points_z1(self, device, dtype):
        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        expected = torch.tensor([0.3333333432674408, 0.6666666865348816], device=device, dtype=dtype)
        self.assert_close(project_points_z1(points), expected)

    def test_project_points_z1_batch(self, device, dtype):
        points = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device, dtype=dtype)
        expected = torch.tensor(
            [[0.3333333432674408, 0.6666666865348816], [0.6666666865348816, 0.8333333730697632]],
            device=device,
            dtype=dtype,
        )
        self.assert_close(project_points_z1(points), expected)

    def test_unproject_points_z1(self, device, dtype):
        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        expected = torch.tensor([1.0, 2.0, 1.0], device=device, dtype=dtype)
        self.assert_close(unproject_points_z1(points), expected)

    def test_unproject_points_z1_batch(self, device, dtype):
        points = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=dtype)
        expected = torch.tensor([[1.0, 2.0, 1.0], [3.0, 4.0, 1.0]], device=device, dtype=dtype)
        self.assert_close(unproject_points_z1(points), expected)

    def test_unproject_points_z1_extension(self, device, dtype):
        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        extension = torch.tensor([2.0], device=device, dtype=dtype)
        expected = torch.tensor([2.0, 4.0, 2.0], device=device, dtype=dtype)
        self.assert_close(unproject_points_z1(points, extension), expected)

    def test_unproject_points_z1_batch_extension(self, device, dtype):
        points = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=dtype)
        extension = torch.tensor([2.0, 3.0], device=device, dtype=dtype)
        expected = torch.tensor([[2.0, 4.0, 2.0], [9.0, 12.0, 3.0]], device=device, dtype=dtype)
        self.assert_close(unproject_points_z1(points, extension), expected)

    def test_dx_proj_x(self, device, dtype):
        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        expected = torch.tensor(
            [[0.3333333432674408, 0.0, -0.1111111119389534], [0.0, 0.3333333432674408, -0.2222222238779068]],
            device=device,
            dtype=dtype,
        )
        self.assert_close(dx_proj_x(points), expected)

    @pytest.mark.skip(reason="Unnecessary test")
    def test_smoke(self, device: Device, dtype: Dtype) -> None:
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_cardinality(self, device: Device, dtype: Dtype) -> None:
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_exception(self, device: Device, dtype: Dtype) -> None:
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_gradcheck(self, device: Device) -> None:
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_jit(self, device: Device, dtype: Dtype) -> None:
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_module(self, device: Device, dtype: Dtype) -> None:
        pass
