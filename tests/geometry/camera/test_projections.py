import pytest
import torch

from kornia.geometry.camera.projection_orthographic import (
    dx_project_points_orthographic,
    project_points_orthographic,
    unproject_points_orthographic,
)
from kornia.geometry.camera.projection_z1 import dx_project_points_z1, project_points_z1, unproject_points_z1

from testing.base import BaseTester


class TestProjectionZ1(BaseTester):
    def test_smoke(self, device, dtype):
        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        assert project_points_z1(points) is not None

    def _test_cardinality_unproject_batch(self, device, dtype, batch_size):
        batch_tuple = (batch_size,) if batch_size is not None else ()
        points = torch.rand(batch_tuple + (3,), device=device, dtype=dtype)
        assert project_points_z1(points).shape == batch_tuple + (2,)

    def _test_cardinality_project_batch(self, device, dtype, batch_size):
        batch_tuple = (batch_size,) if batch_size is not None else ()
        points = torch.rand(batch_tuple + (2,), device=device, dtype=dtype)
        assert unproject_points_z1(points).shape == batch_tuple + (3,)

    @pytest.mark.parametrize("batch_size", [None, 1, 2, 3])
    def test_cardinality(self, device, dtype, batch_size):
        self._test_cardinality_project_batch(device, dtype, batch_size)
        self._test_cardinality_unproject_batch(device, dtype, batch_size)

    def test_project_points_z1(self, device, dtype):
        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        expected = torch.tensor([0.3333333432674408, 0.6666666865348816], device=device, dtype=dtype)
        self.assert_close(project_points_z1(points), expected)

    def test_project_points_z1_batch(self, device, dtype):
        points = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device, dtype=dtype)
        expected = torch.tensor(
            [
                [0.3333333432674408, 0.6666666865348816],
                [0.6666666865348816, 0.8333333730697632],
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(project_points_z1(points), expected)

    def test_project_points_z1_invalid(self, device, dtype):
        # NOTE: this is a corner case where the depth is 0.0 and the point is at infinity
        #      the projection is not defined and the function returns inf. The second point
        #      is behind the camera which is not a valid point and the user should handle it.
        points = torch.tensor([[1.0, 2.0, 0.0], [4.0, 5.0, -1.0]], device=device, dtype=dtype)
        expected = torch.tensor([[float("inf"), float("inf")], [-4.0, -5.0]], device=device, dtype=dtype)
        self.assert_close(project_points_z1(points), expected)

    def test_unproject_points_z1(self, device, dtype):
        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        expected = torch.tensor([1.0, 2.0, 1.0], device=device, dtype=dtype)
        self.assert_close(unproject_points_z1(points), expected)

    def test_unproject_points_z1_batch(self, device, dtype):
        points = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=dtype)
        expected = torch.tensor([[1.0, 2.0, 1.0], [3.0, 4.0, 1.0]], device=device, dtype=dtype)
        self.assert_close(unproject_points_z1(points), expected)

    def test_project_unproject(self, device, dtype):
        points = torch.tensor([1.0, 2.0, 2.0], device=device, dtype=dtype)
        extension = torch.tensor([2.0], device=device, dtype=dtype)
        self.assert_close(unproject_points_z1(project_points_z1(points), extension), points)

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
            [
                [0.3333333432674408, 0.0, -0.1111111119389534],
                [0.0, 0.3333333432674408, -0.2222222238779068],
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(dx_project_points_z1(points), expected)

    def test_exception(self, device, dtype) -> None:
        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        extension = torch.tensor([2.0], device=device, dtype=dtype)
        with pytest.raises(TypeError):
            unproject_points_z1(points, extension)

    def _test_gradcheck_unproject(self, device):
        points = torch.tensor([1.0, 2.0], device=device, dtype=torch.float64)
        extension = torch.tensor([2.0], device=device, dtype=torch.float64)
        self.gradcheck(unproject_points_z1, (points, extension))

    def _test_gradcheck_project(self, device):
        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=torch.float64)
        self.gradcheck(project_points_z1, (points,))

    def test_gradcheck(self, device) -> None:
        self._test_gradcheck_project(device)
        self._test_gradcheck_unproject(device)

    def _test_jit_unproject(self, device, dtype) -> None:
        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        extension = torch.tensor([2.0], device=device, dtype=dtype)
        op_script = torch.jit.script(unproject_points_z1)
        actual = op_script(points, extension)
        expected = unproject_points_z1(points, extension)
        self.assert_close(actual, expected)

    def _test_jit_project(self, device, dtype) -> None:
        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        op_script = torch.jit.script(project_points_z1)
        actual = op_script(points)
        expected = project_points_z1(points)
        self.assert_close(actual, expected)

    def test_jit(self, device, dtype) -> None:
        self._test_jit_project(device, dtype)
        self._test_jit_unproject(device, dtype)


class TestProjectionOrthographic(BaseTester):
    def test_smoke(self, device, dtype):
        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        assert project_points_orthographic(points) is not None

    def _test_cardinality_unproject_batch(self, device, dtype, batch_size):
        batch_tuple = (batch_size,) if batch_size is not None else ()
        points = torch.rand(batch_tuple + (3,), device=device, dtype=dtype)
        assert project_points_orthographic(points).shape == batch_tuple + (2,)

    def _test_cardinality_project_batch(self, device, dtype, batch_size):
        batch_tuple = (batch_size,) if batch_size is not None else ()
        points = torch.rand(batch_tuple + (2,), device=device, dtype=dtype)
        extension = torch.rand(batch_tuple, device=device, dtype=dtype)
        assert unproject_points_orthographic(points, extension).shape == batch_tuple + (3,)

    @pytest.mark.parametrize("batch_size", [None, 1, 2, 3])
    def test_cardinality(self, device, dtype, batch_size):
        self._test_cardinality_project_batch(device, dtype, batch_size)
        self._test_cardinality_unproject_batch(device, dtype, batch_size)

    def test_project_points_orthographic(self, device, dtype):
        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        expected = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        self.assert_close(project_points_orthographic(points), expected)

    def test_project_points_orthographic_batch(self, device, dtype):
        points = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device, dtype=dtype)
        expected = torch.tensor([[1.0, 2.0], [4.0, 5.0]], device=device, dtype=dtype)
        self.assert_close(project_points_orthographic(points), expected)

    def test_unproject_points_orthographic_extension(self, device, dtype):
        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        extension = torch.tensor([2.0], device=device, dtype=dtype)
        expected = torch.tensor([1.0, 2.0, 2.0], device=device, dtype=dtype)
        self.assert_close(unproject_points_orthographic(points, extension), expected)

    def test_unproject_points_orthographic_batch_extension(self, device, dtype):
        points = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=dtype)
        extension = torch.tensor([2.0, 3.0], device=device, dtype=dtype)
        expected = torch.tensor([[1.0, 2.0, 2.0], [3.0, 4.0, 3.0]], device=device, dtype=dtype)
        self.assert_close(unproject_points_orthographic(points, extension), expected)

    def test_project_unproject(self, device, dtype):
        points = torch.tensor([1.0, 2.0, 2.0], device=device, dtype=dtype)
        extension = torch.tensor([2.0], device=device, dtype=dtype)
        self.assert_close(unproject_points_orthographic(project_points_orthographic(points), extension), points)

    def test_dx_proj_x(self, device, dtype):
        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        expected = torch.tensor([1.0], device=device, dtype=dtype)
        self.assert_close(dx_project_points_orthographic(points), expected)

    def test_exception(self, device, dtype) -> None:
        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        extension = torch.tensor([2.0], device=device, dtype=dtype)
        with pytest.raises(TypeError):
            unproject_points_orthographic(points, extension)

    def _test_gradcheck_project(self, device):
        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=torch.float64)
        self.gradcheck(project_points_orthographic, (points,))

    def _test_gradcheck_unproject(self, device):
        points = torch.tensor([1.0, 2.0], device=device, dtype=torch.float64)
        extension = torch.tensor([2.0], device=device, dtype=torch.float64)
        self.gradcheck(unproject_points_orthographic, (points, extension))

    def test_gradcheck(self, device) -> None:
        self._test_gradcheck_project(device)
        self._test_gradcheck_unproject(device)

    def _test_jit_project(self, device, dtype) -> None:
        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        op_script = torch.jit.script(project_points_orthographic)
        actual = op_script(points)
        expected = project_points_orthographic(points)
        self.assert_close(actual, expected)

    def _test_jit_unproject(self, device, dtype) -> None:
        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        extension = torch.tensor([2.0], device=device, dtype=dtype)
        op_script = torch.jit.script(unproject_points_orthographic)
        actual = op_script(points, extension)
        expected = unproject_points_orthographic(points, extension)
        self.assert_close(actual, expected)

    def test_jit(self, device, dtype) -> None:
        self._test_jit_project(device, dtype)
        self._test_jit_unproject(device, dtype)
