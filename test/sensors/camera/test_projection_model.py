import pytest
import torch

from kornia.core import Device, Dtype
from kornia.geometry.vector import Vector2, Vector3
from kornia.sensors.camera.distortion_affine import (
    distort_points_affine,
    dx_distort_points_affine,
    undistort_points_affine,
)
from kornia.sensors.camera.projection_model import Z1Projection
from kornia.sensors.camera.projection_orthographic import (
    dx_project_points_orthographic,
    project_points_orthographic,
    unproject_points_orthographic,
)
from kornia.sensors.camera.projection_z1 import (
    dx_project_points_z1,
    project_points_z1,
    unproject_points_z1,
)
from kornia.testing import BaseTester, tensor_to_gradcheck_var


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
            [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [6.0, 6.0, 2.0], [9.0, 9.0, 3.0]],
            device=device,
            dtype=dtype,
        )
        expected = torch.tensor([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0], [3.0, 3.0]], device=device, dtype=dtype)
        self.assert_close(projection.project(Vector3(points)).data, expected)

    def test_unproject(self, device, dtype):
        projection = Z1Projection()
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0], [3.0, 3.0]], device=device, dtype=dtype)
        expected = torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [6.0, 6.0, 2.0], [9.0, 9.0, 3.0]],
            device=device,
            dtype=dtype,
        )
        self.assert_close(
            projection.unproject(
                Vector2(points),
                torch.tensor([1.0, 1.0, 2.0, 3.0], device=device, dtype=dtype),
            ).data,
            expected,
        )


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
            [
                [0.3333333432674408, 0.0, -0.1111111119389534],
                [0.0, 0.3333333432674408, -0.2222222238779068],
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(dx_project_points_z1(points), expected)

    def test_exception(self, device: Device, dtype: Dtype) -> None:
        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        extension = torch.tensor([2.0], device=device, dtype=dtype)
        with pytest.raises(TypeError):
            unproject_points_z1(points, extension)

    def _test_gradcheck_unproject(self, device):
        points = torch.tensor([1.0, 2.0], device=device, dtype=torch.float64)
        points = tensor_to_gradcheck_var(points)
        extension = torch.tensor([2.0], device=device, dtype=torch.float64)
        extension = tensor_to_gradcheck_var(extension)
        self.gradcheck(unproject_points_z1, (points, extension))

    def _test_gradcheck_project(self, device):
        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=torch.float64)
        points = tensor_to_gradcheck_var(points)
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

    @pytest.mark.skip(reason="Unnecessary test")
    def test_module(self, device: Device, dtype: Dtype) -> None:
        pass


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
        points = tensor_to_gradcheck_var(points)
        self.gradcheck(project_points_orthographic, (points,))

    def _test_gradcheck_unproject(self, device):
        points = torch.tensor([1.0, 2.0], device=device, dtype=torch.float64)
        points = tensor_to_gradcheck_var(points)
        extension = torch.tensor([2.0], device=device, dtype=torch.float64)
        extension = tensor_to_gradcheck_var(extension)
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

    @pytest.mark.skip(reason="Unnecessary test")
    def test_module(self, device: Device, dtype: Dtype) -> None:
        pass


class TestDistortionAffine(BaseTester):
    def test_smoke(self, device, dtype):
        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        params = torch.tensor([600.0, 600.0, 319.5, 239.5], device=device, dtype=dtype)
        assert distort_points_affine(points, params) is not None

    def _test_cardinality_distort_batch(self, device, dtype, batch_size):
        batch_tuple = (batch_size,) if batch_size is not None else ()
        points = torch.rand(batch_tuple + (2,), device=device, dtype=dtype)
        params = torch.rand(batch_tuple + (4,), device=device, dtype=dtype)
        assert distort_points_affine(points, params).shape == batch_tuple + (2,)

    def _test_cardinality_undistort_batch(self, device, dtype, batch_size):
        batch_tuple = (batch_size,) if batch_size is not None else ()
        points = torch.rand(batch_tuple + (2,), device=device, dtype=dtype)
        params = torch.rand(batch_tuple + (4,), device=device, dtype=dtype)
        assert undistort_points_affine(points, params).shape == batch_tuple + (2,)

    @pytest.mark.parametrize("batch_size", [None, 1, 2, 3])
    def test_cardinality(self, device, dtype, batch_size):
        self._test_cardinality_distort_batch(device, dtype, batch_size)
        self._test_cardinality_undistort_batch(device, dtype, batch_size)

    def test_distort_points_affine(self, device, dtype):
        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        params = torch.tensor([600.0, 600.0, 319.5, 239.5], device=device, dtype=dtype)
        expected = torch.tensor([919.5000, 1439.5000], device=device, dtype=dtype)
        self.assert_close(distort_points_affine(points, params), expected)

    def test_distort_points_affine_batch(self, device, dtype):
        points = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=dtype)
        params = torch.tensor([[600.0, 600.0, 319.5, 239.5], [600.0, 600.0, 319.5, 239.5]], device=device, dtype=dtype)
        expected = torch.tensor([[919.5000, 1439.5000], [2119.5000, 2639.5000]], device=device, dtype=dtype)
        self.assert_close(distort_points_affine(points, params), expected)

    def test_undistort_points_affine(self, device, dtype):
        points = torch.tensor([601.0, 602.0], device=device, dtype=dtype)
        params = torch.tensor([600.0, 600.0, 319.5, 239.5], device=device, dtype=dtype)
        expected = torch.tensor([0.4692, 0.6042], device=device, dtype=dtype)
        self.assert_close(undistort_points_affine(points, params), expected)

    def test_undistort_points_affine_batch(self, device, dtype):
        points = torch.tensor([[601.0, 602.0], [1203.0, 1204.0]], device=device, dtype=dtype)
        params = torch.tensor([[600.0, 600.0, 319.5, 239.5], [600.0, 600.0, 319.5, 239.5]], device=device, dtype=dtype)
        expected = torch.tensor([[0.4692, 0.6042], [1.4725, 1.6075]], device=device, dtype=dtype)
        self.assert_close(undistort_points_affine(points, params), expected)

    def test_dx_distort_points_affine(self, device, dtype):
        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        params = torch.tensor([600.0, 600.0, 319.5, 239.5], device=device, dtype=dtype)
        expected = torch.tensor([[600.0, 0.0], [0.0, 600.0]], device=device, dtype=dtype)
        self.assert_close(dx_distort_points_affine(points, params), expected)

    def test_exception(self, device, dtype) -> None:
        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        params = torch.tensor([600.0, 600.0, 319.5], device=device, dtype=dtype)
        with pytest.raises(TypeError):
            distort_points_affine(points, params)

    def _test_gradcheck_distort(self, device):
        points = torch.tensor([1.0, 2.0], device=device, dtype=torch.float64)
        points = tensor_to_gradcheck_var(points)
        params = torch.tensor([600.0, 600.0, 319.5, 239.5], device=device, dtype=torch.float64)
        params = tensor_to_gradcheck_var(params)
        self.gradcheck(distort_points_affine, (points, params))

    def _test_gradcheck_undistort(self, device):
        points = torch.tensor([601.0, 602.0], device=device, dtype=torch.float64)
        points = tensor_to_gradcheck_var(points)
        params = torch.tensor([600.0, 600.0, 319.5, 239.5], device=device, dtype=torch.float64)
        params = tensor_to_gradcheck_var(params)
        self.gradcheck(undistort_points_affine, (points, params))

    def test_gradcheck(self, device) -> None:
        self._test_gradcheck_distort(device)
        self._test_gradcheck_undistort(device)

    def _test_jit_distort(self, device, dtype) -> None:
        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        params = torch.tensor([600.0, 600.0, 319.5, 239.5], device=device, dtype=dtype)
        op_script = torch.jit.script(distort_points_affine)
        actual = op_script(points, params)
        expected = distort_points_affine(points, params)
        self.assert_close(actual, expected)

    def _test_jit_undistort(self, device, dtype) -> None:
        points = torch.tensor([601.0, 602.0], device=device, dtype=dtype)
        params = torch.tensor([600.0, 600.0, 319.5, 239.5], device=device, dtype=dtype)
        op_script = torch.jit.script(undistort_points_affine)
        actual = op_script(points, params)
        expected = undistort_points_affine(points, params)
        self.assert_close(actual, expected)

    def test_jit(self, device, dtype) -> None:
        self._test_jit_distort(device, dtype)
        self._test_jit_undistort(device, dtype)

    @pytest.mark.skip(reason="Unnecessary test")
    def test_module(self, device: Device, dtype: Dtype) -> None:
        pass
