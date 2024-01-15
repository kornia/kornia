import pytest
import torch

from kornia.geometry.camera.distortion_affine import (
    distort_points_affine,
    dx_distort_points_affine,
    undistort_points_affine,
)
from kornia.geometry.camera.distortion_kannala_brandt import (
    distort_points_kannala_brandt,
    dx_distort_points_kannala_brandt,
    undistort_points_kannala_brandt,
)
from kornia.testing import BaseTester, tensor_to_gradcheck_var


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
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 400.0],
                [320.0, 240.0],
                [319.5, 239.5],
                [100.0, 40.0],
                [639.0, 479.0],
            ],
            device=device,
            dtype=dtype,
        )
        params = torch.tensor([[600.0, 600.0, 319.5, 239.5]], device=device, dtype=dtype)
        expected = torch.tensor(
            [
                [319.5, 239.5],
                [919.5, 240239.5],
                [192319.5, 144239.5],
                [192019.5, 143939.5],
                [60319.5, 24239.5],
                [383719.5, 287639.5],
            ],
            device=device,
            dtype=dtype,
        )
        points_distorted = distort_points_affine(points, params)
        self.assert_close(points_distorted, expected)
        self.assert_close(points, undistort_points_affine(points_distorted, params))

    def test_undistort_points_affine(self, device, dtype):
        points = torch.tensor([601.0, 602.0], device=device, dtype=dtype)
        params = torch.tensor([600.0, 600.0, 319.5, 239.5], device=device, dtype=dtype)
        expected = torch.tensor([0.4691666, 0.6041666], device=device, dtype=dtype)
        self.assert_close(undistort_points_affine(points, params), expected)

    def test_undistort_points_affine_batch(self, device, dtype):
        points = torch.tensor([[601.0, 602.0], [1203.0, 1204.0]], device=device, dtype=dtype)
        params = torch.tensor(
            [[600.0, 600.0, 319.5, 239.5], [600.0, 600.0, 319.5, 239.5]],
            device=device,
            dtype=dtype,
        )
        expected = torch.tensor([[0.46916666, 0.60416666], [1.4725, 1.6075]], device=device, dtype=dtype)
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
    def test_module(self, device, dtype) -> None:
        pass


class TestDistortionKannalaBrandt(BaseTester):
    def test_smoke(self, device, dtype) -> None:
        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        params = torch.tensor([600.0, 600.0, 319.5, 239.5, 0.1, 0.2, 0.3, 0.4], device=device, dtype=dtype)
        assert distort_points_kannala_brandt(points, params) is not None

    def _test_cardinality_distort_batch(self, device, dtype, batch_size):
        batch_tuple = (batch_size,) if batch_size is not None else ()
        points = torch.rand(batch_tuple + (2,), device=device, dtype=dtype)
        params = torch.rand(batch_tuple + (8,), device=device, dtype=dtype)
        assert distort_points_kannala_brandt(points, params).shape == batch_tuple + (2,)

    def _test_cardinality_undistort_batch(self, device, dtype, batch_size):
        batch_tuple = (batch_size,) if batch_size is not None else ()
        points = torch.rand(batch_tuple + (2,), device=device, dtype=dtype)
        params = torch.rand(batch_tuple + (8,), device=device, dtype=dtype)
        assert undistort_points_kannala_brandt(points, params).shape == batch_tuple + (2,)

    @pytest.mark.parametrize("batch_size", [None, 1, 2, 3])
    def test_cardinality(self, device, dtype, batch_size):
        self._test_cardinality_distort_batch(device, dtype, batch_size)
        self._test_cardinality_undistort_batch(device, dtype, batch_size)

    def test_distort_points_kannala_brandt(self, device, dtype) -> None:
        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        params = torch.tensor([600.0, 600.0, 319.5, 239.5, 0.1, 0.2, 0.3, 0.4], device=device, dtype=dtype)
        expected = torch.tensor([1369.8710, 2340.2419], device=device, dtype=dtype)
        self.assert_close(distort_points_kannala_brandt(points, params), expected)

    def test_distort_points_kannala_brandt_batch(self, device, dtype) -> None:
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 400.0],
                [320.0, 240.0],
                [319.5, 239.5],
                [100.0, 40.0],
                [639.0, 479.0],
            ],
            device=device,
            dtype=dtype,
        )
        params = torch.tensor(
            [[1000.0, 1000.0, 320.0, 280.0, 0.1, 0.01, 0.001, 0.0001]],
            device=device,
            dtype=dtype,
        )
        expected = torch.tensor(
            [
                [320.0, 280.0],
                [325.1949172763466, 2357.966910538644],
                [1982.378709731326, 1526.7840322984944],
                [1982.6832644475849, 1526.3619462760455],
                [2235.6822069661744, 1046.2728827864696],
                [1984.8663275417607, 1527.9983895031353],
            ],
            device=device,
            dtype=dtype,
        )
        points_distorted = distort_points_kannala_brandt(points, params)
        self.assert_close(points_distorted, expected)
        # TODO: investigate why this fails
        # self.assert_close(points, undistort_points_kannala_brandt(points_distorted, params))

    def test_undistort_points_kannala_brandt(self, device, dtype) -> None:
        points = torch.tensor([919.5000, 1439.5000], device=device, dtype=dtype)
        params = torch.tensor([600.0, 600.0, 319.5, 239.5, 0.1, 0.2, 0.3, 0.4], device=device, dtype=dtype)
        expected = torch.tensor([0.18564432460876581, 0.37128864921753163], device=device, dtype=dtype)
        self.assert_close(undistort_points_kannala_brandt(points, params), expected)

    def test_undistort_points_kannala_brandt_batch(self, device, dtype) -> None:
        points = torch.tensor([[919.5000, 1439.5000], [2119.5000, 2639.5000]], device=device, dtype=dtype)
        params = torch.tensor(
            [
                [600.0, 600.0, 319.5, 239.5, 0.1, 0.2, 0.3, 0.4],
                [600.0, 600.0, 319.5, 239.5, 0.1, 0.2, 0.3, 0.4],
            ],
            device=device,
            dtype=dtype,
        )
        expected = torch.tensor(
            [
                [0.18564432460876581, 0.37128864921753163],
                [0.27051682368562285, 0.3606890982474972],
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(undistort_points_kannala_brandt(points, params), expected)

    def test_dx_distort_points_kannala_brandt(self, device, dtype) -> None:
        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        params = torch.tensor([600.0, 600.0, 319.5, 239.5, 0.1, 0.2, 0.3, 0.4], device=device, dtype=dtype)
        expected = torch.tensor([[1191.5316, 282.3213], [282.3213, 1615.0135]], device=device, dtype=dtype)
        self.assert_close(dx_distort_points_kannala_brandt(points, params), expected)

    def test_exception(self, device, dtype) -> None:
        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        params = torch.tensor([600.0, 600.0, 319.5], device=device, dtype=dtype)
        with pytest.raises(TypeError):
            distort_points_kannala_brandt(points, params)

    def _test_gradcheck_distort(self, device):
        points = torch.tensor([1.0, 2.0], device=device, dtype=torch.float64)
        points = tensor_to_gradcheck_var(points)
        params = torch.tensor(
            [600.0, 600.0, 319.5, 239.5, 0.1, 0.2, 0.3, 0.4],
            device=device,
            dtype=torch.float64,
        )
        params = tensor_to_gradcheck_var(params)
        self.gradcheck(distort_points_kannala_brandt, (points, params))

    def _test_gradcheck_undistort(self, device):
        points = torch.tensor([919.5000, 1439.5000], device=device, dtype=torch.float64)
        points = tensor_to_gradcheck_var(points)
        params = torch.tensor(
            [600.0, 600.0, 319.5, 239.5, 0.1, 0.2, 0.3, 0.4],
            device=device,
            dtype=torch.float64,
        )
        params = tensor_to_gradcheck_var(params)
        self.gradcheck(undistort_points_kannala_brandt, (points, params))

    def test_gradcheck(self, device) -> None:
        self._test_gradcheck_distort(device)
        self._test_gradcheck_undistort(device)

    def _test_jit_distort(self, device, dtype) -> None:
        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        params = torch.tensor([600.0, 600.0, 319.5, 239.5, 0.1, 0.2, 0.3, 0.4], device=device, dtype=dtype)
        op_script = torch.jit.script(distort_points_kannala_brandt)
        actual = op_script(points, params)
        expected = distort_points_kannala_brandt(points, params)
        self.assert_close(actual, expected)

    def _test_jit_undistort(self, device, dtype) -> None:
        points = torch.tensor([919.5000, 1439.5000], device=device, dtype=dtype)
        params = torch.tensor([600.0, 600.0, 319.5, 239.5, 0.1, 0.2, 0.3, 0.4], device=device, dtype=dtype)
        op_script = torch.jit.script(undistort_points_kannala_brandt)
        actual = op_script(points, params)
        expected = undistort_points_kannala_brandt(points, params)
        self.assert_close(actual, expected)

    def test_jit(self, device, dtype) -> None:
        self._test_jit_distort(device, dtype)
        self._test_jit_undistort(device, dtype)

    @pytest.mark.skip(reason="Unnecessary test")
    def test_module(self, device, dtype) -> None:
        pass
