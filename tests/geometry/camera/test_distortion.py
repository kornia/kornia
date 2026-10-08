# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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

from testing.base import BaseTester


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

    # NOTE: data generated with sophus-rs
    def test_distort_points_roundtrip(self, device, dtype):
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

    def test_dx_distort_points(self, device, dtype):
        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        params = torch.tensor([600.0, 600.0, 319.5, 239.5], device=device, dtype=dtype)
        expected = torch.tensor([[600.0, 0.0], [0.0, 600.0]], device=device, dtype=dtype)
        self.assert_close(dx_distort_points_affine(points, params), expected)

    def test_exception(self, device, dtype) -> None:
        from kornia.core.exceptions import ShapeError

        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        params = torch.tensor([600.0, 600.0, 319.5], device=device, dtype=dtype)
        with pytest.raises(ShapeError):
            distort_points_affine(points, params)

    def _test_gradcheck_distort(self, device):
        points = torch.tensor([1.0, 2.0], device=device, dtype=torch.float64)
        params = torch.tensor([600.0, 600.0, 319.5, 239.5], device=device, dtype=torch.float64)
        self.gradcheck(distort_points_affine, (points, params))

    def _test_gradcheck_undistort(self, device):
        points = torch.tensor([601.0, 602.0], device=device, dtype=torch.float64)
        params = torch.tensor([600.0, 600.0, 319.5, 239.5], device=device, dtype=torch.float64)
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

    def test_convention_distort_points_affine_takes_normalized_z1_points(self, device, dtype):
        # Input is a normalized z = 1 point and a flat [fx, fy, cx, cy]: (0.5, 0.25) -> (100*0.5 + 4, 100*0.25 + 3)
        # = (54, 28). The second batch row uses a different camera so a broadcast of element 0 fails.
        points = torch.tensor([0.5, 0.25], device=device, dtype=dtype)
        params = torch.tensor([100.0, 100.0, 4.0, 3.0], device=device, dtype=dtype)
        self.assert_close(distort_points_affine(points, params), torch.tensor([54.0, 28.0], device=device, dtype=dtype))
        batched_points = torch.tensor([[0.5, 0.25], [1.0, 1.0]], device=device, dtype=dtype)
        batched_params = torch.tensor([[100.0, 100.0, 4.0, 3.0], [200.0, 50.0, 6.0, 2.0]], device=device, dtype=dtype)
        self.assert_close(
            distort_points_affine(batched_points, batched_params),
            torch.tensor([[54.0, 28.0], [206.0, 52.0]], device=device, dtype=dtype),
        )

    def test_convention_undistort_points_affine_is_the_closed_form_inverse(self, device, dtype):
        # undistort_points_affine (pixel in, normalized z = 1 point out) inverts distort_points_affine in closed
        # form, with no iteration; the Kannala-Brandt pair below is iterative.
        points = torch.tensor([0.5, 0.25], device=device, dtype=dtype)
        params = torch.tensor([100.0, 100.0, 4.0, 3.0], device=device, dtype=dtype)
        distorted = distort_points_affine(points, params)
        self.assert_close(undistort_points_affine(distorted, params), points)

    def test_convention_dx_distort_points_affine_matches_autograd(self, device, dtype):
        # The (2, 2) Jacobian with respect to the point (rows = outputs, columns = inputs) is diag(fx, fy) and
        # matches autograd.
        points = torch.tensor([0.5, 0.25], device=device, dtype=dtype)
        params = torch.tensor([100.0, 100.0, 4.0, 3.0], device=device, dtype=dtype)
        analytic = dx_distort_points_affine(points, params)
        autograd = torch.autograd.functional.jacobian(lambda q: distort_points_affine(q, params), points)
        self.assert_close(analytic, autograd, atol=0.0, rtol=0.0)
        self.assert_close(
            analytic, torch.tensor([[100.0, 0.0], [0.0, 100.0]], device=device, dtype=dtype), atol=0.0, rtol=0.0
        )


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

    # NOTE: data generated with sophus-rs
    def test_distort_points_roundtrip(self, device, dtype) -> None:
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
        self.assert_close(points, undistort_points_kannala_brandt(points_distorted, params))

    def test_dx_distort_points_kannala_brandt(self, device, dtype) -> None:
        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        params = torch.tensor([600.0, 600.0, 319.5, 239.5, 0.1, 0.2, 0.3, 0.4], device=device, dtype=dtype)
        expected = torch.tensor(
            [
                [1221.1801242852937, 341.6185175810278],
                [341.6185175810278, 1733.6079006568352],
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(dx_distort_points_kannala_brandt(points, params), expected)

    def test_exception(self, device, dtype) -> None:
        from kornia.core.exceptions import ShapeError

        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        params = torch.tensor([600.0, 600.0, 319.5], device=device, dtype=dtype)
        with pytest.raises(ShapeError):
            distort_points_kannala_brandt(points, params)

    def _test_gradcheck_distort(self, device):
        points = torch.tensor([1.0, 2.0], device=device, dtype=torch.float64)
        params = torch.tensor(
            [600.0, 600.0, 319.5, 239.5, 0.1, 0.2, 0.3, 0.4],
            device=device,
            dtype=torch.float64,
        )
        self.gradcheck(distort_points_kannala_brandt, (points, params))

    def _test_gradcheck_undistort(self, device):
        points = torch.tensor([919.5000, 1439.5000], device=device, dtype=torch.float64)
        params = torch.tensor(
            [600.0, 600.0, 319.5, 239.5, 0.1, 0.2, 0.3, 0.4],
            device=device,
            dtype=torch.float64,
        )
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

    def test_convention_undistort_points_kannala_brandt_round_trip_closes_4308(self, device, dtype):
        # undistort_points_kannala_brandt is an iterative (Gauss-Newton) inverse, so the round trip closes to the
        # dtype tolerance, including at a normalized radius of 3 where the polynomial is far from linear (the
        # fixed-point undistort_points does not, #4285). In float64 nonzero radii are rescaled by r itself, so the
        # residual reaches the rounding floor rather than an additive-epsilon bias of ~1e-8 (#4308).
        params = torch.tensor([100.0, 100.0, 4.0, 3.0, 0.1, 0.01, 0.001, 0.0001], device=device, dtype=dtype)
        points = torch.tensor([[0.5, 0.25], [3.0, 0.0], [0.01, 0.0]], device=device, dtype=dtype)
        if dtype == torch.bfloat16:
            points = points[[0, 2]]  # the far-off-axis point closes only to about 6e-02 in bfloat16
        recovered = undistort_points_kannala_brandt(distort_points_kannala_brandt(points, params), params)
        self.assert_close(recovered, points)
        if dtype == torch.float64:
            assert (recovered - points).abs().max().item() < 1e-12

    def test_convention_principal_point_undistorts_to_the_origin(self, device, dtype):
        # A zero distorted radius is masked, so the principal point undistorts to the exact origin, also with
        # float16 points and float32 params (the result keeps the points dtype). The off-centre point changes
        # under a cx/cy or fx/fy swap, so the pin is not frame-invariant.
        params = torch.tensor([100.0, 100.0, 4.0, 3.0, 0.1, 0.01, 0.001, 0.0001], device=device, dtype=dtype)
        principal_point = torch.tensor([4.0, 3.0], device=device, dtype=dtype)
        undistorted = undistort_points_kannala_brandt(principal_point, params)
        assert torch.equal(undistorted, torch.zeros(2, device=device, dtype=dtype))
        if dtype == torch.float16:
            wide_params = params.to(torch.float32)
            mixed = undistort_points_kannala_brandt(principal_point, wide_params)
            assert mixed.dtype == dtype
            assert torch.equal(mixed, torch.zeros(2, device=device, dtype=dtype))
        off_centre = torch.tensor([54.0, 28.0], device=device, dtype=dtype)
        self.assert_close(
            undistort_points_kannala_brandt(off_centre, params),
            torch.tensor([0.5392647981643677, 0.26963239908218384], device=device, dtype=dtype),
        )

    def test_convention_nearest_float16_point_is_not_collapsed_4308(self, device, dtype):
        if device.type != "cpu" or dtype != torch.float16:
            pytest.skip("CPU float16 near-origin regression")
        params = torch.tensor([100.0, 50.0, 4.0, 3.0, 0.1, 0.01, 0.001, 0.0001], device=device, dtype=dtype)
        point = torch.tensor([4.00390625, 3.0], device=device, dtype=dtype)
        undistorted = undistort_points_kannala_brandt(point, params)
        expected_x = (point[0] - params[2]) / params[0]
        assert undistorted[0] != 0
        self.assert_close(undistorted, torch.stack([expected_x, torch.zeros_like(expected_x)]), atol=6e-8, rtol=0.0)

    def test_convention_principal_point_has_finite_gradients_4308(self, device, dtype):
        params = torch.tensor([100.0, 50.0, 4.0, 3.0, 0.1, 0.01, 0.001, 0.0001], device=device, dtype=dtype)
        principal_point = torch.tensor([4.0, 3.0], device=device, dtype=dtype)
        point_jacobian = torch.autograd.functional.jacobian(
            lambda point: undistort_points_kannala_brandt(point, params), principal_point
        )
        params_jacobian = torch.autograd.functional.jacobian(
            lambda camera: undistort_points_kannala_brandt(principal_point, camera), params
        )
        expected_point_jacobian = torch.tensor(
            [[1.0 / params[0], 0.0], [0.0, 1.0 / params[1]]], device=device, dtype=dtype
        )
        expected_params_jacobian = torch.zeros(2, 8, device=device, dtype=dtype)
        expected_params_jacobian[0, 2] = -1.0 / params[0]
        expected_params_jacobian[1, 3] = -1.0 / params[1]
        self.assert_close(point_jacobian, expected_point_jacobian)
        self.assert_close(params_jacobian, expected_params_jacobian)

    def test_dx_distort_points_kannala_brandt_affine_branch(self, device, dtype) -> None:
        # The forward distortion switches to the affine model when radius_sq <= 1e-8.
        # Its Jacobian must therefore be diag(fx, fy), including exactly at the origin.
        points = torch.tensor([[0.0, 0.0], [1e-5, 0.0]], device=device, dtype=dtype)
        params = torch.tensor([100.0, 50.0, 4.0, 3.0, 0.1, 0.01, 0.001, 0.0001], device=device, dtype=dtype)

        expected_single = torch.tensor([[100.0, 0.0], [0.0, 50.0]], device=device, dtype=dtype)
        expected = torch.stack([expected_single, expected_single])

        self.assert_close(
            dx_distort_points_kannala_brandt(points, params),
            expected,
            atol=0.0,
            rtol=0.0,
        )

    def test_convention_dx_distort_points_kannala_brandt_matches_autograd_4277(self, device, dtype):
        # Regression for #4277: the analytic Kannala-Brandt Jacobian must match
        # the derivative of distort_points_kannala_brandt with respect to the point.
        # Asymmetric focal lengths ensure an fx/fy or row/column swap cannot hide.
        params = torch.tensor(
            [100.0, 50.0, 4.0, 3.0, 0.1, 0.01, 0.001, 0.0001],
            device=device,
            dtype=dtype,
        )
        points = torch.tensor([0.5, 0.25], device=device, dtype=dtype)

        analytic = dx_distort_points_kannala_brandt(points, params)

        reference_dtype = dtype
        if dtype in (torch.float16, torch.bfloat16):
            reference_dtype = torch.float32

        reference_points = points.to(reference_dtype)
        reference_params = params.to(reference_dtype)

        autograd = torch.autograd.functional.jacobian(
            lambda q: distort_points_kannala_brandt(q, reference_params),
            reference_points,
        ).to(dtype)

        self.assert_close(analytic, autograd)
