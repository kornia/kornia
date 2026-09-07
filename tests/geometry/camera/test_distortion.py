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
        # Convention pin: distort_points_affine consumes points on the z = 1
        # NORMALIZED plane and a flat ``[fx, fy, cx, cy]`` parameter vector -- it is the pinhole projection, not a
        # pixel-to-pixel map. (0.5, 0.25) with fx = fy = 100, cx = 4, cy = 3 therefore lands on
        # u = 100 * 0.5 + 4 = 54, v = 100 * 0.25 + 3 = 28. cx != cy and x != y, so a transposed reading of either
        # the point or the parameter vector changes both literals; the batched row uses a second, differently
        # scaled camera (fx = 200, fy = 50, cx = 6, cy = 2) so a broadcast of element 0 fails too.
        # Snippet used to generate expected: distort_points_affine(tensor([0.5, 0.25]), tensor([100., 100., 4., 3.]))
        # executed 2026-09-06 at c0b50ad7 (torch 2.14.0, cpu and mps, every dtype) -> [54., 28.].
        points = torch.tensor([0.5, 0.25], device=device, dtype=dtype)
        params = torch.tensor([100.0, 100.0, 4.0, 3.0], device=device, dtype=dtype)
        self.assert_close(distort_points_affine(points, params), torch.tensor([54.0, 28.0], device=device, dtype=dtype))
        batched_points = torch.tensor([[0.5, 0.25], [1.0, 1.0]], device=device, dtype=dtype)
        batched_params = torch.tensor([[100.0, 100.0, 4.0, 3.0], [200.0, 50.0, 6.0, 2.0]], device=device, dtype=dtype)
        self.assert_close(
            distort_points_affine(batched_points, batched_params),
            torch.tensor([[54.0, 28.0], [206.0, 52.0]], device=device, dtype=dtype),
        )

    def test_convention_undistort_points_affine_is_the_exact_inverse(self, device, dtype):
        # Convention pin: undistort_points_affine((u, v), params) is the exact inverse of
        # distort_points_affine -- pixel in, normalized z = 1 point out -- and the round trip recovers the input
        # bit for bit on this asymmetric camera (atol = rtol = 0). The pair is a genuine forward/inverse duo; the
        # Kannala-Brandt twin below is iterative and closes only to a tolerance.
        # Snippet used to generate expected: torch.equal(undistort_points_affine(distort_points_affine(p, par), par), p)
        # executed 2026-09-06 at c0b50ad7 (torch 2.14.0) -> True on cpu for float32/float64/float16/
        # bfloat16 and on mps for float32/float16; recovered value [0.5, 0.25].
        points = torch.tensor([0.5, 0.25], device=device, dtype=dtype)
        params = torch.tensor([100.0, 100.0, 4.0, 3.0], device=device, dtype=dtype)
        distorted = distort_points_affine(points, params)
        self.assert_close(undistort_points_affine(distorted, params), points, atol=0.0, rtol=0.0)

    def test_convention_dx_distort_points_affine_matches_autograd(self, device, dtype):
        # Convention pin: dx_distort_points_affine returns the (2, 2)
        # Jacobian of distort_points_affine with respect to the POINT (rows = output components, columns = input
        # components), byte-identical to torch.autograd.functional.jacobian, and constant in the point because the
        # map is affine -- diag(fx, fy) = diag(100, 100). This is the control probe for the Kannala-Brandt
        # Jacobian pinned in TestDistortionKannalaBrandt below, which is NOT the Jacobian of the function it
        # documents (kornia#4277).
        # Snippet used to generate expected: torch.equal(dx_distort_points_affine(p, par),
        # torch.autograd.functional.jacobian(lambda q: distort_points_affine(q, par), p)) executed 2026-09-06 on
        # c0b50ad7 (torch 2.14.0) -> True on cpu (float32/float64/float16/bfloat16) and on mps
        # (float32/float16); value [[100., 0.], [0., 100.]].
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
                [1191.5316162109375, 282.3212890625],
                [282.3212890625, 1615.0135498046875],
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

    def test_convention_undistort_points_kannala_brandt_round_trip_closes(self, device, dtype):
        # Convention pin: unlike the affine pair in
        # TestDistortionAffine, undistort_points_kannala_brandt is an ITERATIVE inverse (10 Gauss-Newton steps),
        # so the round trip closes to the dtype tolerance rather than bit for bit. The pin asserts closure at
        # assert_close's dtype tolerance and deliberately states no error bound; the executed residuals are
        # recorded, not enforced. The far-off-axis case is the sibling pin below.
        # Snippet used to generate expected: (undistort_points_kannala_brandt(distort_points_kannala_brandt(p, par),
        # par) - p).abs().max() executed 2026-09-06 at c0b50ad7 (torch 2.14.0), differenced in the
        # working dtype -> cpu float32 2.98e-08, float64 9.55e-09, float16 9.77e-04, bfloat16 1.95e-03;
        # mps float32 2.98e-08, float16 9.77e-04.
        params = torch.tensor([100.0, 100.0, 4.0, 3.0, 0.1, 0.01, 0.001, 0.0001], device=device, dtype=dtype)
        points = torch.tensor([0.5, 0.25], device=device, dtype=dtype)
        self.assert_close(
            undistort_points_kannala_brandt(distort_points_kannala_brandt(points, params), params), points
        )

    def test_convention_undistort_points_kannala_brandt_closes_far_off_axis(self, device, dtype):
        # Convention pin: the Gauss-Newton inverse still closes at a normalized radius of
        # 3 focal lengths, where the fish-eye polynomial is far from linear -- the fixed-point iteration in
        # kornia.geometry.calibration.undistort_points does NOT (kornia#4285). Closure is asserted at
        # assert_close's dtype tolerance, with no error bound.
        # Snippet used to generate expected: (undistort_points_kannala_brandt(distort_points_kannala_brandt(
        # tensor([3., 0.]), par), par) - tensor([3., 0.])).abs().max() executed 2026-09-06 at c0b50ad7
        # (torch 2.14.0), differenced in the working dtype -> cpu float32 1.91e-06, float64 2.03e-08,
        # float16 1.95e-03, bfloat16 6.25e-02; mps float32 1.91e-06, float16 1.95e-03.
        if dtype == torch.bfloat16:
            pytest.skip("bfloat16: the far-off-axis round trip closes only to 6.25e-02, outside the bfloat16 atol")
        params = torch.tensor([100.0, 100.0, 4.0, 3.0, 0.1, 0.01, 0.001, 0.0001], device=device, dtype=dtype)
        far = torch.tensor([3.0, 0.0], device=device, dtype=dtype)
        self.assert_close(undistort_points_kannala_brandt(distort_points_kannala_brandt(far, params), params), far)

    def test_convention_principal_point_undistorts_to_the_origin(self, device, dtype):
        # Convention pin: small constants that guard the Gauss-Newton denominator and final radial rescale mean the
        # principal point (cx, cy) undistorts to the EXACT origin instead of the 0/0 that the unguarded
        # arithmetic would give. The assertion is torch.equal, not assert_close, because the guard makes the
        # result exactly zero rather than nearly zero.
        # cx != cy in these params, so the principal point is off the diagonal and a cx/cy swap would move it.
        # The second, off-centre point is what keeps the pin from being frame-invariant: (54, 28) normalizes to
        # (0.5, 0.25) and undistorts to a value that changes under either swap -- the same params with cx and cy
        # exchanged give [0.5507686734199524, 0.2591852843761444] and with fy halved to 50 they give
        # [0.5658687353134155, 0.5658687353134155].
        # Snippet used to generate expected: undistort_points_kannala_brandt(tensor([4., 3.]), params) and the
        # same call on tensor([54., 28.]), executed 2026-09-06 at c0b50ad7 (torch 2.14.0) -> the
        # principal point gives exactly [0.0, 0.0] on cpu float64/float32/bfloat16 and on mps float32; the
        # off-centre point gives [0.5392647981643677, 0.26963239908218384] on cpu and mps float32. In float16
        # the final radial-rescale denominator's 1e-8 guard underflows and the principal point gives [nan, nan]
        # -- kornia#4308, pinned by
        # test_wart_principal_point_undistorts_to_nan_in_float16_4308 below.
        if dtype == torch.float16:
            pytest.skip("float16: the final radial-rescale guard 1e-8 underflows, making the origin 0/0 (#4308)")
        params = torch.tensor([100.0, 100.0, 4.0, 3.0, 0.1, 0.01, 0.001, 0.0001], device=device, dtype=dtype)
        principal_point = torch.tensor([4.0, 3.0], device=device, dtype=dtype)
        undistorted = undistort_points_kannala_brandt(principal_point, params)
        assert torch.equal(undistorted, torch.zeros(2, device=device, dtype=dtype))
        off_centre = torch.tensor([54.0, 28.0], device=device, dtype=dtype)
        self.assert_close(
            undistort_points_kannala_brandt(off_centre, params),
            torch.tensor([0.5392647981643677, 0.26963239908218384], device=device, dtype=dtype),
        )

    def test_wart_principal_point_undistorts_to_nan_in_float16_4308(self, device, dtype):
        # Wart pin for kornia#4308: in float16, the 1e-8 epsilon added to ``rth`` for the final radial rescale
        # underflows to zero (the smallest subnormal is about 5.96e-08). At the principal point, both
        # ``radius_undistorted`` and ``rth`` are zero, so ``mag = 0 / (0 + 0)`` is nan and propagates through
        # the final multiplication. The 1e-16 Newton-start clamp and 1e-12 Newton-step denominator also
        # underflow, but they are not the direct source of this result. The body runs in ``params.dtype``;
        # bfloat16 preserves these values' exponent range, so the wart is specific to float16.
        # Snippet used to generate expected: undistort_points_kannala_brandt(tensor([4., 3.], dtype=torch.
        # float16), params.half()) executed 2026-09-06 at c0b50ad7 (torch 2.14.0) -> [nan, nan] on
        # both cpu and mps. Float16 POINTS with float32
        # params return [0.0, 0.0], because the body casts the points to params.dtype first.
        # Pins the CURRENT behavior; NOT a contract; delete when #4308 is repaired.
        if dtype != torch.float16:
            pytest.skip("float16-only wart: the radial-rescale epsilon is representable in every other dtype")
        params = torch.tensor([100.0, 100.0, 4.0, 3.0, 0.1, 0.01, 0.001, 0.0001], device=device, dtype=dtype)
        principal_point = torch.tensor([4.0, 3.0], device=device, dtype=dtype)
        assert undistort_points_kannala_brandt(principal_point, params).isnan().all()
        wide_params = params.to(torch.float32)
        self.assert_close(
            undistort_points_kannala_brandt(principal_point, wide_params),
            torch.zeros(2, device=device, dtype=dtype),
        )

    def test_wart_dx_distort_points_kannala_brandt_disagrees_with_autograd_4277(self, device, dtype):
        # Wart pin for kornia#4277: the analytic Jacobian is
        # not the Jacobian of distort_points_kannala_brandt. torch.autograd.functional.jacobian and central finite
        # differences agree with each other and both disagree with the analytic matrix -- by 62.58 in absolute
        # terms on entries of order 1e2 at this point. The asymmetric focal lengths also make the analytic matrix
        # nonsymmetric, and its transpose still does not close the gap, so it is not an index-order slip. The
        # sibling dx_distort_points_affine (pinned in
        # TestDistortionAffine) matches autograd byte for byte.
        # The pre-existing test_dx_distort_points_kannala_brandt in this class asserts the same wrong analytic
        # family at a different input; the fix for
        # #4277 has to re-derive both its ``expected`` and the function's docstring example.
        # Snippet used to generate expected: dx_distort_points_kannala_brandt(tensor([0.5, 0.25]), params) executed
        # 2026-09-06 at c0b50ad7 (torch 2.14.0, cpu float32) with fx = 100 and fy = 50 ->
        # [[22.0633487701416, -35.7770881652832], [-17.8885440826416, 37.8644866943359]]; autograd at the same
        # input -> [[84.64064025878906, -4.488434791564941], [-2.24421739578247, 45.6866455078125]].
        # Pins the CURRENT value; NOT a contract; delete when #4277 is repaired.
        params = torch.tensor([100.0, 50.0, 4.0, 3.0, 0.1, 0.01, 0.001, 0.0001], device=device, dtype=dtype)
        points = torch.tensor([0.5, 0.25], device=device, dtype=dtype)
        analytic = dx_distort_points_kannala_brandt(points, params)
        self.assert_close(
            analytic,
            torch.tensor(
                [[22.0633487701416, -35.7770881652832], [-17.8885440826416, 37.8644866943359]],
                device=device,
                dtype=dtype,
            ),
        )
        autograd = torch.autograd.functional.jacobian(lambda q: distort_points_kannala_brandt(q, params), points)
        assert not torch.allclose(analytic.float(), analytic.transpose(-1, -2).float())
        assert not torch.allclose(analytic.float(), autograd.float())
        assert not torch.allclose(analytic.transpose(-1, -2).float(), autograd.float())

    @pytest.mark.xfail(strict=True, reason="kornia#4277: the analytic Kannala-Brandt Jacobian disagrees with autograd")
    def test_convention_dx_distort_points_kannala_brandt_matches_autograd_4277(self, device, dtype):
        # Intended contract, asserted as a strict xfail so the repair makes it XPASS and forces this mark out:
        # dx_distort_points_kannala_brandt returns the Jacobian of the function it is named after, exactly as
        # dx_distort_points_affine and dx_project_points_z1 already do.
        # Settled by #4277's own Expected section ("re-derived so it matches autograd and central differences").
        params = torch.tensor([100.0, 100.0, 4.0, 3.0, 0.1, 0.01, 0.001, 0.0001], device=device, dtype=dtype)
        points = torch.tensor([0.5, 0.25], device=device, dtype=dtype)
        autograd = torch.autograd.functional.jacobian(lambda q: distort_points_kannala_brandt(q, params), points)
        self.assert_close(dx_distort_points_kannala_brandt(points, params), autograd)
