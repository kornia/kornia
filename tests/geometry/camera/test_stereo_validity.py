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

from kornia.geometry.camera.stereo import StereoCamera, reproject_disparity_to_3D
from kornia.geometry.grid import create_meshgrid
from kornia.geometry.linalg import transform_points

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, dynamo_is_available


def _camera(device, dtype):
    left = torch.tensor([[[100, 0, 4, 0], [0, 100, 3, 0], [0, 0, 1, 0]]], device=device, dtype=dtype)
    right = left.clone()
    right[..., 0, 3] = -50
    return StereoCamera(left, right)


def _legacy(disparity, q):
    # The pre-mask implementation delegates the homogeneous divide and final cast
    # to transform_points. Keep that independent entry point as a compatibility oracle.
    b, h, w, _ = disparity.shape
    uv = create_meshgrid(h, w, normalized_coordinates=False, device=disparity.device, dtype=disparity.dtype)
    uvd = torch.cat((uv.expand(b, -1, -1, -1), disparity), -1).reshape(b, h * w, 3)
    return transform_points(q, uvd).reshape(b, h, w, 3)


class _Reproject(torch.nn.Module):
    def __init__(self, return_mask):
        super().__init__()
        self.return_mask = return_mask

    def forward(self, disparity, q):
        return reproject_disparity_to_3D(disparity, q, return_mask=self.return_mask)


class TestStereoReconstructionValidity(BaseTester):
    def test_both_entry_points_and_q_sign(self, device, dtype):
        camera = _camera(device, dtype)
        disparity = torch.tensor([0, 10, -10], device=device, dtype=dtype).reshape(1, 1, 3, 1)
        points, valid = camera.reproject_disparity_to_3D(disparity, return_mask=True)
        function_points, function_valid = reproject_disparity_to_3D(disparity, camera.Q, return_mask=True)
        assert valid.shape == disparity.shape
        assert valid.dtype == torch.bool
        assert valid.device == device
        assert valid.tolist() == [[[[False], [True], [True]]]]
        assert torch.equal(points, camera.reproject_disparity_to_3D(disparity))
        assert torch.equal(points, function_points)
        assert torch.equal(valid, function_valid)
        opposite_points, opposite_valid = reproject_disparity_to_3D(disparity, -camera.Q, return_mask=True)
        assert torch.equal(valid, opposite_valid)
        assert torch.equal(points[..., 1:, :], opposite_points[..., 1:, :])
        assert torch.equal(points[..., :1, :], -opposite_points[..., :1, :])
        self.assert_close(points[0, 0, 0], disparity.new_tensor([200, 150, -5000]))
        # A behind-camera point can be numerically reconstructed.
        assert points[0, 0, 2, 2] < 0

    def test_zero_disparity_with_offset_q(self, device, dtype):
        q = _camera(device, dtype).Q.clone()
        q[..., 3, 3] = -100
        points, valid = reproject_disparity_to_3D(
            torch.zeros(1, 1, 1, 1, device=device, dtype=dtype), q, return_mask=True
        )
        assert valid.all()
        self.assert_close(points.flatten(), q.new_tensor([-2, -1.5, 50]))

    def test_infinite_w_is_invalid_even_with_finite_xyz(self, device, dtype):
        q = _camera(device, dtype).Q.clone()
        q[..., 3, 3] = torch.inf
        d = torch.zeros(1, 1, 1, 1, device=device, dtype=dtype)
        points, valid = reproject_disparity_to_3D(d, q, return_mask=True)
        assert torch.isfinite(points).all()
        assert not valid.any()
        assert torch.equal(points, _legacy(d, q))

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
    @pytest.mark.parametrize("operand", ["disparity", "q"])
    def test_nonfinite_inputs(self, device, dtype, value, operand):
        q = _camera(device, dtype).Q.clone()
        d = torch.ones(1, 1, 2, 1, device=device, dtype=dtype)
        if operand == "disparity":
            d[0, 0, 0, 0] = value
        else:
            q[0, 0, 3] = value
        points, valid = reproject_disparity_to_3D(d, q, return_mask=True)
        assert not valid[0, 0, 0, 0]
        torch.testing.assert_close(points, _legacy(d, q), atol=0, rtol=0, equal_nan=True)

    @pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
    @pytest.mark.parametrize("d_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
    def test_mixed_dtypes_preserve_coordinates(self, device, q_dtype, d_dtype):
        if device.type == "mps" and torch.float64 in (q_dtype, d_dtype):
            pytest.skip("MPS does not support float64")
        q = _camera(device, q_dtype).Q
        d = torch.full((2, 2, 3, 1), 10, device=device, dtype=d_dtype)
        points, valid = reproject_disparity_to_3D(d, q, return_mask=True)
        assert points.dtype == d_dtype
        assert torch.equal(points, _legacy(d, q))
        assert valid.all()

    def test_final_cast_overflow(self, device):
        q = torch.eye(4, device=device, dtype=torch.float32)[None]
        q[..., 0, 3] = 1e5
        d = torch.zeros(1, 1, 1, 1, device=device, dtype=torch.float16)
        points, valid = reproject_disparity_to_3D(d, q, return_mask=True)
        assert torch.isinf(points[..., 0]).all()
        assert not valid.any()

    def test_cutoff_working_precision_and_scaling(self, device, dtype):
        # Test the converter's working dtype, including the float16 cutoff rounding
        # to zero. Choose numerator=W so the division is finite above the cutoff
        # except for reciprocal overflow, which must also make the mask false.
        cutoff = torch.tensor(1e-8, device=device, dtype=dtype)
        below = torch.nextafter(cutoff, cutoff.new_tensor(0.0))
        above = torch.nextafter(cutoff, cutoff.new_tensor(float("inf")))
        w = torch.stack((below, cutoff, above, -above))
        q = torch.zeros(4, 4, 4, device=device, dtype=dtype)
        q[:, 0, 3], q[:, 1, 3], q[:, 2, 3], q[:, 3, 3] = w, w, w, w
        d = torch.zeros(4, 1, 1, 1, device=device, dtype=dtype)
        points, valid = reproject_disparity_to_3D(d, q, return_mask=True)
        expected = (w.abs() > 1e-8) & torch.isfinite(_legacy(d, q)).all(dim=(1, 2, 3))
        assert torch.equal(valid.flatten(), expected)
        assert not valid[:2].any()
        torch.testing.assert_close(points, _legacy(d, q), atol=0, rtol=0, equal_nan=True)
        if dtype in (torch.float32, torch.float64):
            assert valid[2:].all()
            _, scaled_valid = reproject_disparity_to_3D(d, q * 0.1, return_mask=True)
            assert not scaled_valid.any()

    def test_diagnostic_gradients_preserve_placeholders(self, device):
        # Preserve the old nonzero Q gradient at W=0, as well as ordinary valid
        # gradients. Diagnostics must not silently become a filling policy.
        dtype = torch.float32 if device.type == "mps" else torch.float64
        q = _camera(device, dtype).Q.detach().requires_grad_()
        d = torch.tensor([0, 10], device=device, dtype=dtype).reshape(1, 1, 2, 1).requires_grad_()
        points, valid = reproject_disparity_to_3D(d, q, return_mask=True)
        actual = torch.autograd.grad(points.sum(), (d, q))
        expected = torch.autograd.grad(_legacy(d, q).sum(), (d, q))
        assert valid.flatten().tolist() == [False, True]
        for a, b in zip(actual, expected):
            self.assert_close(a, b, atol=0, rtol=0)
        placeholder_grad = torch.autograd.grad(reproject_disparity_to_3D(torch.zeros_like(d), q).sum(), q)[0]
        assert torch.count_nonzero(placeholder_grad) > 0

    @pytest.mark.parametrize("batch,rows,cols", [(0, 2, 3), (2, 0, 3), (2, 3, 0)])
    def test_empty_cardinality(self, device, dtype, batch, rows, cols):
        q = _camera(device, dtype).Q.detach().requires_grad_()
        d = torch.empty(batch, rows, cols, 1, device=device, dtype=dtype, requires_grad=True)
        points, valid = reproject_disparity_to_3D(d, q, return_mask=True)
        assert points.shape == (batch, rows, cols, 3)
        assert valid.shape == d.shape
        assert torch.equal(points, _legacy(d, q))
        actual = torch.autograd.grad(points.sum(), (d, q), allow_unused=True)
        expected = torch.autograd.grad(_legacy(d, q).sum(), (d, q), allow_unused=True)
        for a, b in zip(actual, expected):
            if b is None:
                assert a is None
            else:
                self.assert_close(a, b, atol=0, rtol=0)

    @pytest.mark.device_agnostic
    def test_cpu_autocast_preserves_coordinates_gradients_and_cutoff(self):
        q = _camera(torch.device("cpu"), torch.float32).Q.detach().requires_grad_()
        d = torch.tensor([0.0, 10.0]).reshape(1, 1, 2, 1).requires_grad_()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            points, valid = reproject_disparity_to_3D(d, q, return_mask=True)
            expected_points = _legacy(d, q)
        assert torch.equal(points, expected_points)
        assert valid.flatten().tolist() == [False, True]
        actual = torch.autograd.grad(points.sum(), (d, q))
        expected = torch.autograd.grad(expected_points.sum(), (d, q))
        for a, b in zip(actual, expected):
            self.assert_close(a, b, atol=0, rtol=0)

        # W is above the float32 cutoff but rounds to the cutoff under autocast.
        q = torch.zeros(1, 4, 4)
        q[..., 3] = 1e-8 * (1 + 1e-4)
        d = torch.zeros(1, 1, 1, 1)
        assert reproject_disparity_to_3D(d, q, return_mask=True)[1].all()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            points, valid = reproject_disparity_to_3D(d, q, return_mask=True)
            assert torch.equal(points, _legacy(d, q))
        assert not valid.any()

    def test_mismatched_batch(self, device, dtype):
        d = torch.ones(3, 2, 2, 1, device=device, dtype=dtype)
        q = _camera(device, dtype).Q.expand(2, -1, -1)
        with pytest.raises(ValueError, match="batch size"):
            reproject_disparity_to_3D(d, q, return_mask=True)

    def test_noncontiguous_inputs(self, device, dtype):
        d = torch.arange(24, device=device, dtype=dtype).reshape(2, 3, 4, 1).transpose(1, 2)
        q = _camera(device, dtype).Q.expand(2, -1, -1).transpose(1, 2).contiguous().transpose(1, 2)
        points, valid = reproject_disparity_to_3D(d, q, return_mask=True)
        assert torch.equal(points, _legacy(d, q))
        assert torch.equal(valid, d != 0)

    @pytest.mark.parametrize("return_mask", [False, True])
    def test_dynamo(self, device, dtype, torch_optimizer, return_mask):
        d = torch.tensor([0, 10], device=device, dtype=dtype).reshape(1, 1, 2, 1)
        q = _camera(device, dtype).Q
        operation = _Reproject(return_mask)
        actual = torch_optimizer(operation, fullgraph=True)(d, q)
        expected = operation(d, q)
        if return_mask:
            self.assert_close(actual[0], expected[0])
            self.assert_close(actual[1], expected[1])
        else:
            self.assert_close(actual, expected)

    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    @pytest.mark.parametrize("return_mask", [False, True])
    def test_export_fixed_configuration(self, device, dtype, return_mask):
        d = torch.tensor([0, 10], device=device, dtype=dtype).reshape(1, 1, 2, 1)
        q = _camera(device, dtype).Q
        operation = _Reproject(return_mask)
        exported = torch.export.export(operation, (d, q)).module()
        for values in ([0, 10], [10, 0]):
            changed = d.new_tensor(values).reshape_as(d)
            actual, expected = exported(changed, q), operation(changed, q)
            if return_mask:
                self.assert_close(actual[0], expected[0])
                self.assert_close(actual[1], expected[1])
            else:
                self.assert_close(actual, expected)

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("return_mask", [False, True])
    def test_onnx_fixed_configuration(self, tmp_path, return_mask):
        pytest.importorskip("onnx")
        ort = pytest.importorskip("onnxruntime")
        d = torch.tensor([0.0, 10.0]).reshape(1, 1, 2, 1)
        q = _camera(torch.device("cpu"), torch.float32).Q
        operation = _Reproject(return_mask)
        path = tmp_path / "stereo.onnx"
        torch.onnx.export(
            operation,
            (d, q),
            path,
            input_names=["disparity", "q"],
            output_names=["points", "valid"] if return_mask else ["points"],
            opset_version=17,
            dynamo=False,
        )
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        for values in ([0.0, 10.0], [10.0, 0.0]):
            changed = d.new_tensor(values).reshape_as(d)
            actual = session.run(None, {"disparity": changed.numpy(), "q": q.numpy()})
            expected = operation(changed, q)
            if return_mask:
                self.assert_close(torch.from_numpy(actual[0]), expected[0])
                self.assert_close(torch.from_numpy(actual[1]), expected[1])
            else:
                self.assert_close(torch.from_numpy(actual[0]), expected)
