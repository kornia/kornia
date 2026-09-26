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

from __future__ import annotations

import pytest
import torch
from torch import nn

from kornia.geometry.camera import projection_valid_mask
from kornia.geometry.depth import warp_frame_depth, warp_frame_depth_with_mask

from testing.base import BaseTester, dynamic_export_is_available


class _Warp(nn.Module):
    def forward(self, image, depth, transform, intrinsics):
        return warp_frame_depth_with_mask(image, depth, transform, intrinsics)


def _inputs(device, dtype, height=3, width=4):
    image = torch.arange(height * width, device=device, dtype=dtype).reshape(1, 1, height, width)
    depth = torch.ones_like(image)
    transform = torch.eye(4, device=device, dtype=dtype)[None]
    intrinsics = torch.eye(3, device=device, dtype=dtype)[None]
    return image, depth, transform, intrinsics


class TestProjectionValidMask(BaseTester):
    def test_geometry(self, device, dtype):
        points = torch.tensor(
            [[1, 2, 3], [0, 0, 0], [1, 2, -1], [float("nan"), 0, 1], [0, float("inf"), 1]], device=device, dtype=dtype
        )
        result = projection_valid_mask(points)
        self.assert_close(result, torch.tensor([True, False, False, False, False], device=device))
        assert result.dtype == torch.bool and result.device == points.device
        assert projection_valid_mask(points[0]).shape == ()
        assert projection_valid_mask(points[:0]).shape == (0,)

    def test_threshold(self, device, dtype):
        points = torch.tensor([[0, 0, 0.25], [0, 0, 0.5], [0, 0, 1]], device=device, dtype=dtype)
        self.assert_close(projection_valid_mask(points, 0.5), torch.tensor([False, False, True], device=device))
        assert projection_valid_mask(points, 0).all()

    def test_half_threshold_not_rounded(self, device):
        points = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float16)
        assert projection_valid_mask(points, 0.9999).all()

    def test_exceptions(self, device):
        with pytest.raises(ValueError):
            projection_valid_mask(torch.zeros(1, 2, device=device))
        with pytest.raises(TypeError):
            projection_valid_mask(torch.zeros(1, 3, device=device, dtype=torch.int64))
        for eps in [-1, float("nan"), float("inf")]:
            with pytest.raises(ValueError):
                projection_valid_mask(torch.zeros(1, 3, device=device), eps)

    def test_jit(self, device, dtype):
        points = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], device=device, dtype=dtype)
        self.assert_close(torch.jit.script(projection_valid_mask)(points), projection_valid_mask(points))


class TestWarpFrameDepthWithMask(BaseTester):
    @pytest.mark.parametrize("shape", [(3, 4), (1, 4), (3, 1), (1, 1)])
    def test_identity(self, device, dtype, shape):
        inputs = _inputs(device, dtype, *shape)
        image, valid = warp_frame_depth_with_mask(*inputs)
        assert valid.all()
        self.assert_close(image, inputs[0])
        assert valid.shape == (1, 1, *shape)
        assert image.dtype == dtype and valid.dtype == torch.bool

    def test_translation_and_legacy_difference(self, device, dtype):
        image = torch.ones(1, 1, 3, 3, device=device, dtype=dtype)
        _, depth, transform, intrinsics = _inputs(device, dtype, 3, 3)
        transform[:, 0, 3] = 0.25
        warped, valid = warp_frame_depth_with_mask(image, depth, transform, intrinsics)
        assert valid[0, 0, 1].tolist() == [True, True, False]
        self.assert_close(warped[0, 0, 1], warped.new_tensor([1, 1, 0]))
        if dtype in (torch.float32, torch.float64):
            old = warp_frame_depth(image, depth, transform, intrinsics)
            self.assert_close(old[0, 0, 1], old.new_tensor([1, 1, 0.75]))

    def test_source_size_and_direction(self, device, dtype):
        image, _, transform, intrinsics = _inputs(device, dtype, 5, 6)
        depth = image.new_ones(1, 1, 2, 3)
        transform[:, 0, 3] = 1
        result, valid = warp_frame_depth_with_mask(image, depth, transform, intrinsics)
        self.assert_close(result, image[..., :2, 1:4])
        assert valid.all() and result.shape == (1, 1, 2, 3)

    def test_original_depth_invalid_after_translation(self, device, dtype):
        image, depth, transform, intrinsics = _inputs(device, dtype, 2, 3)
        depth[0, 0] = depth.new_tensor([[0, -1, float("nan")], [float("inf"), -float("inf"), 1]])
        transform[:, 2, 3] = 2
        depth.requires_grad_()
        result, valid = warp_frame_depth_with_mask(image, depth, transform, intrinsics)
        assert valid.flatten().tolist() == [False, False, False, False, False, True]
        assert (result[~valid] == 0).all()
        result.sum().backward()
        assert torch.isfinite(depth.grad).all()

    def test_batch_noncontiguous_and_empty(self, device, dtype):
        image, depth, transform, intrinsics = _inputs(device, dtype, 4, 5)
        inputs = [v.expand(2, *v.shape[1:]).clone() for v in (image, depth, transform, intrinsics)]
        inputs[0], inputs[1] = inputs[0].transpose(-1, -2), inputs[1].transpose(-1, -2)
        inputs[2][1, 0, 3] = 0.5
        actual = warp_frame_depth_with_mask(*inputs)
        for i in range(2):
            expected = warp_frame_depth_with_mask(*(v[i : i + 1] for v in inputs))
            for a, e in zip(actual, expected):
                self.assert_close(a[i : i + 1], e)
        empty = [v[:0].requires_grad_() for v in inputs]
        output, mask = warp_frame_depth_with_mask(*empty)
        assert output.shape == mask.shape == (0, 1, 5, 4)
        output.sum().backward()
        assert all(v.grad is not None for v in empty)

    def test_behind_camera_and_overflow(self, device, dtype):
        image, depth, transform, intrinsics = _inputs(device, dtype)
        transform[:, 2, 3] = -2
        assert not warp_frame_depth_with_mask(image, depth, transform, intrinsics)[1].any()
        if dtype in (torch.float32, torch.float64):
            transform = torch.eye(4, device=device, dtype=dtype)[None].requires_grad_()
            depth = torch.full_like(depth, torch.finfo(dtype).max).requires_grad_()
            result, valid = warp_frame_depth_with_mask(image, depth, transform, intrinsics)
            assert not valid[..., 1:, 2:].any()
            result.sum().backward()
            assert torch.isfinite(depth.grad).all()
            assert torch.isfinite(transform.grad).all()

    def test_intrinsics_and_ray_depth(self, device, dtype):
        image, depth, transform, intrinsics = _inputs(device, dtype, 3, 4)
        intrinsics[:, 0, 0], intrinsics[:, 1, 1] = 2, 3
        intrinsics[:, 0, 2], intrinsics[:, 1, 2] = 1, 1
        result, valid = warp_frame_depth_with_mask(image, depth, transform, intrinsics, normalize_points=True)
        assert valid.all()
        self.assert_close(result, image, atol=0.02 if dtype in (torch.float16, torch.bfloat16) else 1e-5, rtol=0)

    def test_gradcheck(self, device):
        if device.type == "mps":
            pytest.skip("MPS does not support float64 gradcheck")
        image, depth, transform, intrinsics = _inputs(device, torch.float64, 4, 5)
        transform[:, 0, 3] = 0.25
        transform[:, 1, 3] = 0.3
        image.requires_grad_()
        depth.requires_grad_()
        self.gradcheck(
            lambda i, d: warp_frame_depth_with_mask(i, d, transform, intrinsics)[0], (image, depth), fast_mode=True
        )

    def test_exceptions(self, device):
        image, depth, transform, intrinsics = _inputs(device, torch.float32)
        for args in [
            (image[0], depth, transform, intrinsics),
            (image, depth[:, 0], transform, intrinsics),
            (image, depth, transform[:, :3], intrinsics),
            (image, depth, transform, intrinsics.expand(2, -1, -1)),
        ]:
            with pytest.raises((ValueError, TypeError)):
                warp_frame_depth_with_mask(*args)

    def test_dynamo(self, device, dtype, torch_optimizer):
        inputs = _inputs(device, dtype)
        inputs[1][..., 0, 0] = float("nan")
        optimized = torch_optimizer(warp_frame_depth_with_mask, fullgraph=True)
        for a, e in zip(optimized(*inputs), warp_frame_depth_with_mask(*inputs)):
            self.assert_close(a, e)

    def test_export_and_onnx_runtime(self, tmp_path):
        pytest.importorskip("onnx")
        ort = pytest.importorskip("onnxruntime")
        inputs = _inputs(torch.device("cpu"), torch.float32)
        inputs[1][..., 0, 0] = float("nan")
        inputs[2][:, 0, 3] = 0.25
        module = _Warp()
        expected = module(*inputs)
        traced = torch.jit.trace(module, inputs)
        for a, e in zip(traced(*inputs), expected):
            self.assert_close(a, e)
        if dynamic_export_is_available():
            exported = torch.export.export(module, inputs).module()
            for a, e in zip(exported(*inputs), expected):
                self.assert_close(a, e)
        path = tmp_path / "warp_with_mask.onnx"
        torch.onnx.export(
            module,
            inputs,
            str(path),
            opset_version=18,
            dynamo=False,
            input_names=["image", "depth", "transform", "intrinsics"],
            output_names=["warped", "valid"],
        )
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        outputs = session.run(
            None, {n: v.numpy() for n, v in zip(["image", "depth", "transform", "intrinsics"], inputs)}
        )
        for a, e in zip(outputs, expected):
            self.assert_close(torch.from_numpy(a), e)
        # Change the validity pattern after export; it must not have been baked into the graph.
        inputs[1].fill_(1)
        inputs[1][..., 1, 1] = 0
        inputs[2][:, 0, 3] = -0.5
        expected = module(*inputs)
        outputs = session.run(
            None, {n: v.numpy() for n, v in zip(["image", "depth", "transform", "intrinsics"], inputs)}
        )
        for a, e in zip(outputs, expected):
            self.assert_close(torch.from_numpy(a), e)
