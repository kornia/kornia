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

import math

import pytest
import torch

from kornia.geometry.calibration import init_camera_intrinsics_zhang
from kornia.geometry.homography import find_homography_dlt

from testing.base import BaseTester


def _scene(device, dtype):
    # Independent plane-to-image model H = K [r1 r2 t], with R = Rz Ry Rx.
    matrices = []
    for ax, ay, az in [(17, -13, 4), (-19, 16, -7), (10, 24, 11), (-22, -17, 5), (25, 9, -13), (7, -25, 18)]:
        x, y, z = (math.radians(v) for v in (ax, ay, az))
        cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
        rotation = torch.tensor(
            [
                [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
                [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
                [-sy, cy * sx, cy * cx],
            ],
            device=device,
            dtype=dtype,
        )
        matrices.append(torch.cat([rotation[:, :2], rotation.new_tensor([[0.02], [-0.03], [0.9]])], -1))
    k = torch.tensor(
        [
            [[800.0, 0.0, 312.0], [0.0, 820.0, 245.0], [0.0, 0.0, 1.0]],
            [[650.0, 0.0, 301.0], [0.0, 710.0, 231.0], [0.0, 0.0, 1.0]],
        ],
        device=device,
        dtype=dtype,
    )
    return k, k[:, None] @ torch.stack(matrices)[None]


@pytest.fixture
def zhang_dtype(dtype):
    if dtype not in (torch.float32, torch.float64):
        pytest.skip("Zhang initialization supports float32 and float64; rejection is tested separately.")
    return dtype


class TestInitCameraIntrinsicsZhang(BaseTester):
    @pytest.mark.parametrize("views", [3, 6])
    def test_exact_intrinsics(self, device, zhang_dtype, views):
        expected, homographies = _scene(device, zhang_dtype)
        actual = init_camera_intrinsics_zhang(homographies[:, :views], (480, 640))
        self.assert_close(actual, expected, atol=0.005, rtol=1e-5)
        assert (actual[:, 0, 1] == 0).all()
        assert actual.dtype == zhang_dtype and actual.device == device
        assert (actual[:, 2, 2] == 1).all()

    def test_gauge_and_board_scale(self, device, zhang_dtype):
        expected, homographies = _scene(device, zhang_dtype)
        scales = homographies.new_tensor([-100, 0.001, 3, -0.2, 20, 1])
        scaled = homographies * scales[None, :, None, None]
        scaled[..., :, :2] *= 0.01
        self.assert_close(init_camera_intrinsics_zhang(scaled, (480, 640)), expected, atol=0.005, rtol=1e-5)

    def test_image_size_is_conditioning_not_prior(self, device, zhang_dtype):
        expected, homographies = _scene(device, zhang_dtype)
        for shape in [(10, 10), (1080, 1920)]:
            self.assert_close(init_camera_intrinsics_zhang(homographies, shape), expected, atol=0.005, rtol=1e-5)
        change = homographies.new_tensor([[2.0, 0.0, 11.0], [0.0, 0.5, -3.0], [0.0, 0.0, 1.0]])
        actual = init_camera_intrinsics_zhang(change @ homographies, (240, 1280))
        self.assert_close(actual, change @ expected, atol=0.005, rtol=1e-5)

    def test_permutation_noncontiguous(self, device, zhang_dtype):
        expected, homographies = _scene(device, zhang_dtype)
        reversed_views = homographies[:, [5, 3, 1, 4, 0, 2]]
        noncontiguous = reversed_views.transpose(-1, -2).contiguous().transpose(-1, -2)
        self.assert_close(init_camera_intrinsics_zhang(noncontiguous, (480, 640)), expected, atol=0.005, rtol=1e-5)

    def test_degeneracy_and_mixed_batch(self, device, zhang_dtype):
        _, homographies = _scene(device, zhang_dtype)
        for bad in [
            torch.zeros_like(homographies),
            homographies[:, :1].expand_as(homographies),
            torch.eye(3, device=device, dtype=zhang_dtype)[None, None].expand_as(homographies),
        ]:
            with pytest.raises(ValueError, match="degenerate"):
                init_camera_intrinsics_zhang(bad, (480, 640))
        mixed = homographies.clone()
        mixed[1] = mixed[1, 0]
        with pytest.raises(ValueError, match="batch"):
            init_camera_intrinsics_zhang(mixed, (480, 640))

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_nonfinite(self, device, zhang_dtype, value):
        _, homographies = _scene(device, zhang_dtype)
        homographies[0, 0, 0, 0] = value
        with pytest.raises(ValueError, match="finite"):
            init_camera_intrinsics_zhang(homographies, (480, 640))

    def test_exceptions(self, device):
        _, homographies = _scene(device, torch.float32)
        for h in [homographies[0], homographies[:, :2], homographies[:0], homographies[..., :2]]:
            with pytest.raises(ValueError):
                init_camera_intrinsics_zhang(h, (480, 640))
        for size in [(0, 640), (-1, 480), (480,), (True, 640), (480.5, 640)]:
            with pytest.raises((ValueError, TypeError)):
                init_camera_intrinsics_zhang(homographies, size)
        for rtol in [-1, 0, 1, float("nan"), float("inf")]:
            with pytest.raises(ValueError):
                init_camera_intrinsics_zhang(homographies, (480, 640), degeneracy_rtol=rtol)

    def test_inconsistent_constraints(self, device, zhang_dtype):
        h = torch.tensor(
            [
                [
                    [[2.0, 2.0, -3.0], [-3.0, 3.0, -2.0], [3.0, 1.0, 2.0]],
                    [[2.0, 3.0, 0.0], [-3.0, -3.0, 0.0], [3.0, -2.0, 3.0]],
                    [[1.0, 1.0, 1.0], [-2.0, -1.0, -3.0], [3.0, 0.0, -1.0]],
                ]
            ],
            device=device,
            dtype=zhang_dtype,
        )
        with pytest.raises(ValueError, match=r"positive-definite.*batch"):
            init_camera_intrinsics_zhang(h, (480, 640))

    def test_singular_homography(self, device, zhang_dtype):
        _, h = _scene(device, zhang_dtype)
        h[0, 0, :, 2] = 0
        with pytest.raises(ValueError, match="degenerate singular"):
            init_camera_intrinsics_zhang(h, (480, 640))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64, torch.complex64])
    def test_unsupported_dtype(self, device, dtype):
        h = torch.ones(1, 3, 3, 3, device=device, dtype=dtype)
        with pytest.raises(TypeError, match="float32 or float64"):
            init_camera_intrinsics_zhang(h, (480, 640))

    @pytest.mark.parametrize("noise", [0.0, 1e-4])
    def test_gradcheck(self, device, noise):
        if device.type == "mps":
            pytest.skip("MPS does not support float64 gradcheck")
        _, homographies = _scene(device, torch.float64)
        h = homographies[:1] / 800
        h = h + noise * torch.sin(torch.arange(h.numel(), device=device, dtype=h.dtype)).reshape(h.shape)
        h.requires_grad_()
        self.gradcheck(
            lambda x: init_camera_intrinsics_zhang(x, (480, 640)), (h,), fast_mode=True, atol=1e-3, rtol=1e-3
        )

    def test_planar_points_composition(self, device, zhang_dtype):
        expected, h = _scene(device, zhang_dtype)
        xy = h.new_tensor(
            [
                [-0.1, -0.08],
                [0.1, -0.08],
                [0.1, 0.08],
                [-0.1, 0.08],
                [0.0, 0.0],
                [0.04, -0.02],
                [-0.02, 0.06],
                [0.06, 0.05],
            ]
        )
        homogeneous = torch.cat([xy, torch.ones_like(xy[:, :1])], -1)
        pixels_h = homogeneous @ h[0].transpose(-1, -2)
        pixels = pixels_h[..., :2] / pixels_h[..., 2:]
        estimated_h = find_homography_dlt(xy[None].expand(6, -1, -1), pixels, solver="svd")
        actual = init_camera_intrinsics_zhang(estimated_h[None], (480, 640))
        self.assert_close(actual[0], expected[0], atol=0.2, rtol=0.001)

    def test_dynamo(self, device, zhang_dtype, torch_optimizer):
        # Strict data-dependent failure checks intentionally allow graph breaks.
        _, h = _scene(device, zhang_dtype)
        optimized = torch_optimizer(init_camera_intrinsics_zhang, fullgraph=False)
        self.assert_close(optimized(h, (480, 640)), init_camera_intrinsics_zhang(h, (480, 640)), atol=0.005, rtol=1e-5)
