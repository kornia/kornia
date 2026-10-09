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

from kornia.geometry.calibration import intrinsics_from_homographies
from kornia.geometry.conversions import axis_angle_to_rotation_matrix
from kornia.geometry.homography import find_homography_dlt
from kornia.image import ImageSize

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


def _noisy_homographies(noise, tilt=None):
    # The review's 8x6 board, 3 cm spacing, camera and seed; fit from pixel observations.
    dtype = torch.float64
    generator = torch.Generator().manual_seed(0)
    if tilt is None:
        angles = (torch.rand(8, 3, generator=generator, dtype=dtype) - 0.5) * 0.9
    else:
        r = math.radians(tilt)
        angles = torch.tensor([[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [r, r, 0], [-r, r, 0]], dtype=dtype)
    rotation = axis_angle_to_rotation_matrix(angles)
    k = torch.tensor([[800.0, 0, 312], [0, 820, 245], [0, 0, 1]], dtype=dtype)
    t = k.new_tensor([0.02, -0.03, 0.9])[None, :, None].expand(len(angles), -1, -1)
    y, x = torch.meshgrid(torch.arange(6, dtype=dtype), torch.arange(8, dtype=dtype), indexing="ij")
    board = torch.stack(((x - 3.5) * 0.03, (y - 2.5) * 0.03), -1).reshape(-1, 2)
    board_h = torch.cat((board, torch.ones_like(board[:, :1])), -1)
    pixels_h = board_h @ (k @ torch.cat((rotation[:, :, :2], t), -1)).transpose(-1, -2)
    pixels = pixels_h[..., :2] / pixels_h[..., 2:]
    pixels = pixels + noise * torch.randn(pixels.shape, generator=generator, dtype=dtype)
    return find_homography_dlt(board[None].expand(len(angles), -1, -1), pixels, solver="svd")[None]


@pytest.fixture
def intrinsics_dtype(dtype):
    if dtype not in (torch.float32, torch.float64):
        pytest.skip("Zhang initialization supports float32 and float64; rejection is tested separately.")
    return dtype


class TestIntrinsicsFromHomographies(BaseTester):
    def test_image_size_scalar_tensors(self, device, intrinsics_dtype):
        expected, h = _scene(device, intrinsics_dtype)
        actual, valid = intrinsics_from_homographies(
            h, ImageSize(torch.tensor(480, device=device), torch.tensor(640, device=device))
        )
        assert valid.all()
        self.assert_close(actual, expected, atol=0.005, rtol=1e-5)
        for size in [ImageSize(torch.tensor([480]), 640), ImageSize(torch.tensor(480.5), 640), ImageSize(480, 0)]:
            with pytest.raises(ValueError, match="image_size"):
                intrinsics_from_homographies(h, size)

    @pytest.mark.parametrize("tilt,expected_valid", [(1, False), (3, False), (10, True)])
    @pytest.mark.parametrize("noise", [0.1, 0.5])
    def test_near_degenerate_noisy_views(self, device, intrinsics_dtype, tilt, expected_valid, noise):
        h = _noisy_homographies(noise, tilt).to(device=device, dtype=intrinsics_dtype)
        actual, valid = intrinsics_from_homographies(h, (480, 640))
        assert valid.item() == expected_valid
        assert torch.isfinite(actual).all()
        explicit, explicit_valid = intrinsics_from_homographies(h, (480, 640), degeneracy_rtol=0.01)
        self.assert_close(actual, explicit)
        self.assert_close(valid, explicit_valid)

    def test_image_size_changes_noisy_estimate(self, device, intrinsics_dtype):
        h = _noisy_homographies(2.0).to(device=device, dtype=intrinsics_dtype)
        actual, valid = intrinsics_from_homographies(h, ImageSize(480, 640))
        wrong_size, wrong_valid = intrinsics_from_homographies(h, (4000, 4000))
        assert valid.all() and wrong_valid.all()
        # Reproduced with the public initializer at bb327d0 and the observations above.
        self.assert_close(actual[0, 0, 0], h.new_tensor(898.3819099), atol=0.05, rtol=0)
        self.assert_close(wrong_size[0, 0, 0], h.new_tensor(861.9563307), atol=0.05, rtol=0)
        assert (actual - wrong_size).abs().max() > 30

    @pytest.mark.parametrize("value", [0.0, float("nan"), float("inf")])
    def test_zero_weights_remove_views(self, device, intrinsics_dtype, value):
        expected, h = _scene(device, intrinsics_dtype)
        h[0, 2:] = value
        h[1, 3:] = value
        weights = h.new_tensor([[1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0]])
        h.requires_grad_()
        weights.requires_grad_()
        actual, valid = intrinsics_from_homographies(h, (480, 640), weights=weights)
        assert valid.all()
        self.assert_close(actual, expected, atol=0.005, rtol=1e-5)
        actual.sum().backward()
        assert torch.isfinite(h.grad).all() and torch.isfinite(weights.grad).all()
        self.assert_close(h.grad[weights == 0], torch.zeros_like(h.grad[weights == 0]))
        self.assert_close(weights.grad[weights == 0], torch.zeros_like(weights.grad[weights == 0]))

    def test_insufficient_weighted_views(self, device, intrinsics_dtype):
        _, h = _scene(device, intrinsics_dtype)
        weights = torch.zeros(h.shape[:2], device=device, dtype=intrinsics_dtype)
        weights[1, 0] = 1
        actual, valid = intrinsics_from_homographies(h, (480, 640), weights=weights)
        assert not valid.any()
        self.assert_close(actual, torch.eye(3, device=device, dtype=intrinsics_dtype).expand_as(actual))

    def test_weighted_least_squares(self, device, intrinsics_dtype):
        h = _noisy_homographies(0.5).to(device=device, dtype=intrinsics_dtype)
        weights = h.new_tensor([[1, 1, 4, 1, 1, 1, 1, 1]])
        weighted, valid = intrinsics_from_homographies(h, (480, 640), weights=weights)
        # Repeating one observation four times is an independent weighted-LS oracle.
        repeated, repeated_valid = intrinsics_from_homographies(h[:, [0, 1, 2, 2, 2, 2, 3, 4, 5, 6, 7]], (480, 640))
        uniform, _ = intrinsics_from_homographies(h, (480, 640))
        rescaled, _ = intrinsics_from_homographies(h, (480, 640), weights=weights * 100)
        assert valid.all() and repeated_valid.all()
        self.assert_close(weighted, repeated, atol=0.01, rtol=1e-5)
        self.assert_close(weighted, rescaled, atol=0.005, rtol=1e-5)
        assert (weighted - uniform).abs().max() > 0.1

    def test_weight_errors(self, device, intrinsics_dtype):
        _, h = _scene(device, intrinsics_dtype)
        for weights in [h.new_ones(6), h.new_ones(2, 5), h.new_ones(2, 6, 1)]:
            with pytest.raises(ValueError, match="weights"):
                intrinsics_from_homographies(h, (480, 640), weights=weights)
        for weights in [1, torch.ones(2, 6, device=device, dtype=torch.int64)]:
            with pytest.raises(TypeError, match="weights"):
                intrinsics_from_homographies(h, (480, 640), weights=weights)
        for value in [-1, float("nan"), float("inf")]:
            weights = h.new_ones(2, 6)
            weights[0, 0] = value
            with pytest.raises(ValueError, match="weights"):
                intrinsics_from_homographies(h, (480, 640), weights=weights)

    @pytest.mark.parametrize("bad", ["zeros", "repeated", "nonfinite"])
    def test_invalid_camera_does_not_poison_backward(self, device, intrinsics_dtype, bad):
        _, h = _scene(device, intrinsics_dtype)
        if bad == "repeated":
            h[1] = h[1, :1].clone()
        else:
            h[1] = 0 if bad == "zeros" else float("nan")
        h.requires_grad_()
        actual, valid = intrinsics_from_homographies(h, (480, 640))
        assert valid.tolist() == [True, False]
        actual.sum().backward()
        assert torch.isfinite(h.grad).all()
        self.assert_close(h.grad[1], torch.zeros_like(h.grad[1]))
        good_h = h[:1].detach().clone().requires_grad_()
        intrinsics_from_homographies(good_h, (480, 640))[0].sum().backward()
        self.assert_close(h.grad[:1], good_h.grad, atol=0.005, rtol=1e-5)

    def test_weight_gradcheck(self, device):
        if device.type == "mps":
            pytest.skip("MPS does not support float64 gradcheck")
        h = _noisy_homographies(0.5).to(device).requires_grad_()
        weights = h.new_tensor([[0.5, 2, 1, 0.8, 0.7, 1.2, 1.3, 0.9]], requires_grad=True)
        self.gradcheck(
            lambda x, w: intrinsics_from_homographies(x, (480, 640), weights=w)[0],
            (h, weights),
            fast_mode=True,
            atol=1e-3,
            rtol=1e-3,
        )

    @pytest.mark.parametrize("views", [2, 3, 6])
    def test_exact_intrinsics(self, device, intrinsics_dtype, views):
        expected, homographies = _scene(device, intrinsics_dtype)
        actual, valid = intrinsics_from_homographies(homographies[:, :views], (480, 640))
        assert valid.shape == (2,) and valid.dtype == torch.bool and valid.device == device
        assert valid.all()
        self.assert_close(actual, expected, atol=0.005, rtol=1e-5)
        assert (actual[:, 0, 1] == 0).all()
        assert actual.dtype == intrinsics_dtype and actual.device == device
        assert (actual[:, 2, 2] == 1).all()

    def test_gauge_and_board_scale(self, device, intrinsics_dtype):
        expected, homographies = _scene(device, intrinsics_dtype)
        scales = homographies.new_tensor([-100, 0.001, 3, -0.2, 20, 1])
        scaled = homographies * scales[None, :, None, None]
        scaled[..., :, :2] *= 0.01
        actual, valid = intrinsics_from_homographies(scaled, (480, 640))
        assert valid.all()
        self.assert_close(actual, expected, atol=0.005, rtol=1e-5)

    def test_image_size_exact_data(self, device, intrinsics_dtype):
        expected, homographies = _scene(device, intrinsics_dtype)
        for shape in [(10, 10), (1080, 1920)]:
            # Deliberately incorrect sizes can fail the default conditioning check.
            actual, valid = intrinsics_from_homographies(homographies, shape, degeneracy_rtol=1e-5)
            assert valid.all()
            self.assert_close(actual, expected, atol=0.005, rtol=1e-5)
        change = homographies.new_tensor([[2.0, 0.0, 11.0], [0.0, 0.5, -3.0], [0.0, 0.0, 1.0]])
        actual, valid = intrinsics_from_homographies(change @ homographies, (240, 1280))
        assert valid.all()
        self.assert_close(actual, change @ expected, atol=0.005, rtol=1e-5)

    def test_permutation_noncontiguous(self, device, intrinsics_dtype):
        expected, homographies = _scene(device, intrinsics_dtype)
        reversed_views = homographies[:, [5, 3, 1, 4, 0, 2]]
        noncontiguous = reversed_views.transpose(-1, -2).contiguous().transpose(-1, -2)
        actual, valid = intrinsics_from_homographies(noncontiguous, (480, 640))
        assert valid.all()
        self.assert_close(actual, expected, atol=0.005, rtol=1e-5)

    def test_degeneracy_and_mixed_batch(self, device, intrinsics_dtype):
        expected, homographies = _scene(device, intrinsics_dtype)
        for bad in [
            torch.zeros_like(homographies),
            homographies[:, :1].expand_as(homographies),
            torch.eye(3, device=device, dtype=intrinsics_dtype)[None, None].expand_as(homographies),
        ]:
            actual, valid = intrinsics_from_homographies(bad, (480, 640))
            assert not valid.any()
            self.assert_close(actual, torch.eye(3, device=device, dtype=intrinsics_dtype).expand_as(actual))
        mixed = homographies.clone()
        mixed[1] = mixed[1, 0]
        actual, valid = intrinsics_from_homographies(mixed, (480, 640))
        assert valid.tolist() == [True, False]
        self.assert_close(actual[0], expected[0], atol=0.005, rtol=1e-5)
        self.assert_close(actual[1], torch.eye(3, device=device, dtype=intrinsics_dtype))

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_nonfinite(self, device, intrinsics_dtype, value):
        _, homographies = _scene(device, intrinsics_dtype)
        homographies[0, 0, 0, 0] = value
        actual, valid = intrinsics_from_homographies(homographies, (480, 640))
        assert valid.tolist() == [False, True]
        assert torch.isfinite(actual).all()

    def test_exceptions(self, device):
        _, homographies = _scene(device, torch.float32)
        for h in [homographies[0], homographies[:, :1], homographies[:0], homographies[..., :2]]:
            with pytest.raises(ValueError):
                intrinsics_from_homographies(h, (480, 640))
        for size in [(0, 640), (-1, 480), (480,), (True, 640), (480.5, 640)]:
            with pytest.raises((ValueError, TypeError)):
                intrinsics_from_homographies(homographies, size)
        for rtol in [-1, 0, 1, float("nan"), float("inf")]:
            with pytest.raises(ValueError):
                intrinsics_from_homographies(homographies, (480, 640), degeneracy_rtol=rtol)

    @pytest.mark.parametrize("degeneracy_rtol", [1e-2, 1e-4])
    def test_inconsistent_constraints(self, device, intrinsics_dtype, degeneracy_rtol):
        h = torch.tensor(
            [
                [
                    [[2.0, 2.0, -3.0], [-3.0, 3.0, -2.0], [3.0, 1.0, 2.0]],
                    [[2.0, 3.0, 0.0], [-3.0, -3.0, 0.0], [3.0, -2.0, 3.0]],
                    [[1.0, 1.0, 1.0], [-2.0, -1.0, -3.0], [3.0, 0.0, -1.0]],
                ]
            ],
            device=device,
            dtype=intrinsics_dtype,
        )
        # These inconsistent homographies are in normalized image coordinates. Put
        # them at pixel scale so float32 rank checks do not mask the intended
        # positive-definiteness rejection. Check both the default threshold and
        # an explicitly relaxed rank diagnostic.
        normalized_to_pixels = h.new_tensor([[320.0, 0.0, 319.5], [0.0, 320.0, 239.5], [0.0, 0.0, 1.0]])
        h = normalized_to_pixels @ h
        h.requires_grad_()
        actual, valid = intrinsics_from_homographies(h, (480, 640), degeneracy_rtol=degeneracy_rtol)
        assert not valid.any()
        self.assert_close(actual, torch.eye(3, device=device, dtype=intrinsics_dtype)[None])
        actual.sum().backward()
        self.assert_close(h.grad, torch.zeros_like(h))

    def test_singular_homography(self, device, intrinsics_dtype):
        _, h = _scene(device, intrinsics_dtype)
        h[0, 0, :, 2] = 0
        actual, valid = intrinsics_from_homographies(h, (480, 640))
        assert valid.tolist() == [False, True]
        assert torch.isfinite(actual).all()

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64, torch.complex64])
    def test_unsupported_dtype(self, device, dtype):
        h = torch.ones(1, 3, 3, 3, device=device, dtype=dtype)
        with pytest.raises(TypeError, match="float32 or float64"):
            intrinsics_from_homographies(h, (480, 640))

    @pytest.mark.parametrize("noise", [0.0, 1e-4])
    @pytest.mark.parametrize("views", [2, 6])
    def test_gradcheck(self, device, noise, views):
        if device.type == "mps":
            pytest.skip("MPS does not support float64 gradcheck")
        _, homographies = _scene(device, torch.float64)
        h = homographies[:1, :views] / 800
        h = h + noise * torch.sin(torch.arange(h.numel(), device=device, dtype=h.dtype)).reshape(h.shape)
        h.requires_grad_()
        self.gradcheck(
            lambda x: intrinsics_from_homographies(x, (480, 640))[0], (h,), fast_mode=True, atol=1e-3, rtol=1e-3
        )

    def test_planar_points_composition(self, device, intrinsics_dtype):
        expected, h = _scene(device, intrinsics_dtype)
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
        actual, valid = intrinsics_from_homographies(estimated_h[None], (480, 640))
        assert valid.all()
        self.assert_close(actual[0], expected[0], atol=0.2, rtol=0.001)

    @pytest.mark.parametrize("views", [2, 6])
    def test_dynamo(self, device, intrinsics_dtype, torch_optimizer, views):
        _, h = _scene(device, intrinsics_dtype)
        h = h[:, :views]
        optimized = torch_optimizer(intrinsics_from_homographies, fullgraph=True)
        for mixed in (h, torch.cat((h[:1], torch.zeros_like(h[:1])))):
            actual, valid = optimized(mixed, ImageSize(480, 640))
            expected, expected_valid = intrinsics_from_homographies(mixed, (480, 640))
            self.assert_close(actual, expected, atol=0.005, rtol=1e-5)
            self.assert_close(valid, expected_valid)
