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

import kornia
from kornia.core.exceptions import BaseError, TypeCheckError
from kornia.geometry.grid import create_meshgrid, create_meshgrid3d

from testing.base import BaseTester, supports_conv2d, supports_replicate_padding


class TestConvDistanceTransform(BaseTester):
    @pytest.mark.parametrize("kernel_size", [3, 5, 7])
    @pytest.mark.parametrize("shape", [(1, 3, 100, 100), (2, 2, 10, 10, 10)])
    def test_smoke(self, kernel_size, shape, device, dtype):
        sample = torch.rand(*shape, device=device, dtype=dtype)
        distance_transformer = kornia.contrib.DistanceTransform(kernel_size)

        output1 = distance_transformer(sample)
        output2 = kornia.contrib.distance_transform(sample, kernel_size)

        assert isinstance(output1, torch.Tensor)
        assert isinstance(output2, torch.Tensor)
        assert output1.shape == sample.shape
        self.assert_close(output1, output2)

    def test_module(self, device, dtype):
        B, C, H, W = 1, 1, 99, 100
        sample2d = torch.rand(B, C, H, W, device=device, dtype=dtype)
        distance_transformer = kornia.contrib.DistanceTransform().to(device, dtype)

        output1 = distance_transformer(sample2d)
        output2 = kornia.contrib.distance_transform(sample2d)
        self.assert_close(output1, output2)

        B, C, D, H, W = 1, 1, 10, 10, 10
        sample3d = torch.rand(B, C, D, H, W, device=device, dtype=dtype)
        output3 = distance_transformer(sample3d)
        output4 = kornia.contrib.distance_transform(sample3d)
        self.assert_close(output3, output4)

    def test_exception(self, device, dtype):
        B, C, H, W = 1, 1, 32, 32
        sample2d = torch.rand(B, C, H, W, device=device, dtype=dtype)
        sample_wrong_dims = torch.rand(C, H, W, device=device, dtype=dtype)

        # Non-odd kernel size -> BaseError from KORNIA_CHECK
        with pytest.raises(BaseError) as excinfo:
            ConvDT = kornia.contrib.DistanceTransform(6)
            ConvDT.forward(sample2d)
        assert "kernel_size must be an odd integer >= 3" in str(excinfo.value)

        with pytest.raises(BaseError):
            kornia.contrib.distance_transform(sample2d, 4)

        # Kernel size too small
        with pytest.raises(BaseError):
            kornia.contrib.distance_transform(sample2d, 1)

        # Invalid input dimensions (3D tensor instead of 4D or 5D)
        with pytest.raises(BaseError) as excinfo:
            kornia.contrib.distance_transform(sample_wrong_dims)
        assert "Invalid image shape" in str(excinfo.value)

        # Invalid input type (None)
        with pytest.raises(TypeCheckError):
            kornia.contrib.distance_transform(None)

        # Integer input not supported (dtype check)
        sample_int = torch.randint(0, 2, (B, C, H, W), device=device)
        with pytest.raises(BaseError):
            kornia.contrib.distance_transform(sample_int)

        # invalid h
        with pytest.raises(BaseError):
            kornia.contrib.distance_transform(sample2d, kernel_size=3, h=0.0)

    def test_kernel_geometry(self, device, dtype):
        kernel_size = 5
        k_half = kernel_size // 2
        h = 0.35

        # 2D Kernel Check
        grid2d = create_meshgrid(kernel_size, kernel_size, False, device, dtype)
        grid2d = grid2d - k_half
        dist2d = torch.norm(grid2d[0], p=2, dim=-1)
        kernel2d = torch.exp(-dist2d / h)

        self.assert_close(kernel2d[k_half, k_half], torch.tensor(1.0, device=device, dtype=dtype))

        self.assert_close(kernel2d[0, 0], kernel2d[-1, -1])

        # 3D Kernel Check
        grid3d = create_meshgrid3d(kernel_size, kernel_size, kernel_size, False, device, dtype)
        grid3d = grid3d - k_half
        dist3d = torch.norm(grid3d[0], p=2, dim=-1)
        kernel3d = torch.exp(-dist3d / h)

        # Center must be exactly 1.0
        self.assert_close(kernel3d[k_half, k_half, k_half], torch.tensor(1.0, device=device, dtype=dtype))
        self.assert_close(kernel3d[0, 0, 0], kernel3d[-1, -1, -1])

    def test_noncontiguous_multi_channel(self, device, dtype):
        B, C, H, W = 1, 2, 4, 4
        sample = torch.rand(B, C, H, W, device=device, dtype=dtype)
        sample = sample.transpose(2, 3)
        op = kornia.contrib.DistanceTransform()
        out = op(sample)
        assert out.shape == sample.shape

    def test_value_2d(self, device, dtype):
        B, C, H, W = 1, 1, 4, 4
        kernel_size = 7
        h = 0.35
        sample1 = torch.zeros(B, C, H, W, device=device, dtype=dtype)
        sample1[:, :, 1, 1] = 1.0
        expected_output1 = torch.tensor(
            [
                [
                    [
                        [1.4142135382, 1.0000000000, 1.4142135382, 2.2360680103],
                        [1.0000000000, 0.0000000000, 1.0000000000, 2.0000000000],
                        [1.4142135382, 1.0000000000, 1.4142135382, 2.2360680103],
                        [2.2360680103, 2.0000000000, 2.2360680103, 2.8284270763],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        output1 = kornia.contrib.distance_transform(sample1, kernel_size, h)
        self.assert_close(expected_output1, output1)

    def test_value_3d(self, device, dtype):
        B, C, D, H, W = 1, 1, 3, 3, 3
        kernel_size = 3
        h = 0.35
        sample1 = torch.zeros(B, C, D, H, W, device=device, dtype=dtype)

        sample1[:, :, 1, 1, 1] = 1.0

        output1 = kornia.contrib.distance_transform(sample1, kernel_size, h)

        self.assert_close(output1[0, 0, 1, 1, 1], torch.tensor(0.0, device=device, dtype=dtype))

        self.assert_close(output1[0, 0, 0, 1, 1], torch.tensor(1.0, device=device, dtype=dtype))

        expected_corner = torch.tensor(1.7320508, device=device, dtype=dtype)
        self.assert_close(output1[0, 0, 0, 0, 0], expected_corner)

    def test_gradcheck(self, device):
        B, C, H, W = 1, 1, 32, 32
        sample2d = torch.ones(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(kornia.contrib.distance_transform, (sample2d,))

        B, C, D, H, W = 1, 1, 5, 5, 5
        sample3d = torch.ones(B, C, D, H, W, device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(kornia.contrib.distance_transform, (sample3d,))

    def test_loss_grad(self, device, dtype):
        B, C, H, W = 1, 1, 32, 32
        sample1 = torch.rand(B, C, H, W, device=device, dtype=dtype, requires_grad=True)
        sample2 = torch.rand(B, C, H, W, device=device, dtype=dtype, requires_grad=True)
        tiny_module = torch.nn.Conv2d(1, 1, (3, 3), (1, 1), (1, 1)).to(device=device, dtype=dtype)
        out1 = kornia.contrib.distance_transform(tiny_module(sample1))
        out2 = kornia.contrib.distance_transform(sample2)
        loss = torch.nn.functional.mse_loss(out1, out2)
        loss.backward()

        B, C, D, H, W = 1, 1, 10, 10, 10
        sample3d_1 = torch.rand(B, C, D, H, W, device=device, dtype=dtype, requires_grad=True)
        sample3d_2 = torch.rand(B, C, D, H, W, device=device, dtype=dtype, requires_grad=True)
        out3 = kornia.contrib.distance_transform(sample3d_1)
        out4 = kornia.contrib.distance_transform(sample3d_2)
        loss_3d = torch.nn.functional.mse_loss(out3, out4)
        loss_3d.backward()

    def test_offset_parenthesis_fix(self, device, dtype):
        # h=0.01 at kernel_size=3 was not itself numerically wrong on CPU/float32 (it tracked the
        # analytic reference to ~2e-4, carried through the subnormal range), but #4152's fix now
        # explicitly rejects that combination -- the same call silently disagreed on MPS, which
        # flushes subnormals to zero -- so this test can no longer use it regardless. h=0.1 keeps
        # the offset accumulation this test exists to pin (n - 0.00313 per row, strictly increasing
        # by 1, so a mis-parenthesized `offset + cdt * mask` would fail this) while staying well
        # clear of the underflow guard; reference values generated at float64 via:
        #   img = torch.zeros(1, 1, 8, 4, dtype=torch.float64); img[0, 0, 1, :] = 1.0
        #   kornia.contrib.distance_transform(img, kernel_size=3, h=0.1)[0, 0, :, 0]
        if not supports_replicate_padding(device, dtype):
            pytest.skip("This device does not support replicate padding for the regression dtype")
        if not supports_conv2d(device, dtype):
            pytest.skip("This device does not support convolution for the regression dtype")

        img = torch.zeros(1, 1, 8, 4, device=device, dtype=dtype)
        img[0, 0, 1, :] = 1.0
        out = kornia.contrib.distance_transform(img, kernel_size=3, h=0.1)
        expected = torch.tensor(
            [
                [0.996872, 0.996872, 0.996872, 0.996872],
                [0.000000, 0.000000, 0.000000, 0.000000],
                [0.996872, 0.996872, 0.996872, 0.996872],
                [1.996872, 1.996872, 1.996872, 1.996872],
                [2.996872, 2.996872, 2.996872, 2.996872],
                [3.996872, 3.996872, 3.996872, 3.996872],
                [4.996872, 4.996872, 4.996872, 4.996872],
                [5.996872, 5.996872, 5.996872, 5.996872],
            ],
            device=device,
            dtype=dtype,
        )

        tolerance = 5e-3 if dtype == torch.bfloat16 else 1e-3
        self.assert_close(out[0, 0], expected, rtol=tolerance, atol=tolerance)

    # test_convention_small_h_keeps_nonzero_distances_4152, a strict xfail pinning the pre-fix silent
    # wrong-output behavior for h=0.01 at kernel_size=3 in float16/bfloat16/mps-float32, is removed
    # here: this PR fixes #4152, so that combination now raises BaseError (a different exception than
    # the pinned `raises=AssertionError`) instead of returning a wrong value, which would make the
    # xfail fail outright rather than pass. test_h_too_small_raises and
    # test_h_too_small_raises_float32_regression below pin the new, fixed behavior instead.

    def test_half_precision_matches_float64_reference(self, device, dtype):
        # #4152: the exp(-dist/h) kernel's far taps underflow to exactly zero in float16/bfloat16
        # at the documented default h=0.35 once kernel_size >= 15, and the convolution silently
        # drops them -- the peak distance below came back as ~15.7 instead of ~13.5 before the fix.
        # distance_transform now runs the cascade in float32 for half-precision inputs, so the
        # result should track the float64 reference to about float32 precision regardless of the
        # input dtype. The reference is built on its own CPU float64 tensor -- MPS has no float64,
        # so a `.double()` of the fixture's own `img` would fail there.
        if not supports_replicate_padding(device, dtype):
            pytest.skip("This device does not support replicate padding for the regression dtype")
        if not supports_conv2d(device, dtype):
            pytest.skip("This device does not support convolution for the regression dtype")

        img = torch.zeros(1, 1, 16, 4, device=device, dtype=dtype)
        img[0, 0, 1, :] = 1.0
        ref_img = torch.zeros(1, 1, 16, 4, dtype=torch.float64)
        ref_img[0, 0, 1, :] = 1.0
        reference = kornia.contrib.distance_transform(ref_img, kernel_size=15).max()
        out = kornia.contrib.distance_transform(img, kernel_size=15).max()
        self.assert_close(out.cpu(), reference.to(dtype).cpu(), rtol=1e-2, atol=1e-2)

    def test_h_too_small_raises(self, device, dtype):
        # #4152: an h too small for kernel_size at the working precision used to underflow the
        # farthest kernel tap to zero and silently return a wrong or all-zero field. It now raises
        # instead of guessing. h=1e-30 underflows at any working precision, including float64.
        img = torch.zeros(1, 1, 8, 4, device=device, dtype=dtype)
        img[0, 0, 1, :] = 1.0
        with pytest.raises(BaseError, match="too small for kernel_size"):
            kornia.contrib.distance_transform(img, kernel_size=3, h=1e-30)

    def test_h_too_small_raises_float32_regression(self, device):
        # #4152: h=0.01 at kernel_size=3 in float32 (or float16/bfloat16, upcast to float32) is not
        # itself numerically wrong on CPU -- it tracks the analytic Euclidean distance to ~2e-4,
        # carried through the subnormal range -- so the reason to reject it is not accuracy. It is
        # that MPS flushes subnormals to zero while CPU does not, so the same call would silently
        # disagree across backends; raising here makes that disagreement impossible rather than
        # trusting the accident of which backend a caller happens to run on. Checked at float32
        # explicitly, independent of the dtype fixture: float64's much smaller `finfo.tiny` does not
        # underflow at this h, so this combination legitimately only raises at float32-or-coarser
        # working precision.
        img = torch.zeros(1, 1, 8, 4, device=device, dtype=torch.float32)
        img[0, 0, 1, :] = 1.0
        with pytest.raises(BaseError, match="too small for kernel_size"):
            kornia.contrib.distance_transform(img, kernel_size=3, h=0.01)

    def test_dynamo(self, device, dtype, torch_optimizer):
        input2d = torch.rand(1, 1, 16, 16, device=device, dtype=dtype)
        op = kornia.contrib.distance_transform
        op_optimized = torch_optimizer(op)
        self.assert_close(op(input2d), op_optimized(input2d))

        input3d = torch.rand(1, 1, 8, 8, 8, device=device, dtype=dtype)
        self.assert_close(op(input3d), op_optimized(input3d))
