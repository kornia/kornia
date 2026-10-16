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

import importlib
import math

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

import kornia
from kornia.core.exceptions import BaseError, TypeCheckError
from kornia.geometry.grid import create_meshgrid, create_meshgrid3d

from testing.base import BaseTester, supports_conv2d, supports_replicate_padding


class TestConvDistanceTransform(BaseTester):
    @pytest.mark.parametrize("ndim", [2, 3])
    def test_settled_inference_stops_work(self, ndim, device, dtype, monkeypatch):
        module = importlib.import_module("kornia.contrib.distance_transform")
        name = "filter2d" if ndim == 2 else "filter3d"
        original = getattr(module, name)
        calls = []

        def observed(*args, **kwargs):
            calls.append(None)
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, observed)
        sample = torch.ones((1, 1) + (32,) * ndim, device=device, dtype=dtype)
        output = kornia.contrib.distance_transform(sample)
        self.assert_close(output, torch.zeros_like(sample))
        expected_calls = 8 if device.type in ("cpu", "cuda") else 32
        assert len(calls) == expected_calls

    @pytest.mark.parametrize("ndim", [2, 3])
    @pytest.mark.parametrize("signal", ["empty", "dense", "sparse", "fractional"])
    def test_inference_matches_differentiable_cascade(self, ndim, signal, device, dtype):
        sample = torch.zeros((2, 1) + (17,) * ndim, device=device, dtype=dtype)
        if signal == "dense":
            sample.fill_(1)
        elif signal != "empty":
            sample[(0, 0) + (0,) * ndim] = 1 if signal == "sparse" else 0.25
            sample[(1, 0) + (8,) * ndim] = 1
        expected = kornia.contrib.distance_transform(sample.clone().requires_grad_())
        actual = kornia.contrib.distance_transform(sample)
        self.assert_close(actual, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_trace_keeps_propagating_on_new_data(self, ndim, device, dtype):
        dense = torch.ones((1, 1) + (17,) * ndim, device=device, dtype=dtype)
        traced = torch.jit.trace(kornia.contrib.DistanceTransform(), dense)
        sparse = torch.zeros_like(dense)
        sparse[(0, 0) + (0,) * ndim] = 1
        self.assert_close(traced(sparse), kornia.contrib.distance_transform(sparse))

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_make_fx_real_trace_keeps_propagating_on_new_data(self, ndim, device, dtype):
        def distance_transform(image):
            return kornia.contrib.distance_transform(image)

        empty = torch.zeros((1, 1) + (9,) * ndim, device=device, dtype=dtype)
        traced = make_fx(distance_transform, tracing_mode="real")(empty)
        sparse = torch.zeros_like(empty)
        sparse[(0, 0) + (0,) * ndim] = 1
        self.assert_close(traced(sparse), kornia.contrib.distance_transform(sparse))

    def test_function_transforms(self, device, dtype):
        sample = torch.rand(2, 1, 9, 9, device=device, dtype=dtype)
        op = kornia.contrib.distance_transform
        actual = torch.vmap(lambda x: op(x.unsqueeze(0)).squeeze(0))(sample)
        self.assert_close(actual, op(sample))
        tangent = torch.ones_like(sample)
        _, expected_jvp = torch.func.jvp(op, (sample,), (tangent,))
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(sample, tangent)
            _, actual_jvp = torch.autograd.forward_ad.unpack_dual(op(dual))
        self.assert_close(actual_jvp, expected_jvp)

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_no_grad_inference_stops_work(self, ndim, device, dtype, monkeypatch):
        # Inside no_grad nothing is recorded, so a requires_grad input may stop early as well.
        module = importlib.import_module("kornia.contrib.distance_transform")
        name = "filter2d" if ndim == 2 else "filter3d"
        original = getattr(module, name)
        calls = []

        def observed(*args, **kwargs):
            calls.append(None)
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, observed)
        sample = torch.ones((1, 1) + (32,) * ndim, device=device, dtype=dtype, requires_grad=True)
        with torch.no_grad():
            output = kornia.contrib.distance_transform(sample)
        self.assert_close(output, torch.zeros_like(sample))
        expected_calls = 8 if device.type in ("cpu", "cuda") else 32
        assert len(calls) == expected_calls

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("ndim", [2, 3])
    def test_shape_only_tensors(self, ndim):
        # Meta and fake tensors carry no data, so the eager early exit must not read a flag from them.
        shape = (1, 1) + (17,) * ndim
        assert kornia.contrib.distance_transform(torch.zeros(shape, device="meta")).shape == shape
        with torch._subclasses.fake_tensor.FakeTensorMode():
            assert kornia.contrib.distance_transform(torch.zeros(shape)).shape == shape

    @pytest.mark.parametrize("spatial_shape", [(33, 33), (17, 17, 17)])
    def test_cuda_graph_capture(self, spatial_shape, device, dtype):
        if device.type != "cuda":
            pytest.skip("CUDA graphs need a CUDA device")
        sample = torch.zeros(2, 1, *spatial_shape, device=device, dtype=dtype)
        sample[(0, 0) + (4,) * len(spatial_shape)] = 1
        sample[(1, 0) + (12,) * len(spatial_shape)] = 1
        expected = kornia.contrib.distance_transform(sample)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            kornia.contrib.distance_transform(sample)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = kornia.contrib.distance_transform(sample)
        graph.replay()
        self.assert_close(captured, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("kernel_size", [3, 5, 7])
    @pytest.mark.parametrize("shape", [(1, 3, 100, 100), (2, 2, 10, 10, 10)])
    def test_smoke(self, kernel_size, shape, device, dtype, cudnn_tf32_follows_option):
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
            kornia.contrib.DistanceTransform(6).forward(sample2d)
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
        # 0.5, off the kink at 1 where the mask is clamped and a finite difference is one-sided.
        B, C, H, W = 1, 1, 32, 32
        sample2d = torch.full((B, C, H, W), 0.5, device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(kornia.contrib.distance_transform, (sample2d,))

        B, C, D, H, W = 1, 1, 5, 5, 5
        sample3d = torch.full((B, C, D, H, W), 0.5, device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(kornia.contrib.distance_transform, (sample3d,))

    @pytest.mark.parametrize(
        ("shape", "source", "neighbours"),
        [((1, 1, 4, 5), (1, 2), 8), ((1, 1, 3, 4, 5), (1, 1, 2), 26)],
        ids=["2d", "3d"],
    )
    def test_gradient_reaches_a_unit_source(self, shape, source, neighbours, device, dtype):
        # The mask is clamped to [0, 1] with torch.where, whose derivative at a value of exactly 1 is 1 on every torch
        # version; torch.clamp's is 0 there on torch 2.14 (AGENTS.md). Each neighbour of the interior source reads
        # -h * log(tap * v) in the first cascade step, so d/dv = -h / v = -0.35 per neighbour at v = 1, and nothing
        # later depends on v: the gradient at the source is 8 * -0.35 = -2.8 in 2-D and 26 * -0.35 = -9.1 in 3-D.
        sample = torch.zeros(*shape, device=device, dtype=dtype)
        sample[(0, 0, *source)] = 1.0
        sample.requires_grad_(True)
        kornia.contrib.distance_transform(sample).sum().backward()
        expected = torch.tensor(neighbours * -0.35, device=device, dtype=dtype)
        self.assert_close(sample.grad[(0, 0, *source)], expected)

    @pytest.mark.parametrize("shape", [(1, 1, 4, 4), (1, 1, 4, 4, 4)], ids=["2d", "3d"])
    @pytest.mark.parametrize("has_signal", [False, True], ids=["all-zero", "one-hot"])
    def test_zero_convolution_has_finite_gradients(self, shape, has_signal, device, dtype):
        sample = torch.zeros(*shape, device=device, dtype=dtype)
        if has_signal:
            sample[(0, 0, *((1,) * (len(shape) - 2)))] = 1.0
        sample.requires_grad_(True)

        output = kornia.contrib.distance_transform(sample)
        assert torch.isfinite(output).all()
        if not has_signal:
            assert torch.equal(output, torch.zeros_like(output))

        output.sum().backward()
        assert sample.grad is not None
        assert sample.grad.shape == sample.shape
        assert torch.isfinite(sample.grad).all()

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

    @pytest.mark.parametrize(
        ("shape", "kernel_size", "h"),
        [((1, 1, 40, 50), 31, 0.18), ((1, 1, 40, 50), 31, 0.22), ((1, 1, 12, 40, 50), 21, 0.125)],
    )
    def test_h_too_small_for_the_corner_tap_raises(self, shape, kernel_size, h, device):
        # A guard on the axis tap exp(-k/h) alone let these h through while the corner tap
        # exp(-sqrt(ndim) k/h) underflowed (float32 17.78 vs exact 21.21 at ks 31, h 0.18; MPS vs CPU 4.1 at h 0.22).
        with pytest.raises(BaseError, match="corner kernel tap"):
            kornia.contrib.distance_transform(torch.zeros(shape, device=device), kernel_size=kernel_size, h=h)

    @pytest.mark.parametrize(
        ("shape", "kernel_size", "h"),
        [
            ((1, 1, 23, 37), 3, 0.6),
            ((1, 1, 23, 37), 7, 0.53),
            ((1, 1, 9, 11, 13), 3, 0.42),
            ((1, 1, 9, 11, 13), 5, 0.39),
        ],
    )
    def test_h_too_large_raises(self, shape, kernel_size, h, device):
        # A pixel counts as reached when -h*log(sum of its window's taps over reached pixels) > 0.
        # Once the off-centre taps can sum to >= 1, a pixel whose window is full of reached pixels reads as
        # foreground (0). Random 60 % masks, ks 3: 140 of 2805 background pixels were 0 at h 0.591, 2726 at h 2.0.
        # Bounds, float32 working precision (float64): 2-D 0.5624 (0.5626) at ks 3, 0.5084 (0.5085) at ks 5, ~0.501 at
        # ks >= 7; 3-D 0.4070 (0.4071) at ks 3, 0.3843 (0.3844) at ks 5.
        with pytest.raises(BaseError, match="too large for kernel_size"):
            kornia.contrib.distance_transform(torch.zeros(shape, device=device), kernel_size=kernel_size, h=h)

    def test_h_below_the_upper_bound_leaves_no_background_pixel_at_zero(self, device):
        g = torch.Generator().manual_seed(0)
        m = (torch.rand(8, 1, 23, 37, generator=g) < 0.6).float().to(device)
        out = kornia.contrib.distance_transform(m, kernel_size=3, h=0.55)
        assert not ((m == 0) & (out <= 0)).any()

    @pytest.mark.parametrize(
        ("shape", "accepted", "rejected"), [((1, 1, 7, 7), 0.5, 0.6), ((1, 1, 7, 7, 7), 0.35, 0.42)]
    )
    def test_largest_accepted_h_leaves_a_surrounded_background_pixel_nonzero(
        self, shape, accepted, rejected, device, dtype
    ):
        # A background pixel whose whole window is reached sums every off-centre tap. The guard checks that sum in
        # float64, but the cascade sums it in the working precision: at h=0.56256205 (2-D) the float64 sum is
        # 0.99999991, float32 rounds it to 1, and the pixel read as foreground (0). Bisect for the largest h the guard
        # accepts and check the pixel there, so the test pins the guarantee rather than the margin formula.
        if not supports_replicate_padding(device, dtype):
            pytest.skip("This device does not support replicate padding for the regression dtype")
        if not supports_conv2d(device, dtype):
            pytest.skip("This device does not support convolution for the regression dtype")

        img = torch.ones(shape, device=device, dtype=dtype)
        img[(0, 0) + (3,) * (len(shape) - 2)] = 0.0
        largest_out = None
        for _ in range(50):
            h = (accepted + rejected) / 2
            try:
                out = kornia.contrib.distance_transform(img, kernel_size=3, h=h)
            except BaseError:
                rejected = h
            else:
                accepted, largest_out = h, out
        assert largest_out is not None
        assert largest_out[(0, 0) + (3,) * (len(shape) - 2)] > 0

    @pytest.mark.parametrize(("shape", "h"), [((1, 1, 7, 7), 0.56256205), ((1, 1, 7, 7, 7), 0.40709874)])
    def test_h_whose_tap_sum_rounds_to_one_raises(self, shape, h, device):
        # The off-centre taps sum to 0.99999991 (2-D) and 0.99999994 (3-D) in float64 at these h, and to 1 in the
        # float32 cascade, which float16 and bfloat16 inputs also use.
        with pytest.raises(BaseError, match="too large for kernel_size"):
            kornia.contrib.distance_transform(torch.zeros(shape, device=device), kernel_size=3, h=h)

    def test_dynamo(self, device, dtype, torch_optimizer):
        input2d = torch.rand(1, 1, 16, 16, device=device, dtype=dtype)
        op = kornia.contrib.distance_transform
        op_optimized = torch_optimizer(op)
        self.assert_close(op(input2d), op_optimized(input2d))

        input3d = torch.rand(1, 1, 8, 8, 8, device=device, dtype=dtype)
        self.assert_close(op(input3d), op_optimized(input3d))

    def test_convention_distance_is_to_the_nearest_nonzero_pixel(self, device, dtype):
        image = torch.zeros(1, 1, 23, 37, device=device, dtype=dtype)
        image[0, 0, 4, 9] = 1.0
        out = kornia.contrib.distance_transform(image)
        # SciPy 1.17.1 measures to the nearest zero; distance_transform_edt(image == 0) at (4, 9), (4, 10), (5, 10)
        # gives [0, 1, 1.4142135]: exact within kernel_size // 2.
        values = torch.stack([out[0, 0, 4, 9], out[0, 0, 4, 10], out[0, 0, 5, 10]])
        self.assert_close(values, torch.tensor([0.0, 1.0, 2.0**0.5], device=device, dtype=dtype))

    def test_convention_empty_mask_gives_zeros_like_a_full_mask(self, device, dtype):
        empty = kornia.contrib.distance_transform(torch.zeros(2, 3, 5, 7, device=device, dtype=dtype))
        full = kornia.contrib.distance_transform(torch.ones(2, 3, 5, 7, device=device, dtype=dtype))
        assert torch.equal(empty, torch.zeros_like(empty))
        assert torch.equal(full, empty)

    def test_convention_a_value_below_one_shifts_the_distances_near_it(self, device, dtype):
        image = torch.zeros(1, 1, 23, 37, device=device, dtype=dtype)
        image[0, 0, 4, 9] = 0.5
        binary = (image > 0).to(dtype)
        out, ref = kornia.contrib.distance_transform(image), kornia.contrib.distance_transform(binary)
        shift = -0.35 * math.log(0.5)  # 0.2426, -h * log(v) at the default h
        values = torch.stack([out[0, 0, 4, 9], out[0, 0, 4, 10]])
        self.assert_close(values, torch.tensor([shift, 1.0 + shift], device=device, dtype=dtype))
        self.assert_close(out[0, 0, 4, 30], ref[0, 0, 4, 30])  # beyond kernel_size // 2 the value does not matter
        # A source sharing the window adds to the soft minimum before the log, so the shift shrinks: with a 1 beside
        # the 0.5, (2, 3) rises by 0.1689, not 0.2426. Literals from dt(pair.double())[0, 0, 2, 3] and
        # dt((pair > 0).double())[0, 0, 2, 3] at the default h.
        pair = torch.zeros(1, 1, 7, 7, device=device, dtype=dtype)
        pair[0, 0, 3, 3] = 0.5
        pair[0, 0, 3, 4] = 1.0
        dt = kornia.contrib.distance_transform
        values = torch.stack([dt(pair)[0, 0, 2, 3], dt((pair > 0).to(dtype))[0, 0, 2, 3]])
        self.assert_close(values, torch.tensor([1.0753916230203424, 0.9065031851948744], device=device, dtype=dtype))

    def test_wart_distance_transform_diagonal_underestimate_5956(self, device, dtype):
        image = torch.zeros(1, 1, 23, 37, device=device, dtype=dtype)
        image[0, 0, 4, 9] = 1.0
        out = kornia.contrib.distance_transform(image)
        # Beyond kernel_size // 2 the result follows the ring count: (16, 21) is 12 rings away and comes out as
        # 11 + sqrt(2) = 12.414, while scipy.ndimage.distance_transform_edt(image == 0) gives 12 * sqrt(2) = 16.971.
        self.assert_close(out[0, 0, 16, 21], torch.tensor(11.0 + 2.0**0.5, device=device, dtype=dtype))

    def test_wart_distance_transform_soft_min_bias_5956(self, device, dtype):
        # Where several reached pixels share a window, the soft-min pulls the value below the nearest distance
        # (exact: 1, 14, 1, 1). Literals from the same calls in float64 at the default h = 0.35:
        #   dt = distance_transform; [dt(line.double())[0, 0, 7, 10], dt(line.double())[0, 0, 20, 10],
        #   dt(two.double())[0, 0, 2, 3], dt(hole.double())[0, 0, 2, 3]]
        line = torch.zeros(1, 1, 23, 37, device=device, dtype=dtype)
        line[0, 0, 6] = 1.0
        two = torch.zeros(1, 1, 5, 7, device=device, dtype=dtype)
        two[0, 0, 2, 2] = 1.0
        two[0, 0, 2, 4] = 1.0
        hole = torch.ones(1, 1, 5, 7, device=device, dtype=dtype)
        hole[0, 0, 2, 3] = 0.0
        dt = kornia.contrib.distance_transform
        values = torch.stack([dt(line)[0, 0, 7, 10], dt(line)[0, 0, 20, 10], dt(two)[0, 0, 2, 3], dt(hole)[0, 0, 2, 3]])
        expected = torch.tensor(
            [0.8327901098243616, 13.832790109824362, 0.7573984868040191, 0.42130015880291255],
            device=device,
            dtype=dtype,
        )
        self.assert_close(values, expected)

    @pytest.mark.parametrize(
        ("shape", "source"), [((1, 1, 9, 13), (4, 6)), ((1, 1, 5, 7, 9), (2, 3, 4))], ids=["2d", "3d"]
    )
    def test_convention_values_above_one_are_full_sources(self, shape, source, device, dtype):
        # The mask is clamped to [0, 1], so a value above 1 is the same source as 1. Read unclamped, a source of value
        # v added -h * log(v) to the distances near it (its neighbour read 1 - 0.35 * log(2) = 0.7574 at v = 2), and
        # at v = 255 every pixel read 0.
        image = torch.zeros(*shape, device=device, dtype=dtype)
        image[(0, 0, *source)] = 1.0
        binary = kornia.contrib.distance_transform(image)
        assert binary.max() > 3
        for v in (2.0, 20.0, 255.0):
            self.assert_close(kornia.contrib.distance_transform(v * image), binary, rtol=0, atol=0)
        assert (kornia.contrib.distance_transform(255 * image) != 0).any()

    @pytest.mark.parametrize(
        ("shape", "source", "negatives"),
        [((1, 1, 9, 13), (4, 6), [(4, 8), (0, 0)]), ((1, 1, 5, 7, 9), (2, 3, 4), [(2, 3, 6), (0, 0, 0)])],
        ids=["2d", "3d"],
    )
    def test_convention_negative_values_are_background(self, shape, source, negatives, device, dtype):
        # The mask is clamped to [0, 1], so a negative value is background. Read unclamped, in 2-D the -1 two pixels
        # from the source cancelled its contribution to the pixel between them, which read 2.17 instead of 1, and the
        # -1 and the -5 read 0 themselves, as did the -5's neighbours (float64, default kernel_size and h).
        image = torch.zeros(*shape, device=device, dtype=dtype)
        image[(0, 0, *source)] = 1.0
        negative = image.clone()
        negative[(0, 0, *negatives[0])] = -1.0
        negative[(0, 0, *negatives[1])] = -5.0
        dt = kornia.contrib.distance_transform
        self.assert_close(dt(negative), dt(image), rtol=0, atol=0)
