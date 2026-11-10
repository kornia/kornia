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

import math

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

import kornia
from kornia.core.exceptions import BaseError, TypeCheckError

from testing.base import BaseTester


class TestConvDistanceTransform(BaseTester):
    @pytest.mark.parametrize("ndim", [2, 3])
    @pytest.mark.parametrize("signal", ["empty", "dense", "sparse", "fractional"])
    def test_inference_matches_autograd(self, ndim, signal, device, dtype):
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

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("ndim", [2, 3])
    def test_shape_only_tensors(self, ndim):
        # Reductions must not branch on tensor data, including when no concrete values exist.
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

    def test_noncontiguous_multi_channel(self, device, dtype):
        B, C, H, W = 1, 2, 4, 4
        sample = torch.rand(B, C, H, W, device=device, dtype=dtype)
        sample = sample.transpose(2, 3)
        op = kornia.contrib.DistanceTransform()
        out = op(sample)
        assert out.shape == sample.shape
        self.assert_close(out, kornia.contrib.distance_transform(sample.contiguous()))

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
        # Stay inside (0, 1), away from clipping and minimum ties. At h=3, the small
        # sources lose to neighbours, exercising backward through multiple spatial passes.
        for shape in [(1, 1, 3, 4), (1, 1, 2, 3, 4)]:
            sample = torch.linspace(0.01, 0.9, math.prod(shape), device=device, dtype=torch.float64)
            sample = sample.reshape(shape).requires_grad_()
            self.gradcheck(kornia.contrib.distance_transform, (sample, 3, 3.0))

    @pytest.mark.parametrize(
        ("shape", "above", "below"),
        [((1, 1, 4, 5), (1, 1), (2, 3)), ((1, 1, 3, 4, 5), (1, 1, 1), (1, 2, 3))],
        ids=["2d", "3d"],
    )
    def test_gradient_is_zero_outside_the_unit_range(self, shape, above, below, device, dtype):
        # Only the source above 1 is present, so clipping makes every input gradient zero.
        sample = torch.zeros(*shape, device=device, dtype=dtype)
        sample[(0, 0, *above)] = 2.0
        sample[(0, 0, *below)] = -1.0
        sample.requires_grad_(True)
        kornia.contrib.distance_transform(sample).sum().backward()
        assert sample.grad[(0, 0, *above)] == 0
        assert sample.grad[(0, 0, *below)] == 0
        self.assert_close(sample.grad, torch.zeros_like(sample))

    @pytest.mark.parametrize("shape", [(1, 1, 4, 4), (1, 1, 4, 4, 4)], ids=["2d", "3d"])
    @pytest.mark.parametrize("has_signal", [False, True], ids=["all-zero", "one-hot"])
    def test_empty_and_sparse_masks_have_finite_gradients(self, shape, has_signal, device, dtype):
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
        # Elementwise MSE also supports CPU half backward on the oldest supported Torch.
        loss = (out1 - out2).square().mean()
        loss.backward()

        B, C, D, H, W = 1, 1, 10, 10, 10
        sample3d_1 = torch.rand(B, C, D, H, W, device=device, dtype=dtype, requires_grad=True)
        sample3d_2 = torch.rand(B, C, D, H, W, device=device, dtype=dtype, requires_grad=True)
        out3 = kornia.contrib.distance_transform(sample3d_1)
        out4 = kornia.contrib.distance_transform(sample3d_2)
        loss_3d = (out3 - out4).square().mean()
        loss_3d.backward()

    def test_half_precision_matches_float64_reference(self, device, dtype):
        # Squaring a 300-pixel displacement overflows float16 without the working-precision upcast.
        img = torch.zeros(1, 1, 1, 301, device=device, dtype=dtype)
        img[0, 0, 0, 0] = 1.0
        out = kornia.contrib.distance_transform(img)
        self.assert_close(out[0, 0, 0, -1], torch.tensor(300.0, device=device, dtype=dtype))

    def test_dynamo(self, device, dtype, torch_optimizer):
        op = kornia.contrib.distance_transform
        op_optimized = torch_optimizer(op)
        for shape in [(1, 1, 4, 35), (1, 1, 3, 4, 5)]:
            for dense in (True, False):
                sample = torch.full(shape, 0.5 if dense else 0.0, device=device, dtype=dtype)
                sample[(0, 0) + (0,) * (len(shape) - 2)] = 0.75
                sample.requires_grad_()
                expected = op(sample)
                actual = op_optimized(sample)
                self.assert_close(actual, expected)
                self.assert_close(
                    torch.autograd.grad(actual.sum(), sample)[0], torch.autograd.grad(expected.sum(), sample)[0]
                )

    def test_convention_distance_is_to_the_nearest_nonzero_pixel(self, device, dtype):
        image = torch.zeros(1, 1, 23, 37, device=device, dtype=dtype)
        image[0, 0, 4, 9] = 1.0
        out = kornia.contrib.distance_transform(image)
        # SciPy 1.17.1 measures to the nearest zero; distance_transform_edt(image == 0) at (4, 9), (4, 10), (5, 10)
        # gives [0, 1, 1.4142135].
        values = torch.stack([out[0, 0, 4, 9], out[0, 0, 4, 10], out[0, 0, 5, 10]])
        self.assert_close(values, torch.tensor([0.0, 1.0, 2.0**0.5], device=device, dtype=dtype))

    def test_convention_empty_mask_gives_zeros_like_a_full_mask(self, device, dtype):
        empty = kornia.contrib.distance_transform(torch.zeros(2, 3, 5, 7, device=device, dtype=dtype))
        full = kornia.contrib.distance_transform(torch.ones(2, 3, 5, 7, device=device, dtype=dtype))
        assert torch.equal(empty, torch.zeros_like(empty))
        assert torch.equal(full, empty)

    def test_convention_euclidean_diagonal_5956(self, device, dtype):
        image = torch.zeros(1, 1, 23, 37, device=device, dtype=dtype)
        image[0, 0, 4, 9] = 1.0
        out = kornia.contrib.distance_transform(image)
        # The displacement is (12, 12), so the Euclidean distance is sqrt(12**2 + 12**2).
        # The former cascade returned 11 + sqrt(2) = 12.414.
        self.assert_close(out[0, 0, 16, 21], torch.tensor(12.0 * 2.0**0.5, device=device, dtype=dtype))

    def test_convention_no_cluster_bias_5956(self, device, dtype):
        # Independent geometry: distances to the line, midpoint and hole boundary are 1, 14, 1, 1.
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
            [1.0, 14.0, 1.0, 1.0],
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

    def test_convention_euclidean_3d_diagonal_5956(self, device, dtype):
        image = torch.zeros(1, 1, 9, 13, 17, device=device, dtype=dtype)
        image[0, 0, 8, 12, 16] = 1
        # Distance from (0, 0, 0): sqrt(8**2 + 12**2 + 16**2).
        expected = torch.tensor(21.540659228538015, device=device, dtype=dtype)
        self.assert_close(kornia.contrib.distance_transform(image)[0, 0, 0, 0, 0], expected)

    @pytest.mark.parametrize("ndim", [2, 3])
    @pytest.mark.parametrize("strength", [0.5, 1.0])
    def test_gradient_reaches_a_distant_source(self, ndim, strength, device, dtype):
        image = torch.zeros((1, 1) + (6,) * ndim, device=device, dtype=dtype)
        image[(0, 0) + (0,) * ndim] = strength
        image.requires_grad_()
        distance = kornia.contrib.distance_transform(image)[(0, 0) + (5,) * ndim]
        # D = sqrt(sum(5**2) - h log(v)); dD/dv = -h / (2 v D).
        expected = math.sqrt(25 * ndim - 0.35 * math.log(strength))
        self.assert_close(distance, torch.tensor(expected, device=device, dtype=dtype))
        distance.backward()
        expected_grad = torch.zeros_like(image)
        expected_grad[(0, 0) + (0,) * ndim] = -0.35 / (2 * strength * expected)
        self.assert_close(image.grad, expected_grad)

    def test_convention_fractional_source_cost(self, device, dtype):
        image = torch.zeros(1, 1, 1, 5, device=device, dtype=dtype)
        image[0, 0, 0, 0] = 0.5
        image[0, 0, 0, 4] = 1.0
        # sqrt(min(x**2 - 0.35*log(0.5), (x-4)**2)), x=0,...,4.
        expected = torch.tensor(
            [0.49254595033964177, 1.114720374441941, 2.0, 1.0, 0.0], device=device, dtype=dtype
        ).reshape_as(image)
        self.assert_close(kornia.contrib.distance_transform(image), expected)

    def test_convention_tie_selects_one_source_gradient(self, device, dtype):
        image = torch.tensor([0.5, 0.0, 0.5], device=device, dtype=dtype).reshape(1, 1, 1, 3).requires_grad_()
        result = kornia.contrib.distance_transform(image)
        result[0, 0, 0, 1].backward()
        # Both sources have cost 1 - h*log(0.5). The first minimum receives the subgradient.
        expected = torch.zeros_like(image)
        expected[0, 0, 0, 0] = -0.35 / math.sqrt(1 - 0.35 * math.log(0.5))
        self.assert_close(image.grad, expected)

    def test_zero_distance_has_zero_finite_gradient(self, device, dtype):
        image = torch.ones(1, 1, 3, 4, device=device, dtype=dtype, requires_grad=True)
        result = kornia.contrib.distance_transform(image)
        result.sum().backward()
        self.assert_close(result, torch.zeros_like(image))
        self.assert_close(image.grad, torch.zeros_like(image))

    @pytest.mark.parametrize("kernel_size", [3, 7, 15])
    @pytest.mark.parametrize("h", [0.001, 0.35, 2.0])
    def test_binary_geometry_is_independent_of_parameters(self, kernel_size, h, device, dtype):
        image = torch.zeros(1, 1, 4, 40, device=device, dtype=dtype)
        image[0, 0, 0, 0] = 1
        output = kornia.contrib.distance_transform(image, kernel_size, h)
        # 3-4-5 triangle and a query past the internal tile boundary: sqrt(3**2 + 39**2).
        actual = torch.stack([output[0, 0, 3, 4], output[0, 0, 3, 39]])
        self.assert_close(actual, torch.tensor([5.0, 39.11521443121589], device=device, dtype=dtype))

    @pytest.mark.parametrize("h", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_h(self, h, device, dtype):
        with pytest.raises(BaseError, match="positive and finite"):
            kornia.contrib.distance_transform(torch.ones(1, 1, 3, 3, device=device, dtype=dtype), h=h)

    @pytest.mark.parametrize("shape", [(0, 2, 3, 4), (2, 0, 3, 4), (1, 2, 0, 4)])
    def test_empty_dimensions(self, shape, device, dtype):
        image = torch.empty(shape, device=device, dtype=dtype)
        if 0 in shape[2:]:
            with pytest.raises(BaseError, match="Spatial dimensions"):
                kornia.contrib.distance_transform(image)
        else:
            assert kornia.contrib.distance_transform(image).shape == image.shape

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_convention_multiple_sources_batched(self, ndim, device, dtype):
        # Literals from SciPy 1.17.1:
        # x = np.zeros(shape); for q in sources: x[q] = 1
        # expected = scipy.ndimage.distance_transform_edt(x == 0).tolist()
        if ndim == 2:
            shape, sources = (3, 4), [(0, 0), (2, 3)]
            expected = [[0, 1, 2, 2], [1, 1.4142135623730951, 1.4142135623730951, 1], [2, 2, 1, 0]]
        else:
            shape, sources = (3, 3, 4), [(0, 0, 0), (2, 2, 3), (1, 1, 2)]
            expected = [
                [
                    [0, 1, 1.4142135623730951, 1.7320508075688772],
                    [1, 1.4142135623730951, 1, 1.4142135623730951],
                    [2, 1.7320508075688772, 1.4142135623730951, 1.7320508075688772],
                ],
                [
                    [1, 1.4142135623730951, 1, 1.4142135623730951],
                    [1.4142135623730951, 1, 0, 1],
                    [2.23606797749979, 1.4142135623730951, 1, 1],
                ],
                [
                    [2, 1.7320508075688772, 1.4142135623730951, 1.7320508075688772],
                    [2.23606797749979, 1.4142135623730951, 1, 1],
                    [2.449489742783178, 1.7320508075688772, 1, 0],
                ],
            ]
        image = torch.zeros((2, 2) + shape, device=device, dtype=dtype)
        for q in sources:
            image[(0, 0) + q] = 1
        image[0, 1] = image[0, 0].flip(-1)
        image[1, 1] = 1  # one empty and one full channel alongside the two different source sets
        reference = torch.zeros_like(image)
        reference[0, 0] = torch.tensor(expected, device=device, dtype=dtype)
        reference[0, 1] = reference[0, 0].flip(-1)
        self.assert_close(kornia.contrib.distance_transform(image), reference)

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_partial_query_and_line_tiles_preserve_gradients(self, ndim, device, dtype):
        # Cross the 1024-query CUDA tile, including a partial line block and a partial query block.
        # Independent corner-source references, with v=0.5 and h=0.35:
        # d = [math.sqrt(sum((n - 1)**2 for n in shape[:-1]) + q*q - 0.35*math.log(0.5))
        #      for q in (0, 1023, 1024, 1030)]
        # source_grad = sum(-0.35 / (2 * 0.5 * value) for value in d)
        if ndim == 2:
            shape = (9, 1031)
            distances = [8.015148252727206, 1023.0313986391195, 1024.0313679773258, 1030.0311852567927]
            source_grad = -0.044691016989478055
        else:
            shape = (2, 5, 1031)
            distances = [4.152421162791171, 1023.0084274342578, 1024.0084192044092, 1030.0083701608999]
            source_grad = -0.08531189990185774
        image = torch.zeros((1, 1) + shape, device=device, dtype=dtype)
        source = (0, 0) + (0,) * ndim
        image[source] = 0.5
        image.requires_grad_()
        output = kornia.contrib.distance_transform(image)
        far = (0, 0) + tuple(n - 1 for n in shape[:-1])
        probes = torch.stack([output[far + (q,)] for q in (0, 1023, 1024, 1030)])
        self.assert_close(probes, torch.tensor(distances, device=device, dtype=dtype))
        probes.sum().backward()
        expected_grad = torch.zeros_like(image)
        expected_grad[source] = source_grad
        self.assert_close(image.grad, expected_grad)

    def test_line_tiles_preserve_batch_and_channel_layout(self, device, dtype):
        # This batched rectangular image crosses both row and query tiles on CPU and CUDA.
        # Each channel has one source, in a different corner; probe across both tile boundaries.
        image = torch.zeros(3, 2, 257, 353, device=device, dtype=dtype)
        image[:, 0, 0, 0] = 1
        image[:, 1, -1, -1] = 1
        actual = kornia.contrib.distance_transform(image)
        # sqrt(256**2 + 352**2), sqrt(200**2 + 300**2), sqrt(56**2 + 52**2).
        expected = torch.tensor([435.2470562795342, 360.5551275463989, 76.4198926981712], device=device, dtype=dtype)
        for batch in range(3):
            self.assert_close(actual[batch, 0, -1, -1], expected[0])
            self.assert_close(actual[batch, 1, 0, 0], expected[0])
            self.assert_close(actual[batch, 0, 200, 300], expected[1])
            self.assert_close(actual[batch, 1, 200, 300], expected[2])

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_tile_boundaries_preserve_source_gradients(self, ndim, device, dtype):
        # Independent corner-source references, spanning row and query tiles in both dimensions:
        # for v in (0.25, 0.5, 0.75, 1.0):
        #     d = math.sqrt(sum((n - 1)**2 for n in shape) - 0.35 * math.log(v))
        #     grad = -0.35 / (2 * v * d)
        if ndim == 2:
            shape = (65, 257)
            distances = [263.8796794052668, 263.87921972279895, 263.8789508254218, 263.8787600395303]
            gradients = [-0.0026527241566219233, -0.0013263643888581662, -0.0008842438269648231, -0.0006631833497087229]
        else:
            shape = (9, 17, 257)
            distances = [256.6251842727568, 256.62471159557725, 256.62443509674864, 256.62423891752707]
            gradients = [-0.002727713579568237, -0.0013638593018725949, -0.0009092405142377245, -0.0006819309069874761]
        image = torch.zeros((2, 2) + shape, device=device, dtype=dtype)
        near, far = (0,) * ndim, tuple(n - 1 for n in shape)
        for index, strength in enumerate((0.25, 0.5, 0.75, 1.0)):
            image[(index // 2, index % 2) + (near if index % 2 == 0 else far)] = strength
        image.requires_grad_()
        output = kornia.contrib.distance_transform(image)
        probes = torch.stack(
            [output[(index // 2, index % 2) + (far if index % 2 == 0 else near)] for index in range(4)]
        )
        self.assert_close(probes, torch.tensor(distances, device=device, dtype=dtype))
        probes.sum().backward()
        expected_grad = torch.zeros_like(image)
        for index, gradient in enumerate(gradients):
            expected_grad[(index // 2, index % 2) + (near if index % 2 == 0 else far)] = gradient
        self.assert_close(image.grad, expected_grad)
