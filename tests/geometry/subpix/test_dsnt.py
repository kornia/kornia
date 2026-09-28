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

from testing.base import BaseTester


class TestRenderGaussian2d(BaseTester):
    @pytest.fixture
    def gaussian(self, device, dtype):
        # For a standard gaussian on 5 points [-1, -0.5, 0, 0.5, 1] with std=0.25
        # The equation is exp( -x^2 / (2 * std^2) ) -> exp( -x^2 * 8 )
        # x=0   -> exp(0)  = 1.0
        # x=0.5 -> exp(-2) ≈ 0.135335
        # x=1.0 -> exp(-8) ≈ 0.000335

        vec = torch.tensor([0.00033546, 0.13533528, 1.00000000, 0.13533528, 0.00033546], device=device, dtype=dtype)

        # Create 2D from 1D (Outer Product)
        grid = vec.unsqueeze(1) * vec.unsqueeze(0)

        # Normalize sum to 1
        return grid / grid.sum()

    def test_normalized_coordinates(self, gaussian, device, dtype):
        mean = torch.tensor([0.0, 0.0], dtype=dtype, device=device)
        std = torch.tensor([0.25, 0.25], dtype=dtype, device=device)

        actual = kornia.geometry.subpix.render_gaussian2d(mean.view(1, 2), std.view(1, 2), (5, 5), True)

        self.assert_close(actual[0], gaussian, rtol=1e-5, atol=1e-5)

    def test_pixel_coordinates(self, gaussian, device, dtype):
        mean = torch.tensor([2.0, 2.0], dtype=dtype, device=device)
        std = torch.tensor([0.5, 0.5], dtype=dtype, device=device)

        actual = kornia.geometry.subpix.render_gaussian2d(mean.view(1, 2), std.view(1, 2), (5, 5), False)

        self.assert_close(actual[0], gaussian, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("normalized", [False, True])
    def test_in_image_sums_to_one(self, device, dtype, normalized):
        # An off-centre, anisotropic Gaussian well inside a 9 x 11 grid: the per-axis renormalisation makes it sum to
        # one up to roundoff (a +1e-8 bias in the denominators left this one 8.3e-9 short, visible only in float64).
        size = (9, 11)
        mean_px, std_px = [4.3, 3.7], [1.2, 0.8]
        if normalized:
            mean = [2 * mean_px[0] / (size[1] - 1) - 1, 2 * mean_px[1] / (size[0] - 1) - 1]
            std = [2 * std_px[0] / (size[1] - 1), 2 * std_px[1] / (size[0] - 1)]
        else:
            mean, std = mean_px, std_px
        mean_t = torch.tensor([mean], device=device, dtype=dtype)
        std_t = torch.tensor([std], device=device, dtype=dtype)

        heatmap = kornia.geometry.subpix.render_gaussian2d(mean_t, std_t, size, normalized)

        total = heatmap.cpu().double().sum().item()  # on CPU: MPS has no float64
        tol = 1e-12 if dtype == torch.float64 else 4 * torch.finfo(dtype).eps
        assert abs(total - 1.0) < tol, f"sum {total!r} is not 1 within {tol}"

    @pytest.mark.parametrize("mean_x", [-14.0, -60.0])
    def test_mean_far_off_grid_renders_on_border(self, device, dtype, mean_x):
        # With std 1, the nearest x sample of a mean at -14 has exp(-98) = 2.7e-43, subnormal in float32; at -60,
        # exp(-1800) is 0 in every dtype. Either way the heatmap is the grid part of the Gaussian rescaled to one:
        # all of the x mass on column 0, and there the y profile of a mean at y = 2. The gradient stays finite.
        mean = torch.tensor([[mean_x, 2.0]], device=device, dtype=dtype, requires_grad=True)
        std = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype, requires_grad=True)

        heatmap = kornia.geometry.subpix.render_gaussian2d(mean, std, (5, 5), False)

        y_profile = torch.softmax(-0.5 * (torch.arange(5, dtype=torch.float64) - 2.0) ** 2, dim=-1)
        self.assert_close(heatmap[0, :, 0], y_profile.to(device=device, dtype=dtype))
        self.assert_close(heatmap[..., 1:], torch.zeros_like(heatmap[..., 1:]))
        (heatmap * torch.arange(25, device=device, dtype=dtype).view(1, 5, 5)).sum().backward()
        assert mean.grad is not None and std.grad is not None
        assert torch.isfinite(mean.grad).all() and torch.isfinite(std.grad).all()

    def test_dynamo(self, device, dtype, torch_optimizer):
        mean = torch.tensor([0.0, 0.0], dtype=dtype, device=device)
        std = torch.tensor([0.25, 0.25], dtype=dtype, device=device)

        op = kornia.geometry.subpix.render_gaussian2d
        op_optimized = torch_optimizer(op)

        res_orig = op(mean.view(1, 2), std.view(1, 2), (5, 5), True)
        res_opt = op_optimized(mean.view(1, 2), std.view(1, 2), (5, 5), True)

        self.assert_close(res_orig, res_opt)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("normalized", [False, True])
    @pytest.mark.parametrize("axis", ["x", "y"])
    # 2049.3 is above float16's exact-integer limit (2048); 1500.2 is below it, where only bfloat16 (limit 256)
    # collapses. One mean is not enough: bfloat16 in normalized mode is 1 px off at 2049.3 but 7 px off at 1500.2.
    @pytest.mark.parametrize("mu", [2049.3, 1500.2])
    def test_large_grid_peak_not_distorted(self, device, dtype, normalized, axis, mu):
        """The coordinate grid must not collapse in half precision (float16 above 2048, bfloat16 above ~256)."""
        n = 2200
        size = (10, n) if axis == "x" else (n, 10)
        mu_norm = mu / (n - 1) * 2 - 1
        mean_xy = [mu, 5.0] if axis == "x" else [5.0, mu]
        std_xy = [2.0, 2.0]
        if normalized:
            mean_xy = [mu_norm, 0.0] if axis == "x" else [0.0, mu_norm]
            # A sigma of 6 pixels on EACH axis, expressed in that axis's own normalized units. Wide enough to keep
            # 1 / sigma**2 inside the float16 range on the long axis, and (unlike one shared value) still wide enough
            # to cover grid points on the short axis, where a tiny sigma makes the whole heatmap round to zero.
            (h, w) = size
            std_xy = [6.0 * 2 / (w - 1), 6.0 * 2 / (h - 1)]
        mean = torch.tensor([mean_xy], dtype=dtype, device=device)
        std = torch.tensor([std_xy], dtype=dtype, device=device)

        heatmap = kornia.geometry.subpix.render_gaussian2d(mean, std, size, normalized)

        assert heatmap.dtype == dtype
        # Compare against the mean as actually stored (post half rounding), so only the grid is under test.
        stored = mean[0, 0 if axis == "x" else 1].item()
        expected = round((stored + 1) / 2 * (n - 1)) if normalized else round(stored)
        line = heatmap[0, 5] if axis == "x" else heatmap[0, :, 5]
        # Guard against a vacuous pass: an all-zero line satisfies `line[expected] == line.max()` as 0 == 0.
        assert line.max() > 0, "heatmap rounded entirely to zero, so the peak position is not being tested"
        # A collapsed coordinate grid does not move the peak to one wrong pixel, it smears the maximum over a
        # plateau of tied pixels, so `line[expected] == line.max()` alone accepts any plateau containing `expected`.
        # Require every tied maximum to sit within one pixel of it (a tie between two neighbours is legitimate).
        assert line[expected] == line.max()
        tied = (line == line.max()).nonzero().flatten()
        assert (tied - expected).abs().max() <= 1, f"peak plateau {tied.tolist()} is not centred on pixel {expected}"


class TestSpatialSoftmax2d(BaseTester):
    @pytest.fixture(params=[torch.ones(1, 1, 5, 7), torch.randn(2, 3, 16, 16)])
    def input(self, request, device, dtype):
        return request.param.to(device, dtype)

    def test_forward(self, input):
        actual = kornia.geometry.subpix.spatial_softmax2d(input)
        assert actual.lt(0).sum().item() == 0, "expected no negative values"
        sums = actual.sum(-1).sum(-1)
        self.assert_close(sums, torch.ones_like(sums))

    def test_non_contiguous(self, device, dtype):
        input = torch.randn(2, 3, 4, 6, device=device, dtype=dtype).transpose(-2, -1)
        assert not input.is_contiguous()

        expected = kornia.geometry.subpix.spatial_softmax2d(input.contiguous())
        actual = kornia.geometry.subpix.spatial_softmax2d(input)

        self.assert_close(actual, expected)

    def test_dynamo(self, input, torch_optimizer):
        op = kornia.geometry.subpix.spatial_softmax2d
        op_optimized = torch_optimizer(op)

        self.assert_close(op(input), op_optimized(input))

    @pytest.mark.parametrize("as_tensor", [False, True])
    @pytest.mark.parametrize("temperature", [0.5, 2.0])
    def test_temperature_divides_input(self, device, dtype, temperature, as_tensor):
        # An asymmetric map, so dividing by T and multiplying by T give different distributions for T != 1.
        input = torch.tensor([[[[0.0, 1.0, 3.0], [2.0, -1.0, 0.5]]]], device=device, dtype=dtype)
        t = torch.tensor(temperature, device=device, dtype=dtype) if as_tensor else temperature

        actual = kornia.geometry.subpix.spatial_softmax2d(input, t)

        # The float64 reference is computed on CPU: MPS has no float64.
        reference = input.cpu().double().reshape(1, 1, -1)
        expected = torch.softmax(reference / temperature, dim=-1).view(1, 1, 2, 3)
        self.assert_close(actual, expected.to(device=device, dtype=dtype))
        wrong = torch.softmax(reference * temperature, dim=-1).view(1, 1, 2, 3)
        assert not torch.allclose(actual.cpu().double(), wrong, atol=1e-2)

    @pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), "tensor", "nan_tensor"])
    def test_nonpositive_temperature_raises(self, device, dtype, temperature):
        input = torch.zeros(1, 1, 2, 3, device=device, dtype=dtype)
        if temperature == "tensor":
            temperature = torch.tensor(0.0, device=device, dtype=dtype)
        elif temperature == "nan_tensor":
            temperature = torch.tensor(float("nan"), device=device, dtype=dtype)
        with pytest.raises(ValueError, match="Temperature should be positive"):
            kornia.geometry.subpix.spatial_softmax2d(input, temperature)


class TestSpatialExpectation2d(BaseTester):
    @pytest.fixture(
        params=[
            (
                torch.tensor([[[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]]]),
                torch.tensor([[[1.0, -1.0]]]),
                torch.tensor([[[2.0, 0.0]]]),
            )
        ]
    )
    def example(self, request, device, dtype):
        input, expected_norm, expected_px = request.param
        return input.to(device, dtype), expected_norm.to(device, dtype), expected_px.to(device, dtype)

    def test_forward(self, example):
        input, expected_norm, expected_px = example
        actual_norm = kornia.geometry.subpix.spatial_expectation2d(input, True)
        self.assert_close(actual_norm, expected_norm)
        actual_px = kornia.geometry.subpix.spatial_expectation2d(input, False)
        self.assert_close(actual_px, expected_px)

    def test_non_contiguous(self, device, dtype):
        input = torch.rand(2, 3, 4, 6, device=device, dtype=dtype)
        input = input / input.sum(dim=(-2, -1), keepdim=True)
        input = input.transpose(-2, -1)
        assert not input.is_contiguous()

        expected = kornia.geometry.subpix.spatial_expectation2d(input.contiguous())
        actual = kornia.geometry.subpix.spatial_expectation2d(input)

        self.assert_close(actual, expected)

    def test_float64_normalized_grid_is_exact_5019(self, device):
        if device.type == "mps":
            pytest.skip("MPS does not support float64")
        # #5019: the normalised grid was built in float32 and cast afterwards, so float64 coordinates carried
        # float32 rounding error (4e-8 here). A width of 7 has spacing 1/3, which float32 cannot represent.
        heatmap = torch.zeros(1, 1, 4, 7, device=device, dtype=torch.float64)
        heatmap[0, 0, 1, 5] = 1.0
        out = kornia.geometry.subpix.spatial_expectation2d(heatmap, True)
        assert out.dtype == torch.float64
        expected = torch.tensor([[[2 / 3, -1 / 3]]], device=device, dtype=torch.float64)
        self.assert_close(out, expected, rtol=0.0, atol=1e-15)

        probs = torch.softmax(torch.randn(1, 1, 48, 64, device=device, dtype=torch.float64).flatten(-2), -1)
        probs = probs.view(1, 1, 48, 64)
        xs = torch.linspace(-1, 1, 64, device=device, dtype=torch.float64)
        ys = torch.linspace(-1, 1, 48, device=device, dtype=torch.float64)
        reference = torch.stack([(probs.sum(-2) * xs).sum(-1), (probs.sum(-1) * ys).sum(-1)], -1)
        self.assert_close(kornia.geometry.subpix.spatial_expectation2d(probs, True), reference, rtol=0.0, atol=1e-14)

    def test_bfloat16_grid_rounds_once_5019(self, device):
        # bfloat16 keeps building the grid in float32 and rounds each coordinate once. Built directly in bfloat16,
        # the pixel coordinate 2057 comes out as 2048 instead of its nearest bfloat16 value 2064.
        heatmap = torch.zeros(1, 1, 1, 3001, device=device, dtype=torch.bfloat16)
        heatmap[0, 0, 0, 2057] = 1.0
        out = kornia.geometry.subpix.spatial_expectation2d(heatmap, False)
        expected = torch.tensor([[[2064.0, 0.0]]], device=device, dtype=torch.bfloat16)
        self.assert_close(out, expected, rtol=0.0, atol=0.0)

    @pytest.mark.skip("After the op be optimized the results are not the same")
    def test_dynamo(self, dtype, device, torch_optimizer):
        data = torch.tensor([[[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]]], device=device, dtype=dtype)
        op = kornia.geometry.subpix.spatial_expectation2d
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data, True), op_optimized(data, True))
