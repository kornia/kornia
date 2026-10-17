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

inf = float("inf")


class TestHistMatch(BaseTester):
    def test_interp(self, device, dtype):
        xp = torch.tensor([1, 2, 3], device=device, dtype=dtype)
        fp = torch.tensor([4, 2, 0], device=device, dtype=dtype)
        x = torch.tensor([4, 5, 6], device=device, dtype=dtype)
        x_hat_expected = torch.tensor([-2.0, -4.0, -6.0], device=device, dtype=dtype)
        x_hat = kornia.contrib.interp(x, xp, fp)
        self.assert_close(x_hat_expected, x_hat)

    def test_interp_zero_width_last_segment(self, device, dtype):
        # A 2**24 + 1 pixel template with one pixel in its last bin has its last two float32 quantiles both round to 1.
        # A source quantile of 1 selects that zero-width segment, which divided by zero and gave NaN.
        xp = torch.tensor([0.0, 1.0, 1.0], device=device, dtype=dtype)
        fp = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        x = torch.tensor([0.5, 1.0], device=device, dtype=dtype)
        x_hat = kornia.contrib.interp(x, xp, fp)
        self.assert_close(x_hat, torch.tensor([0.0, 1.0], device=device, dtype=dtype))

    def test_interp_zero_width_last_segment_gradient(self, device, dtype):
        # torch.where differentiates the discarded quotient too: dividing by the raw zero width gave NaN gradients.
        xp = torch.tensor([0.0, 1.0, 1.0], device=device, dtype=dtype)
        fp = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype, requires_grad=True)
        x = torch.tensor([0.5, 1.0], device=device, dtype=dtype, requires_grad=True)
        kornia.contrib.interp(x, xp, fp).sum().backward()
        self.assert_close(fp.grad, torch.tensor([0.5, 0.5, 1.0], device=device, dtype=dtype))
        self.assert_close(x.grad, torch.zeros(2, device=device, dtype=dtype))

    def test_histmatch(self, device, dtype):
        torch.manual_seed(44)
        # generate random value by CPU.
        src = torch.randn(1, 4, 4).to(device=device, dtype=dtype)
        dst = torch.randn(1, 16, 16).to(device=device, dtype=dtype)
        out = kornia.contrib.histogram_matching(src, dst)
        exp = torch.tensor(
            [
                [
                    [1.5902, 0.9295, 2.9409, 0.1211],
                    [0.2472, 1.2137, -0.1098, -0.4272],
                    [-0.2644, -1.1983, -0.6065, -0.8091],
                    [-1.4999, 0.6370, -0.9800, 0.4474],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        assert exp.shape == out.shape
        self.assert_close(out, exp, rtol=1e-4, atol=1e-4)

    def test_histmatch_stays_in_template_range(self, device, dtype):
        # The lowest source quantile (1/4) is below the template's (3/4): it maps to the template minimum, as np.interp
        # does in the referenced implementation, instead of being extrapolated below it.
        src = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device, dtype=dtype)
        dst = torch.tensor([10.0, 10.0, 10.0, 20.0], device=device, dtype=dtype)
        out = kornia.contrib.histogram_matching(src, dst)
        exp = torch.tensor([10.0, 10.0, 10.0, 20.0], device=device, dtype=dtype)
        self.assert_close(out, exp)

    def test_histmatch_constant_template(self, device, dtype):
        src = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device, dtype=dtype)
        dst = torch.full((5,), 7.0, device=device, dtype=dtype)
        out = kornia.contrib.histogram_matching(src, dst)
        self.assert_close(out, torch.full_like(src, 7.0))

    def test_histmatch_clipped_black_template(self, device, dtype):
        # An 8-bit template with crushed shadows: 60 of its 256 pixels are 0. The darkest source pixels map to 0, as in
        # skimage.exposure.match_histograms (60 zeros); they used to be extrapolated below it (57 negative outputs).
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, 16, device=device, dtype=dtype),
            torch.linspace(0, 1, 16, device=device, dtype=dtype),
            indexing="ij",
        )
        src = ((0.1 + 0.5 * xx + 0.3 * yy) * 255).round() / 255
        dst = ((1.2 * xx + 0.3 * yy - 0.4).clamp(0, 1) * 255).round() / 255
        out = kornia.contrib.histogram_matching(src, dst)
        assert out.min() == 0
        assert int((out == 0).sum()) == 60

    def test_histmatch_large_image(self, device, dtype):
        # 64x64 is past the exact-integer range of float16 (2048) and bfloat16 (256), where summing the counts in the
        # input dtype repeated quantiles and gave NaN. The result must match the float64 computation on the same values.
        # Seed 5 is one of the pairs that gave NaN in both float16 and bfloat16.
        g = torch.Generator().manual_seed(5)
        src = torch.rand(64, 64, generator=g).to(device=device, dtype=dtype)
        dst = torch.rand(64, 64, generator=g).to(device=device, dtype=dtype)
        out = kornia.contrib.histogram_matching(src, dst)
        exp = kornia.contrib.histogram_matching(src.cpu().double(), dst.cpu().double()).to(device=device, dtype=dtype)
        assert out.dtype == dtype
        assert out.isfinite().all()
        self.assert_close(out, exp)

    def test_histmatch_float64_keeps_template_values(self, device, dtype):
        # A float64 source is matched in float64: float32 would round the template value 1 + 2**-40 to 1.
        if dtype != torch.float64:
            pytest.skip("float64 only")
        src = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
        dst = torch.tensor([1.0, 1.0 + 2**-40], device=device, dtype=dtype)
        assert torch.equal(kornia.contrib.histogram_matching(src, dst), dst)

    @pytest.mark.parametrize("size", [50, 64])
    def test_histmatch_uint8(self, device, size):
        # A uint8 count sum wraps past 255: 50x50 mapped to values up to 266, and 64x64 (4096 = 0 mod 256) to NaN.
        g = torch.Generator().manual_seed(0)
        src = (torch.rand(size, size, generator=g) * 255).round().to(device=device, dtype=torch.uint8)
        dst = (torch.rand(size, size, generator=g) * 255).round().to(device=device, dtype=torch.uint8)
        out = kornia.contrib.histogram_matching(src, dst)
        assert out.dtype == torch.float32
        assert out.isfinite().all()
        assert out.min() >= dst.min()
        assert out.max() <= dst.max()
        self.assert_close(out, kornia.contrib.histogram_matching(src.float(), dst.float()))

    @pytest.mark.skip(reason="not differentiable now.")
    def test_grad(self, device):
        B, C, H, W = 1, 3, 32, 32
        src = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        dst = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(kornia.contrib.histogram_matching, (src, dst))

    def test_interp_is_exact_at_its_knots(self, device, dtype):
        xp = torch.tensor([0.0, 0.1, 0.35, 0.7, 1.0], device=device, dtype=dtype)
        fp = torch.tensor([10.0, 20.0, 30.0, 47.0, 51.0], device=device, dtype=dtype)
        assert torch.equal(kornia.contrib.interp(xp, xp, fp), fp)

    def test_interp_left_of_a_repeated_first_knot(self, device, dtype):
        # np.interp([-1, 0, 0.5], [0, 0, 1], [5, 6, 7]) == [5, 6, 6.5]; 6.0 was returned left of the step.
        x = torch.tensor([-1.0, 0.0, 0.5], device=device, dtype=dtype)
        xp = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        fp = torch.tensor([5.0, 6.0, 7.0], device=device, dtype=dtype)
        assert kornia.contrib.interp(x, xp, fp).tolist() == [5.0, 6.0, 6.5]

    def test_matching_an_image_to_itself_is_the_identity(self, device, dtype):
        # An interp inexact at its knots changed 132 (float32) / 178 (float64) of these 5328 pixels.
        g = torch.Generator().manual_seed(0)
        x = (torch.rand(2, 3, 24, 37, generator=g) * 255).round().to(device, dtype)
        assert torch.equal(kornia.contrib.histogram_matching(x, x), x)

    def test_uint8_result_survives_a_uint8_cast(self, device):
        # An interp inexact at its knots gave [29.999998, 10, 20] for [3, 1, 2] onto [10, 20, 30], 29 as uint8.
        src = torch.tensor([3, 1, 2], dtype=torch.uint8, device=device)
        tpl = torch.tensor([10, 20, 30], dtype=torch.uint8, device=device)
        assert kornia.contrib.histogram_matching(src, tpl).to(torch.uint8).tolist() == [30, 10, 20]

    def test_float32_output_stays_inside_the_template_range(self, device):
        # An interp inexact at its knots left the template range by one ulp for some random float32 pairs.
        outside = 0
        for seed in range(300):
            g = torch.Generator().manual_seed(seed)
            n1, n2 = torch.randint(2, 60, (2,), generator=g).tolist()
            src = (torch.rand(n1, generator=g) * 3).to(device)
            tpl = (torch.rand(n2, generator=g) * 3).to(device)
            out = kornia.contrib.histogram_matching(src, tpl)
            outside += int(bool((out > tpl.max()).any() or (out < tpl.min()).any()))
        assert outside == 0

    def test_empty_source_gives_an_empty_output(self, device, dtype):
        out = kornia.contrib.histogram_matching(
            torch.zeros(0, 3, 4, 5, device=device, dtype=dtype), torch.rand(1, 3, 4, 5, device=device, dtype=dtype)
        )
        assert out.shape == (0, 3, 4, 5)
        assert out.dtype == dtype

    def test_empty_template_raises(self, device, dtype):
        with pytest.raises(ValueError, match="template"):
            kornia.contrib.histogram_matching(
                torch.rand(1, 3, 4, 5, device=device, dtype=dtype), torch.zeros(0, 3, 4, 5, device=device, dtype=dtype)
            )

    def test_interp_integer_inputs_return_floats(self, device):
        # All-integer inputs interpolate in the default float dtype: lerp has no integer kernel.
        x = torch.tensor([4, 5, 6], device=device)
        xp = torch.tensor([1, 2, 3], device=device)
        fp = torch.tensor([4, 2, 0], device=device)
        out = kornia.contrib.interp(x, xp, fp)
        assert out.dtype == torch.get_default_dtype()
        assert out.tolist() == [-2.0, -4.0, -6.0]

    @pytest.mark.parametrize(
        ("fp", "x", "expected"),
        [
            # np.interp(x, [0, 1, 2, 3], fp) for each case
            ([3.0, 4.0, inf, inf], [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], [3.0, 3.5, 4.0, inf, inf, inf, inf]),
            ([-inf, 2.0, 3.0, 4.0], [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0], [-inf, -inf, -inf, 2.0, 2.5, 3.0, 4.0]),
            ([3.0, 4.0, 5.0, 5.0], [0.5, 1.5, 2.5, 3.0], [3.5, 4.5, 5.0, 5.0]),
        ],
    )
    def test_interp_non_finite_fp_matches_numpy(self, fp, x, expected, device, dtype):
        xp = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device, dtype=dtype)
        out = kornia.contrib.interp(
            torch.tensor(x, device=device, dtype=dtype), xp, torch.tensor(fp, device=device, dtype=dtype)
        )
        assert out.tolist() == expected

    def test_template_with_minus_inf_maps_the_lowest_value_to_minus_inf(self, device):
        src = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device)
        tpl = torch.tensor([-inf, 2.0, 3.0, 4.0], device=device)
        out = kornia.contrib.histogram_matching(src, tpl)
        assert out.tolist() == [-inf, 2.0, 3.0, 4.0]

    def test_interp_propagates_nan(self, device, dtype):
        xp = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
        nan = float("nan")
        out = kornia.contrib.interp(torch.tensor([nan, 0.5], device=device, dtype=dtype), xp, xp)
        assert out[0].isnan()
        assert out[1] == 0.5
        out = kornia.contrib.interp(
            torch.tensor([0.5, 1.0], device=device, dtype=dtype),
            xp,
            torch.tensor([0.0, nan], device=device, dtype=dtype),
        )
        assert out.isnan().all()
        # A NaN query stays NaN on a flat segment and on a repeated last knot too: np.interp([nan], [0, 1], [5, 5])
        # and np.interp([nan], [0, 1, 1], [5, 6, 7]) are both nan (they gave 5.0 and 7.0).
        for xp_values, fp_values in (([0.0, 1.0], [5.0, 5.0]), ([0.0, 1.0, 1.0], [5.0, 6.0, 7.0])):
            out = kornia.contrib.interp(
                torch.tensor([nan], device=device, dtype=dtype),
                torch.tensor(xp_values, device=device, dtype=dtype),
                torch.tensor(fp_values, device=device, dtype=dtype),
            )
            assert out.isnan().all()

    def test_interp_flat_segment_with_an_infinite_knot_keeps_its_value(self, device, dtype):
        # Only a NaN query is excluded from the flat-segment value: here the weight is inf / inf = nan, but the
        # segment is flat, and np.interp([-0.5], [-inf, 0, 1], [5, 5, 6]) == [5.0].
        out = kornia.contrib.interp(
            torch.tensor([-0.5], device=device, dtype=dtype),
            torch.tensor([-inf, 0.0, 1.0], device=device, dtype=dtype),
            torch.tensor([5.0, 5.0, 6.0], device=device, dtype=dtype),
        )
        assert out.tolist() == [5.0]

    def test_interp_integer_coordinates_above_2_24_are_exact(self, device):
        # Integer coordinates are compared and subtracted as integers: as float32, 16777217 rounds to 16777216.
        # np.interp([16777217], [16777216, 16777218], [0, 2]) == [1.0]; the float32 cast gave [0.0].
        out = kornia.contrib.interp(
            torch.tensor([16777217], device=device),
            torch.tensor([16777216, 16777218], device=device),
            torch.tensor([0, 2], device=device),
        )
        assert out.tolist() == [1.0]
        # A knot whose right neighbour rounds onto it in float32: np.interp([16777216], [0, 16777216, 16777217],
        # [0, 5, 1]) == [5.0]; the float32 cast saw a zero-width segment and gave [1.0].
        out = kornia.contrib.interp(
            torch.tensor([16777216], device=device),
            torch.tensor([0, 16777216, 16777217], device=device),
            torch.tensor([0, 5, 1], device=device),
        )
        assert out.tolist() == [5.0]
        # uint8 coordinates are widened before they are subtracted: 0 - 1 does not wrap to 255.
        out = kornia.contrib.interp(
            torch.tensor([0, 3], dtype=torch.uint8, device=device),
            torch.tensor([1, 2], dtype=torch.uint8, device=device),
            torch.tensor([10.0, 20.0], device=device),
        )
        assert out.tolist() == [0.0, 30.0]

    @pytest.mark.parametrize("complex_dtype", [torch.complex64, torch.complex128])
    def test_interp_complex_fp(self, complex_dtype, device):
        # The real and imaginary parts are interpolated separately, as np.interp does:
        # np.interp([0, 0.5, 1, 1.5], [0, 1], [1+2j, 3+4j]) gives [1+2j, 2+3j, 3+4j, 3+4j]; kornia extrapolates the
        # last point to 4+5j.
        if device.type == "mps" and complex_dtype == torch.complex128:
            pytest.skip("MPS has no float64")
        real_dtype = torch.float32 if complex_dtype == torch.complex64 else torch.float64
        x = torch.tensor([0.0, 0.5, 1.0, 1.5], device=device, dtype=real_dtype)
        xp = torch.tensor([0.0, 1.0], device=device, dtype=real_dtype)
        out = kornia.contrib.interp(x, xp, torch.tensor([1 + 2j, 3 + 4j], device=device, dtype=complex_dtype))
        assert out.dtype == complex_dtype
        assert out.tolist() == [1 + 2j, 2 + 3j, 3 + 4j, 4 + 5j]

    def test_interp_half_x_with_float32_fp_keeps_float32_precision(self, device):
        # np.interp([0.3, 0.77, 1.4], [0, 1, 2], [10, 20.001, 45.123456]) in float64 with the half x rounded first:
        # [13.000788165, 17.700964891, 30.059795149]. Computing the weight in half precision was off by 2e-3.
        x = torch.tensor([0.3, 0.77, 1.4], device=device, dtype=torch.float16)
        xp = torch.tensor([0.0, 1.0, 2.0], device=device, dtype=torch.float16)
        fp = torch.tensor([10.0, 20.001, 45.123456], device=device)
        out = kornia.contrib.interp(x, xp, fp)
        expected = torch.tensor([13.000788165, 17.700964891, 30.059795149], device=device)
        assert out.dtype == torch.float32
        assert (out - expected).abs().max() < 1e-5

    def test_interp_gradient_at_knots_is_the_segment_slope(self, device):
        x = torch.tensor([1.0, 2.0, 2.5, 4.0], device=device, requires_grad=True)
        xp = torch.arange(5.0, device=device)
        kornia.contrib.interp(x, xp, 3 * xp).sum().backward()
        assert x.grad.tolist() == [3.0, 3.0, 3.0, 3.0]

    def test_interp_gradcheck(self, device):
        if device.type == "mps":
            pytest.skip("MPS has no float64")
        xp = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device, dtype=torch.float64)
        x = torch.tensor([0.3, 1.7, 2.2, 3.4], device=device, dtype=torch.float64, requires_grad=True)
        fp = torch.tensor([1.0, -2.0, 5.0, 4.0], device=device, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda a, c: kornia.contrib.interp(a, xp, c), (x, fp))

    @pytest.mark.parametrize(("float_dtype", "big"), [(torch.float32, 3e38), (torch.float16, 40000.0)])
    def test_interp_finite_ends_give_finite_values_when_their_difference_overflows(self, float_dtype, big, device):
        # f1 - f0 overflows (6e38 > float32 max, 8e4 > float16 max) but every interpolated value is finite.
        x = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device, dtype=float_dtype)
        xp = torch.tensor([0.0, 1.0], device=device, dtype=float_dtype)
        out = kornia.contrib.interp(x, xp, torch.tensor([-big, big], device=device, dtype=float_dtype))
        expected = torch.tensor([-big, -big / 2, 0.0, big / 2, big], device=device, dtype=float_dtype)
        self.assert_close(out, expected)

    @pytest.mark.parametrize(
        ("float_dtype", "big"), [(torch.float64, 1e308), (torch.float32, 3e38), (torch.float16, 40000.0)]
    )
    def test_interp_gradient_is_the_slope_when_the_difference_overflows(self, float_dtype, big, device):
        # f1 - f0 = 2 * big overflows, but the slope 2 * big / 4 = big / 2 is representable, at the knots too. The
        # formulas the overflow branch replaces used to multiply their zero gradient by the infinite difference: nan.
        if device.type == "mps" and float_dtype == torch.float64:
            pytest.skip("MPS has no float64")
        x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], device=device, dtype=float_dtype, requires_grad=True)
        xp = torch.tensor([0.0, 4.0], device=device, dtype=float_dtype)
        out = kornia.contrib.interp(x, xp, torch.tensor([-big, big], device=device, dtype=float_dtype))
        out.sum().backward()
        expected = torch.tensor([-big, -big / 2, 0.0, big / 2, big], device=device, dtype=float_dtype)
        self.assert_close(out.detach(), expected)
        assert torch.equal(x.grad, torch.full_like(x, big) / 2)

    def test_interp_extrapolation_that_overflows_stays_infinite(self, device):
        out = kornia.contrib.interp(
            torch.tensor([10.0], device=device),
            torch.tensor([0.0, 1.0], device=device),
            torch.tensor([3e38, 3.4e38], device=device),
        )
        assert out.tolist() == [float("inf")]

    def test_jit_script(self, device, dtype):
        # A recursive interp (for the complex parts) made both public functions unscriptable, also for real inputs.
        interp_script = torch.jit.script(kornia.contrib.interp)
        x = torch.linspace(-1.0, 2.0, 13, device=device, dtype=dtype)
        xp = torch.tensor([0.0, 0.5, 1.0], device=device, dtype=dtype)
        fp = torch.tensor([1.0, 3.0, -2.0], device=device, dtype=dtype)
        self.assert_close(interp_script(x, xp, fp), kornia.contrib.interp(x, xp, fp))
        ints = torch.tensor([4, 5, 6], device=device), torch.tensor([1, 2, 3], device=device)
        out = interp_script(*ints, torch.tensor([4, 2, 0], device=device))
        assert out.dtype == torch.get_default_dtype()  # all-integer inputs, as in eager
        assert out.tolist() == [-2.0, -4.0, -6.0]
        g = torch.Generator().manual_seed(0)
        source = torch.rand(1, 3, 8, 9, generator=g).to(device, dtype)
        template = (torch.rand(1, 3, 10, 7, generator=g) ** 2).to(device, dtype)
        matched = torch.jit.script(kornia.contrib.histogram_matching)(source, template)
        self.assert_close(matched, kornia.contrib.histogram_matching(source, template))

    def test_convention_histogram_matching_pools_every_element(self, device, dtype):
        source = torch.tensor([0.1, 0.4, 0.2, 0.3, 0.6, 0.5], device=device, dtype=dtype).view(2, 1, 1, 3)
        template = torch.arange(1, 7, device=device, dtype=dtype).view(2, 1, 1, 3)
        # skimage 0.26.0: match_histograms(source, template, channel_axis=None) -> [[1, 4, 2], [3, 6, 5]].
        # Matching each sample on its own would give [2, 6, 4] for both.
        expected = torch.tensor([1.0, 4.0, 2.0, 3.0, 6.0, 5.0], device=device, dtype=dtype).view(2, 1, 1, 3)
        self.assert_close(kornia.contrib.histogram_matching(source, template), expected)

    def test_convention_integer_source_returns_float32_between_template_values(self, device):
        source = torch.tensor([0, 1, 2], dtype=torch.uint8, device=device)
        out = kornia.contrib.histogram_matching(source, torch.tensor([0, 10], dtype=torch.uint8, device=device))
        assert out.dtype == torch.float32
        # skimage 0.26.0 match_histograms gives [0, 3.333, 10] (in float64)
        self.assert_close(out, torch.tensor([0.0, 10.0 / 3.0, 10.0], device=device))

    def test_convention_interp_extrapolates_where_np_interp_clamps(self, device, dtype):
        xp = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        fp = torch.tensor([4.0, 2.0, 0.0], device=device, dtype=dtype)
        x = torch.tensor([[0.0, 0.5], [4.0, 6.0]], device=device, dtype=dtype)
        # np.interp([0, 0.5, 4, 6], [1, 2, 3], [4, 2, 0]) -> [4, 4, 0, 0]
        expected = torch.tensor([[6.0, 5.0], [-2.0, -6.0]], device=device, dtype=dtype)
        self.assert_close(kornia.contrib.interp(x, xp, fp), expected)

    def test_half_overflow_segment_is_correctly_rounded(self, device):
        # f1 - f0 = 80000 overflows float16 (max 65504) while both ends are finite. Every value, the extrapolated ones
        # included, equals the float64 result rounded to float16. Reference: fp0 + (fp1 - fp0) * (x - xp0) / (xp1 - xp0)
        # evaluated in float64 on the float16 inputs.
        x = torch.linspace(-0.4, 4.4, 481, dtype=torch.float64).to(torch.float16)
        xp = torch.tensor([0.0, 4.0], dtype=torch.float16)
        fp = torch.tensor([-40000.0, 40000.0], dtype=torch.float16)
        x64, xp64, fp64 = x.double(), xp.double(), fp.double()
        expected = (fp64[0] + (fp64[1] - fp64[0]) * (x64 - xp64[0]) / (xp64[1] - xp64[0])).to(torch.float16)
        out = kornia.contrib.interp(x.to(device), xp.to(device), fp.to(device))
        assert out.dtype == torch.float16
        assert torch.equal(out.cpu(), expected)

    @pytest.mark.parametrize(
        ("x", "xp", "expected"),
        [
            # Each expected value is fp0 + (fp1 - fp0) * (x - xp0) / (xp1 - xp0) in float64, rounded to float16
            # (fp = [0, 2]); the rows are the four overflow outcomes that were measured.
            (100000, [0, 200000], 1.0),  # the weight 100000 / 200000 used to overflow to NaN
            (60000, [0, 200000], 0.6),  # the knot spacing used to overflow to inf, so the weight was a silent 0
            (1000, [0, 200000], 0.01),  # the same silent 0
            (100000, [0, 10], 20000.0),  # a weight of 10000 times a spacing of 10 used to be inf
        ],
    )
    def test_integer_coordinates_with_half_fp(self, x, xp, expected, device):
        out = kornia.contrib.interp(
            torch.tensor([x], device=device),
            torch.tensor(xp, device=device),
            torch.tensor([0.0, 2.0], device=device, dtype=torch.float16),
        )
        assert out.dtype == torch.float16
        assert torch.equal(out.cpu(), torch.tensor([expected], dtype=torch.float64).to(torch.float16))

    def test_interp_integer_coordinates_with_half_fp_are_exact_at_knots_and_keep_the_slope(self, device):
        # Integer coordinates and float16 values whose difference overflows: exact knots, fp gradient = weight.
        x = torch.tensor([0, 100000, 200000], device=device)
        xp = torch.tensor([0, 200000], device=device)
        fp = torch.tensor([-40000.0, 40000.0], device=device, dtype=torch.float16, requires_grad=True)
        out = kornia.contrib.interp(x, xp, fp)
        assert out.tolist() == [-40000.0, 0.0, 40000.0]
        out.sum().backward()
        # weights (1, .5, 0) and (0, .5, 1) summed
        assert fp.grad.tolist() == [1.5, 1.5]
