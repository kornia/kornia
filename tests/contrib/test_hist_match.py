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
        # A-31: np.interp([-1, 0, 0.5], [0, 0, 1], [5, 6, 7]) == [5, 6, 6.5]; kornia returned 6.0 left of the step.
        x = torch.tensor([-1.0, 0.0, 0.5], device=device, dtype=dtype)
        xp = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        fp = torch.tensor([5.0, 6.0, 7.0], device=device, dtype=dtype)
        assert kornia.contrib.interp(x, xp, fp).tolist() == [5.0, 6.0, 6.5]

    def test_matching_an_image_to_itself_is_the_identity(self, device, dtype):
        # A-21: 209 (float32) / 283 (float64) of 5328 pixels changed.
        g = torch.Generator().manual_seed(0)
        x = (torch.rand(2, 3, 24, 37, generator=g) * 255).round().to(device, dtype)
        assert torch.equal(kornia.contrib.histogram_matching(x, x), x)

    def test_uint8_result_survives_a_uint8_cast(self, device):
        # A-21: [3, 1, 2] onto [10, 20, 30] gave [29.999998, 10, 20], i.e. 29 after .to(torch.uint8).
        src = torch.tensor([3, 1, 2], dtype=torch.uint8, device=device)
        tpl = torch.tensor([10, 20, 30], dtype=torch.uint8, device=device)
        assert kornia.contrib.histogram_matching(src, tpl).to(torch.uint8).tolist() == [30, 10, 20]

    def test_float32_output_stays_inside_the_template_range(self, device):
        # A-22: #5794's range guarantee leaked one ulp in 24-49 of 300 random float32 pairs.
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
