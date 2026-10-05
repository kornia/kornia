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

from kornia.core._compat import torch_version
from kornia.filters.otsu_thresholding import OtsuThreshold, otsu_threshold

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, assert_close, dynamo_is_available


class _FunctionalOtsuThreshold(torch.nn.Module):
    def __init__(self, return_mask, nbins=2):
        super().__init__()
        self.return_mask = return_mask
        self.nbins = nbins

    def forward(self, image):
        return otsu_threshold(image, nbins=self.nbins, return_mask=self.return_mask)


class TestOtsuThreshold(BaseTester):
    def test_smoke(self, device, dtype):
        img = torch.rand(1, 3, 5, 5, device=device, dtype=dtype)
        op = OtsuThreshold()
        thresh_result, _thresh_value = op(img, nbins=4)
        assert thresh_result.shape == img.shape

    @pytest.mark.parametrize("input_shape", [(3, 3), (1, 3, 3), (1, 1, 3, 3), (2, 1, 1, 3, 3)])
    def test_transform_input_shapes(self, input_shape, device, dtype):
        img = torch.rand(input_shape, device=device, dtype=dtype)
        op = OtsuThreshold()
        flat, orig_shape = op.transform_input(img)
        assert orig_shape == img.shape
        assert flat.ndim == 2

    def test_otsu_threshold_consistency(self, device, dtype):
        torch.manual_seed(0)
        img = torch.rand(1, 4, 6, 1, device=device, dtype=dtype)
        out_func_tensor, _out_func_value = otsu_threshold(img, nbins=3, return_mask=False)
        out_class_tensor, _out_class_value = OtsuThreshold()(img, nbins=3)
        assert_close(out_func_tensor, out_class_tensor)

    def test_invalid_dim(self, device, dtype):
        img = torch.rand(1, 1, 1, 1, 3, 3, device=device, dtype=dtype)
        op = OtsuThreshold()
        with pytest.raises(ValueError, match="Unsupported tensor dimensionality"):
            op.transform_input(img)

    @staticmethod
    def _separated_image(device, dtype):
        """A fixed 5x5 image whose ``nbins=3`` slow-path Otsu threshold is 2/3 and keeps 10 of 25 pixels.

        Every pixel is at least 6.3e-3 from the threshold in each tested dtype (the smallest margin is 6.35e-3, in
        float16), so a finite-difference step cannot move a pixel across it. The values are fixed rather than drawn: a
        random draw often puts the threshold at the data maximum (empty output) or at a near-tie between two splits,
        and gradcheck then fails.
        """
        values = [
            [0.00, 0.00, 0.00, 0.502, 1.00],
            [1.00, 0.08, 0.13, 0.21, 0.27],
            [0.33, 0.38, 0.44, 0.47, 0.56],
            [0.61, 0.66, 0.71, 0.77, 0.83],
            [0.88, 0.91, 0.94, 0.96, 0.98],
        ]
        return torch.tensor(values, device=device, dtype=dtype).view(1, 1, 5, 5)

    def test_gradcheck(self, device, dtype):
        img = self._separated_image(device, dtype)
        # gradcheck evaluates the op in float64 on the (possibly half-rounded) fixture, so check that input.
        img64 = img.to(torch.float64)
        out, threshold = otsu_threshold(img64, 3, True)
        # Guard against a vacuous pass: pixels must lie on both sides of the threshold, and none within 1e-3 of it,
        # which is far above the finite-difference step.
        assert (out > 0).any()
        assert not (img64 > threshold).all()
        assert (img64 - threshold).abs().min() > 1e-3
        # The analytical gradient of the thresholded image is just the mask (no gradient flows from the image output
        # through the threshold), so this only checks that no pixel crosses the threshold under the finite-difference
        # perturbation. The threshold output is left out: its value is piecewise constant and its gradient a
        # straight-through surrogate, which finite differences cannot check (see TestOtsuThresholdDifferentiable).
        self.gradcheck(lambda img: otsu_threshold(img, 3, True, False)[0], (img,))

    def test_differentiable_tensor_otsu(self, device, dtype):
        img = self._separated_image(device, dtype).requires_grad_(True)

        op = OtsuThreshold()
        result, threshold = op(img, slow_and_differentiable=True)
        mask = (img.detach() > threshold).to(dtype)
        # A non-degenerate fixture: some pixels are kept and some are dropped.
        assert mask.any()
        assert not mask.all()

        # The slow path returns the pixels above the threshold and zeros elsewhere.
        self.assert_close(result.detach(), img.detach() * mask)

        result.sum().backward()
        # The thresholded image is ``mask * x`` with a constant mask, so its gradient is the mask.
        assert img.grad is not None
        self.assert_close(img.grad, mask)

    def test_threshold_result(self, device, dtype):
        input = torch.tensor(
            [[10, 10, 10, 10], [10, 10, 10, 10], [200, 200, 200, 200], [200, 200, 200, 200]], device=device, dtype=dtype
        )

        expected = torch.tensor(
            [[0, 0, 0, 0], [0, 0, 0, 0], [200, 200, 200, 200], [200, 200, 200, 200]], device=device, dtype=dtype
        )

        op = OtsuThreshold()
        thresh_result, _thresh_value = op(input)
        self.assert_close(thresh_result, expected)

    def test_gradual_threshold(self, device, dtype):
        input = torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]], device=device, dtype=dtype)

        expected = torch.tensor([[0, 0, 0], [0, 50, 60], [70, 80, 90]], device=device, dtype=dtype)

        op = OtsuThreshold()
        thresh_result, _thresh_value = op(input)
        self.assert_close(thresh_result, expected)

    def test_uniform_result(self, device, dtype):
        input = torch.tensor(
            [[10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10]], device=device, dtype=dtype
        )

        expected = torch.zeros_like(input)

        op = OtsuThreshold()
        thresh_result, _thresh_value = op(input)
        self.assert_close(thresh_result, expected)

    def test_large_image_threshold_matches_float32(self, device, dtype):
        # #5196: the histogram counts of a 300 x 300 image (90000 pixels) came back in the input dtype, so in float16
        # their sum overflowed to inf, the normalised histogram was all zeros and the threshold was 0; bfloat16 counts
        # were rounded to 8 bits. The pixel levels k / 256 are exact in every dtype, so the float32 call sees the same
        # pixels. For float16, bfloat16 and float32 the histogram arithmetic is then the same float32 computation, and
        # the two thresholds agree up to rounding the float32 one to `dtype`. float64 bins and sums in float64, and
        # the two best splits of this image differ by only about 50 float32 eps, so it is allowed one bin (1 / 256).
        levels = torch.randint(0, 256, (1, 1, 300, 300), generator=torch.Generator().manual_seed(0))
        img = (levels / 256).to(device=device, dtype=dtype)

        _, threshold = otsu_threshold(img)
        _, expected = otsu_threshold(img.float())

        assert 0.25 < expected.item() < 0.75
        atol = 1 / 256 if dtype == torch.float64 else 0.0
        self.assert_close(threshold, expected.to(dtype), rtol=0.0, atol=atol)

    def test_two_bins_uses_histogram_split_5172(self, device, dtype):
        image = torch.tensor([[0.0, 0.1, 0.2, 0.55, 0.9, 1.0]], device=device, dtype=dtype)
        mask, threshold = otsu_threshold(image, nbins=2, return_mask=True)
        # The two histc bins split [0, 1] at 0.5. The previous nbins-point linspace returned 1 instead.
        self.assert_close(threshold, image.new_tensor([0.5]), rtol=0, atol=0)
        assert mask.tolist() == [[False, False, False, True, True, True]]

    def test_uint8_pixel_above_selected_bin_5172(self, device):
        image = torch.cat([torch.arange(256), torch.full((300,), 60), torch.full((300,), 190)])
        image = image.to(device=device, dtype=torch.uint8).view(1, 1, 8, 107)
        mask, threshold = otsu_threshold(image, return_mask=True)
        assert threshold.item() == 125
        assert mask.flatten()[126]

    @pytest.mark.parametrize(
        "input_dtype,sign,offset,nbins,expected,kept,dropped",
        [
            (torch.int16, 1, -128, 256, -3, 126, 125),
            (torch.int16, -1, 0, 256, -126, 125, 126),
            (torch.uint8, 1, 0, 255, 125, 126, 125),
        ],
        ids=["shifted_below_zero", "mirrored", "edge_on_an_integer"],
    )
    def test_integer_threshold_is_the_largest_integer_below_the_edge_5172(
        self, input_dtype, sign, offset, nbins, expected, kept, dropped, device
    ):
        # The image above, shifted below zero, mirrored, or with 255 bins, whose selected upper edge is 126 exactly.
        # Truncating the edge toward zero rounds a negative edge up and keeps an integer edge, so the level just
        # above the split (scikit-image and OpenCV keep it) was dropped from the foreground.
        values = torch.cat([torch.arange(256), torch.full((300,), 60), torch.full((300,), 190)])
        image = (sign * values + offset).to(device=device, dtype=input_dtype).view(1, 1, 8, 107)
        mask, threshold = otsu_threshold(image, nbins=nbins, return_mask=True)
        assert threshold.item() == expected
        assert mask.flatten()[kept]
        assert not mask.flatten()[dropped]

    def test_float64_threshold_uses_float64_edges_5172(self, device):
        if device.type == "mps":
            pytest.skip("MPS has no float64")
        # The single split of two bins is the midpoint 0.5 + 2**-31, which float32 rounds to 0.5.
        image = torch.tensor([[0.0, 0.1, 0.2, 0.55, 0.9, 1.0 + 2**-30]], device=device, dtype=torch.float64)
        _, threshold = otsu_threshold(image, nbins=2)
        assert threshold.item() == 0.5 + 2**-31

    @pytest.mark.parametrize("shape", [(2, 6, 10), (2, 1, 6, 10), (1, 2, 6, 10), (1, 2, 1, 6, 10)])
    @pytest.mark.parametrize("slow_and_differentiable", [False, True])
    def test_planes_use_independent_ranges_5172(self, shape, slow_and_differentiable, device, dtype):
        first = torch.linspace(0, 1, 60, device=device, dtype=dtype).square().view(6, 10)
        second = torch.linspace(0.6, 4.3, 60, device=device, dtype=dtype).view(6, 10)
        image = torch.stack([first, second]).reshape(shape)
        expected = [otsu_threshold(plane, slow_and_differentiable=slow_and_differentiable) for plane in [first, second]]
        result, threshold = otsu_threshold(image, slow_and_differentiable=slow_and_differentiable)
        self.assert_close(threshold, torch.cat([item[1] for item in expected]), rtol=0, atol=0)
        self.assert_close(result, torch.stack([item[0] for item in expected]).reshape(shape), rtol=0, atol=0)

    @pytest.mark.parametrize("value", [-0.4, 0.0, 0.4])
    @pytest.mark.parametrize("slow_and_differentiable", [False, True])
    def test_constant_threshold_is_plane_value_5172(self, value, slow_and_differentiable, device, dtype):
        image = torch.full((1, 1, 4, 5), value, device=device, dtype=dtype)
        mask, threshold = otsu_threshold(image, slow_and_differentiable=slow_and_differentiable, return_mask=True)
        self.assert_close(threshold, image.flatten()[:1], rtol=0, atol=0)
        assert not mask.any()

    @pytest.mark.parametrize("slow_and_differentiable", [False, True])
    @pytest.mark.parametrize(
        "input_dtype,value", [(torch.int32, 2**24 + 1), (torch.int64, 2**24 + 1), (torch.int64, 2**53 + 1)]
    )
    @pytest.mark.parametrize("sign", [-1, 1])
    def test_constant_integer_preserves_exact_value_5172(
        self, input_dtype, value, sign, slow_and_differentiable, device
    ):
        image = torch.full((2, 3), sign * value, dtype=input_dtype, device=device)
        mask, threshold = otsu_threshold(image, slow_and_differentiable=slow_and_differentiable, return_mask=True)
        assert torch.equal(threshold, image.flatten()[:1])
        assert not mask.any()

    @pytest.mark.parametrize("offset", [-1.0, -126 / 256, 0.0], ids=["negative", "zero", "positive"])
    def test_float_pixel_on_selected_edge_is_foreground_5422(self, offset, device, dtype):
        levels = torch.cat([torch.arange(257), torch.full((300,), 60), torch.full((300,), 190)])
        image = (levels / 256 + offset).to(device=device, dtype=dtype).view(1, -1)
        mask, threshold = otsu_threshold(image, return_mask=True)
        edge = torch.tensor([126 / 256 + offset], dtype=dtype)
        expected = torch.nextafter(edge, torch.full_like(edge, -torch.inf)).to(device)
        if device.type == "mps" and dtype in (torch.float32, torch.bfloat16) and offset == -126 / 256:
            # Metal comparisons flush subnormals to zero; the closest usable negative threshold is normal.
            expected = image.new_tensor([-torch.finfo(dtype).tiny])
        # histc places the pixel at the upper edge in the next bin. Strict comparison must keep it too,
        # including when that foreground pixel is zero or negative.
        self.assert_close(threshold, expected, rtol=0, atol=0)
        assert mask.sum().item() == 431
        assert mask[0, 126]
        assert not mask[0, 125]
        self.assert_close(mask, image > threshold)

    def test_float_and_integer_edges_keep_same_pixels_5422(self, device):
        levels = torch.cat([torch.arange(257), torch.full((300,), 60), torch.full((300,), 190)])
        image = levels.to(device=device, dtype=torch.int16).view(1, -1)
        integer_mask, integer_threshold = otsu_threshold(image, return_mask=True)
        float_mask, _ = otsu_threshold(image.float() / 256, return_mask=True)
        assert integer_threshold.item() == 125
        assert integer_mask.sum().item() == 431
        self.assert_close(float_mask, integer_mask)

    def test_rounded_bfloat16_edge_keeps_foreground_pixel_5422(self, device):
        # The float32 histogram edge is 0.505859375, which rounds up to the existing pixel 0.5078125.
        image = torch.tensor([[0.01171875, 0.01171875, 0.5078125, 1.0, 1.0]], device=device, dtype=torch.bfloat16)
        mask, threshold = otsu_threshold(image, nbins=2, return_mask=True)
        self.assert_close(threshold, image.new_tensor([0.50390625]), rtol=0, atol=0)
        assert mask.tolist() == [[False, False, True, True, True]]

    def test_downward_rounded_edge_keeps_background_pixel_excluded_5422(self, device):
        # The float32 edge 0.501953125 rounds down to 0.5. That pixel belongs to the background bin,
        # so an unconditional nextafter would incorrectly promote it to the foreground.
        image = torch.tensor([[0.00390625, 0.00390625, 0.5, 1.0, 1.0]], device=device, dtype=torch.bfloat16)
        mask, threshold = otsu_threshold(image, nbins=2, return_mask=True)
        self.assert_close(threshold, image.new_tensor([0.5]), rtol=0, atol=0)
        assert mask.tolist() == [[False, False, False, True, True]]

    def test_background_pixel_above_interpolated_edge_stays_background_5422(self, device):
        # histc and the bin index put -1.1049999 in the lower of two bins over [-9.03, 6.82], but the interpolated
        # float32 edge -1.1050000 lies one ulp below it. Comparing against the edge alone made it foreground.
        image = torch.tensor([[-9.03, -9.03, -1.1049998998641968, 6.82, 6.82]], device=device, dtype=torch.float32)
        mask, threshold = otsu_threshold(image, nbins=2, return_mask=True)
        assert mask.tolist() == [[False, False, False, True, True]]
        self.assert_close(threshold, image[:, 2], rtol=0, atol=0)

    @pytest.mark.parametrize("middle", [0.5, 0.6], ids=["on_edge", "above_edge"])
    def test_slow_path_edge_correction_keeps_foreground_pixel_5422(self, middle, device, dtype):
        # Three bins over [0, 1.5] put the first upper edge at 0.5, the split the KDE selects. A pixel on that edge
        # belongs to the foreground and must not be removed by the strict comparison. Without a collision, keep the
        # edge exactly.
        image = torch.tensor([[0.0, 0.0, 0.0, middle, 1.5, 1.5]], device=device, dtype=dtype)
        mask, threshold = otsu_threshold(image, nbins=3, slow_and_differentiable=True, return_mask=True)
        expected = torch.tensor([0.5], dtype=dtype)
        if middle == 0.5:
            expected = torch.nextafter(expected, torch.full_like(expected, -torch.inf))
        self.assert_close(threshold, expected.to(device), rtol=0, atol=0)
        assert mask.tolist() == [[False, False, False, True, True, True]]

    def test_slow_path_downward_rounded_edge_keeps_background_pixel_excluded_5422(self, device):
        # The float32 edge 0.50065 over [2**-10, 1.5] rounds down to the pixel 0.5 in bfloat16. That pixel lies below
        # the edge, so lowering the threshold to keep it, as for an upward-rounded edge, would promote it.
        image = torch.tensor([[2**-10] * 3 + [0.5, 1.5, 1.5]], device=device, dtype=torch.bfloat16)
        mask, threshold = otsu_threshold(image, nbins=3, slow_and_differentiable=True, return_mask=True)
        self.assert_close(threshold, image.new_tensor([0.5]), rtol=0, atol=0)
        assert mask.tolist() == [[False, False, False, False, True, True]]

    def test_edge_correction_is_independent_for_each_plane_5422(self, device, dtype):
        image = torch.tensor(
            [[[0.0, 0.25, 0.5, 1.0]], [[0.0, 0.25, 0.75, 1.0]], [[0.5, 0.5, 0.5, 0.5]]],
            device=device,
            dtype=dtype,
        )
        mask, threshold = otsu_threshold(image, nbins=2, return_mask=True)
        edge = torch.tensor(0.5, dtype=dtype)
        corrected = torch.nextafter(edge, torch.full_like(edge, -torch.inf))
        expected = torch.stack([corrected, edge, edge]).to(device)
        # Only the first plane has a pixel equal to its edge. The unmatched edge and constant plane stay exact.
        self.assert_close(threshold, expected, rtol=0, atol=0)
        assert mask.tolist() == [[[False, False, True, True]], [[False, False, True, True]], [[False] * 4]]

    @pytest.mark.parametrize("distribution", ["bimodal", "squared"])
    def test_empty_gap_uses_first_occupied_split_5421(self, distribution, device):
        # Generate exactly the same float32 pixels on every backend. Empty bins after the winning occupied bin
        # leave both classes unchanged, but parallel cumsum roundoff used to give one of them a higher score on MPS.
        if distribution == "bimodal":
            noise = torch.rand(1000, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
            image = torch.cat([0.3 + 0.1 * noise[:500], 0.6 + 0.1 * noise[500:]]).float().view(20, 50)
            expected_threshold, foreground = 0.4000208377838135, 500
        else:
            image = torch.linspace(0, 1, 60).square().view(6, 10)
            expected_threshold, foreground = 101 / 256, 22
        # Independent torch.histc Otsu references, excluding empty candidate bins, select bins 63 and 100.
        image = image.to(device)
        mask, threshold = otsu_threshold(image, return_mask=True)
        self.assert_close(threshold, image.new_tensor([expected_threshold]), rtol=0, atol=1e-7)
        assert mask.sum().item() == foreground

    @pytest.mark.parametrize("nbins", [2, 17, 256])
    def test_tensor_histogram_matches_histc_at_bin_boundaries_5425(self, nbins, device, dtype):
        stats_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        planes = []
        for low, high in [(-0.7, 1.1), (32.0, 34.5), (0.0, 1.0 + 2**-30)]:
            edges = torch.linspace(low, high, nbins + 1, dtype=stats_dtype).to(dtype)
            interior = edges[1:-1]
            planes.append(
                torch.cat(
                    [
                        edges,
                        torch.nextafter(interior, torch.full_like(interior, -torch.inf)),
                        torch.nextafter(interior, torch.full_like(interior, torch.inf)),
                    ]
                )
            )
        planes.append(torch.full_like(planes[0], -0.375))
        image = torch.stack(planes).to(device)
        histograms, edges, _ = OtsuThreshold._OtsuThreshold__histogram(image, nbins)
        for i, plane in enumerate(image.to(stats_dtype)):
            low, high = plane.min().item(), plane.max().item()
            if low == high:
                expected = torch.zeros_like(histograms[i])
                expected[0] = 1
            else:
                # torch.histc independently pins the bin-assignment arithmetic, including adjacent floating values.
                expected = torch.histc(plane, bins=nbins, min=low, max=high)
                expected = expected / expected.sum()
            self.assert_close(histograms[i], expected, rtol=0, atol=0)
            expected_edges = torch.linspace(low, high, nbins + 1, device=device, dtype=stats_dtype)
            self.assert_close(edges[i], expected_edges)

    def test_integer_histogram_matches_histc_with_offsets_5425(self, device):
        image = torch.stack([torch.arange(-128, 129), torch.arange(1000, 1257)]).to(device=device, dtype=torch.int16)
        histograms, _, _ = OtsuThreshold._OtsuThreshold__histogram(image, 17)
        for i, plane in enumerate(image.float()):
            expected = torch.histc(plane, bins=17, min=plane.min().item(), max=plane.max().item())
            self.assert_close(histograms[i], expected / expected.sum(), rtol=0, atol=0)

    @staticmethod
    def _capture_images(device, dtype):
        image = torch.tensor([[[[0.0, 0.25, 0.5, 1.0]], [[0.5, 0.5, 0.5, 0.5]]]], device=device, dtype=dtype)
        different = torch.tensor([[[[2.0, 2.0, 2.0, 2.0]], [[-1.0, -0.5, 0.0, 1.0]]]], device=device, dtype=dtype)
        return image, different

    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    @pytest.mark.parametrize("api", ["module", "function", "mask"])
    def test_export_reuses_graph_with_new_ranges_5425(self, api, device, dtype):
        image, different = self._capture_images(device, dtype)
        op = OtsuThreshold() if api == "module" else _FunctionalOtsuThreshold(return_mask=api == "mask")
        extra_args = (2,) if api == "module" else ()
        exported = torch.export.export(op, (image, *extra_args), strict=True).module()
        # Swap which plane is constant and change both ranges; capture must not specialize on the observed values.
        for sample in (image, different):
            expected = op(sample, *extra_args)
            actual = exported(sample, *extra_args)
            self.assert_close(actual[0], expected[0], rtol=0, atol=0)
            self.assert_close(actual[1], expected[1], rtol=0, atol=0)

    @pytest.mark.parametrize("return_mask", [False, True])
    def test_dynamo_fullgraph_reuses_new_ranges_5425(self, device, dtype, torch_optimizer, return_mask):
        image, different = self._capture_images(device, dtype)
        op = _FunctionalOtsuThreshold(return_mask=return_mask)
        compiled = torch_optimizer(op, fullgraph=True)
        for sample in (image, different):
            expected = op(sample)
            actual = compiled(sample)
            self.assert_close(actual[0], expected[0], rtol=0, atol=0)
            self.assert_close(actual[1], expected[1], rtol=0, atol=0)

    def test_dynamo_bin_edge_pixels_match_eager_5425(self, device, dtype, torch_optimizer):
        generator = torch.Generator().manual_seed(0)
        low = torch.rand(20, 1, generator=generator, dtype=dtype) * 8 - 4
        high = low + torch.rand(20, 1, generator=generator, dtype=dtype) * 5 + 0.01
        image = (torch.linspace(0, 1, 257, dtype=dtype)[None, :] * (high - low) + low).reshape(20, 1, 257)
        image = image.to(device)
        op = _FunctionalOtsuThreshold(return_mask=True, nbins=256)
        expected = op(image)
        actual = torch_optimizer(op, fullgraph=True)(image)
        # Fusing the edge interpolation can move the threshold by one ULP, but must not change bin membership.
        self.assert_close(actual[0], expected[0], rtol=0, atol=0)
        self.assert_close(actual[1], expected[1])


def test_mask(device, dtype):
    input = torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]], device=device, dtype=dtype)

    expected = torch.tensor([[0, 0, 0], [0, 1, 1], [1, 1, 1]], device=device, dtype=torch.bool)

    thresh_result, _thresh_value = otsu_threshold(input, return_mask=True)
    assert_close(thresh_result, expected)


@pytest.mark.parametrize("slow_and_differentiable", [False, True])
@pytest.mark.parametrize(
    "values",
    [(-1.0, 0.0), (-2.0, -1.0), (0.0, 1.0), (1.0, 2.0)],
)
def test_mask_is_the_comparison_5173(values, slow_and_differentiable, device, dtype):
    # #5173: the mask used to be `result > 0`, so a foreground pixel of value 0 or below, above the threshold but not
    # above 0, came out False. The mask is `x > threshold`, whatever the sign of the data.
    low, high = values
    x = torch.tensor([[low, low, high, high]], device=device, dtype=dtype)
    mask, threshold = otsu_threshold(x, slow_and_differentiable=slow_and_differentiable, return_mask=True)
    assert mask.dtype == torch.bool
    assert_close(mask, x > threshold)
    assert mask.tolist() == [[False, False, True, True]]


def test_mask_per_channel_threshold_5173(device, dtype):
    # Each channel of a (B, C, H, W) input has its own threshold; the mask follows its channel's comparison and keeps
    # the input shape. Channel 0 is non-positive data, channel 1 positive.
    x = torch.tensor([[[[-1.0, -1.0], [0.0, 0.0]], [[1.0, 1.0], [2.0, 2.0]]]], device=device, dtype=dtype)
    mask, threshold = otsu_threshold(x, return_mask=True)
    assert mask.shape == x.shape
    assert threshold.shape == (2,)
    assert_close(mask, x > threshold.reshape(1, 2, 1, 1))
    assert mask[0, 0].tolist() == [[False, False], [True, True]]
    assert mask[0, 1].tolist() == [[False, False], [True, True]]


@pytest.mark.parametrize("slow_and_differentiable", [False, True])
def test_mask_agrees_with_the_thresholded_image_5173(slow_and_differentiable, device):
    # Integer input: the threshold is truncated to the input dtype, so on the fast path one pixel (0) equals it, and a
    # `>=` in the mask would keep that pixel while the thresholded image drops it.
    x = torch.tensor([[-30, -20, -10], [0, 10, 20], [30, 40, 50]], device=device)
    image, threshold = otsu_threshold(x, slow_and_differentiable=slow_and_differentiable)
    mask, _ = otsu_threshold(x, slow_and_differentiable=slow_and_differentiable, return_mask=True)
    assert torch.equal(mask, x > threshold)
    assert torch.equal(image, torch.where(mask, x, torch.zeros_like(x)))
    assert slow_and_differentiable or (x == threshold).any()  # the fixture does contain a pixel equal to the threshold


@pytest.mark.parametrize("shape", [(1, 3, 5, 5), (2, 1, 10, 10)])
def test_otsu_threshold_basic(shape, device, dtype):
    img = torch.rand(shape, device=device, dtype=dtype)
    thresh_result, _thresh_value = otsu_threshold(img)
    assert thresh_result.shape == img.shape


def _bimodal_uint8_image(device):
    # the image of #5172: every value 0..255 once, plus 300 pixels at 60 and 300 at 190
    values = torch.cat([torch.arange(256), torch.full((300,), 60), torch.full((300,), 190)])
    return values.to(device=device, dtype=torch.uint8).view(1, 1, 8, 107)


def _quantile_mixture(device, dtype, dark=(0.3, 0.06, 700), bright=(0.7, 0.1, 300), bright_shift=0.0, levels=None):
    """A (1, 1, 20, 50) image of two modes, each ``(mean, std, count)``, at the quantiles of a normal distribution.

    Quantiles rather than a random draw: the image is the same on every device and in every run. The default modes
    overlap, so the threshold sits between them and follows them. With ``levels``, the values are rounded to multiples
    of ``1 / levels``, as in an 8-bit image scaled to [0, 1].
    """

    def quantiles(n):
        p = (torch.arange(n, dtype=torch.float64) + 0.5) / n
        return 2**0.5 * torch.erfinv(2 * p - 1)

    (dark_mean, dark_std, dark_count), (bright_mean, bright_std, bright_count) = dark, bright
    x = torch.cat(
        [
            dark_mean + dark_std * quantiles(dark_count),
            bright_mean + bright_shift + bright_std * quantiles(bright_count),
        ]
    )
    if levels is not None:
        x = (x * levels).round().clamp(0, levels) / levels
    return x.view(1, 1, 20, 50).to(device=device, dtype=dtype)


def _documented_soft_threshold(x, nbins, bandwidth=0.5, temperature=0.01):
    # The formula of otsu_threshold's note, written independently: Gaussian bin masses with the tails kept in the end
    # bins, the between-class variance in bins, and a softmax at `temperature` times its maximum over the upper edges
    # k + 1 of the splits.
    v = x.flatten()
    lo, hi = v.min(), v.max()
    c = (v - lo) * nbins / (hi - lo)
    edges = torch.arange(1, nbins, dtype=v.dtype, device=v.device)
    above = 0.5 * (1 + torch.erf((c[:, None] - edges) / (bandwidth * 2**0.5))).mean(0)
    p = -torch.diff(torch.cat([above.new_ones(1), above, above.new_zeros(1)]))
    k = torch.arange(nbins, dtype=v.dtype, device=v.device)
    w0, s0 = p.cumsum(0)[:-1], (p * k).cumsum(0)[:-1]
    w1, s1 = 1 - w0, (p * k).sum() - s0
    var = w0 * w1 * (s0 / w0 - s1 / w1) ** 2
    return lo + (torch.softmax(var / (temperature * var.max()), 0) * edges).sum() * (hi - lo) / nbins


class TestOtsuThresholdDifferentiable(BaseTester):
    # No gradcheck on the threshold: its value comes from a hard split, while its gradient is that of
    # a soft-argmax over a smoother between-class variance curve (straight-through), so finite differences of the
    # threshold do not match it by design. These tests check that the gradient exists, is finite, is consistent with
    # translating a plane, reaches every pixel and points the right way.
    def test_threshold_has_a_gradient_5174(self, device, dtype):
        x = _quantile_mixture(device, dtype).requires_grad_(True)
        out, threshold = otsu_threshold(x, slow_and_differentiable=True)
        assert threshold.requires_grad
        (grad,) = torch.autograd.grad(threshold.sum(), x, retain_graph=True)
        assert grad.isfinite().all()
        assert grad.count_nonzero() > 0

        # the gradient leaves the value alone: the same input without a gradient gets the same threshold
        _, detached = otsu_threshold(x.detach(), slow_and_differentiable=True)
        assert not detached.requires_grad
        self.assert_close(threshold.detach(), detached, rtol=0.0, atol=0.0)

        # the thresholded image is x * (x > threshold): its gradient is the mask
        (grad,) = torch.autograd.grad(out.sum(), x)
        self.assert_close(grad, (x.detach() > threshold.detach()).to(dtype))

        # the default path's threshold has no gradient
        _, threshold = otsu_threshold(x)
        assert not threshold.requires_grad

    @pytest.mark.parametrize("scale_kind", ["large", "small", "subnormal"])
    @pytest.mark.parametrize("signed", [False, True])
    def test_finite_extreme_ranges_5426(self, scale_kind, signed, device, dtype):
        if scale_kind == "subnormal" and device.type == "mps" and dtype != torch.float16:
            pytest.skip("MPS flushes float32 and bfloat16 subnormals to zero")
        limits = torch.finfo(dtype)
        scale = {"large": limits.max, "small": limits.tiny * 64, "subnormal": limits.tiny / 64}[scale_kind]
        levels = torch.tensor([[0.0, 0.1, 0.2, 0.6, 0.8, 1.0]], device=device, dtype=dtype)
        if signed:
            levels = 2 * levels - 1
        x = (levels * scale).requires_grad_(True)
        out, threshold = otsu_threshold(x, slow_and_differentiable=True)
        (grad,) = torch.autograd.grad(threshold.sum(), x, retain_graph=True)
        assert threshold.isfinite().all()
        assert grad.isfinite().all()
        self.assert_close(grad.float().sum(), grad.new_tensor(1.0).float(), rtol=0, atol=1e-2)
        assert torch.equal(threshold.detach(), otsu_threshold(x.detach(), slow_and_differentiable=True)[1])
        (image_grad,) = torch.autograd.grad(out.sum(), x)
        self.assert_close(image_grad, (x.detach() > threshold.detach()).to(dtype))

        # Compare against the independent documented formula on the same quantized pixels in ordinary units.
        # The derivative of s * f(x / s) is f'(x / s), so a finite scale must leave this gradient unchanged.
        work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        normalized = (x.detach().to(work_dtype) / scale).requires_grad_(True)
        (expected_grad,) = torch.autograd.grad(_documented_soft_threshold(normalized, 256), normalized)
        self.assert_close(grad, expected_grad.to(dtype))
        expected = otsu_threshold(normalized.detach(), slow_and_differentiable=True)[1]
        self.assert_close(
            threshold.detach().to(work_dtype) / scale,
            expected,
            rtol=max(8 * limits.eps, 1e-11),
            # Subnormal thresholds round in units of the smallest subnormal, rather than relative to their value.
            atol=limits.tiny * limits.eps / scale,
        )

    @pytest.mark.parametrize("scale_kind", ["large", "subnormal"])
    def test_extreme_constant_plane_gradient_5426(self, scale_kind, device, dtype):
        if scale_kind == "subnormal" and device.type == "mps" and dtype != torch.float16:
            pytest.skip("MPS flushes float32 and bfloat16 subnormals to zero")
        limits = torch.finfo(dtype)
        value = limits.max if scale_kind == "large" else limits.tiny / 64
        x = torch.full((2, 3), value, device=device, dtype=dtype, requires_grad=True)
        mask, threshold = otsu_threshold(x, slow_and_differentiable=True, return_mask=True)
        (grad,) = torch.autograd.grad(threshold.sum(), x)
        assert torch.equal(threshold.detach(), x.detach().flatten()[:1])
        assert not mask.any()
        self.assert_close(grad, torch.full_like(x, 1 / x.numel()))

    def test_low_contrast_offset_plane_keeps_its_gradient_and_split_5426(self, device, dtype):
        # Levels one ulp apart around 1.5: scaling by the largest magnitude before subtracting the minimum rounds their
        # differences, which changed the gradient and promoted level 2 at 256 bins. Translating the plane must not.
        levels = torch.tensor([[0, 1, 2, 6, 8, 10]], device=device, dtype=dtype)
        x = (1.5 + levels * torch.finfo(dtype).eps).requires_grad_(True)
        (grad,) = torch.autograd.grad(otsu_threshold(x, 8, slow_and_differentiable=True)[1].sum(), x)
        reference = x.detach().cpu().double().requires_grad_(True)
        (expected,) = torch.autograd.grad(_documented_soft_threshold(reference, 8), reference)
        self.assert_close(grad, expected.to(device=device, dtype=dtype), rtol=1e-3, atol=1e-3)
        mask, _ = otsu_threshold(x.detach(), 256, slow_and_differentiable=True, return_mask=True)
        expected_mask, _ = otsu_threshold(levels, 256, slow_and_differentiable=True, return_mask=True)
        assert mask.tolist() == expected_mask.tolist() == [[False, False, False, True, True, True]]

    def test_narrow_two_level_plane_keeps_background_when_edge_rounds_down_5426(self, device, dtype):
        # The first upper edge lies less than half an ulp above the minimum and rounds down onto it. That threshold
        # already separates the levels, so it must not be lowered below the background pixel.
        x = 1.5 + torch.tensor([[0, 10, 10]], device=device, dtype=dtype) * torch.finfo(dtype).eps
        mask, threshold = otsu_threshold(x, 256, slow_and_differentiable=True, return_mask=True)
        assert mask.tolist() == [[False, True, True]]
        self.assert_close(threshold, x[:, 0], rtol=0, atol=0)

    def test_subnormal_range_keeps_background_when_edge_rounds_down_5426(self, device, dtype):
        if device.type == "mps" and dtype != torch.float16:
            pytest.skip("MPS flushes float32 and bfloat16 subnormals to zero")
        # The selected edge is 86 / 256 * 3u = 1.0078125u, which rounds down onto the background pixel u.
        u = torch.nextafter(torch.zeros((), dtype=dtype), torch.ones((), dtype=dtype)).item()
        x = torch.tensor([[0, 1, 3]], device=device, dtype=dtype) * u
        mask, _ = otsu_threshold(x, 256, slow_and_differentiable=True, return_mask=True)
        assert mask.tolist() == [[False, False, True]]

    def test_threshold_gradient_sums_to_one_per_plane_5174(self, device, dtype):
        # Moving every pixel of a plane by c moves its threshold by c and no other plane's, so the gradient of each
        # plane's threshold sums to 1 over that plane and is 0 on the others. A constant plane's threshold is its value,
        # whose gradient its pixels share.
        planes = [
            _quantile_mixture(device, dtype),
            _quantile_mixture(device, dtype, dark=(0.3, 0.08, 600), bright=(0.7, 0.08, 400), levels=255),
            torch.linspace(0, 1, 1000, dtype=torch.float64).square().view(1, 1, 20, 50).to(device=device, dtype=dtype),
            torch.full((1, 1, 20, 50), 0.4, device=device, dtype=dtype),
        ]
        x = torch.cat(planes).view(2, 2, 20, 50).requires_grad_(True)
        _, thresholds = otsu_threshold(x, slow_and_differentiable=True)
        (grad,) = torch.autograd.grad(thresholds.sum(), x, retain_graph=True)
        # bfloat16 rounds each pixel's gradient to 8 bits, and the rounding errors add up over the plane
        atol = 3e-3 if dtype == torch.bfloat16 else 1e-3
        self.assert_close(grad.float().sum(dim=(-2, -1)), torch.ones(2, 2, device=device), rtol=0.0, atol=atol)
        self.assert_close(grad[1, 1], torch.full_like(grad[1, 1], 1 / 1000))

        (grad,) = torch.autograd.grad(thresholds[0], x)
        assert grad.flatten(0, 1)[1:].count_nonzero() == 0

    def test_threshold_gradient_reaches_every_pixel_5174(self, device, dtype):
        # An 8-bit image scaled to [0, 1]: at 256 bins its levels sit at fixed positions inside their bins. The gradient
        # comes from a kernel density estimate 0.5 bin wide, so it does not concentrate on the pixels near a bin edge.
        # The 0.1-bin estimate that selects the threshold's value would leave about half of this image's pixels, those
        # more than about 0.3 bin from every bin edge, below 1e-3 of the largest gradient. A pixel where the gradient
        # changes sign, near the threshold, can get an arbitrarily small gradient under any estimate; in this image the
        # smallest is 1.5e-2 of the largest.
        x = _quantile_mixture(device, dtype, dark=(0.3, 0.08, 600), bright=(0.7, 0.08, 400), levels=255)
        x.requires_grad_(True)
        _, threshold = otsu_threshold(x, slow_and_differentiable=True)
        (grad,) = torch.autograd.grad(threshold.sum(), x)
        magnitude = grad.abs().float()
        assert (magnitude > 1e-3 * magnitude.max()).all()

    def test_threshold_gradient_is_the_documented_surrogate_5174(self, device, dtype):
        # The gradient is that of the note's soft-argmax: 0.5-bin kernel density estimate, temperature 0.01 of the
        # curve's maximum. The other tests check its properties, which other bandwidths and temperatures share too.
        if dtype != torch.float64:
            pytest.skip("checks the formula, not dtype rounding")
        x = _quantile_mixture(device, dtype).requires_grad_(True)
        (grad,) = torch.autograd.grad(otsu_threshold(x, nbins=64, slow_and_differentiable=True)[1].sum(), x)
        (expected,) = torch.autograd.grad(_documented_soft_threshold(x, 64), x)
        self.assert_close(grad, expected, rtol=1e-9, atol=1e-12)

    def test_threshold_gradient_follows_the_bright_mode_5174(self, device, dtype):
        x = _quantile_mixture(device, dtype).requires_grad_(True)
        _, threshold = otsu_threshold(x, slow_and_differentiable=True)
        (grad,) = torch.autograd.grad(threshold.sum(), x)
        up_image = _quantile_mixture(device, dtype, bright_shift=0.05)
        down_image = _quantile_mixture(device, dtype, bright_shift=-0.05)
        _, up = otsu_threshold(up_image, slow_and_differentiable=True)
        _, down = otsu_threshold(down_image, slow_and_differentiable=True)
        # shifting the bright mode moves the threshold the same way ...
        assert up.item() > threshold.item() > down.item()

        # ... and the gradient along that shift, the sum over the bright pixels, agrees in sign and is close to the
        # threshold's actual response, (up - down) / 0.1 = 0.45, against a gradient of 0.39. The response is measured
        # on float32 copies of half-precision images: rounding the two thresholds themselves to bfloat16 (2**-8 near
        # 0.5) would move it by up to 2 * 2**-8 / 0.1 = 0.08.
        reference_dtype = torch.promote_types(dtype, torch.float32)
        _, up = otsu_threshold(up_image.to(reference_dtype), slow_and_differentiable=True)
        _, down = otsu_threshold(down_image.to(reference_dtype), slow_and_differentiable=True)
        response = (up.item() - down.item()) / 0.1
        along_shift = grad.flatten()[700:].float().sum().item()
        assert along_shift > 0
        assert abs(along_shift - response) < 0.1
        assert grad.flatten()[:700].float().sum().item() > 0

    def test_slow_and_fast_thresholds_agree_within_one_bin_5174(self, device, dtype):
        # The slow histogram is the mass a Gaussian kernel density estimate of bandwidth 0.1 bin puts in each bin. It
        # differs from histc's only by the part of a pixel's mass, within a fraction of a bin of an edge, that falls in
        # the neighbouring bin. On the images of #5174 and #5172 below, that moves the best split up by at most one
        # bin: 0 <= slow - fast <= one bin, plus the rounding of both thresholds to the dtype (the two paths also
        # compute the same edge with different float arithmetic, which differs in the last bits). Not a property of
        # every image: two near-tied splits far apart can swap under any change of histogram estimator. The default
        # path runs on the CPU, where it takes the lowest of splits that give the same partition; on MPS its
        # cumulative sums can make a later one win (0.4766 instead of 0.4000 on the bimodal image, #5421).
        generator = torch.Generator().manual_seed(0)
        noise = torch.rand(1000, generator=generator, dtype=torch.float64)
        ramp_a = torch.linspace(0, 1, 60, dtype=torch.float64).square().view(1, 1, 6, 10)
        ramp_b = torch.linspace(0.6, 4.3, 60, dtype=torch.float64).view(1, 1, 6, 10)
        cases = [
            ("bimodal", torch.cat([0.3 + 0.1 * noise[:500], 0.6 + 0.1 * noise[500:]]).view(1, 1, 20, 50), 256),
            ("mass at 0.3", torch.tensor([0.0] * 10 + [0.3] * 80 + [1.0] * 10).view(1, 1, 1, -1), 16),
            ("mass at 0.7", torch.tensor([0.0] * 10 + [0.7] * 80 + [1.0] * 10).view(1, 1, 1, -1), 16),
            ("uint8 values", _bimodal_uint8_image("cpu").double(), 256),
            ("nbins=2", torch.tensor([[0.0, 0.1, 0.2, 0.55, 0.9, 1.0]]), 2),
            ("ramp a", ramp_a, 256),
            ("ramp b", ramp_b, 256),
        ]
        thresholds = {}
        for name, x, nbins in cases:
            if name == "ramp a" and dtype == torch.bfloat16:
                # bfloat16 rounds one pixel of this ramp onto the bin edge 106 / 256: histc counts it in bin 106, the
                # kernel splits it between bins 105 and 106, and the splits on its two sides are within 0.003 % of each
                # other in between-class variance, so the slow path takes the other one, 6 bins up
                continue
            x = x.to(device=device, dtype=dtype)
            _, fast = otsu_threshold(x.cpu(), nbins=nbins)
            _, slow = otsu_threshold(x, nbins=nbins, slow_and_differentiable=True)
            bin_width = (x.max().item() - x.min().item()) / nbins
            rounding = torch.finfo(dtype).eps * max(abs(fast.item()), abs(slow.item()))
            assert -rounding <= slow.item() - fast.item() <= bin_width + rounding, name
            thresholds[name] = slow.item()
        # every pixel contributes to the slow histogram: the 80 pixels at 0.3 or 0.7 decide between the two images
        assert thresholds["mass at 0.3"] != thresholds["mass at 0.7"]

        # an integer image gets the largest integer below the edge on both paths
        image = _bimodal_uint8_image(device)
        assert otsu_threshold(image, slow_and_differentiable=True)[1].item() == 125
        assert otsu_threshold(image.cpu())[1].item() == 125

    def test_equivalent_splits_take_the_lowest_5174(self, device, dtype):
        # 1 / x on 60 evenly spaced x leaves runs of empty bins between its upper levels, so several splits give the
        # same partition. The kernel tails put about 1e-23 of a pixel in each empty bin; counted as mass, they would
        # leave the choice between those splits to the rounding of the cumulative sums, and in float16 on the CPU, or
        # in float32 on MPS, the highest of them, 6 bins up, would win. Only a bin above 1 % of a pixel counts as
        # non-empty, so the lowest wins on every device, as on the default path on the CPU.
        x = (1 / torch.linspace(1, 10, 60, dtype=torch.float64)).view(1, 1, 6, 10).to(dtype)
        _, expected = otsu_threshold(x)
        _, slow = otsu_threshold(x.to(device), slow_and_differentiable=True)
        self.assert_close(slow.cpu(), expected, rtol=4 * torch.finfo(dtype).eps, atol=0.0)

    @pytest.mark.parametrize("slow_and_differentiable", [False, True])
    def test_dynamo(self, slow_and_differentiable, device, dtype, torch_optimizer, optimizer_backend):
        x = _quantile_mixture(device, dtype)
        # neither path calls .item(), so each compiles as one graph
        op = torch_optimizer(
            lambda x: otsu_threshold(x, nbins=64, slow_and_differentiable=slow_and_differentiable), fullgraph=True
        )
        expected = otsu_threshold(x, nbins=64, slow_and_differentiable=slow_and_differentiable)
        actual = op(x)
        self.assert_close(actual[0], expected[0])
        self.assert_close(actual[1], expected[1])

        if slow_and_differentiable:
            if (
                optimizer_backend == "inductor"
                and dtype == torch.float64
                and device.type == "cpu"
                and torch_version().startswith("2.5.")
            ):
                pytest.skip(
                    "PyTorch 2.5.1 CPU Inductor has no `fmadd` overload for `VectorizedN<double, 2>` in the softmax "
                    "backward"
                )
            x = x.clone().requires_grad_(True)
            (expected_grad,) = torch.autograd.grad(otsu_threshold(x, 64, True)[1].sum(), x)
            (actual_grad,) = torch.autograd.grad(op(x)[1].sum(), x)
            # The plane's minimum pixel collects the backward of the minimum, which cancels terms of order 1 down to
            # about 1e-3. In float32, eager and compiled each land within about 6e-6 of the float64 surrogate there,
            # rounding in opposite directions on MPS, so allow both errors.
            tolerance = {"rtol": 1e-4, "atol": 2e-5} if dtype == torch.float32 else {}
            self.assert_close(actual_grad, expected_grad, **tolerance)

            # Compiler rewrites must preserve the scale cancellation at both ends of the finite dtype range.
            limits = torch.finfo(dtype)
            levels = torch.tensor([[-1.0, -0.8, -0.6, 0.2, 0.6, 1.0]], device=device, dtype=dtype)
            x = torch.stack([levels * limits.max, levels * (limits.tiny * 64)]).requires_grad_(True)
            expected = otsu_threshold(x, 64, True)[1]
            actual = op(x)[1]
            assert actual.isfinite().all()
            self.assert_close(actual, expected)
            (expected_grad,) = torch.autograd.grad(expected.sum(), x)
            (actual_grad,) = torch.autograd.grad(actual.sum(), x)
            assert actual_grad.isfinite().all()
            self.assert_close(actual_grad, expected_grad)
