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
        """A fixed 5x5 image whose ``nbins=3`` Otsu threshold is 0.5 and keeps 14 of 25 pixels.

        Every pixel is at least 1.9e-3 from the threshold in each tested dtype (the smallest margin is 1.95e-3, in
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
        # The analytical gradient of the output is just the mask (no gradient flows through the threshold), so
        # this only checks that no pixel crosses the threshold under the finite-difference perturbation.
        self.gradcheck(otsu_threshold, (img, 3, True, False))

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
        # Three zeros give the first KDE split a unique maximum when middle=0.5. Its next sample is 0.5,
        # which must not remove that foreground pixel. Without a collision, preserve the sample exactly.
        image = torch.tensor([[0.0, 0.0, 0.0, middle, 1.0, 1.0]], device=device, dtype=dtype)
        mask, threshold = otsu_threshold(image, nbins=3, slow_and_differentiable=True, return_mask=True)
        expected = torch.tensor([0.5], dtype=dtype)
        if middle == 0.5:
            expected = torch.nextafter(expected, torch.full_like(expected, -torch.inf))
        self.assert_close(threshold, expected.to(device), rtol=0, atol=0)
        assert mask.tolist() == [[False, False, False, True, True, True]]

    def test_slow_path_downward_rounded_sample_keeps_background_pixel_excluded_5422(self, device):
        # The middle KDE sample 0.501953125 rounds down to the pixel 0.5 in bfloat16. That pixel lies below the
        # sample, so lowering the threshold to keep it, as for an upward-rounded sample, would promote it.
        image = torch.tensor([[0.00390625] * 3 + [0.5, 1.0, 1.0]], device=device, dtype=torch.bfloat16)
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
