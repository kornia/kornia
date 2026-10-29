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

import pytest
import torch

from kornia.feature import GeometryAwareDescriptorMatcher, laf_from_center_scale_ori, match_adalam
from kornia.feature.adalam import AdalamFilter, get_adalam_default_config
from kornia.feature.adalam.core import _merge_duplicate_matches
from kornia.feature.adalam.ransac import (
    _select_inliers,
    confidence_based_inlier_selection,
    ransac,
    stable_sort_residuals,
)
from kornia.feature.adalam.utils import draw_first_k_couples

from testing.base import BaseTester


class TestAdalamExact(BaseTester):
    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.parametrize("sigma", [0.0, 0.001, 0.01, 0.02, 0.05, 0.5])
    def test_translation_5866(self, device, dtype, seed, sigma):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("filter_matches requires full-precision coordinates")
        g = torch.Generator().manual_seed(seed)
        n = 200
        xy = torch.rand(n, 2, generator=g, dtype=torch.float64) * torch.tensor([639.0, 479.0])
        scores = 0.5 * torch.rand(n, generator=g, dtype=torch.float64)
        noise = torch.randn(n, 2, generator=g, dtype=torch.float64)
        moved = (xy + torch.tensor([20.0, 10.0]) + sigma * noise).to(device, dtype)
        xy, scores = xy.to(device, dtype), scores.to(device, dtype)
        zero, scale = torch.zeros(n, device=device, dtype=dtype), torch.full((n,), 10.0, device=device, dtype=dtype)
        indices = torch.arange(n, device=device)
        config = get_adalam_default_config()
        config["device"] = device
        matches, quality = AdalamFilter(config).filter_matches(
            xy, moved, indices, scores, None, (480, 640), (480, 640), zero, zero, scale, scale, return_dist=True
        )
        if sigma <= 0.001:
            self.assert_close(matches, torch.stack([indices, indices], dim=-1))
        else:
            assert len(matches) >= 0.9 * n
            self.assert_close(matches[:, 0], matches[:, 1])
        assert torch.isfinite(quality).all()
        assert (quality <= 1 / config["min_confidence"]).all()

    @pytest.mark.parametrize("api", ["functional", "module"])
    @pytest.mark.parametrize("translated", [False, True])
    def test_public_matchers_5866(self, device, dtype, api, translated):
        g = torch.Generator().manual_seed(4)
        n = 200
        # Even coordinates below 512 stay exact in bfloat16 after the +16 shift; odd values in [240, 256) do not.
        xy = (2 * torch.randint(0, 200, (1, n, 2), generator=g)).to(device, dtype)
        desc = torch.randn(n, 32, generator=g).to(device, dtype)
        lafs = laf_from_center_scale_ori(xy, torch.full((1, n, 1, 1), 10.0, device=device, dtype=dtype))
        moved = lafs.clone()
        if translated:
            moved[..., 2] += 16
        if api == "functional":
            quality, matches = match_adalam(desc, desc, lafs, moved, hw1=(480, 640), hw2=(480, 640))
        else:
            quality, matches = GeometryAwareDescriptorMatcher("adalam")(desc, desc, lafs, moved)
        indices = torch.arange(n, device=device)
        self.assert_close(matches, torch.stack([indices, indices], dim=-1))
        assert torch.isfinite(quality).all()

    @pytest.mark.parametrize("refit", [False, True])
    @pytest.mark.parametrize("n_inliers,n_outliers,seed", [(40, 8, 9), (9, 0, 0)])
    def test_affine_with_outliers(self, device, dtype, refit, n_inliers, n_outliers, seed):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("local RANSAC uses full-precision coordinates")
        g = torch.Generator().manual_seed(seed)
        x = torch.rand(n_inliers + n_outliers, 2, generator=g).to(device, dtype) - 0.5
        x[0] = 0
        affine = torch.tensor([[1.1, -0.2], [0.1, 0.9]], device=device, dtype=dtype)
        y = x @ affine.T
        y[n_inliers:] += 10
        config = get_adalam_default_config()
        config["device"] = device
        indices, fitted, confidence, counts = ransac(x, y, torch.tensor([len(x)], device=device), config, refit=refit)
        self.assert_close(indices.sort().values, torch.arange(n_inliers, device=device))
        self.assert_close(fitted[0], affine)
        assert counts.tolist() == [n_inliers]
        assert torch.isfinite(confidence).all()
        assert (confidence >= config["min_confidence"]).all()

    @pytest.mark.parametrize("case", ["duplicates", "minimal", "collinear", "bad_scale", "one_target"])
    @pytest.mark.parametrize("refit", [False, True])
    def test_no_manufactured_support(self, device, dtype, case, refit):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("local RANSAC uses full-precision coordinates")
        g = torch.Generator().manual_seed(17)
        x = torch.rand(40, 2, generator=g).to(device, dtype) - 0.5
        x[0] = 0
        if case == "duplicates":
            x = x[:5].repeat(8, 1)
        elif case == "collinear":
            x[:, 1] = 0
        y = x.clone()
        if case == "minimal":
            y[3:] = 10 + torch.rand(37, 2, generator=g).to(device, dtype)
        elif case == "bad_scale":
            y *= 10
        elif case == "one_target":
            y[:] = 0
        config = get_adalam_default_config()
        config["device"] = device
        _, _, confidence, counts = ransac(x, y, torch.tensor([len(x)], device=device), config, refit=refit)
        accepted = (confidence >= config["min_confidence"]) & (counts * (1 - 1 / confidence) >= config["min_inliers"])
        assert not accepted.any()

    def test_neighbourhood_isolation(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("local RANSAC uses full-precision coordinates")
        n = 20
        g = torch.Generator().manual_seed(7)
        # The same source/target geometry occurs in both groups and must count separately.
        xy = torch.rand(n, 2, generator=g).to(device, dtype).repeat(2, 1)
        residuals = torch.zeros(1, 2 * n, device=device, dtype=dtype)
        residuals[0, n:] = torch.linspace(0.001, 0.01, n, device=device, dtype=dtype)
        groups = torch.arange(2, device=device).repeat_interleave(n)
        dims = torch.full((2,), n, device=device, dtype=torch.long)
        offsets = torch.tensor([0, n], device=device)
        config = get_adalam_default_config()
        legacy = confidence_based_inlier_selection(residuals, groups, dims, offsets, device, config["min_confidence"])
        result = _select_inliers(
            residuals,
            xy,
            xy,
            groups,
            dims,
            offsets,
            torch.eye(2, device=device, dtype=dtype).expand(1, 2, 2, 2),
            config,
        )
        self.assert_close(result[1][result[0] == 0], torch.arange(n, device=device))
        self.assert_close(result[1][result[0] == 1], legacy[1][legacy[0] == 1])
        for actual, expected in zip(result[2:], legacy[2:]):
            self.assert_close(actual[1:], expected[1:], atol=0, rtol=0)
        residuals.zero_()
        result = _select_inliers(
            residuals,
            xy,
            xy,
            groups,
            dims,
            offsets,
            torch.eye(2, device=device, dtype=dtype).expand(1, 2, 2, 2),
            config,
        )
        self.assert_close(result[1], torch.arange(2 * n, device=device))
        assert result[2].tolist() == [n, n]

    def test_equal_residual_sort(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("local RANSAC uses full-precision coordinates")
        groups = torch.tensor([1, 0, 1, 0], device=device)
        residuals = torch.zeros(1, 4, device=device, dtype=dtype)
        sorted_residuals, order = stable_sort_residuals(residuals, groups)
        self.assert_close(groups[order[0]], groups.sort().values)
        self.assert_close(sorted_residuals, residuals)

    def test_noisy_fits_skip_recovery_scan(self, device, dtype, monkeypatch):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("local RANSAC uses full-precision coordinates")
        dims = torch.tensor([6, 8, 20], device=device)
        offsets = torch.tensor([0, 6, 14], device=device)
        groups = torch.arange(3, device=device).repeat_interleave(dims)
        residuals = torch.full((1, 34), 0.01, device=device, dtype=dtype)
        residuals[:, offsets[:, None] + torch.arange(3, device=device)] = 0
        points = torch.zeros(34, 2, device=device, dtype=dtype)
        config = get_adalam_default_config()
        expected = confidence_based_inlier_selection(residuals, groups, dims, offsets, device, config["min_confidence"])

        def no_recovery_scan(*args, **kwargs):
            pytest.fail("Ordinary fits must not scan all residuals again for near-exact support")

        monkeypatch.setattr("kornia.feature.adalam.ransac.group_sum_and_cumsum", no_recovery_scan)
        monkeypatch.setattr("kornia.feature.adalam.ransac._confidence_from_sorted_residuals", lambda *args: expected)
        result = _select_inliers(
            residuals, points, points, groups, dims, offsets, torch.eye(2, device=device, dtype=dtype), config
        )
        assert result is expected

    def test_complete_exact_fits_skip_legacy_confidence(self, device, dtype, monkeypatch):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("local RANSAC uses full-precision coordinates")

        def no_legacy_confidence(*args, **kwargs):
            pytest.fail("Fully supported exact fits must not run the residual-duplicate heuristic")

        monkeypatch.setattr("kornia.feature.adalam.ransac._confidence_from_sorted_residuals", no_legacy_confidence)
        dims = torch.tensor([9, 20], device=device)
        offsets = torch.tensor([0, 9], device=device)
        groups = torch.arange(2, device=device).repeat_interleave(dims)
        residuals = torch.zeros(1, 29, device=device, dtype=dtype)
        points = torch.rand(29, 2, generator=torch.Generator().manual_seed(11)).to(device, dtype)
        affinities = torch.eye(2, device=device, dtype=dtype).expand(1, 2, 2, 2)
        result = _select_inliers(
            residuals, points, points, groups, dims, offsets, affinities, get_adalam_default_config()
        )
        self.assert_close(result[1], torch.arange(29, device=device))
        self.assert_close(result[2], dims)


class TestDrawFirstKCouples(BaseTester):
    @pytest.mark.parametrize("k", [1, 2, 3, 4, 10, 100, 128, 129])
    def test_couples_are_the_first_k_distinct_pairs(self, k, device):
        # The k couples are the pairs (i, j), j < i, in row order. The residual row after the complete ones used
        # its length as its first index, so it repeated couples already drawn: 8 of the default 128.
        couples = draw_first_k_couples(k, torch.tensor([1000], device=device), device)
        expected = [(i, j) for i in range(1, k + 2) for j in range(i)][:k]
        assert couples.shape == (k, 2, 1)
        assert [tuple(c) for c in couples[:, :, 0].tolist()] == expected

    def test_couples_wrap_per_region(self, device):
        # Each region's couples are reduced modulo its own number of samples, so a region smaller than the couples
        # need wraps: here (3, 0) becomes (0, 0), a rank-deficient pair that the minimal fit marks as NaN.
        couples = draw_first_k_couples(4, torch.tensor([1000, 3], device=device), device)
        self.assert_close(couples[:, :, 0], torch.tensor([[1, 0], [2, 0], [2, 1], [3, 0]], device=device))
        self.assert_close(couples[:, :, 1], torch.tensor([[1, 0], [2, 0], [2, 1], [0, 0]], device=device))


class TestMergeDuplicateMatches(BaseTester):
    def test_keeps_each_match_once_with_its_smallest_distance_ratio(self, device, dtype):
        # overlapping neighbourhoods accept a match in several local RANSACs; the most confident copy is reported,
        # whichever comes first
        matches = torch.tensor([[0, 1], [2, 3], [0, 1], [0, 1], [2, 0]], device=device)
        dists = torch.tensor([0.5, 0.2, 0.1, 0.9, 0.4], device=device, dtype=dtype)
        for order in ([0, 1, 2, 3, 4], [3, 4, 1, 2, 0]):
            unique, best = _merge_duplicate_matches(matches[order], dists[order], 5)
            assert unique.tolist() == [[0, 1], [2, 0], [2, 3]]
            self.assert_close(best, torch.tensor([0.1, 0.4, 0.2], device=device, dtype=dtype))
