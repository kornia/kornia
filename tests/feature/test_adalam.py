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

    @pytest.mark.parametrize("refit", [False, True])
    def test_mixed_exact_and_noisy_consensus(self, device, dtype, refit):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("local RANSAC uses full-precision coordinates")
        g = torch.Generator().manual_seed(0)
        x = torch.rand(60, 2, generator=g).to(device, dtype) - 0.5
        x[0] = 0
        y = x @ torch.tensor([[1.1, -0.2], [0.1, 0.9]], device=device, dtype=dtype).T
        y[20:50] += 0.004 * torch.randn(30, 2, generator=g).to(device, dtype)
        y[50:] += 10
        config = get_adalam_default_config()
        config["device"] = device
        indices, _, _, counts = ransac(x, y, torch.tensor([60], device=device), config, refit=refit)
        # The first 50 correspondences follow the specified affine; the others are displaced by 10.
        self.assert_close(indices.sort().values, torch.arange(50, device=device))
        assert counts.tolist() == [50]

    @pytest.mark.parametrize("sigma", [0.1, 0.3, 0.5])
    def test_high_resolution_subpixel_matches(self, device, dtype, sigma):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("filter_matches requires full-precision coordinates")
        g = torch.Generator().manual_seed(0)
        n = 2000
        xy = torch.rand(n, 2, generator=g, dtype=torch.float64) * torch.tensor([3999.0, 2999.0])
        scores = 0.5 * torch.rand(n, generator=g, dtype=torch.float64)
        moved = xy + torch.tensor([20.0, 10.0]) + sigma * torch.randn(n, 2, generator=g, dtype=torch.float64)
        config = get_adalam_default_config()
        config["device"] = device
        matches = AdalamFilter(config).filter_matches(
            xy.to(device, dtype),
            moved.to(device, dtype),
            torch.arange(n, device=device),
            scores.to(device, dtype),
            None,
            (3000, 4000),
            (3000, 4000),
        )
        assert len(matches) >= 0.99 * n
        self.assert_close(matches[:, 0], matches[:, 1])

    @pytest.mark.parametrize("case", ["noisy_duplicates", "random", "minimal_plus_duplicates"])
    def test_noisy_false_fits_have_no_independent_support(self, device, dtype, case):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("local RANSAC uses full-precision coordinates")
        g = torch.Generator().manual_seed(37)
        x = torch.rand(60, 2, generator=g).to(device, dtype) - 0.5
        y = torch.rand(60, 2, generator=g).to(device, dtype) - 0.5
        if case == "noisy_duplicates":
            x = x[:5].repeat(12, 1)
            y = x + 0.004 * torch.randn(60, 2, generator=g).to(device, dtype)
        elif case == "minimal_plus_duplicates":
            x[:30] = x[:3].repeat(10, 1)
            y[:30] = x[:30]
        x[0] = y[0] = 0
        config = get_adalam_default_config()
        config["device"] = device
        for refit in (False, True):
            indices, _, confidence, counts = ransac(x, y, torch.tensor([60], device=device), config, refit=refit)
            assert counts.sum().item() == len(indices)
            assert not (confidence >= config["min_confidence"]).any()

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
        self.assert_close(result[1][result[0] == 1].sort().values, torch.arange(n, 2 * n, device=device))
        assert result[2].tolist() == [n, n]
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

    @pytest.mark.parametrize("preferred_correct", [False, True])
    def test_conflicting_duplicates_keep_descriptor_preference(self, device, dtype, preferred_correct):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("local RANSAC uses full-precision coordinates")
        points = torch.arange(9, device=device, dtype=dtype).unsqueeze(1).repeat(1, 2)
        x = points.repeat(2, 1)
        y = torch.cat([points + 20, points])
        if preferred_correct:
            y = y.flip(0)
            x = x.flip(0)
        residuals = (y - x).norm(dim=-1).unsqueeze(0)
        result = _select_inliers(
            residuals,
            x,
            y,
            torch.zeros(18, device=device, dtype=torch.long),
            torch.tensor([18], device=device),
            torch.tensor([0], device=device),
            torch.eye(2, device=device, dtype=dtype).expand(1, 1, 2, 2),
            get_adalam_default_config(),
        )
        # Samples arrive in descriptor-score order. A later duplicate cannot replace
        # the preferred observation merely because a hypothesis fits it better.
        assert result[2].tolist() == ([9] if preferred_correct else [0])
        assert len(result[1]) == result[2].sum()

    def test_equal_support_prefers_tighter_consensus(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("local RANSAC uses full-precision coordinates")
        points = torch.arange(20, device=device, dtype=dtype).unsqueeze(1).repeat(1, 2)
        residuals = torch.tensor([[0.02], [0.01]], device=device, dtype=dtype).expand(2, 20)
        result = _select_inliers(
            residuals,
            points,
            points,
            torch.zeros(20, device=device, dtype=torch.long),
            torch.tensor([20], device=device),
            torch.tensor([0], device=device),
            torch.eye(2, device=device, dtype=dtype).expand(2, 1, 2, 2),
            get_adalam_default_config(),
        )
        assert result[2].tolist() == [20]
        assert result[3].tolist() == [1]
        # 20 distinct points minus the 3 reserved samples, over 20 * 0.01**2.
        self.assert_close(result[4], torch.tensor([8500.0], device=device, dtype=dtype))

    def test_empty_consensus_between_supported_neighbourhoods(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("local RANSAC uses full-precision coordinates")
        points = torch.arange(36, device=device, dtype=dtype).unsqueeze(1).repeat(1, 2)
        residuals = torch.zeros(1, 36, device=device, dtype=dtype)
        residuals[:, 12:24] = 2
        groups = torch.arange(3, device=device).repeat_interleave(12)
        result = _select_inliers(
            residuals,
            points,
            points,
            groups,
            torch.tensor([12, 12, 12], device=device),
            torch.tensor([0, 12, 24], device=device),
            torch.eye(2, device=device, dtype=dtype).expand(1, 3, 2, 2),
            get_adalam_default_config(),
        )
        assert result[2].tolist() == [12, 0, 12]
        expected = torch.cat([torch.arange(12, device=device), torch.arange(24, 36, device=device)])
        self.assert_close(result[1], expected)
        self.assert_close(result[0], groups[expected])
        assert result[4][1] == 0

    def test_residual_order_survives_large_group_ids(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("local RANSAC uses full-precision coordinates")
        groups = torch.full((4,), 5000, device=device, dtype=torch.long)
        residuals = torch.tensor([[0.0100001, 0.01, 2.0, 0.0]], device=device, dtype=dtype)
        sorted_residuals, order = stable_sort_residuals(residuals, groups)
        self.assert_close(order, torch.tensor([[3, 1, 0, 2]], device=device))
        self.assert_close(sorted_residuals, residuals.sort().values)

    def test_equal_residual_sort(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("local RANSAC uses full-precision coordinates")
        groups = torch.tensor([1, 0, 1, 0], device=device)
        residuals = torch.zeros(1, 4, device=device, dtype=dtype)
        sorted_residuals, order = stable_sort_residuals(residuals, groups)
        self.assert_close(groups[order[0]], groups.sort().values)
        self.assert_close(sorted_residuals, residuals)


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
        # overlapping neighbourhoods accept a match in several local RANSACs; the most confident copy is reported, in
        # whatever order the copies come
        matches = torch.tensor([[0, 1], [2, 3], [0, 1], [0, 1], [2, 0]], device=device)
        dists = torch.tensor([0.5, 0.2, 0.1, 0.9, 0.4], device=device, dtype=dtype)
        for order in ([0, 1, 2, 3, 4], [3, 4, 1, 2, 0]):
            unique, best = _merge_duplicate_matches(matches[order], dists[order], 5)
            assert unique.tolist() == [[0, 1], [2, 0], [2, 3]]
            self.assert_close(best, torch.tensor([0.1, 0.4, 0.2], device=device, dtype=dtype))

    def test_many_copies_and_ties_give_the_minimum_of_each_match(self, device, dtype):
        g = torch.Generator().manual_seed(0)
        matches = torch.randint(0, 6, (300, 2), generator=g)
        dists = torch.randint(0, 10, (300,), generator=g) / 10  # ties within and across matches
        unique, best = _merge_duplicate_matches(matches.to(device), dists.to(device, dtype), 6)
        expected = {}
        for (i, j), d in zip(matches.tolist(), dists.tolist()):
            expected[(i, j)] = min(d, expected.get((i, j), d))
        assert unique.tolist() == [list(key) for key in sorted(expected)]
        self.assert_close(best, torch.tensor([expected[key] for key in sorted(expected)], device=device, dtype=dtype))
