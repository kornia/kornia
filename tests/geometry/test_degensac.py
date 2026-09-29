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

import kornia.geometry._degensac as degensac_module
from kornia.geometry._degensac import (
    _best_candidate,
    _h_degenerate_sample,
    _homographies_from_fundamental,
    _inner_homography,
    _left_epipole,
    _msac_gain,
    _pair_draws,
    _plane_parallax_fundamentals,
    _plane_parallax_search,
)
from kornia.geometry.epipolar.fundamental import _epipolar_design_rows, _seven_point_candidates
from kornia.geometry.epipolar.numeric import cross_product_matrix
from kornia.geometry.homography import oneway_transfer_error, sampson_homography_distance
from kornia.geometry.ransac import RANSAC

# The kernels run on the host in float64 by design (RANSAC moves one sample there), so these tests do not take the
# device and dtype fixtures; RANSAC's own tests cover the device paths.
F64 = torch.float64


def _two_view_geometry():
    """A calibrated pair, the plane z - 0.2 x = 6 and its homography, and F; all in pixels, float64."""
    K = torch.tensor([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=F64)
    a, b = math.radians(8.0), math.radians(3.0)
    Ry = torch.tensor([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]], dtype=F64)
    Rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, math.cos(b), -math.sin(b)], [0.0, math.sin(b), math.cos(b)]], dtype=F64)
    R, t = Ry @ Rx, torch.tensor([-1.0, 0.1, 0.15], dtype=F64)
    normal, distance = torch.tensor([-0.2, 0.0, 1.0], dtype=F64), 6.0
    K_inv = torch.linalg.inv(K)
    H = K @ (R + torch.outer(t, normal) / distance) @ K_inv
    F = K_inv.T @ cross_product_matrix(t) @ R @ K_inv
    return K, R, t, H / H[2, 2], F / F.norm()


def _points(K, R, t, count, planar, generator):
    if planar:
        xy = (torch.rand(count, 2, generator=generator, dtype=F64) - 0.5) * torch.tensor([4.0, 3.0], dtype=F64)
        X = torch.cat([xy, 6.0 + 0.2 * xy[:, :1]], 1)
    else:
        X = torch.cat(
            [
                (torch.rand(count, 2, generator=generator, dtype=F64) - 0.5) * 2,
                3 + 5 * torch.rand(count, 1, generator=generator, dtype=F64),
            ],
            1,
        )
    x1, x2 = X @ K.T, (X @ R.T + t) @ K.T
    return x1 / x1[:, 2:], x2 / x2[:, 2:]


def _same_up_to_scale(A, B):
    A, B = A / A.norm(), B / B.norm()
    return float(torch.minimum((A - B).abs().max(), (A + B).abs().max()))


class TestLeftEpipole:
    def test_null_vector_of_transpose(self):
        _, _, _, _, F = _two_view_geometry()
        e = _left_epipole(F[None])[0]
        assert float((F.T @ e).abs().max()) < 1e-12
        assert abs(float(e.norm()) - 1.0) < 1e-12


class TestHomographiesFromFundamental:
    def test_recovers_plane_homography(self):
        K, R, t, H, F = _two_view_geometry()
        x1, x2 = _points(K, R, t, 3, True, torch.Generator().manual_seed(0))
        estimate = _homographies_from_fundamental(F, x1[None], x2[None])[0]
        assert _same_up_to_scale(estimate, H) < 1e-10
        compatibility = estimate.T @ F  # skew-symmetric exactly when H is compatible with F
        assert float((compatibility + compatibility.T).abs().max()) < 1e-10 * float(compatibility.abs().max())

    def test_collinear_triplet_is_nan(self):
        _, _, _, _, F = _two_view_geometry()
        x1 = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 1.0]]], dtype=F64)
        assert torch.isnan(_homographies_from_fundamental(F, x1, x1)).any()


class TestHDegenerateSample:
    THRESHOLD = 3.0  # 3 t with t = 1 px^2; the sample is noise-free

    def test_five_plane_pairs_flag_the_compatible_root(self):
        # Five H-related pairs and two others: only one seven-point root is guaranteed compatible with H (paper,
        # section 2.3), so only that one must flag.
        K, R, t, H, _ = _two_view_geometry()
        generator = torch.Generator().manual_seed(1)
        p1, p2 = _points(K, R, t, 5, True, generator)
        o1, o2 = _points(K, R, t, 2, False, generator)
        x1, x2 = torch.cat([p1, o1]), torch.cat([p2, o2])
        candidates, valid = _seven_point_candidates(_epipolar_design_rows(x1[None], x2[None]))
        compatible = []
        for root in range(3):
            if not bool(valid[0, root]):
                continue
            product = H.T @ candidates[0, root]
            skew = float((product + product.T).abs().max() / product.abs().max())
            compatible.append((skew, root))
        _, root = min(compatible)
        assert _h_degenerate_sample(candidates[0, root], x1, x2, self.THRESHOLD) is not None

    @pytest.mark.parametrize("on_plane", [6, 7])
    def test_six_and_seven_plane_pairs_flag_a_compatible_model(self, on_plane):
        # Exact 6- and 7-of-7 data make the seven-point cubic vanish identically (paper, sections 2.1-2.2), so the
        # model is an explicit member of the compatible family F = [e']_x H rather than a solver root.
        K, R, t, H, _ = _two_view_geometry()
        generator = torch.Generator().manual_seed(2)
        p1, p2 = _points(K, R, t, on_plane, True, generator)
        o1, o2 = _points(K, R, t, 7 - on_plane, False, generator)
        x1, x2 = torch.cat([p1, o1]), torch.cat([p2, o2])
        F = cross_product_matrix(K @ t) @ H
        homography = _h_degenerate_sample(F / F.norm(), x1, x2, self.THRESHOLD)
        assert homography is not None
        assert _same_up_to_scale(homography, H) < 1e-6

    def test_general_position_does_not_flag(self):
        K, R, t, _, _ = _two_view_geometry()
        x1, x2 = _points(K, R, t, 7, False, torch.Generator().manual_seed(3))
        candidates, valid = _seven_point_candidates(_epipolar_design_rows(x1[None], x2[None]))
        for root in range(3):
            if bool(valid[0, root]):
                assert _h_degenerate_sample(candidates[0, root], x1, x2, self.THRESHOLD) is None


class TestPlaneParallax:
    def test_two_off_plane_pairs_give_the_fundamental_matrix(self):
        K, R, t, H, F = _two_view_geometry()
        o1, o2 = _points(K, R, t, 4, False, torch.Generator().manual_seed(4))
        models = _plane_parallax_fundamentals(H, o1, o2, torch.tensor([0, 2]), torch.tensor([1, 3]))
        for model in models:
            assert _same_up_to_scale(model, F) < 1e-10
            assert abs(float(model.norm()) - 1.0) < 1e-12

    def test_parallel_lines_give_nan(self):
        _, _, _, H, _ = _two_view_geometry()
        x1 = torch.tensor([[100.0, 100.0, 1.0]], dtype=F64)
        x2 = torch.tensor([[150.0, 120.0, 1.0]], dtype=F64)
        models = _plane_parallax_fundamentals(H, x1, x2, torch.tensor([0]), torch.tensor([0]))
        assert torch.isnan(models).all()


def _plane_with_outliers(H, generator, noise=1.0, count=300, outliers=200):
    clean = torch.rand(count, 2, generator=generator, dtype=F64) * 100
    mapped = torch.cat([clean, torch.ones(count, 1, dtype=F64)], 1) @ H.T
    p1 = clean + noise * torch.randn(count, 2, generator=generator, dtype=F64)
    p2 = mapped[:, :2] / mapped[:, 2:] + noise * torch.randn(count, 2, generator=generator, dtype=F64)
    o1 = torch.rand(outliers, 2, generator=generator, dtype=F64) * 100
    o2 = torch.rand(outliers, 2, generator=generator, dtype=F64) * 400
    return torch.cat([p1, o1]), torch.cat([p2, o2])


class TestInnerHomography:
    THRESHOLD = 16.0  # 16 t with t = 1 px^2

    def test_zoom_keeps_plane_inliers_a_transfer_cutoff_drops(self):
        # At a 4x zoom the one-way transfer error is about 17 times the Sampson one along the zoom, so the fixed
        # 32 t transfer cutoff the first design used would drop a third of the plane; innerH keeps it.
        H = torch.tensor([[4.0, 0.3, 10.0], [0.1, 4.0, -5.0], [1e-4, 0.0, 1.0]], dtype=F64)
        generator = torch.Generator().manual_seed(0)
        x1, x2 = _plane_with_outliers(H, generator)
        start = H * (1 + 1e-3 * torch.randn(3, 3, generator=generator, dtype=F64))
        _, errors = _inner_homography(start, x1, x2, self.THRESHOLD, torch.Generator().manual_seed(1))
        assert int((errors[:300] <= self.THRESHOLD).sum()) >= 297
        assert int((errors[300:] <= self.THRESHOLD).sum()) <= 5
        transfer = oneway_transfer_error(x1[None], x2[None], H[None])[0]
        assert int((transfer[:300] <= 2 * self.THRESHOLD).sum()) < 250

    def test_shear_keeps_plane_inliers(self):
        H = torch.tensor([[1.0, 1.5, 3.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]], dtype=F64)
        generator = torch.Generator().manual_seed(2)
        x1, x2 = _plane_with_outliers(H, generator)
        start = H * (1 + 1e-3 * torch.randn(3, 3, generator=generator, dtype=F64))
        _, errors = _inner_homography(start, x1, x2, self.THRESHOLD, torch.Generator().manual_seed(3))
        assert int((errors[:300] <= self.THRESHOLD).sum()) >= 297

    def test_seeded_generator_is_reproducible(self):
        H = torch.tensor([[1.1, 0.1, 3.0], [0.0, 0.9, 2.0], [1e-4, 0.0, 1.0]], dtype=F64)
        x1, x2 = _plane_with_outliers(H, torch.Generator().manual_seed(4))
        first = _inner_homography(H, x1, x2, self.THRESHOLD, torch.Generator().manual_seed(5))
        second = _inner_homography(H, x1, x2, self.THRESHOLD, torch.Generator().manual_seed(5))
        assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])

    def test_repetitions_without_inliers_contribute_nothing(self, monkeypatch):
        # When every subset model h0 has fewer than 4 inliers, iterH returns an empty score, h0 included, so innerH
        # keeps the homography it was given. Were h0 kept as a candidate, its finite gain of 0 would win instead.
        H = torch.tensor([[1.1, 0.1, 3.0], [0.0, 0.9, 2.0], [1e-4, 0.0, 1.0]], dtype=F64)
        x1, x2 = _plane_with_outliers(H, torch.Generator().manual_seed(11))
        far = torch.tensor([[1.0, 0.0, 1e4], [0.0, 1.0, 1e4], [0.0, 0.0, 1.0]], dtype=F64)
        monkeypatch.setattr(
            degensac_module,
            "find_homography_dlt",
            lambda p1, p2, weights=None, solver="svd": far.expand(p1.shape[0], 3, 3).clone(),
        )
        homography, _ = _inner_homography(H, x1, x2, self.THRESHOLD, torch.Generator().manual_seed(12))
        assert torch.equal(homography, H)

    def test_too_few_inliers_keep_the_homography(self):
        H = torch.eye(3, dtype=F64)
        x1 = torch.rand(20, 2, generator=torch.Generator().manual_seed(6), dtype=F64) * 100
        x2 = x1 + 50.0  # no inliers of H at 16 t
        homography, errors = _inner_homography(H, x1, x2, self.THRESHOLD, None)
        assert torch.equal(homography, H)
        assert torch.equal(errors, sampson_homography_distance(x1[None], x2[None], H[None])[0])


class TestCandidateRanking:
    def test_gain_not_count_decides(self):
        # Candidate 0 has more inliers at the threshold but larger errors; SC_M's gain prefers candidate 1.
        threshold = 1.0
        errors = torch.tensor([[0.9, 0.9, 0.9, 0.9], [0.0, 0.0, 0.0, 5.0]], dtype=F64)
        assert (errors <= threshold).sum(1).tolist() == [4, 3]
        gains = _msac_gain(errors, threshold)
        assert float(gains[1]) > float(gains[0])
        assert _best_candidate(gains, torch.tensor([True, True])) == 1

    def test_ties_keep_the_first_and_dead_candidates_never_win(self):
        gains = torch.tensor([1.0, 3.0, 3.0, 5.0], dtype=F64)
        assert _best_candidate(gains, torch.tensor([True, True, True, False])) == 1
        assert _best_candidate(gains, torch.zeros(4, dtype=torch.bool)) is None
        assert _best_candidate(torch.tensor([float("nan"), 2.0], dtype=F64), torch.tensor([True, True])) == 1


class TestPlaneParallaxSearch:
    def test_pair_draws_match_ransac_bound(self):
        for support, total in [(5, 100), (30, 430), (99, 100), (2, 50)]:
            assert _pair_draws(support, total, 0.999) == RANSAC.max_samples_by_conf(support, total, 2, 0.999)

    def test_finds_the_epipolar_geometry_among_outliers(self):
        K, R, t, H, F = _two_view_geometry()
        generator = torch.Generator().manual_seed(7)
        o1, o2 = _points(K, R, t, 40, False, generator)
        r1 = torch.cat([torch.rand(200, 2, generator=generator, dtype=F64) * 640, torch.ones(200, 1, dtype=F64)], 1)
        r2 = torch.cat([torch.rand(200, 2, generator=generator, dtype=F64) * 480, torch.ones(200, 1, dtype=F64)], 1)
        x1, x2 = torch.cat([o1, r1]), torch.cat([o2, r2])
        # Unit-norm models on pixel coordinates lose accuracy in the quadratic Sampson basis; normalize like RANSAC.
        scale = 1.0 / 300.0
        T = torch.tensor([[scale, 0.0, -320 * scale], [0.0, scale, -240 * scale], [0.0, 0.0, 1.0]], dtype=F64)
        y1, y2 = x1 @ T.T, x2 @ T.T
        H_normalized = T @ H @ torch.linalg.inv(T)
        found = _plane_parallax_search(H_normalized, y1, y2, 2.0 * scale**2, 256, torch.Generator().manual_seed(8))
        assert found is not None
        models, support = found
        assert len(models) <= 8 and bool((support > 4).all()) and int(support[0]) >= 38
        best = T.T @ models[0] @ T  # y = T x, so y2^T F y1 = x2^T (T^T F T) x1
        assert _same_up_to_scale(best, F) < 1e-6

    def test_no_consensus_returns_none(self):
        generator = torch.Generator().manual_seed(9)
        x1 = torch.cat([torch.rand(6, 2, generator=generator, dtype=F64), torch.ones(6, 1, dtype=F64)], 1)
        x2 = torch.cat([torch.rand(6, 2, generator=generator, dtype=F64), torch.ones(6, 1, dtype=F64)], 1)
        found = _plane_parallax_search(torch.eye(3, dtype=F64), x1, x2, 1e-12, 4096, torch.Generator().manual_seed(10))
        assert found is None
