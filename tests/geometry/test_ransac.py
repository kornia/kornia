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
import sys

import pytest
import torch

import kornia
from kornia.geometry import RANSAC, transform_points
from kornia.geometry.conversions import axis_angle_to_rotation_matrix, convert_points_from_homogeneous
from kornia.geometry.epipolar import find_fundamental, project_to_essential, sampson_epipolar_distance

from testing.base import BaseTester
from testing.casts import dict_to

# RANSAC's batched minimal solvers take the Metal shader compiler down on the paravirtualized GPU
# of the hosted macOS runners, and the next MPS allocation aborts the pytest process (SIGABRT).
# The abort site moves between runs — observed inside the 5-point solver and, with that class
# deselected, later in an unrelated `F.pad` — and SIGABRT carries no exception type, so these
# cannot be recorded as strict xfails in testing/known_failure_xfails/mps_float32.txt: the process
# dies and every test after it is lost. conftest skips this module on MPS; real Apple hardware
# does not abort, so `--run-mps-process-abort` runs it anyway. Tracked in #4204.
pytestmark = pytest.mark.mps_process_abort


class TestRANSACHomography(BaseTester):
    def test_smoke(self, device, dtype):
        torch.random.manual_seed(0)
        points1 = torch.rand(4, 2, device=device, dtype=dtype)
        points2 = torch.rand(4, 2, device=device, dtype=dtype)
        ransac = RANSAC("homography").to(device=device, dtype=dtype)
        torch.random.manual_seed(0)
        H, _ = ransac(points1, points2)
        assert H.shape == (3, 3)

    @pytest.mark.parametrize("model_found", [False, True])
    def test_forward_output_shapes_are_stable(self, device, dtype, model_found):
        generator = torch.Generator().manual_seed(123)
        # A 10 px extent keeps bfloat16 rounding of the transformed points below inl_th; at 100 px it drops inliers.
        points1 = 10.0 * torch.rand(16, 2, generator=generator).to(device=device, dtype=dtype)
        unrelated_points = 10.0 * torch.rand(16, 2, generator=generator).to(device=device, dtype=dtype)
        homography = torch.tensor([[1.0, 0.1, 2.0], [0.05, 1.0, -1.0], [0.001, 0.002, 1.0]], device=device, dtype=dtype)
        points2 = transform_points(homography[None], points1[None])[0] if model_found else unrelated_points
        ransac = RANSAC("homography", inl_th=0.5, batch_size=4, max_iter=1, max_lo_iters=0, seed=0)

        model, inliers = ransac(points1, points2)

        assert model.shape == (3, 3)
        assert inliers.shape == (16,)
        assert inliers.dtype == torch.bool
        assert inliers.device == points1.device
        selected_points = points1[inliers]
        assert selected_points.shape == ((16 if model_found else 0), 2)

    @pytest.mark.xfail(reason="might slightly and randomly imprecise due to RANSAC randomness")
    def test_dirty_points(self, device, dtype):
        # generate input data
        torch.random.manual_seed(0)

        H = torch.eye(3, dtype=dtype, device=device)
        H[:2] = H[:2] + 0.1 * torch.rand_like(H[:2])
        H[2:, :2] = H[2:, :2] + 0.001 * torch.rand_like(H[2:, :2])

        points_src = 100.0 * torch.rand(1, 20, 2, device=device, dtype=dtype)
        points_dst = transform_points(H[None], points_src)

        # making last point an outlier
        points_dst[:, -1, :] += 800
        ransac = RANSAC("homography", inl_th=0.5, max_iter=20).to(device=device, dtype=dtype)
        # compute transform from source to target
        dst_homo_src, _ = ransac(points_src[0], points_dst[0])

        self.assert_close(
            transform_points(dst_homo_src[None], points_src[:, :-1]), points_dst[:, :-1], rtol=1e-3, atol=1e-3
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("data", ["loftr_homo"], indirect=True)
    def test_real_clean(self, device, dtype, data):
        # generate input data
        torch.random.manual_seed(0)
        data_dev = dict_to(data, device, dtype)
        homography_gt = torch.inverse(data_dev["H_gt"])
        homography_gt = homography_gt / homography_gt[2, 2]
        pts_src = data_dev["pts0"]
        pts_dst = data_dev["pts1"]
        ransac = RANSAC("homography", inl_th=0.5, max_iter=20).to(device=device, dtype=dtype)
        # compute transform from source to target
        dst_homo_src, _ = ransac(pts_src, pts_dst)

        self.assert_close(transform_points(dst_homo_src[None], pts_src[None]), pts_dst[None], rtol=1e-2, atol=1.0)

    @pytest.mark.slow
    @pytest.mark.xfail(reason="might slightly and randomly imprecise due to RANSAC randomness")
    @pytest.mark.parametrize("data", ["loftr_homo"], indirect=True)
    def test_real_dirty(self, device, dtype, data):
        # generate input data
        torch.random.manual_seed(0)
        data_dev = dict_to(data, device, dtype)
        homography_gt = torch.inverse(data_dev["H_gt"])
        homography_gt = homography_gt / homography_gt[2, 2]
        pts_src = data_dev["pts0"]
        pts_dst = data_dev["pts1"]

        kp1 = data_dev["loftr_outdoor_tentatives0"]
        kp2 = data_dev["loftr_outdoor_tentatives1"]

        ransac = RANSAC("homography", inl_th=3.0, max_iter=30, max_lo_iters=10).to(device=device, dtype=dtype)
        # compute transform from source to target
        dst_homo_src, _ = ransac(kp1, kp2)

        # Reprojection error of 5px is OK
        self.assert_close(transform_points(dst_homo_src[None], pts_src[None]), pts_dst[None], rtol=0.15, atol=5)

    @pytest.mark.skip(reason="find_homography_dlt is using try/except block")
    def test_jit(self, device, dtype):
        torch.random.manual_seed(0)
        points1 = torch.rand(4, 2, device=device, dtype=dtype)
        points2 = torch.rand(4, 2, device=device, dtype=dtype)
        model = RANSAC("homography").to(device=device, dtype=dtype)
        model_jit = torch.jit.script(RANSAC("homography").to(device=device, dtype=dtype))
        self.assert_close(model(points1, points2)[0], model_jit(points1, points2)[0], rtol=1e-4, atol=1e-4)


class TestRANSACHomographyLineSegments(BaseTester):
    def test_smoke(self, device, dtype):
        torch.random.manual_seed(0)
        points1 = torch.rand(4, 2, 2, device=device, dtype=dtype)
        points2 = torch.rand(4, 2, 2, device=device, dtype=dtype)
        ransac = RANSAC("homography_from_linesegments").to(device=device, dtype=dtype)
        torch.random.manual_seed(0)
        H, _ = ransac(points1, points2)
        assert H.shape == (3, 3)

    @pytest.mark.xfail(reason="might slightly and randomly imprecise due to RANSAC randomness")
    def test_dirty_points(self, device, dtype):
        # generate input data
        torch.random.manual_seed(0)

        H = torch.eye(3, dtype=dtype, device=device)
        H[:2] = H[:2] + 0.1 * torch.rand_like(H[:2])
        H[2:, :2] = H[2:, :2] + 0.001 * torch.rand_like(H[2:, :2])

        points_src_st = 100.0 * torch.rand(1, 20, 2, device=device, dtype=dtype)
        points_src_end = 100.0 * torch.rand(1, 20, 2, device=device, dtype=dtype)

        points_dst_st = transform_points(H[None], points_src_st)
        points_dst_end = transform_points(H[None], points_src_end)

        # making last point an outlier
        points_dst_st[:, -1, :] += 800
        ls1 = torch.stack([points_src_st, points_src_end], dim=2)
        ls2 = torch.stack([points_dst_st, points_dst_end], dim=2)

        ransac = RANSAC("homography_from_linesegments", inl_th=0.5, max_iter=20).to(device=device, dtype=dtype)
        # compute transform from source to target
        dst_homo_src, _ = ransac(ls1[0], ls2[0])

        self.assert_close(
            transform_points(dst_homo_src[None], points_src_st[:, :-1]), points_dst_st[:, :-1], rtol=1e-3, atol=1e-3
        )

    @pytest.mark.skip(reason="find_homography_dlt is using try/except block")
    def test_jit(self, device, dtype):
        torch.random.manual_seed(0)
        points1 = torch.rand(4, 2, 2, device=device, dtype=dtype)
        points2 = torch.rand(4, 2, 2, device=device, dtype=dtype)
        model = RANSAC("homography_from_linesegments").to(device=device, dtype=dtype)
        model_jit = torch.jit.script(RANSAC("homography_from_linesegments").to(device=device, dtype=dtype))
        self.assert_close(model(points1, points2)[0], model_jit(points1, points2)[0], rtol=1e-4, atol=1e-4)


class TestRANSACFundamental(BaseTester):
    def test_smoke(self, device, dtype):
        torch.random.manual_seed(0)
        points1 = torch.rand(8, 2, device=device, dtype=dtype)
        points2 = torch.rand(8, 2, device=device, dtype=dtype)
        ransac = RANSAC("fundamental").to(device=device, dtype=dtype)
        Fm, _ = ransac(points1, points2)
        assert Fm.shape == (3, 3)

    @pytest.mark.slow
    @pytest.mark.parametrize("data", ["loftr_fund"], indirect=True)
    def test_real_clean_8pt(self, device, dtype, data):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("find_fundamental calls torch.linalg.eigh, which has no float16/bfloat16 kernel")
        torch.random.manual_seed(0)
        # generate input data
        data_dev = dict_to(data, device, dtype)
        pts_src = data_dev["pts0"]
        pts_dst = data_dev["pts1"]
        # The ground-truth F leaves up to 0.58px on these 10 points. At 0.5px the best eight-point fit has
        # only its own sample as inliers, which is no consensus, so RANSAC returns the zero failure matrix;
        # at 1px all ten are inliers, and over 100 seeds the refit's largest error is at most 1px.
        ransac = RANSAC("fundamental", inl_th=1.0, max_iter=20, max_lo_iters=10).to(device=device, dtype=dtype)
        fundamental_matrix, mask = ransac(pts_src, pts_dst)
        # A zero matrix is the failure result and has zero Sampson error for every point.
        assert fundamental_matrix.abs().amax() > 0
        assert mask.all()
        gross_errors = (
            sampson_epipolar_distance(pts_src[None], pts_dst[None], fundamental_matrix[None], squared=False) > 2.0
        )
        assert gross_errors.sum().item() == 0

    @pytest.mark.slow
    @pytest.mark.xfail(reason="might fail, because out F-RANSAC is not yet 7pt")
    @pytest.mark.parametrize("data", ["loftr_fund"], indirect=True)
    def test_real_clean_7pt(self, device, dtype, data):
        torch.random.manual_seed(0)
        # generate input data
        data_dev = dict_to(data, device, dtype)
        pts_src = data_dev["pts0"]
        pts_dst = data_dev["pts1"]
        # compute transform from source to target
        ransac = RANSAC("fundamental_7pt", inl_th=1.0, max_iter=100, max_lo_iters=10).to(device=device, dtype=dtype)
        fundamental_matrix, _ = ransac(pts_src, pts_dst)
        assert fundamental_matrix.abs().amax() > 0
        gross_errors = (
            sampson_epipolar_distance(pts_src[None], pts_dst[None], fundamental_matrix[None], squared=False) > 1.0
        )
        assert gross_errors.sum().item() == 0

    @pytest.mark.slow
    @pytest.mark.xfail(reason="might slightly and randomly imprecise due to RANSAC randomness")
    @pytest.mark.parametrize("data", ["loftr_fund"], indirect=True)
    def test_real_dirty_8pt(self, device, dtype, data):
        torch.random.manual_seed(0)
        # generate input data
        data_dev = dict_to(data, device, dtype)
        pts_src = data_dev["pts0"]
        pts_dst = data_dev["pts1"]

        kp1 = data_dev["loftr_indoor_tentatives0"]
        kp2 = data_dev["loftr_indoor_tentatives1"]

        ransac = RANSAC("fundamental", inl_th=1.0, max_iter=20, max_lo_iters=10).to(device=device, dtype=dtype)
        # compute transform from source to target
        fundamental_matrix, _ = ransac(kp1, kp2)
        assert fundamental_matrix.abs().amax() > 0
        gross_errors = (
            sampson_epipolar_distance(pts_src[None], pts_dst[None], fundamental_matrix[None], squared=False) > 10.0
        )
        assert gross_errors.sum().item() < 2

    @pytest.mark.slow
    @pytest.mark.xfail(reason="might fail, because this F-RANSAC is not 7pt")
    @pytest.mark.parametrize("data", ["loftr_fund"], indirect=True)
    def test_real_dirty_7pt(self, device, dtype, data):
        torch.random.manual_seed(0)
        # generate input data
        data_dev = dict_to(data, device, dtype)
        pts_src = data_dev["pts0"]
        pts_dst = data_dev["pts1"]

        kp1 = data_dev["loftr_indoor_tentatives0"]
        kp2 = data_dev["loftr_indoor_tentatives1"]

        ransac = RANSAC("fundamental_7pt", inl_th=1.0, max_iter=20, max_lo_iters=10).to(device=device, dtype=dtype)
        # compute transform from source to target
        fundamental_matrix, _ = ransac(kp1, kp2)
        assert fundamental_matrix.abs().amax() > 0
        gross_errors = (
            sampson_epipolar_distance(pts_src[None], pts_dst[None], fundamental_matrix[None], squared=False) > 10.0
        )
        assert gross_errors.sum().item() < 2

    @pytest.mark.skip(reason="try except block in python version")
    def test_jit(self, device, dtype):
        torch.random.manual_seed(0)
        points1 = torch.rand(8, 2, device=device, dtype=dtype)
        points2 = torch.rand(8, 2, device=device, dtype=dtype)
        model = RANSAC("fundamental").to(device=device, dtype=dtype)
        model_jit = torch.jit.script(model)
        self.assert_close(model(points1, points2)[0], model_jit(points1, points2)[0], rtol=1e-3, atol=1e-3)

    @pytest.mark.skip(reason="RANSAC is random algorithm, so Jacobian is not defined")
    def test_gradcheck(self, device):
        torch.random.manual_seed(0)
        points1 = torch.rand(8, 2, device=device, dtype=torch.float64, requires_grad=True)
        points2 = torch.rand(8, 2, device=device, dtype=torch.float64)
        model = RANSAC("fundamental").to(device=device, dtype=torch.float64)

        def gradfun(p1, p2):
            return model(p1, p2)[0]

        self.gradcheck(gradfun, (points1, points2), fast_mode=False, requires_grad=(True, False, False))


class TestRANSACLocalOptimization(BaseTester):
    @pytest.mark.parametrize("good_idx", [1, 2])
    def test_polish_step_uses_verify_best_model(self, device, dtype, good_idx):
        """Regression test for RANSAC.forward()'s local-optimization loop.

        It must adopt the model verify() actually selects as best-scoring, not blindly take a fixed index of whatever
        the polisher solver returns. This is only observable when a polisher returns more than one candidate, so the
        polisher is patched to return three: two deliberately bad matrices and the least-squares fit of the data, placed
        at ``good_idx`` so that neither the first nor the last position alone satisfies the test. The minimal solver is
        patched to a plausible but worse fit, so the local-optimization branch adopts a candidate by construction
        rather than depending on how the random minimal samples fall.
        """
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("find_fundamental needs linalg_eigh, which has no float16/bfloat16 kernel")
        torch.random.manual_seed(0)
        points1 = torch.rand(60, 2, device=device, dtype=dtype) * 100.0
        H = torch.tensor([[1.0, 0.05, 3.0], [-0.03, 1.0, -2.0], [1e-4, 1e-4, 1.0]], device=device, dtype=dtype)
        ph = torch.cat([points1, torch.ones(60, 1, device=device, dtype=dtype)], 1) @ H.T
        points2 = ph[:, :2] / ph[:, 2:] + 0.3 * torch.randn(60, 2, device=device, dtype=dtype)

        good_fit = find_fundamental(points1[None], points2[None])
        # A plausible but worse starting model: the least-squares fit to a much noisier copy of the data.
        worse_fit = find_fundamental(points1[None], points2[None] + torch.randn_like(points2[None]))
        bad_fits = [
            torch.eye(3, device=device, dtype=dtype)[None],
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)[None],
        ]
        candidates = torch.cat([*bad_fits[:good_idx], good_fit, *bad_fits[good_idx:]], dim=0)
        calls = []

        def fake_polisher(kp1, kp2, weights):
            calls.append(1)
            return candidates

        ransac = RANSAC("fundamental", inl_th=1.0, batch_size=4, max_iter=1, max_lo_iters=1, score_type="msac")
        ransac.minimal_solver = lambda kp1, kp2, w: worse_fit.expand(kp1.shape[0], 3, 3).clone()
        ransac.polisher_solver = fake_polisher

        Fm, _ = ransac(points1, points2)
        assert len(calls) > 0, "the local-optimization step never ran"
        self.assert_close(Fm, good_fit[0])


class TestRansacMethods:
    def test_max_samples_by_conf(self):
        """Test max_samples_by_conf with realistic scenarios."""
        conf = 0.99

        # Test 1: Very few inliers (1 out of 1000) with sample_size=7
        # An all-inlier minimal sample is impossible.
        x = RANSAC.max_samples_by_conf(n_inl=1, num_tc=1000, sample_size=7, conf=conf)
        assert x == sys.maxsize

        # Test 2: Low inlier ratio (10 out of 1000, 1%) with sample_size=7
        # Should require many iterations
        x = RANSAC.max_samples_by_conf(n_inl=10, num_tc=1000, sample_size=7, conf=conf)
        assert x > 0
        # With 1% inliers, probability of all 7 samples being inliers is very low
        # So we need a huge number of samples
        assert x > 1_000_000  # Should be a very large number

        # Test 3: Medium inlier ratio (500 out of 1000, 50%) with sample_size=4
        # Should require moderate number of iterations
        x = RANSAC.max_samples_by_conf(n_inl=500, num_tc=1000, sample_size=4, conf=conf)
        assert x > 0
        # With 50% inliers, probability of all 4 samples being inliers is ~0.0625
        # So we need log(1-0.99)/log(1-0.0625) ≈ 70 samples
        assert 50 < x < 100

        # Test 4: High inlier ratio (900 out of 1000, 90%) with sample_size=4
        # Should require very few iterations
        x = RANSAC.max_samples_by_conf(n_inl=900, num_tc=1000, sample_size=4, conf=conf)
        assert x > 0
        # With 90% inliers, probability of all 4 samples being inliers is ~0.66
        # So we need log(1-0.99)/log(1-0.66) ≈ 4 samples
        assert x < 10

        # Test 5: Edge case - all points are inliers
        x = RANSAC.max_samples_by_conf(n_inl=100, num_tc=100, sample_size=4, conf=conf)
        assert x == 1  # Only need 1 sample if all are inliers

        # Test 6: Edge case - not enough points for sample
        x = RANSAC.max_samples_by_conf(n_inl=10, num_tc=10, sample_size=15, conf=conf)
        assert x == sys.maxsize

        # Test 7: Edge case - too few inliers (n_inl <= sample_size)
        x = RANSAC.max_samples_by_conf(n_inl=2, num_tc=1000, sample_size=4, conf=conf)
        assert x == sys.maxsize

        # Test 8: Edge case - confidence at boundaries
        x = RANSAC.max_samples_by_conf(n_inl=50, num_tc=100, sample_size=4, conf=1.0)
        assert x == sys.maxsize

        x = RANSAC.max_samples_by_conf(n_inl=50, num_tc=100, sample_size=4, conf=0.0)
        assert x == 1  # Returns 1 when conf <= 0.0

        # Test 9: Verify monotonicity - more inliers should require fewer samples
        x1 = RANSAC.max_samples_by_conf(n_inl=200, num_tc=1000, sample_size=4, conf=conf)
        x2 = RANSAC.max_samples_by_conf(n_inl=500, num_tc=1000, sample_size=4, conf=conf)
        x3 = RANSAC.max_samples_by_conf(n_inl=800, num_tc=1000, sample_size=4, conf=conf)
        assert x1 > x2 > x3  # More inliers = fewer samples needed


class TestRANSACSeed:
    def test_same_seed_reproducible(self, device, dtype):
        """Same seed should produce identical results across two calls."""
        torch.manual_seed(42)
        points1 = torch.rand(20, 2, device=device, dtype=dtype)
        points2 = torch.rand(20, 2, device=device, dtype=dtype)

        ransac = RANSAC("homography", inl_th=2.0, max_iter=5, seed=123).to(device=device, dtype=dtype)
        H1, inliers1 = ransac(points1, points2)
        H2, inliers2 = ransac(points1, points2)

        assert torch.allclose(H1, H2)
        assert torch.equal(inliers1, inliers2)

    def test_different_seeds_differ(self, device, dtype):
        """Different seeds should (very likely) produce different results."""
        torch.manual_seed(42)
        points1 = torch.rand(20, 2, device=device, dtype=dtype)
        points2 = torch.rand(20, 2, device=device, dtype=dtype)

        # Every point is an inlier at this threshold, so local optimization would return the same
        # least-squares fit for any seed; compare the seed-dependent minimal-sample models instead.
        ransac_a = RANSAC("homography", inl_th=2.0, max_iter=5, max_lo_iters=0, seed=1).to(device=device, dtype=dtype)
        ransac_b = RANSAC("homography", inl_th=2.0, max_iter=5, max_lo_iters=0, seed=2).to(device=device, dtype=dtype)

        H_a, _ = ransac_a(points1, points2)
        H_b, _ = ransac_b(points1, points2)

        assert not torch.allclose(H_a, H_b)

    def test_no_seed_differs_from_seeded(self, device, dtype):
        """Without a seed, repeated calls should not be forced to be identical to a seeded call."""
        torch.manual_seed(0)
        points1 = torch.rand(20, 2, device=device, dtype=dtype)
        points2 = torch.rand(20, 2, device=device, dtype=dtype)

        ransac_seeded = RANSAC("homography", inl_th=2.0, max_iter=5, seed=42).to(device=device, dtype=dtype)
        H_s1, _ = ransac_seeded(points1, points2)
        H_s2, _ = ransac_seeded(points1, points2)
        # Seeded should be reproducible
        assert torch.allclose(H_s1, H_s2)

    def test_seed_stored_as_attribute(self, device, dtype):
        ransac = RANSAC("homography", seed=7)
        assert ransac.seed == 7

    def test_none_seed_stored(self, device, dtype):
        ransac = RANSAC("homography")
        assert ransac.seed is None


class TestRANSACEssential(BaseTester):
    def test_smoke(self, device, dtype):
        torch.random.manual_seed(0)
        points1 = torch.rand(8, 2, device=device, dtype=dtype)
        points2 = torch.rand(8, 2, device=device, dtype=dtype)
        ransac = RANSAC("essential").to(device=device, dtype=dtype)
        E, _ = ransac(points1, points2)
        assert E.shape == (3, 3)

    def test_polish_enforces_essential_constraint(self, device, dtype):
        """Regression test for https://github.com/kornia/kornia/issues/3874.

        The polishing (local-optimization) step must return an essential matrix, i.e. its two
        non-zero singular values must be (approximately) equal and the third must be
        (approximately) zero. The buggy version fit a fundamental matrix and returned it as-is,
        violating the constraint and silently breaking decompose_essential_matrix.
        """
        torch.random.manual_seed(0)
        ransac = RANSAC("essential").to(device=device, dtype=dtype)
        kp1 = torch.rand(20, 2, device=device, dtype=dtype)
        kp2 = torch.rand(20, 2, device=device, dtype=dtype)
        inliers = torch.ones(20, dtype=torch.bool, device=device)
        E = ransac.polish_model(kp1, kp2, inliers)  # (1, 3, 3)
        sv = torch.linalg.svdvals(E[0])
        # two non-zero singular values must be equal (essential-matrix constraint)
        assert (sv[0] / sv[1] - 1.0).abs() < 1e-2
        # third singular value must be ~0 (relative to the largest singular value)
        assert (sv[2] / sv[0]).abs() < 1e-4

    def test_forward_enforces_essential_constraint(self, device, dtype):
        """Regression test for https://github.com/kornia/kornia/issues/3874 (forward path).

        forward() must return an essential matrix even when local optimization does not win and
        the best model is the minimal-solver estimate (find_essential), which is not guaranteed
        to be on the essential manifold. The returned model is projected onto the manifold
        before being returned, so the constraint holds for the model forward() selects.

        ``batch_size``/``max_iter`` are trimmed for speed, but ``max_lo_iters`` keeps its default
        so the local-optimization loop still runs: on this input it is entered once per draw and
        never beats the minimal-solver model, exactly as with the default sampler settings.
        """
        torch.random.manual_seed(0)
        ransac = RANSAC("essential", batch_size=32, max_iter=1).to(device=device, dtype=dtype)
        for _ in range(5):
            kp1 = torch.rand(20, 2, device=device, dtype=dtype)
            kp2 = torch.rand(20, 2, device=device, dtype=dtype)
            E, _ = ransac(kp1, kp2)
            sv = torch.linalg.svdvals(E)
            assert (sv[0] / sv[1] - 1.0).abs() < 1e-2
            assert (sv[2] / sv[0]).abs() < 1e-4

    def test_project_to_essential(self, device, dtype):
        """project_to_essential must enforce the essential-matrix constraint."""
        torch.random.manual_seed(0)
        M = torch.rand(4, 3, 3, device=device, dtype=dtype)
        E = project_to_essential(M)
        svd_input = E.float() if E.dtype in (torch.float16, torch.bfloat16) else E
        sv = torch.linalg.svdvals(svd_input)
        assert torch.all((sv[..., 0] / sv[..., 1] - 1.0).abs() < 1e-2)
        zero_tolerance = max(1e-4, torch.finfo(E.dtype).eps)
        assert torch.all((sv[..., 2] / sv[..., 0]).abs() < zero_tolerance)

    @pytest.mark.skip(reason="try except block in python version")
    def test_jit(self, device, dtype):
        torch.random.manual_seed(0)
        points1 = torch.rand(8, 2, device=device, dtype=dtype)
        points2 = torch.rand(8, 2, device=device, dtype=dtype)
        model = RANSAC("essential").to(device=device, dtype=dtype)
        model_jit = torch.jit.script(model)
        self.assert_close(model(points1, points2)[0], model_jit(points1, points2)[0], rtol=1e-3, atol=1e-3)

    @pytest.mark.skip(reason="RANSAC is random algorithm, so Jacobian is not defined")
    def test_gradcheck(self, device):
        torch.random.manual_seed(0)
        points1 = torch.rand(8, 2, device=device, dtype=torch.float64, requires_grad=True)
        points2 = torch.rand(8, 2, device=device, dtype=torch.float64)
        model = RANSAC("essential").to(device=device, dtype=torch.float64)

        def gradfun(p1, p2):
            return model(p1, p2)[0]

        self.gradcheck(gradfun, (points1, points2), fast_mode=False, requires_grad=(True, False, False))


# Sixteen exact planar matches for the convention pins. _MATCHES_1 is an asymmetric 4 x 4 lattice with a per-point
# diagonal jitter, generated by
#   torch.stack(torch.meshgrid(torch.tensor([160., 230., 330., 440.]), torch.tensor([95., 150., 240., 320.]),
#   indexing="xy"), -1).reshape(16, 2) + torch.linspace(0, 3, 16)[:, None]
# and _H_TRUE is a fixed homography (a mild projective warp) whose entries are all distinct and non-zero, so swapping
# the images changes the model.
_MATCHES_1 = [
    [160.0, 95.0], [230.2, 95.2], [330.4, 95.4], [440.6, 95.6], [160.8, 150.8], [231.0, 151.0], [331.2, 151.2],
    [441.4, 151.4], [161.6, 241.6], [231.8, 241.8], [332.0, 242.0], [442.2, 242.2], [162.4, 322.4], [232.6, 322.6],
    [332.8, 322.8], [443.0, 323.0],
]  # fmt: skip
_H_TRUE = [
    [1.034685, -0.04160598, -63.61482],
    [0.1039753, 1.135451, -62.87412],
    [2.851519e-04, 1.404178e-04, 1.0],
]
# Sixteen fixed points per image that no homography relates to each other or to the matches above.
_UNRELATED_1 = [
    [75.0, 410.0], [390.0, 30.0], [505.0, 260.0], [120.0, 205.0], [285.0, 380.0], [455.0, 60.0], [30.0, 140.0],
    [350.0, 290.0], [205.0, 25.0], [480.0, 395.0], [95.0, 330.0], [260.0, 175.0], [410.0, 225.0], [180.0, 450.0],
    [55.0, 70.0], [300.0, 120.0],
]  # fmt: skip
_UNRELATED_2 = [
    [310.0, 95.0], [60.0, 280.0], [150.0, 30.0], [420.0, 330.0], [25.0, 190.0], [240.0, 400.0], [380.0, 170.0],
    [100.0, 60.0], [470.0, 240.0], [200.0, 130.0], [330.0, 20.0], [45.0, 420.0], [270.0, 300.0], [495.0, 110.0],
    [140.0, 260.0], [215.0, 355.0],
]  # fmt: skip
# A non-planar two-view scene for the fundamental-matrix pin: K1 != K2, fx != fy, cx != cy, a rotation about a
# non-axis direction and a translation off every axis. Twelve points at depth 4 to 6 in camera 1, no two sharing a
# coordinate; the matches are their pixel projections through P1 = K1 [I | 0] and P2 = K2 [R | t].
_SCENE_K1 = [[800.0, 0.0, 320.0], [0.0, 760.0, 200.0], [0.0, 0.0, 1.0]]
_SCENE_K2 = [[700.0, 0.0, 300.0], [0.0, 740.0, 240.0], [0.0, 0.0, 1.0]]
_SCENE_AXIS_ANGLE = [0.1, -0.2, 0.05]
_SCENE_T = [0.5, 0.1, 0.02]
_SCENE_X = [
    [-0.0075, 0.5364, 4.4163], [-0.7359, -0.3852, 5.8596], [-0.0198, 0.7929, 5.4462], [0.2646, -0.3022, 5.4847],
    [-0.9553, -0.6623, 5.0526], [0.0370, 0.3953, 4.4873], [-0.6779, -0.4355, 5.1692], [0.8304, -0.2058, 4.0663],
    [-0.1612, 0.1058, 4.2774], [-0.9277, -0.6295, 4.4845], [-0.3898, 0.8640, 5.6309], [-0.4603, -0.6986, 5.5863],
]  # fmt: skip
_HALF_LINES = (
    "line_segment_transfer_error_one_way builds the unnormalised image-2 line in the input dtype; its constant term "
    "is of order (pixel coordinate)**2, which overflows float16 and which bfloat16 cannot resolve"
)


def _cpu_only(device: torch.device) -> None:
    if device.type != "cpu":
        pytest.skip("the RANSAC convention pins are verified on the CPU only")


def _planar_matches(device, dtype):
    """Sixteen exact matches ``kp2 = H(kp1)``, generated in float64 on the CPU and cast, shape ``(16, 2)`` each."""
    kp1 = torch.tensor([_MATCHES_1], dtype=torch.float64)
    kp2 = transform_points(torch.tensor([_H_TRUE], dtype=torch.float64), kp1)
    return kp1[0].to(device, dtype), kp2[0].to(device, dtype)


def _scene_matches(device, dtype):
    """Pixel matches of the two-view scene, projected in float64 on the CPU and cast, shape ``(12, 2)`` each."""
    f64 = torch.float64
    R = axis_angle_to_rotation_matrix(torch.tensor([_SCENE_AXIS_ANGLE], dtype=f64))[0]
    X = torch.tensor(_SCENE_X, dtype=f64)
    kp1 = convert_points_from_homogeneous(X @ torch.tensor(_SCENE_K1, dtype=f64).T)
    kp2 = convert_points_from_homogeneous(
        (X @ R.T + torch.tensor(_SCENE_T, dtype=f64)) @ torch.tensor(_SCENE_K2, dtype=f64).T
    )
    return kp1.to(device, dtype), kp2.to(device, dtype)


def _planar_segments(device, dtype, stretch: float):
    """Eight segments made of consecutive lattice points and their exact images; segment 0 is ``stretch`` x longer."""
    ls1 = torch.tensor(_MATCHES_1, dtype=torch.float64).reshape(8, 2, 2)
    ls1[0, 1] = ls1[0, 0] + stretch * (ls1[0, 1] - ls1[0, 0])
    ls2 = transform_points(torch.tensor([_H_TRUE], dtype=torch.float64), ls1.reshape(1, 16, 2)).reshape(8, 2, 2)
    return ls1.to(device, dtype), ls2.to(device, dtype)


def _fixed_model(ransac: RANSAC, model: torch.Tensor) -> RANSAC:
    """Make every minimal sample return ``model``, so that forward's thresholding is observed on a known model."""
    ransac.minimal_solver = lambda kp1, kp2, weights: model.expand(kp1.shape[0], 3, 3).clone()
    return ransac


def _max_transfer(model: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    return (transform_points(model[None], src[None]) - dst[None]).abs().max()


class TestConventionRANSAC(BaseTester):
    def test_convention_ransac_model_maps_kp1_to_kp2(self, device, dtype, monkeypatch):
        _cpu_only(device)
        kp1, kp2 = _planar_matches(device, dtype)
        # inl_th=5 keeps every exact match an inlier in bfloat16 too, whose pixel coordinates carry up to 2 px of
        # rounding here.
        model, mask = RANSAC("homography", inl_th=5.0, seed=0, max_iter=3, batch_size=64)(kp1, kp2)
        # A (3, 3) model and an (N,) bool mask; the model maps kp1 onto kp2, and the control (applying it to kp2)
        # misses kp1 by about 213 px.
        assert model.shape == (3, 3)
        assert mask.shape == (16,) and mask.dtype == torch.bool and bool(mask.all())
        assert _max_transfer(model, kp1, kp2) < 0.05 * _max_transfer(model, kp2, kp1)
        swapped, _ = RANSAC("homography", inl_th=5.0, seed=0, max_iter=3, batch_size=64)(kp2, kp1)
        assert _max_transfer(swapped, kp2, kp1) < 0.05 * _max_transfer(swapped, kp1, kp2)
        # No sample reaches consensus on unrelated points: an all-zero model and no inliers.
        unrelated_1 = torch.tensor(_UNRELATED_1[:12], device=device, dtype=dtype)
        unrelated_2 = torch.tensor(_UNRELATED_2[:12], device=device, dtype=dtype)
        model, mask = RANSAC("homography", inl_th=0.5, seed=0, max_iter=3, batch_size=64)(unrelated_1, unrelated_2)
        assert bool((model == 0).all())
        assert mask.shape == (12,) and mask.dtype == torch.bool and not bool(mask.any())
        # "homography" screens its minimal samples with sample_is_valid_for_homography: when every sample is
        # rejected, no model is estimated from the exact matches either.
        samples = []

        def reject_all(points1, points2):
            samples.append(tuple(points1.shape))
            return torch.zeros(points1.shape[0], dtype=torch.bool, device=points1.device)

        monkeypatch.setattr(kornia.geometry.ransac, "sample_is_valid_for_homography", reject_all)
        model, mask = RANSAC("homography", inl_th=5.0, seed=0, max_iter=3, batch_size=64)(kp1, kp2)
        # Every sample is rejected, so nothing stops early: max_iter=3 batches of batch_size=64 minimal samples,
        # the documented maximum.
        assert len(samples) == 3 and all(shape == (64, 4, 2) for shape in samples)
        assert bool((model == 0).all()) and not bool(mask.any())

    @pytest.mark.parametrize("model_type", ["homography", "fundamental"])
    def test_convention_ransac_inl_th_is_pixels_for_point_models(self, model_type, device, dtype):
        _cpu_only(device)
        if model_type == "homography":
            # inl_th is compared with the one-way transfer error in pixels (forward passes inl_th**2 to the squared
            # error): a match displaced by 10 px is an outlier at inl_th=4 and an inlier at inl_th=15. RANSAC may
            # absorb part of the displacement into a compromise model, so the flip lies at or below 10 px; both
            # thresholds stay clear of it, and in squared-pixel units both would reject the match.
            kp1, kp2 = _planar_matches(device, dtype)
            kp2_bad = kp2.clone()
            kp2_bad[9, 0] += 10.0  # an interior match, displaced along x
            for inl_th, expected in ((4.0, False), (15.0, True)):
                _, mask = RANSAC("homography", inl_th=inl_th, max_iter=20, seed=0)(kp1, kp2_bad)
                assert mask.dtype == torch.bool and mask.shape == (16,)
                assert bool(mask[9]) is expected
                assert int(mask.sum()) == 15 + int(expected)
            # With the true H fixed, the moved match's one-way error is exactly 10 px (its symmetric error is
            # about 15 px): an inlier at inl_th=11 and an outlier at inl_th=9. The minimal samples still pass
            # sample_is_valid_for_homography, which rejects some lattice samples, so a seeded batch of 64 is drawn.
            H = torch.tensor([_H_TRUE], device=device, dtype=dtype)
            for inl_th, expected in ((9.0, False), (11.0, True)):
                ransac = RANSAC("homography", inl_th=inl_th, batch_size=64, max_iter=1, max_lo_iters=0, seed=0)
                _, mask = _fixed_model(ransac, H)(kp1, kp2_bad)
                assert bool(mask[9]) is expected
                assert int(mask.sum()) == 15 + int(expected)
            return
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("find_fundamental calls torch.linalg.eigh, which has no float16/bfloat16 kernel")
        kp1, kp2 = _scene_matches(device, dtype)
        if dtype == torch.float64:
            # kp1 is the estimator's first image: the returned F explains every match in this order
            # (x2^T F x1 = 0), and its transpose, the other image order, explains none. float64 only: in float32
            # the fit on these exact matches already misses by a pixel or two.
            F_est, mask_est = RANSAC("fundamental", inl_th=1.0, seed=0, max_iter=3, batch_size=64)(kp1, kp2)
            assert bool(mask_est.all())
            assert sampson_epipolar_distance(kp1[None], kp2[None], F_est[None], squared=False).max() < 1e-2
            assert sampson_epipolar_distance(kp1[None], kp2[None], F_est.mT[None], squared=False).min() > 1.0
        # For the fundamental models inl_th is compared with the Sampson distance in pixels. The model is fixed to
        # the F of the twelve exact matches, so the threshold is observed without sampling.
        F = find_fundamental(kp1[None], kp2[None])
        kp2_bad = kp2.clone()
        kp2_bad[6] += torch.tensor([4.0, -3.0], device=device, dtype=dtype)
        distance = sampson_epipolar_distance(kp1[None], kp2_bad[None], F, squared=False)[0, 6]
        # About 2.6 px: an outlier at inl_th=2 and an inlier at inl_th=4. Compared in squared-pixel units, or with
        # the distance itself against inl_th**2, the two verdicts would not both hold.
        assert 2.0 < distance < 4.0
        for inl_th, expected in ((2.0, False), (4.0, True)):
            ransac = RANSAC("fundamental", inl_th=inl_th, batch_size=1, max_iter=1, max_lo_iters=0, seed=0)
            model, mask = _fixed_model(ransac, F)(kp1, kp2_bad)
            assert torch.equal(model, F[0])
            assert bool(mask[6]) is expected
            assert int(mask.sum()) == 11 + int(expected)

    def test_convention_ransac_seed_is_private(self, device, dtype):
        _cpu_only(device)
        kp1, kp2 = _planar_matches(device, dtype)
        kp2[9, 0] += 10.0

        def run(seed):
            return RANSAC("homography", inl_th=2.0, seed=seed, max_iter=3, batch_size=16)(kp1, kp2)

        # A seeded call draws from a private generator: it leaves the global RNG state unchanged, and a second
        # instance with the same seed reproduces it whatever the global generator holds.
        state = torch.get_rng_state()
        model_a, mask_a = run(11)
        assert torch.equal(torch.get_rng_state(), state)
        torch.manual_seed(123)
        model_b, mask_b = run(11)
        assert torch.equal(model_a, model_b) and torch.equal(mask_a, mask_b)
        # seed=None draws from the global generator: it advances it, and reseeding it reproduces the result.
        torch.manual_seed(5)
        state = torch.get_rng_state()
        model_c, mask_c = run(None)
        assert not torch.equal(torch.get_rng_state(), state)
        torch.manual_seed(5)
        model_d, mask_d = run(None)
        assert torch.equal(model_c, model_d) and torch.equal(mask_c, mask_d)

    def test_convention_ransac_linesegment_threshold_units_4867(self, device, dtype):
        _cpu_only(device)
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip(_HALF_LINES)
        H = torch.tensor([_H_TRUE], device=device, dtype=dtype)
        # One segment moved 3 px off its line in image 2, scored by the true H. The inlier test uses the mean
        # endpoint-to-line distance in pixels, whatever the segment's length: the 3-px offset is an inlier at
        # inl_th=4 and an outlier at inl_th=2 on a 66-px and on a 129-px segment alike. #4867 compared the distance
        # times the segment length with inl_th**2, which rejected both at inl_th=4.
        for stretch in (1.0, 2.0):
            ls1, ls2 = _planar_segments(device, dtype, stretch)
            d = ls2[0, 1] - ls2[0, 0]
            ls2[0] += 3.0 * torch.stack([-d[1], d[0]]) / d.norm()
            for inl_th, expected in ((4.0, True), (2.0, False)):
                ransac = RANSAC("homography_from_linesegments", inl_th=inl_th, batch_size=1, max_iter=1, max_lo_iters=0)
                _, mask = _fixed_model(ransac, H)(ls1, ls2)
                assert bool(mask[1:].all())
                assert bool(mask[0]) is expected

    def test_convention_ransac_msac_finds_model_with_outliers_4868(self, device, dtype):
        _cpu_only(device)
        kp1, kp2 = _planar_matches(device, dtype)
        kp1 = torch.cat([kp1, torch.tensor(_UNRELATED_1, device=device, dtype=dtype)])
        kp2 = torch.cat([kp2, torch.tensor(_UNRELATED_2, device=device, dtype=dtype)])
        # 16 exact matches and 16 outliers. score_type="ransac" finds the homography of the sixteen.
        model, mask = RANSAC("homography", inl_th=5.0, score_type="ransac", seed=0, max_iter=5, batch_size=256)(
            kp1, kp2
        )
        assert bool(mask[:16].all()) and not bool(mask[16:].any())
        # MSAC ranks candidates by their truncated residuals and accepts them by inlier count, so it finds the same
        # sixteen. #4868 compared the MSAC score N - sum(min(err, inl_th**2)) with the minimal sample size, a count,
        # and returned no model here.
        model, mask = RANSAC("homography", inl_th=5.0, score_type="msac", seed=0, max_iter=5, batch_size=256)(kp1, kp2)
        assert bool(model.abs().amax() > 0)
        assert bool(mask[:16].all()) and not bool(mask[16:].any())

    def test_convention_ransac_prosac_sampling_4869(self, device, dtype):
        _cpu_only(device)
        # PROSAC draws its first sample from the four top-ranked correspondences and then grows the prefix it samples
        # from, one newest correspondence per sample, so its first samples differ from uniform ones. #4869 stored
        # prosac_sampling and never read it.
        prosac = RANSAC("homography", inl_th=2.0, seed=3, prosac_sampling=True, max_iter=3, batch_size=16)
        uniform = RANSAC("homography", inl_th=2.0, seed=3, prosac_sampling=False, max_iter=3, batch_size=16)
        samples = prosac.sample(4, 32, 6, 0)
        assert samples[0].sort().values.tolist() == [0, 1, 2, 3]
        assert samples.amax(dim=1).tolist() == [3, 4, 5, 6, 7, 8]
        assert not torch.equal(samples, uniform.sample(4, 32, 6, 0))

    @pytest.mark.parametrize(
        "model_type, n, validated_type, validated_n",
        [("fundamental_7pt", 6, "fundamental", 7), ("essential", 4, "homography", 3)],
    )
    def test_convention_ransac_size_validation_4872(self, model_type, n, validated_type, validated_n, device, dtype):
        _cpu_only(device)
        kp1, kp2 = _planar_matches(device, dtype)
        # Too few correspondences for the validated model types raise kornia's ValueError before sampling.
        with pytest.raises(ValueError):
            RANSAC(validated_type, max_iter=1, batch_size=4, seed=0)(kp1[:validated_n], kp2[:validated_n])
        # fundamental_7pt and essential are validated the same way. #4872 skipped them, and they failed inside the
        # sampler with a RuntimeError.
        with pytest.raises(ValueError):
            RANSAC(model_type, max_iter=1, batch_size=4, seed=0)(kp1[:n], kp2[:n])


class TestRANSACScoringAndStopping(BaseTester):
    @pytest.mark.parametrize("threshold", [0.5, 1.0, 2.0])
    def test_msac_accepts_partial_consensus(self, device, dtype, threshold):
        # A perfect 40% consensus must be accepted regardless of threshold units.
        points = torch.arange(200, device=device, dtype=dtype).reshape(100, 2)
        target = points.clone()
        target[40:] += 100
        identity = torch.eye(3, device=device, dtype=dtype)[None]
        ransac = RANSAC("homography", inl_th=threshold, batch_size=1, max_iter=1, max_lo_iters=0, score_type="msac")
        ransac.remove_bad_samples = lambda a, b: (a, b)
        ransac.minimal_solver = lambda a, b, w: identity
        model, mask = ransac(points, target)
        self.assert_close(model, identity[0])
        assert mask.shape == (100,)
        assert mask.sum() == 40

    @pytest.mark.parametrize("model_type,n", [("homography", 5), ("essential", 6), ("essential", 7)])
    def test_keep_minimal_consensus_without_polishing(self, device, dtype, model_type, n):
        # Acceptance needs one inlier beyond the minimal sample, not the eight-point polisher's point count.
        points = torch.rand(n, 2, device=device, dtype=dtype)
        matrix = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)
        if model_type == "homography":
            matrix = torch.eye(3, device=device, dtype=dtype)
        ransac = RANSAC(model_type, batch_size=1, max_iter=1)
        ransac.remove_bad_samples = lambda a, b: (a, b)
        ransac.minimal_solver = lambda a, b, w: matrix[None]
        ransac.polisher_solver = lambda a, b, w: matrix[None]
        model, mask = ransac(points, points)
        assert model.abs().sum() > 0
        assert mask.shape == (n,)
        assert mask.all()

    def test_invalid_models(self, device, dtype):
        models = torch.eye(3, device=device, dtype=dtype).repeat(4, 1, 1)
        models[0] = 0
        models[1, 0, 0] = float("nan")
        models[2, 0, 0] = float("inf")
        valid = RANSAC("fundamental_7pt").remove_bad_models(models)
        self.assert_close(valid, models[3:])

    def test_msac_nonfinite_residual_is_outlier(self, device, dtype):
        points = torch.zeros(10, 2, device=device, dtype=dtype)
        candidates = torch.eye(3, device=device, dtype=dtype).repeat(2, 1, 1)
        candidates[1, 0, 2] = 1
        ransac = RANSAC(score_type="msac")
        errors = torch.zeros(2, 10, device=device, dtype=dtype)
        errors[0, 0] = float("nan")
        ransac.error_fn = lambda a, b, m: errors
        best, _, score, count = ransac.verify(points, points, candidates, 4.0)
        self.assert_close(best, candidates[1])
        assert math.isfinite(score)
        assert count == 10

    def test_stopping_uses_support_after_lo_and_checks_unchanged_best(self, device, dtype):
        # LO increases support to 70/100. The MSAC score is intentionally different.
        points = torch.rand(100, 2, device=device, dtype=dtype)
        matrix = torch.eye(3, device=device, dtype=dtype)[None]
        ransac = RANSAC("fundamental", batch_size=16, max_iter=10, max_lo_iters=1, score_type="msac")
        sampled = []
        ransac.minimal_solver = lambda a, b, w: sampled.append(1) or matrix
        ransac.polisher_solver = lambda a, b, w: matrix
        calls = []

        def verify(a, b, models, threshold):
            calls.append(1)
            count = 70 if len(calls) == 2 else 50
            mask = torch.arange(100, device=device) < count
            return matrix[0], mask, float(count - 20), float(count)

        ransac.verify = verify
        ransac(points, points)
        expected = math.ceil(RANSAC.max_samples_by_conf(70, 100, 8, 0.99) / 16)
        assert len(sampled) == expected

    def test_exact_minimal_support_confidence(self):
        # Exactly four inliers among ten: success probability is 1 / C(10, 4).
        expected = math.ceil(math.log1p(-0.99) / math.log1p(-1 / math.comb(10, 4)))
        assert RANSAC.max_samples_by_conf(4, 10, 4, 0.99) == expected

    @pytest.mark.parametrize("confidence,outliers,batches", [(0.99, 1, 1), (1.0, 1, 3), (0.99, 0, 1), (1.0, 0, 3)])
    def test_unit_confidence_runs_full_budget(self, device, dtype, confidence, outliers, batches):
        # 19 of 20 inliers: 0.99 confidence needs three minimal samples, i.e. one batch. With every
        # point an inlier the bound is one sample, but confidence=1 must still run the whole budget.
        points = torch.rand(20, 2, device=device, dtype=dtype)
        target = points.clone()
        target[:outliers] += 100
        matrix = torch.eye(3, device=device, dtype=dtype)[None]
        ransac = RANSAC("homography", batch_size=8, max_iter=3, max_lo_iters=0, confidence=confidence)
        calls = []
        ransac.remove_bad_samples = lambda a, b: (a, b)
        ransac.minimal_solver = lambda a, b, w: calls.append(1) or matrix
        ransac(points, target)
        assert len(calls) == batches

    def test_failure_mask_shape(self, device, dtype):
        points = torch.zeros(10, 2, device=device, dtype=dtype)
        ransac = RANSAC("fundamental", batch_size=2, max_iter=1)
        ransac.minimal_solver = lambda a, b, w: torch.zeros(1, 3, 3, device=device, dtype=dtype)
        _, mask = ransac(points, points)
        assert mask.shape == (10,)
        assert not mask.any()

    @pytest.mark.parametrize("score_type", ["ransac", "msac"])
    def test_under_supported_best_is_rejected(self, device, dtype, score_type):
        # Four agreeing points are no more than a homography's minimal sample, which any sample fits.
        points = torch.rand(10, 2, device=device, dtype=dtype)
        target = points + 100
        target[:4] = points[:4]
        identity = torch.eye(3, device=device, dtype=dtype)[None]
        ransac = RANSAC("homography", batch_size=1, max_iter=2, max_lo_iters=0, score_type=score_type)
        ransac.remove_bad_samples = lambda a, b: (a, b)
        ransac.minimal_solver = lambda a, b, w: identity
        model, mask = ransac(points, target)
        assert not model.any()
        assert not mask.any()

    def test_stopping_bound_follows_incumbent_support(self, device, dtype):
        # Under MSAC a better-scoring model can have less support than the one it replaces. The
        # stopping bound must then follow the new incumbent, not the larger support it displaced.
        points = torch.rand(100, 2, device=device, dtype=dtype)
        matrix = torch.eye(3, device=device, dtype=dtype)[None]
        ransac = RANSAC("fundamental", batch_size=1, max_iter=50, max_lo_iters=0, score_type="msac")
        sampled = []
        ransac.minimal_solver = lambda a, b, w: sampled.append(1) or matrix
        results = {1: (1.0, 95), 2: (2.0, 60)}

        def verify(a, b, models, threshold):
            score, count = results.get(len(sampled), (0.0, 50))
            return matrix[0], torch.arange(100, device=device) < count, score, float(count)

        ransac.verify = verify
        ransac(points, points)
        assert RANSAC.max_samples_by_conf(95, 100, 8, 0.99) < 50 < RANSAC.max_samples_by_conf(60, 100, 8, 0.99)
        assert len(sampled) == 50

    @pytest.mark.parametrize("lo_sample_size,expected_calls", [(None, 1), (8, 2)])
    def test_invalid_refit_is_not_repeated(self, device, dtype, lo_sample_size, expected_calls):
        # A failed full-inlier refit would fail identically on the unchanged inliers. A failed subset
        # batch still leaves the full refit to try.
        points = torch.rand(20, 2, device=device, dtype=dtype)
        identity = torch.eye(3, device=device, dtype=dtype)[None]
        ransac = RANSAC("homography", batch_size=1, max_iter=1, max_lo_iters=5, lo_sample_size=lo_sample_size, seed=0)
        ransac.remove_bad_samples = lambda a, b: (a, b)
        ransac.minimal_solver = lambda a, b, w: identity
        calls = []
        ransac.polisher_solver = lambda a, b, w: (
            calls.append(1) or torch.full_like(identity, float("nan")).expand(len(a), 3, 3)
        )
        model, mask = ransac(points, points)
        self.assert_close(model, identity[0])
        assert mask.all()
        assert len(calls) == expected_calls

    @pytest.mark.parametrize("lo_sample_size", [None, 8])
    def test_full_inlier_refit_wins_ties(self, device, dtype, lo_sample_size):
        # RANSAC scoring ties whenever the refit keeps the same support; the least-squares refit
        # must still replace the minimal-sample model, which is only as precise as its sample.
        points = torch.rand(20, 2, device=device, dtype=dtype)
        minimal = torch.eye(3, device=device, dtype=dtype)[None]
        refit = 2 * minimal
        ransac = RANSAC("homography", batch_size=1, max_iter=1, max_lo_iters=3, lo_sample_size=lo_sample_size, seed=0)
        ransac.remove_bad_samples = lambda a, b: (a, b)
        ransac.minimal_solver = lambda a, b, w: minimal
        polished = []
        ransac.polisher_solver = lambda a, b, w: polished.append(1) or refit.expand(len(a), 3, 3)
        mask = torch.ones(20, device=device, dtype=torch.bool)
        ransac.verify = lambda a, b, m, t: (m[0].clone(), mask, 20.0, 20.0)
        model, _ = ransac(points, points)
        self.assert_close(model, refit[0])
        # A tie ends local optimization: it cannot increase support any further.
        assert len(polished) == (2 if lo_sample_size else 1)


class TestRANSACSampling(BaseTester):
    def test_prosac_growth_per_draw(self, device):
        # Chum & Matas (CVPR 2005), eqs. 3--6; m=4, N=10, T_N=64.
        ransac = RANSAC(batch_size=16, max_iter=4, prosac_sampling=True, seed=7)
        samples = torch.cat([ransac.sample(4, 10, 16, i, device) for i in range(4)])
        ends = [1, 3, 7, 14, 25, 43, 69]
        expected = torch.tensor(
            [next(n + 3 for n, end in enumerate(ends) if t <= end) for t in range(1, 65)], device=device
        )
        assert torch.equal(samples.amax(1), expected)
        assert torch.all(samples.sort(1).values.diff(dim=1) > 0)
        # A different batch partition must have the same prefix schedule at equal budget.
        other = RANSAC(batch_size=32, max_iter=2, prosac_sampling=True, seed=7)
        larger = torch.cat([other.sample(4, 10, 32, i, device) for i in range(2)])
        assert torch.equal(larger.amax(1), expected)

    def test_sample_accepts_device_string(self, device):
        ransac = RANSAC(seed=0)
        by_name = ransac.sample(4, 10, 3, 0, str(device))
        assert by_name.device.type == device.type
        assert torch.equal(by_name, ransac.sample(4, 10, 3, 0, device))

    def test_prosac_eventually_uniform(self, device):
        ransac = RANSAC(batch_size=128, max_iter=1, prosac_sampling=True, seed=3)
        samples = ransac.sample(4, 10, 128, 10, device)
        assert torch.any(samples.amax(1) < 9)
        assert torch.all(samples.sort(1).values.diff(dim=1) > 0)

    @pytest.mark.parametrize("population", [4, 10, 1000])
    def test_uniform_without_replacement(self, device, population):
        ransac = RANSAC(seed=0)
        samples = ransac.sample(4, population, 10000, 0, device)
        assert samples.min() >= 0
        assert samples.max() < population
        assert torch.all(samples.sort(1).values.diff(dim=1) > 0)
        # Check marginal inclusion, independent of ordering within each sampled set.
        counts = torch.bincount(samples.flatten(), minlength=population).float()
        expected = 40000 / population
        assert (counts - expected).abs().max() < 7 * math.sqrt(expected)

    def test_prosac_does_not_use_uniform_confidence(self, device, dtype):
        points = torch.rand(20, 2, device=device, dtype=dtype)
        matrix = torch.eye(3, device=device, dtype=dtype)[None]
        ransac = RANSAC("homography", batch_size=8, max_iter=3, max_lo_iters=0, prosac_sampling=True)
        calls = []
        ransac.remove_bad_samples = lambda a, b: (a, b)
        ransac.minimal_solver = lambda a, b, w: calls.append(1) or matrix
        ransac(points, points)
        assert len(calls) == 3


class TestRANSACBoundedLO(BaseTester):
    def test_subset_refits_and_final_full_refit(self, device, dtype):
        points = torch.rand(100, 2, device=device, dtype=dtype)
        matrix = torch.eye(3, device=device, dtype=dtype)[None]
        ransac = RANSAC("fundamental", batch_size=1, max_iter=1, max_lo_iters=3, lo_sample_size=16, seed=0)
        ransac.minimal_solver = lambda a, b, w: matrix
        sizes = []
        ransac.polisher_solver = lambda a, b, w: sizes.append(a.shape[:2]) or matrix
        mask = torch.ones(100, device=device, dtype=torch.bool)
        ransac.verify = lambda a, b, m, t: (matrix[0], mask, 100.0, 100.0)
        ransac(points, points)
        assert sizes == [(3, 16), (1, 100)]

    def test_bad_subset_does_not_replace_best(self, device, dtype):
        points = torch.rand(40, 2, device=device, dtype=dtype)
        identity = torch.eye(3, device=device, dtype=dtype)[None]
        wrong = identity.clone()
        wrong[:, 0, 2] = 100
        ransac = RANSAC(
            "homography", batch_size=1, max_iter=1, max_lo_iters=2, lo_sample_size=8, score_type="msac", seed=3
        )
        ransac.minimal_solver = lambda a, b, w: identity
        ransac.remove_bad_samples = lambda a, b: (a, b)
        ransac.polisher_solver = lambda a, b, w: wrong
        model, mask = ransac(points, points)
        self.assert_close(model, identity[0])
        assert mask.all()

    def test_seeded_subsets_reproducible(self, device, dtype):
        points = torch.rand(50, 2, device=device, dtype=dtype)
        identity = torch.eye(3, device=device, dtype=dtype)[None]
        ransac = RANSAC("homography", batch_size=1, max_iter=1, max_lo_iters=2, lo_sample_size=8, seed=13)
        ransac.minimal_solver = lambda a, b, w: identity
        ransac.remove_bad_samples = lambda a, b: (a, b)
        subsets = []
        ransac.polisher_solver = lambda a, b, w: subsets.append(a.clone()) or identity
        ransac(points, points)
        ransac(points, points)
        self.assert_close(subsets[0], subsets[2])
        self.assert_close(subsets[1], subsets[3])
        assert not torch.equal(subsets[0][0], subsets[0][1])


class TestRANSACValidation(BaseTester):
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"inl_th": 0.0},
            {"inl_th": float("nan")},
            {"batch_size": 0},
            {"max_iter": 0},
            {"max_lo_iters": -1},
            {"score_type": "magsac"},
            {"confidence": 1.5},
            {"confidence": 0.0},
            {"lo_sample_size": 3},
        ],
    )
    def test_invalid_configuration(self, kwargs):
        with pytest.raises(ValueError):
            RANSAC(**kwargs)

    @pytest.mark.parametrize("model", ["fundamental_7pt", "essential"])
    def test_mismatched_correspondences(self, device, dtype, model):
        points = torch.zeros(9, 2, device=device, dtype=dtype)
        with pytest.raises(ValueError):
            RANSAC(model).validate_inputs(points, points[:8])

    @staticmethod
    def _projection_case(device, dtype, extra_exact_points):
        # Every correspondence fits this rank-two F exactly. Its projection onto the essential manifold
        # fits only the points with y = 0 and moves the others to a squared Sampson error of 0.5.
        points1 = torch.tensor(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [2.0, 0.0], [3.0, 1.0], [3.0, 0.0]]
            + [[4.0 + i, 0.0] for i in range(extra_exact_points)],
            device=device,
            dtype=dtype,
        )
        points2 = points1.clone()
        points2[:, 1] *= 2
        candidate = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 2.0, 0.0]], device=device, dtype=dtype)[None]
        ransac = RANSAC("essential", inl_th=0.01, batch_size=1, max_iter=1, max_lo_iters=0)
        ransac.minimal_solver = lambda a, b, w: candidate
        return ransac, points1, points2

    def test_essential_mask_matches_projected_output(self, device, dtype):
        ransac, points1, points2 = self._projection_case(device, dtype, extra_exact_points=2)
        model, mask = ransac(points1, points2)
        expected = sampson_epipolar_distance(points1[None], points2[None], model[None])[0] <= 0.01**2
        assert model.abs().amax() > 0
        assert torch.equal(mask, expected)
        assert torch.equal(mask, points1[:, 1] == 0)

    def test_essential_projection_below_minimal_support_fails(self, device, dtype):
        # Four of the eight points survive the projection, fewer than the five-point minimal sample.
        ransac, points1, points2 = self._projection_case(device, dtype, extra_exact_points=0)
        model, mask = ransac(points1, points2)
        assert not model.any()
        assert mask.shape == (8,)
        assert not mask.any()


class TestRANSACMSACSelection(BaseTester):
    def test_skip_candidates_with_insufficient_support(self, device, dtype):
        points = torch.zeros(10, 2, device=device, dtype=dtype)
        candidates = torch.eye(3, device=device, dtype=dtype).repeat(2, 1, 1)
        candidates[1, 0, 2] = 1
        errors = torch.full((2, 10), 100.0, device=device, dtype=dtype)
        # Candidate 0 fits only a minimal sample's worth of points, exactly; candidate 1 has one more inlier.
        errors[0, :4] = 0.0
        errors[1, :5] = 2.0
        ransac = RANSAC("homography", score_type="msac")
        ransac.error_fn = lambda a, b, m: errors
        model, _, _, count = ransac.verify(points, points, candidates, 4.0)
        self.assert_close(model, candidates[1])
        assert count == 5

    def test_msac_score_accumulates_in_float32(self, device):
        # 999 is not a bfloat16 value: summing the per-point scores in bfloat16 rounds it to 1000.
        points = torch.zeros(999, 2, device=device, dtype=torch.bfloat16)
        model = torch.eye(3, device=device, dtype=torch.bfloat16)[None]
        ransac = RANSAC("homography", score_type="msac")
        ransac.error_fn = lambda a, b, m: torch.zeros(1, 999, device=device, dtype=torch.bfloat16)
        _, _, score, count = ransac.verify(points, points, model, 4.0)
        assert score == 999.0
        assert count == 999

    @pytest.mark.parametrize("threshold", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_verify_threshold(self, device, dtype, threshold):
        points = torch.zeros(4, 2, device=device, dtype=dtype)
        with pytest.raises(ValueError):
            RANSAC(score_type="msac").verify(points, points, torch.eye(3, device=device, dtype=dtype)[None], threshold)


class TestRANSACLineScoring(BaseTester):
    def test_squared_distance_independent_of_segment_length(self, device, dtype):
        # Horizontal segments all displaced vertically by 3px: squared distance = 9,
        # regardless of whether the segment is 1px or 10px long.
        source = torch.tensor(
            [[[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [3.0, 0.0]], [[0.0, 0.0], [5.0, 0.0]], [[0.0, 0.0], [10.0, 0.0]]]],
            device=device,
            dtype=dtype,
        )
        target = source.clone()
        target[..., 1] += 3
        estimator = RANSAC("homography_from_linesegments", score_type="msac")
        errors = estimator.error_fn(source, target, torch.eye(3, device=device, dtype=dtype)[None])
        self.assert_close(errors, torch.full((1, 4), 9.0, device=device, dtype=dtype))

    def test_zero_length_target_is_outlier(self, device, dtype):
        source = torch.tensor([[[[0.0, 0.0], [1.0, 0.0]]]], device=device, dtype=dtype)
        target = torch.zeros_like(source)
        estimator = RANSAC("homography_from_linesegments")
        errors = estimator.error_fn(source, target, torch.eye(3, device=device, dtype=dtype)[None])
        assert torch.isinf(errors).all()


class TestRANSACEssentialInvalidPolisher(BaseTester):
    @pytest.mark.parametrize("lo_sample_size", [None, 8])
    def test_discard_nonfinite_polisher_before_projection(self, device, dtype, lo_sample_size):
        points = torch.rand(20, 2, device=device, dtype=dtype)
        valid = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)[None]
        candidates = torch.cat([torch.full_like(valid, float("nan")), valid])
        estimator = RANSAC("essential", batch_size=1, max_iter=1, max_lo_iters=2, lo_sample_size=lo_sample_size, seed=0)
        estimator.minimal_solver = lambda a, b, w: valid
        estimator.polisher_solver = lambda a, b, w: candidates
        model, mask = estimator(points, points)
        assert torch.isfinite(model).all()
        assert mask.all()
