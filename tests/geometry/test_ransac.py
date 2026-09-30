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
import kornia.geometry.ransac as ransac_module
from kornia.geometry import RANSAC, transform_points
from kornia.geometry._degensac import _h_degenerate_sample
from kornia.geometry.conversions import axis_angle_to_rotation_matrix, convert_points_from_homogeneous
from kornia.geometry.epipolar import find_fundamental, project_to_essential, sampson_epipolar_distance
from kornia.geometry.epipolar._metrics import _sampson_errors, _sampson_quadratic_basis
from kornia.geometry.epipolar.fundamental import _rank2_projection, _refine_fundamental_lm
from kornia.geometry.homography import _refine_homography_lm, _transfer_errors, oneway_transfer_error
from kornia.geometry.ransac import _normalize_correspondences

from testing.base import BaseTester
from testing.casts import dict_to
from testing.geometry.create import create_dominant_plane_scene

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
        ransac = RANSAC("fundamental_8pt", inl_th=1.0, max_iter=20, max_lo_iters=10).to(device=device, dtype=dtype)
        fundamental_matrix, mask = ransac(pts_src, pts_dst)
        # A zero matrix is the failure result and has zero Sampson error for every point.
        assert fundamental_matrix.abs().amax() > 0
        assert mask.all()
        gross_errors = (
            sampson_epipolar_distance(pts_src[None], pts_dst[None], fundamental_matrix[None], squared=False) > 2.0
        )
        assert gross_errors.sum().item() == 0

    @pytest.mark.slow
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

        ransac = RANSAC("fundamental_8pt", inl_th=1.0, max_iter=20, max_lo_iters=10).to(device=device, dtype=dtype)
        # compute transform from source to target
        fundamental_matrix, _ = ransac(kp1, kp2)
        assert fundamental_matrix.abs().amax() > 0
        gross_errors = (
            sampson_epipolar_distance(pts_src[None], pts_dst[None], fundamental_matrix[None], squared=False) > 10.0
        )
        assert gross_errors.sum().item() < 2

    @pytest.mark.slow
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

        ransac = RANSAC(
            "fundamental_8pt",
            inl_th=1.0,
            batch_size=4,
            max_iter=1,
            max_lo_iters=1,
            score_type="msac",
            local_optimization="dlt",
        )
        ransac.minimal_solver = lambda kp1, kp2, w: worse_fit.expand(kp1.shape[0], 3, 3).clone()
        ransac.polisher_solver = fake_polisher

        Fm, _ = ransac(points1, points2)
        assert len(calls) > 0, "the local-optimization step never ran"
        self.assert_close(Fm, good_fit[0])

    def test_line_segment_polish_improves_the_fit(self, device, dtype):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("line_segment_transfer_error_one_way overflows half precision on 600 px coordinates")
        # Segments about 100 px long, half of them outliers. The polisher's Gaussian weights are on the perpendicular
        # distance in pixels, not on the length-scaled residual of line_segment_transfer_error_one_way (#4867), so
        # local optimization refines the minimal-sample model instead of fitting a handful of segments.
        torch.manual_seed(0)
        H = torch.tensor([[1.1, 0.05, 20.0], [0.02, 0.95, -10.0], [1e-4, 2e-4, 1.0]], device=device, dtype=dtype)
        centers = torch.rand(300, 2, device=device, dtype=dtype) * 600
        half = torch.randn(300, 2, device=device, dtype=dtype)
        half = 50 * half / half.norm(dim=-1, keepdim=True)
        ls1 = torch.stack([centers - half, centers + half], 1)
        ls2_clean = transform_points(H[None], ls1.reshape(1, -1, 2)).reshape(300, 2, 2)
        ls2 = ls2_clean + 0.3 * torch.randn_like(ls2_clean)
        ls2[150:] = torch.rand(150, 2, 2, device=device, dtype=dtype) * 600

        def endpoint_error(model):
            mapped = transform_points(model[None], ls1[:150].reshape(1, -1, 2)).reshape(150, 2, 2)
            return float((mapped - ls2_clean[:150]).norm(dim=-1).mean())

        errors = {}
        for max_lo_iters in (0, 5):
            ransac = RANSAC(
                "homography_from_linesegments",
                inl_th=2.0,
                batch_size=1024,
                max_iter=4,
                max_lo_iters=max_lo_iters,
                seed=0,
                confidence=1.0,
            )
            errors[max_lo_iters] = endpoint_error(ransac(ls1, ls2)[0])
        assert errors[5] < errors[0]
        assert errors[5] < 1.0


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
        # Under RANSAC scoring every all-inlier sample ties and the first one drawn wins, so the
        # result follows the seed; the MSAC optimum over an exhaustive draw is the same for any seed.
        common = {"inl_th": 2.0, "max_iter": 5, "max_lo_iters": 0, "score_type": "ransac"}
        ransac_a = RANSAC("homography", seed=1, **common).to(device=device, dtype=dtype)
        ransac_b = RANSAC("homography", seed=2, **common).to(device=device, dtype=dtype)

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

    def test_local_optimization_has_its_own_seed_stream(self, device, dtype, monkeypatch):
        # More batches than max_iter: the subset-refit generator must not reuse a sampling generator's seed.
        seeds = []

        class Spy(torch.Generator):
            def manual_seed(self, seed):
                seeds.append(seed)
                return super().manual_seed(seed)

        monkeypatch.setattr(kornia.geometry.ransac.torch, "Generator", Spy)
        torch.manual_seed(0)
        kp1 = torch.rand(200, 2, device=device, dtype=dtype) * 100
        ransac = RANSAC(
            "homography",
            inl_th=1.0,
            batch_size=16,
            max_iter=1,
            max_samples=64,
            seed=0,
            lo_sample_size=8,
            confidence=1.0,
            local_optimization="dlt",
        )
        ransac(kp1, kp1.clone())
        assert len(seeds) > 4  # four sampling batches and at least one local optimization
        assert len(seeds) == len(set(seeds))


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
        ransac = RANSAC("essential", batch_size=32, max_iter=1, local_optimization="dlt").to(device=device, dtype=dtype)
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
                ransac = RANSAC(
                    "homography",
                    inl_th=inl_th,
                    batch_size=64,
                    max_iter=1,
                    max_lo_iters=0,
                    seed=0,
                    local_optimization="dlt",
                )
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
            ransac = RANSAC(
                "fundamental",
                inl_th=inl_th,
                batch_size=1,
                max_iter=1,
                max_lo_iters=0,
                seed=0,
                local_optimization="dlt",
            )
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
        [
            ("fundamental_7pt", 6, "fundamental_8pt", 7),
            ("fundamental", 6, "fundamental_8pt", 7),
            ("essential", 4, "homography", 3),
        ],
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
        ransac = RANSAC(
            "homography",
            inl_th=threshold,
            batch_size=1,
            max_iter=1,
            max_lo_iters=0,
            score_type="msac",
            local_optimization="dlt",
        )
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
        ransac = RANSAC(model_type, batch_size=1, max_iter=1, local_optimization="dlt")
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
        ransac = RANSAC(
            "fundamental_8pt", batch_size=16, max_iter=10, max_lo_iters=1, score_type="msac", local_optimization="dlt"
        )
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
        ransac = RANSAC(
            "homography", batch_size=8, max_iter=3, max_lo_iters=0, confidence=confidence, local_optimization="dlt"
        )
        calls = []
        ransac.remove_bad_samples = lambda a, b: (a, b)
        ransac.minimal_solver = lambda a, b, w: calls.append(1) or matrix
        ransac(points, target)
        assert len(calls) == batches

    def test_failure_mask_shape(self, device, dtype):
        points = torch.zeros(10, 2, device=device, dtype=dtype)
        ransac = RANSAC("fundamental", batch_size=2, max_iter=1, local_optimization="dlt")
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
        ransac = RANSAC(
            "homography", batch_size=1, max_iter=2, max_lo_iters=0, score_type=score_type, local_optimization="dlt"
        )
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
        ransac = RANSAC(
            "fundamental_8pt", batch_size=1, max_iter=50, max_lo_iters=0, score_type="msac", local_optimization="dlt"
        )
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
        ransac = RANSAC(
            "homography",
            batch_size=1,
            max_iter=1,
            max_lo_iters=5,
            lo_sample_size=lo_sample_size,
            seed=0,
            local_optimization="dlt",
        )
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
        ransac = RANSAC(
            "homography",
            batch_size=1,
            max_iter=1,
            max_lo_iters=3,
            lo_sample_size=lo_sample_size,
            seed=0,
            local_optimization="dlt",
        )
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

    @pytest.mark.parametrize("confidence,batches", [(0.99, 1), (1.0, 3)])
    def test_prosac_confidence(self, device, dtype, confidence, batches):
        points = torch.rand(20, 2, device=device, dtype=dtype)
        matrix = torch.eye(3, device=device, dtype=dtype)[None]
        ransac = RANSAC(
            "homography",
            batch_size=8,
            max_iter=3,
            max_lo_iters=0,
            prosac_sampling=True,
            confidence=confidence,
            local_optimization="dlt",
        )
        calls = []
        ransac.remove_bad_samples = lambda a, b: (a, b)
        ransac.minimal_solver = lambda a, b, w: calls.append(1) or matrix
        ransac(points, points)
        assert len(calls) == batches

    def test_prosac_stopping_uses_ranking(self, device):
        ransac = RANSAC("fundamental_8pt", batch_size=16, max_iter=256, prosac_sampling=True)
        # 150 inliers of 500: ranked first, one draw from the top-150 prefix certifies the model.
        # Ranked last, no prefix qualifies and the uniform bound (over 60,000 draws) caps at the budget.
        ranked = torch.arange(500, device=device) < 150
        assert ransac._prosac_max_samples(ranked, 150) == 1
        assert ransac._prosac_max_samples(ranked.flip(0), 150) == 4096

    @pytest.mark.parametrize("support", [0, 8, 9, 50, 99])
    def test_prosac_stopping_rejects_small_prefix_support(self, device, support):
        ransac = RANSAC("fundamental_8pt", batch_size=16, max_iter=256, prosac_sampling=True)
        # Fewer than 20% of all correspondences cannot terminate, however perfect the prefix
        # (OpenCV USAC guard): the top ten all-inlier matches of a locally fitted model are no evidence.
        inliers = torch.arange(500, device=device) < support
        assert ransac._prosac_max_samples(inliers, support) == 4096

    def test_prosac_stopping_without_replacement(self, device):
        ransac = RANSAC("homography", batch_size=16, max_iter=256, prosac_sampling=True)
        # The best prefix is the whole set. C(10,4)/C(20,4) requires 104 draws
        # for 99% confidence; the with-replacement approximation gives only 72.
        inliers = torch.arange(20, device=device) >= 10
        assert ransac._prosac_max_samples(inliers, 10) == 104
        assert ransac.max_samples_by_conf(10, 20, 4, 0.99) == 104
        ransac.confidence = 1
        assert ransac._prosac_max_samples(inliers, 10) == 4096

    def test_prosac_stopping_never_exceeds_uniform_bound(self, device):
        ransac = RANSAC("homography", batch_size=16, max_iter=256, prosac_sampling=True)
        generator = torch.Generator().manual_seed(0)
        for _ in range(20):
            inliers = (torch.rand(300, generator=generator) < 0.4).to(device)
            support = int(inliers.sum())
            uniform = min(4096, ransac.max_samples_by_conf(support, 300, 4, 0.99))
            assert 1 <= ransac._prosac_max_samples(inliers, support) <= uniform

    @pytest.mark.parametrize("replace_incumbent", [False, True])
    def test_prosac_stopping_after_multiple_batches(self, device, dtype, replace_incumbent):
        points = torch.rand(20, 2, device=device, dtype=dtype)
        matrix = torch.eye(3, device=device, dtype=dtype)[None]
        ransac = RANSAC(
            "homography",
            batch_size=16,
            max_iter=16,
            max_lo_iters=0,
            score_type="msac",
            prosac_sampling=True,
            local_optimization="dlt",
        )
        calls = []
        ransac.remove_bad_samples = lambda a, b: (a, b)
        ransac.minimal_solver = lambda a, b, w: calls.append(1) or matrix

        def verify(a, b, models, threshold):
            # The initial incumbent requires 104 draws (seven batches). A
            # better MSAC score with less support needs the entire budget.
            replace = replace_incumbent and len(calls) >= 2
            mask = torch.arange(20, device=device) >= (15 if replace else 10)
            return matrix[0], mask, 2.0 if replace else 1.0, float(mask.sum())

        ransac.verify = verify
        for _ in range(2):
            calls.clear()
            ransac(points, points)
            assert len(calls) == (16 if replace_incumbent else 7)

    def test_prosac_minimal_population(self, device):
        ransac = RANSAC("homography", batch_size=16, max_iter=256, prosac_sampling=True)
        assert ransac._prosac_max_samples(torch.ones(4, device=device, dtype=torch.bool), 4) == 4096

    def test_prosac_stopping_respects_growth(self, device):
        ransac = RANSAC("fundamental_8pt", batch_size=16, max_iter=256, prosac_sampling=True)
        # 45 inliers among the top 100 of 200 pass the support guards, but the 3970 draws that
        # prefix needs exceed the ~106 draws PROSAC takes from it: later draws sampled larger
        # prefixes and must not be credited to the top 100. The uniform bound caps at the budget.
        inliers = torch.zeros(200, device=device, dtype=torch.bool)
        inliers[:90:2] = True
        assert ransac._prosac_max_samples(inliers, 45) == 4096
        assert ransac._prosac_schedule(8, 200, inliers.device)[100 - 8].item() < 3970

    def test_prosac_stops_on_a_certified_prefix(self, device, dtype):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("a 1 px threshold on 500 px coordinates is below half-precision resolution")
        # 150 of 500 correspondences follow the homography and are ranked first: PROSAC stops after
        # its first batch, while uniform sampling needs far more than one batch of 16 at 30% inliers.
        torch.manual_seed(0)
        matrix = torch.tensor([[1.1, 0.05, 20.0], [0.02, 0.95, -10.0], [1e-4, -2e-4, 1.0]], device=device, dtype=dtype)
        points1 = torch.rand(500, 2, device=device, dtype=dtype) * 800
        points2 = transform_points(matrix[None], points1[None])[0]
        points2[150:] = torch.rand(350, 2, device=device, dtype=dtype) * 800
        calls = []
        for prosac in (True, False):
            ransac = RANSAC(
                "homography", inl_th=1.0, batch_size=16, max_iter=64, max_lo_iters=0, prosac_sampling=prosac, seed=0
            )
            sample = ransac.sample
            count = []
            ransac.sample = lambda *a, _s=sample, _c=count, **k: _c.append(1) or _s(*a, **k)
            _, mask = ransac(points1, points2)
            calls.append(len(count))
            assert mask[:150].all()
        assert calls[0] == 1
        assert calls[1] > 1


class TestRANSACDefaults(BaseTester):
    def test_defaults_are_msac_and_bounded_lo(self):
        ransac = RANSAC("fundamental")
        assert ransac.score_type == "msac"
        assert ransac.lo_sample_size == 32
        assert ransac.max_lo_iters == 5
        assert not ransac.prosac_sampling

    @pytest.mark.parametrize(
        "model_type, sample_size, local_optimization",
        [
            ("homography", 4, "lm"),
            ("fundamental", 7, "lm"),
            ("fundamental_7pt", 7, "lm"),
            ("fundamental_8pt", 8, "lm"),
            ("essential", 5, "lm"),
            ("homography_from_linesegments", 4, "dlt"),
        ],
    )
    def test_defaults_per_model(self, model_type, sample_size, local_optimization):
        # "fundamental" draws seven-point samples; Levenberg-Marquardt local optimization is the default wherever it
        # is implemented.
        ransac = RANSAC(model_type)
        assert ransac.minimal_sample_size == sample_size
        assert ransac.local_optimization == local_optimization
        assert ransac.refine_iters == 3

    def test_legacy_configuration_is_still_available(self):
        ransac = RANSAC("fundamental", score_type="ransac", lo_sample_size=None)
        assert ransac.score_type == "ransac"
        assert ransac.lo_sample_size is None


class TestRANSACAutoBatch(BaseTester):
    def test_default_is_auto_with_the_historical_budget(self):
        ransac = RANSAC("homography")
        assert ransac.batch_size == "auto"
        assert ransac.sample_budget == 2048 * 10

    @pytest.mark.parametrize("bad", [0, -1, 2.5, True, "large"])
    def test_rejects_bad_batch_size(self, bad):
        with pytest.raises(ValueError):
            RANSAC("homography", batch_size=bad)

    @pytest.mark.parametrize("bad", [0, -5, True, 1000.0])
    def test_rejects_bad_max_samples(self, bad):
        with pytest.raises(ValueError):
            RANSAC("homography", max_samples=bad)

    def test_explicit_batch_keeps_its_budget(self):
        ransac = RANSAC("fundamental", batch_size=512, max_iter=4)
        assert ransac.sample_budget == 2048
        assert ransac.resolve_batch_size(5000, torch.device("cpu")) == 512
        assert ransac.resolve_batch_size(5000, torch.device("cuda")) == 512

    def test_max_samples_overrides_the_budget(self):
        assert RANSAC("homography", batch_size=256, max_iter=100, max_samples=1000).sample_budget == 1000
        assert RANSAC("homography", max_samples=1000).sample_budget == 1000

    def test_budget_follows_later_attribute_changes(self):
        # The budget is read at call time, like confidence, so configuring the module after construction works.
        ransac = RANSAC("homography", batch_size=256)
        ransac.max_iter = 100
        assert ransac.sample_budget == 25600
        ransac.max_samples = 1000
        assert ransac.sample_budget == 1000

    def test_accelerator_batch_shrinks_for_many_correspondences(self):
        # The residual matrix is batch x N: past 2**27 entries (1 GiB in float32 at peak) the batch shrinks, down to
        # the 2048 of the fixed historical batch, so the default cannot run out of memory where it did not before.
        cuda = torch.device("cuda")
        ransac = RANSAC("homography")
        assert ransac.resolve_batch_size(16384, cuda) == 8192
        assert ransac.resolve_batch_size(50000, cuda) == 2684
        assert ransac.resolve_batch_size(1_000_000, cuda) == 2048
        assert RANSAC("fundamental").resolve_batch_size(1_000_000, cuda) == 2048

    def test_accelerators_take_large_batches(self):
        ransac = RANSAC("homography", max_samples=5000)
        for device in (torch.device("cuda"), torch.device("mps")):
            assert ransac.resolve_batch_size(500, device) == 5000
        assert RANSAC("homography").resolve_batch_size(500, torch.device("cuda")) == 8192
        assert RANSAC("homography_from_linesegments").resolve_batch_size(500, torch.device("cuda")) == 8192
        # The epipolar solvers are compute-bound past 2048 hypotheses on the GPU.
        assert RANSAC("fundamental").resolve_batch_size(500, torch.device("cuda")) == 2048
        assert RANSAC("essential", max_samples=1000).resolve_batch_size(500, torch.device("mps")) == 1000

    @pytest.mark.parametrize("backend", ["xla", "privateuseone"])
    def test_other_devices_keep_historical_batch(self, backend):
        device = torch.device(backend)
        assert RANSAC("homography").resolve_batch_size(500, device) == 2048
        assert RANSAC("fundamental").resolve_batch_size(500, device) == 2048
        assert RANSAC("homography", max_samples=1000).resolve_batch_size(500, device) == 1000

    @pytest.mark.parametrize(
        "model_type,num_tc,expected",
        [
            ("homography", 100, 2048),
            ("homography", 500, 1048),
            ("homography", 4000, 256),
            ("fundamental", 100, 512),
            ("fundamental", 500, 262),
            ("fundamental", 4000, 128),
            ("essential", 4000, 128),
        ],
    )
    def test_cpu_batch_follows_model_and_point_count(self, model_type, num_tc, expected):
        assert RANSAC(model_type).resolve_batch_size(num_tc, torch.device("cpu")) == expected
        assert RANSAC(model_type, max_samples=200).resolve_batch_size(num_tc, torch.device("cpu")) == min(expected, 200)

    def test_last_batch_is_truncated_to_the_budget(self, device, dtype):
        points = torch.rand(20, 2, device=device, dtype=dtype)
        matrix = torch.eye(3, device=device, dtype=dtype)[None]
        ransac = RANSAC(
            "homography", batch_size=16, max_samples=40, max_lo_iters=0, confidence=1.0, local_optimization="dlt"
        )
        sizes = []
        ransac.remove_bad_samples = lambda a, b: sizes.append(len(a)) or (a, b)
        ransac.minimal_solver = lambda a, b, w: matrix
        ransac(points, points)
        assert sizes == [16, 16, 8]

    def test_truncated_last_batch_continues_the_prosac_schedule(self, device, dtype):
        # The PROSAC schedule counts draws, not batches: the truncated last batch of a 1000-draw budget takes draws
        # 801 to 1000, from the largest prefixes, rather than restarting at an earlier point of the schedule.
        num_tc = 100
        points = torch.arange(num_tc, device=device, dtype=dtype)[:, None].expand(num_tc, 2)
        matrix = torch.eye(3, device=device, dtype=dtype)[None]
        ransac = RANSAC(
            "homography",
            batch_size=400,
            max_samples=1000,
            prosac_sampling=True,
            max_lo_iters=0,
            confidence=1.0,
            local_optimization="dlt",
        )
        batches = []
        ransac.remove_bad_samples = lambda a, b: (a, b)
        ransac.minimal_solver = lambda a, b, w: batches.append(a[..., 0].long()) or matrix.expand(len(a), 3, 3)
        ransac(points, points)
        assert [len(b) for b in batches] == [400, 400, 200]
        ends = ransac._prosac_schedule(4, num_tc, points.device)
        assert int(ends[-1]) >= 1000  # the schedule is still growing at the last draw
        # Every growth draw includes the newest correspondence of its prefix, so the largest index a batch draws is
        # the prefix its last draw belongs to.
        last_draws = torch.tensor([400, 800, 1000], device=points.device)
        expected = (torch.searchsorted(ends, last_draws) + 4).tolist()
        assert [int(b.max()) + 1 for b in batches] == expected

    def test_auto_batch_forward_matches_explicit_batch(self, device, dtype):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("a 1 px threshold on 500 px coordinates is below half-precision resolution")
        # The auto batch is a plain batch size: the same seed gives the same draws as that size given explicitly.
        torch.manual_seed(0)
        kp1 = torch.rand(200, 2, device=device, dtype=dtype) * 500
        matrix = torch.tensor([[1.0, 0.1, 5.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
        kp2 = transform_points(matrix[None], kp1[None])[0]
        kp2[:60] = torch.rand(60, 2, device=device, dtype=dtype) * 500
        auto = RANSAC("homography", inl_th=1.0, seed=0, local_optimization="dlt")
        batch = auto.resolve_batch_size(200, kp1.device)
        explicit = RANSAC("homography", inl_th=1.0, batch_size=batch, seed=0, local_optimization="dlt")
        H_auto, mask_auto = auto(kp1, kp2)
        H_explicit, mask_explicit = explicit(kp1, kp2)
        assert torch.equal(mask_auto, mask_explicit)
        self.assert_close(H_auto, H_explicit)
        assert mask_auto[60:].all()


class TestRANSACPolisherScale(BaseTester):
    @pytest.mark.parametrize("model_type", ["homography", "homography_from_linesegments"])
    @pytest.mark.parametrize("inl_th", [0.5, 2.0])
    def test_polisher_gaussian_scale_is_inlier_threshold(self, model_type, inl_th):
        # The IRLS polisher's Gaussian re-weighting uses the inlier threshold as its standard deviation.
        ransac = RANSAC(model_type, inl_th=inl_th)
        assert ransac.polisher_solver.keywords == {"soft_inl_th": inl_th}


class TestRANSACBoundedLO(BaseTester):
    def test_subset_refits_and_final_full_refit(self, device, dtype):
        points = torch.rand(100, 2, device=device, dtype=dtype)
        matrix = torch.eye(3, device=device, dtype=dtype)[None]
        ransac = RANSAC(
            "fundamental",
            batch_size=1,
            max_iter=1,
            max_lo_iters=3,
            lo_sample_size=16,
            seed=0,
            local_optimization="dlt",
        )
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
            "homography",
            batch_size=1,
            max_iter=1,
            max_lo_iters=2,
            lo_sample_size=8,
            score_type="msac",
            seed=3,
            local_optimization="dlt",
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
        ransac = RANSAC(
            "homography",
            batch_size=1,
            max_iter=1,
            max_lo_iters=2,
            lo_sample_size=8,
            seed=13,
            local_optimization="dlt",
        )
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
            {"local_optimization": "irls"},
            {"refine_iters": -1},
            {"refine_iters": True},
            {"refine_iters": 1.5},
        ],
    )
    def test_invalid_configuration(self, kwargs):
        with pytest.raises(ValueError):
            RANSAC(**kwargs)

    def test_lm_local_optimization_needs_a_supported_model(self):
        with pytest.raises(ValueError):
            RANSAC("homography_from_linesegments", local_optimization="lm")

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
        ransac = RANSAC("essential", inl_th=0.01, batch_size=1, max_iter=1, max_lo_iters=0, local_optimization="dlt")
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
        estimator = RANSAC(
            "essential",
            batch_size=1,
            max_iter=1,
            max_lo_iters=2,
            lo_sample_size=lo_sample_size,
            seed=0,
            local_optimization="dlt",
        )
        estimator.minimal_solver = lambda a, b, w: valid
        estimator.polisher_solver = lambda a, b, w: candidates
        model, mask = estimator(points, points)
        assert torch.isfinite(model).all()
        assert mask.all()


def _skew(v: torch.Tensor) -> torch.Tensor:
    return torch.tensor([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=v.dtype)


def _two_view_scene(n: int, outliers: int, noise: float, seed: int):
    """Pixel matches of a 640x480 two-view scene, the first ``outliers`` replaced by random points. float64 CPU.

    Returns the matches, their noise-free second-image positions, the ground-truth F and the inlier mask.
    """
    generator = torch.Generator().manual_seed(seed)
    f64 = torch.float64
    K = torch.tensor([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=f64)
    R = axis_angle_to_rotation_matrix(torch.tensor([[0.05, -0.1, 0.02]], dtype=f64))[0]
    t = torch.tensor([1.0, 0.1, 0.2], dtype=f64)
    X = torch.randn(n, 3, generator=generator, dtype=f64) * torch.tensor([2.0, 2.0, 1.0], dtype=f64)
    X = X + torch.tensor([0.0, 0.0, 8.0], dtype=f64)
    kp1 = convert_points_from_homogeneous(X @ K.T)
    clean = convert_points_from_homogeneous((X @ R.T + t) @ K.T)
    kp2 = clean + noise * torch.randn(n, 2, generator=generator, dtype=f64)
    kp2[:outliers] = torch.rand(outliers, 2, generator=generator, dtype=f64) * torch.tensor([640.0, 480.0], dtype=f64)
    F = torch.linalg.inv(K).T @ _skew(t) @ R @ torch.linalg.inv(K)
    inliers = torch.arange(n) >= outliers
    return kp1, kp2, clean, F / F.norm(), inliers


def _planar_scene(n: int, outliers: int, noise: float, seed: int):
    """Matches ``kp2 = H(kp1)`` over 600x600 px, the first ``outliers`` replaced by random points. float64 CPU."""
    generator = torch.Generator().manual_seed(seed)
    f64 = torch.float64
    H = torch.tensor([[1.1, 0.05, 20.0], [0.02, 0.95, -10.0], [1e-4, 2e-4, 1.0]], dtype=f64)
    kp1 = torch.rand(n, 2, generator=generator, dtype=f64) * 600
    clean = transform_points(H[None], kp1[None])[0]
    kp2 = clean + noise * torch.randn(n, 2, generator=generator, dtype=f64)
    kp2[:outliers] = torch.rand(outliers, 2, generator=generator, dtype=f64) * 600
    return kp1, kp2, clean, H, torch.arange(n) >= outliers


def _model_errors(model_type: str, model: torch.Tensor, kp1: torch.Tensor, kp2: torch.Tensor) -> torch.Tensor:
    """Unsquared pixel errors in float64: Sampson distances, or one-way transfer errors for homographies."""
    model, kp1, kp2 = model.cpu().double(), kp1.cpu().double(), kp2.cpu().double()
    if model_type == "homography":
        return oneway_transfer_error(kp1[None], kp2[None], model[None], squared=False, eps=0.0)[0]
    return sampson_epipolar_distance(kp1[None], kp2[None], model[None], squared=False, eps=0.0)[0]


_FOCAL, _CENTER = 800.0, (320.0, 240.0)


def _px(model_type: str, pixels: float) -> float:
    """A distance given in pixels, in the input units of ``model_type``: calibrated units for ``"essential"``."""
    return pixels / _FOCAL if model_type == "essential" else pixels


def _to_input_units(model_type: str, kp: torch.Tensor) -> torch.Tensor:
    """Pixel coordinates of :func:`_two_view_scene`'s camera, as normalized camera coordinates for ``"essential"``."""
    if model_type != "essential":
        return kp
    return (kp - torch.tensor(_CENTER, dtype=kp.dtype, device=kp.device)) / _FOCAL


def _scene(model_type: str, n: int, outliers: int, noise: float, seed: int):
    """``kp1, kp2, clean, model, inliers``; for ``"essential"`` the two-view scene in normalized camera coordinates."""
    if model_type == "homography":
        return _planar_scene(n, outliers, noise, seed)
    kp1, kp2, clean, F, inliers = _two_view_scene(n, outliers, noise, seed)
    if model_type != "essential":
        return kp1, kp2, clean, F, inliers
    K = torch.tensor([[_FOCAL, 0.0, _CENTER[0]], [0.0, _FOCAL, _CENTER[1]], [0.0, 0.0, 1.0]], dtype=F.dtype)
    E = K.T @ F @ K
    return (*(_to_input_units(model_type, kp) for kp in (kp1, kp2, clean)), E / E.norm(), inliers)


_LM_MODELS = ["homography", "fundamental", "fundamental_8pt", "essential"]


class TestRANSACLevenbergMarquardt(BaseTester):
    """``local_optimization="lm"``, the default for homographies, fundamental and essential matrices.

    Essential-matrix cases use the two-view scene in normalized camera coordinates (focal length 800 px), with every
    pixel threshold and tolerance converted by :func:`_px`.
    """

    @pytest.mark.parametrize("model_type", _LM_MODELS)
    def test_recovers_exact_model_among_outliers(self, device, dtype, model_type):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("sub-pixel checks on 600 px coordinates are below half-precision resolution")
        kp1, kp2, _, _, inliers = _scene(model_type, 200, 60, 0.0, seed=0)
        ransac = RANSAC(model_type, inl_th=_px(model_type, 1.0), seed=0)
        model, mask = ransac(kp1.to(device, dtype), kp2.to(device, dtype))
        assert model.shape == (3, 3) and model.dtype == dtype and model.device == kp1.to(device, dtype).device
        assert mask.shape == (200,) and mask.dtype == torch.bool and mask.device == model.device
        assert torch.equal(mask.cpu(), inliers)
        tolerance = _px(model_type, 1e-4 if dtype == torch.float64 else 5e-2)
        assert _model_errors(model_type, model, kp1[inliers], kp2[inliers]).max() < tolerance

    @pytest.mark.parametrize("model_type", _LM_MODELS)
    def test_mask_holds_the_inliers_of_the_returned_model(self, device, dtype, model_type):
        kp1, kp2, _, _, _ = _scene(model_type, 300, 120, 0.7, seed=1)
        kp1, kp2 = kp1.to(device, dtype), kp2.to(device, dtype)
        model, mask = RANSAC(model_type, inl_th=_px(model_type, 1.5), seed=0)(kp1, kp2)
        errors = _model_errors(model_type, model, kp1, kp2)
        # Classify the returned model on the actual input coordinates, including rounding in half precision.
        assert mask.any()
        assert torch.equal(mask.cpu(), errors <= _px(model_type, 1.5))

    def test_msac_does_not_discard_supported_candidates(self, device, dtype, monkeypatch):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("this tight-threshold regression uses float32/float64 coordinates")
        generator = torch.Generator().manual_seed(31)
        kp1 = (torch.rand(40, 2, generator=generator) * 100).to(device, dtype)
        kp2 = (torch.rand(40, 2, generator=generator) * 100).to(device, dtype)
        # Fix the sampled sets across backends: CPU and CUDA generators give different draws for the same seed.
        ransac = RANSAC("fundamental_8pt", seed=0, max_samples=256, confidence=1, inl_th=0.3)
        samples = ransac.sample(8, 40, 256, 0, device="cpu").to(device)
        monkeypatch.setattr(ransac, "sample", lambda *args, **kwargs: samples)
        # The eight highest MSAC scores belong to models supported by at most their eight-point sample.
        # A lower-scoring candidate has additional support and must remain eligible for local optimization.
        model, mask = ransac(kp1, kp2)
        assert mask.sum() > 8
        assert torch.isfinite(model).all()

    def test_failure_when_model_cast_loses_support(self, device, dtype):
        if dtype != torch.bfloat16:
            pytest.skip("the model loses consensus when rounded to bfloat16")
        kp1, kp2, _, _, _ = _scene("homography", 12, 0, 0.0, seed=1)
        model, mask = RANSAC("homography", seed=0, max_samples=256, inl_th=0.5)(
            kp1.to(device, dtype), kp2.to(device, dtype)
        )
        assert not mask.any()
        assert torch.equal(model, torch.zeros_like(model))

    @pytest.mark.parametrize("model_type", _LM_MODELS)
    def test_refinement_improves_on_the_minimal_model(self, device, dtype, model_type):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("sub-pixel accuracy on 600 px coordinates is below half-precision resolution")
        kp1, kp2, clean, _, inliers = _scene(model_type, 300, 90, 0.5, seed=2)
        args = kp1.to(device, dtype), kp2.to(device, dtype)
        th = _px(model_type, 1.5)
        minimal, _ = RANSAC(model_type, inl_th=th, seed=0, max_lo_iters=0, refine_iters=0)(*args)
        refined, _ = RANSAC(model_type, inl_th=th, seed=0)(*args)
        # The distance of the noise-free matches from each model: the best of the minimal models misses them by 0.2
        # to 0.4 px at 0.5 px of noise, the Levenberg-Marquardt refits on two hundred inliers by about 0.1 px.
        miss_minimal = _model_errors(model_type, minimal, kp1[inliers], clean[inliers]).mean()
        miss_refined = _model_errors(model_type, refined, kp1[inliers], clean[inliers]).mean()
        assert miss_refined < 0.75 * miss_minimal
        assert miss_refined < _px(model_type, 0.15)

    @pytest.mark.parametrize("model_type", _LM_MODELS)
    def test_no_consensus_returns_the_failure_result(self, device, dtype, model_type):
        generator = torch.Generator().manual_seed(0)
        kp1 = _to_input_units(model_type, torch.rand(40, 2, generator=generator) * 600).to(device, dtype)
        kp2 = _to_input_units(model_type, torch.rand(40, 2, generator=generator) * 600).to(device, dtype)
        # A sample's model fits its own points; at 1e-4 px no other random point is expected on an epipolar line.
        model, mask = RANSAC(model_type, inl_th=_px(model_type, 1e-4), seed=0, max_samples=512)(kp1, kp2)
        assert bool((model == 0).all()) and model.dtype == dtype
        assert mask.shape == (40,) and mask.dtype == torch.bool and not bool(mask.any())

    @pytest.mark.parametrize("model_type", _LM_MODELS)
    def test_nonfinite_correspondences_are_outliers(self, device, dtype, model_type):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("sub-pixel checks on 600 px coordinates are below half-precision resolution")
        kp1, kp2, _, _, _ = _scene(model_type, 150, 0, 0.0, seed=3)
        kp1[:3] = float("nan")
        kp2[3:6, 0] = float("inf")
        ransac = RANSAC(model_type, inl_th=_px(model_type, 1.0), seed=0)
        model, mask = ransac(kp1.to(device, dtype), kp2.to(device, dtype))
        assert torch.isfinite(model).all()
        assert not bool(mask[:6].any()) and bool(mask[6:].all())

    @pytest.mark.parametrize("model_type", _LM_MODELS)
    def test_seeded_call_is_reproducible_and_private(self, device, dtype, model_type):
        kp1, kp2, _, _, _ = _scene(model_type, 120, 50, 0.5, seed=4)
        kp1, kp2 = kp1.to(device, dtype), kp2.to(device, dtype)
        state = torch.get_rng_state()
        th = _px(model_type, 2.0)
        model_a, mask_a = RANSAC(model_type, inl_th=th, seed=11, max_samples=1024)(kp1, kp2)
        assert torch.equal(torch.get_rng_state(), state)
        torch.manual_seed(123)
        model_b, mask_b = RANSAC(model_type, inl_th=th, seed=11, max_samples=1024)(kp1, kp2)
        assert torch.equal(model_a, model_b) and torch.equal(mask_a, mask_b)

    @pytest.mark.parametrize("score_type", ["msac", "ransac"])
    def test_score_types(self, device, dtype, score_type):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("sub-pixel checks on 600 px coordinates are below half-precision resolution")
        kp1, kp2, _, _, inliers = _planar_scene(200, 80, 0.0, seed=5)
        _, mask = RANSAC("homography", inl_th=1.0, score_type=score_type, seed=0)(
            kp1.to(device, dtype), kp2.to(device, dtype)
        )
        assert torch.equal(mask.cpu(), inliers)

    def test_half_precision_inputs(self, device, dtype):
        if dtype not in (torch.float16, torch.bfloat16):
            pytest.skip("half-precision inputs only")
        # A 10 px extent keeps bfloat16 rounding of the exact matches below inl_th; the pipeline computes in float32.
        kp1, kp2, _, _, inliers = _planar_scene(64, 16, 0.0, seed=6)
        kp1, kp2 = kp1 / 60, kp2 / 60
        kp2[:16] = torch.rand(16, 2, generator=torch.Generator().manual_seed(0), dtype=torch.float64) * 10 + 20
        model, mask = RANSAC("homography", inl_th=0.5, seed=0)(kp1.to(device, dtype), kp2.to(device, dtype))
        assert model.dtype == dtype and torch.isfinite(model).all()
        assert torch.equal(mask.cpu(), inliers)

    @pytest.mark.parametrize(
        "model_type, max_samples, expected",
        [
            ("homography", 2000, [512, 1024, 464]),
            ("fundamental", 2000, [256, 512, 1024, 208]),
            ("essential", 2000, [64, 128, 256, 512, 1024, 16]),
        ],
    )
    def test_auto_batches_grow_on_cpu_and_cover_the_budget(self, device, model_type, max_samples, expected):
        if device.type != "cpu":
            pytest.skip("the CPU batch schedule")
        # confidence=1 draws the whole budget: batches double from the first one, the last is cut to the budget.
        kp1, kp2, _, _, _ = _scene(model_type, 100, 30, 0.5, seed=7)
        ransac = RANSAC(model_type, max_samples=max_samples, confidence=1.0, seed=0)
        sizes = []
        sample = ransac.sample
        ransac.sample = lambda m, n, batch, *a, _s=sample, **k: sizes.append(batch) or _s(m, n, batch, *a, **k)
        ransac(kp1.float(), kp2.float())
        assert sizes == expected

    def test_essential_auto_batches_per_device(self):
        # A five-point sample needs few draws at high inlier ratios: small first batches on every device, the CPU
        # one smallest (tuned on PhotoTourism, where a first batch of 256 doubled the time at equal accuracy).
        ransac = RANSAC("essential", max_samples=100000)
        assert ransac._lm_batch_range(100, torch.device("cpu")) == (64, 1024)
        assert ransac._lm_batch_range(100, torch.device("cuda")) == (256, 8192)
        assert ransac._lm_batch_range(100, torch.device("mps")) == (256, 8192)

    def test_explicit_batch_size_is_kept(self, device):
        kp1, kp2, _, _, _ = _planar_scene(100, 30, 0.5, seed=8)
        ransac = RANSAC("homography", batch_size=300, max_samples=1000, confidence=1.0, seed=0)
        sizes = []
        sample = ransac.sample
        ransac.sample = lambda m, n, batch, *a, _s=sample, **k: sizes.append(batch) or _s(m, n, batch, *a, **k)
        ransac(kp1.to(device, torch.float32), kp2.to(device, torch.float32))
        assert sizes == [300, 300, 300, 100]

    def test_prosac_offset_continues_the_schedule(self, device):
        # Batches of different sizes pass the number of sets drawn so far: each growth draw includes the newest
        # correspondence of its prefix, so its largest index follows the schedule from that offset.
        ransac = RANSAC("homography", max_samples=1000, prosac_sampling=True, seed=0)
        ends = ransac._prosac_schedule(4, 100, device)
        samples = ransac.sample(4, 100, 50, 3, device, offset=120)
        draws = torch.arange(121, 171, device=device)
        assert samples.amax(1).tolist() == (torch.searchsorted(ends, draws) + 3).tolist()
        # Without an offset, the batch index times the batch size is assumed, as before.
        assert torch.equal(ransac.sample(4, 100, 40, 3, device), ransac.sample(4, 100, 40, 3, device, offset=120))

    def test_prosac_finds_the_ranked_consensus(self, device, dtype):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("sub-pixel checks on 600 px coordinates are below half-precision resolution")
        # 60 of 300 matches follow the homography and are ranked first.
        kp1, kp2, _, _, inliers = _planar_scene(300, 240, 0.0, seed=9)
        kp1, kp2, inliers = kp1.flip(0), kp2.flip(0), inliers.flip(0)
        _, mask = RANSAC("homography", inl_th=1.0, prosac_sampling=True, seed=0)(
            kp1.to(device, dtype), kp2.to(device, dtype)
        )
        assert torch.equal(mask.cpu(), inliers)


class TestRANSACEssentialLevenbergMarquardt(BaseTester):
    """What is specific to essential matrices with ``local_optimization="lm"``: no normalization, unit-norm output."""

    def _skip_half(self, dtype):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("calibrated thresholds of 1e-3 are below half-precision resolution")

    @pytest.mark.parametrize("seed", [0, 2, 3])
    def test_returns_a_unit_frobenius_essential_matrix(self, device, dtype, seed):
        self._skip_half(dtype)
        kp1, kp2, _, _, _ = _scene("essential", 200, 60, 0.5, seed=seed)
        E, _ = RANSAC("essential", inl_th=_px("essential", 1.5), seed=0)(kp1.to(device, dtype), kp2.to(device, dtype))
        singular_values = torch.linalg.svdvals(E.cpu().double())
        expected = torch.tensor([2**-0.5, 2**-0.5, 0.0], dtype=torch.float64)
        self.assert_close(singular_values, expected, atol=1e-6 if dtype == torch.float64 else 1e-3, rtol=0)
        # E and -E are the same model; the sign is fixed so that equal inputs give equal outputs on every backend.
        # Without the rule, the scene of seed 2 returns a negative largest entry on the CPU in either dtype.
        assert E.flatten()[E.abs().argmax()] > 0

    @pytest.mark.parametrize("n", [6, 7, 8])
    def test_small_inputs(self, device, dtype, n):
        self._skip_half(dtype)
        kp1, kp2, _, truth, _ = _scene("essential", n, 0, 0.0, seed=3)
        E, mask = RANSAC("essential", inl_th=_px("essential", 1.0), seed=0)(
            kp1.to(device, dtype), kp2.to(device, dtype)
        )
        assert E.shape == (3, 3) and mask.shape == (n,)
        # Noise-free: every correspondence is an inlier and E is the true one, up to sign.
        assert bool(mask.all())
        E = E.cpu().double()
        tolerance = 1e-9 if dtype == torch.float64 else 1e-4
        assert min(float((E - truth).norm()), float((E + truth).norm())) < tolerance

    def test_pixel_coordinates_by_mistake_stay_finite(self, device, dtype):
        self._skip_half(dtype)
        # Pixel coordinates are not a valid input for "essential", but must not produce NaN or raise.
        kp1, kp2, _, _, _ = _two_view_scene(100, 30, 0.0, seed=4)
        E, mask = RANSAC("essential", inl_th=1.0, seed=0)(kp1.to(device, dtype), kp2.to(device, dtype))
        assert torch.isfinite(E).all() and mask.shape == (100,)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_planar_scene(self, device, dtype, seed):
        self._skip_half(dtype)
        # A calibrated plane admits multiple essential matrices with the same epipolar fit. Check the recovered
        # consensus and essential manifold, not equality to one generating motion (which depends on sampled ties).
        generator = torch.Generator().manual_seed(12)
        f64 = torch.float64
        XY = (torch.rand(120, 2, generator=generator, dtype=f64) - 0.5) * 4
        X = torch.cat([XY, (4.0 + 0.3 * XY[:, :1] - 0.2 * XY[:, 1:])], 1)
        R = axis_angle_to_rotation_matrix(torch.tensor([[0.05, -0.1, 0.02]], dtype=f64))[0]
        t = torch.tensor([1.0, 0.1, 0.2], dtype=f64)
        Y = X @ R.T + t
        kp1, kp2 = X[:, :2] / X[:, 2:], Y[:, :2] / Y[:, 2:]
        kp2[:30] = (torch.rand(30, 2, generator=generator, dtype=f64) - 0.5) * 1.2
        E, mask = RANSAC("essential", inl_th=1e-4, seed=seed)(kp1.to(device, dtype), kp2.to(device, dtype))
        assert torch.equal(mask.cpu(), torch.arange(120) >= 30)
        E = E.cpu().double()
        tolerance = 1e-9 if dtype == torch.float64 else 1e-6
        self.assert_close(
            torch.linalg.svdvals(E), E.new_tensor([0.5**0.5, 0.5**0.5, 0.0]), atol=tolerance, rtol=tolerance
        )
        error = sampson_epipolar_distance(kp1[None, 30:], kp2[None, 30:], E[None], eps=0.0)
        assert error.max() < (1e-16 if dtype == torch.float64 else 1e-10)

    def test_batch_of_one_sample(self, device, dtype):
        self._skip_half(dtype)
        kp1, kp2, _, _, inliers = _scene("essential", 60, 10, 0.0, seed=5)
        ransac = RANSAC("essential", inl_th=_px("essential", 1.0), seed=0, batch_size=1, max_samples=64)
        _, mask = ransac(kp1.to(device, dtype), kp2.to(device, dtype))
        assert mask.shape == (60,)
        assert torch.equal(mask.cpu(), inliers)


class TestRANSACLevenbergMarquardtKernels(BaseTester):
    """RANSAC's normalization and the Levenberg-Marquardt refiners it calls."""

    def _skip_half(self, dtype):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("the kernels run in float32 or float64")

    def test_normalize_correspondences(self, device, dtype):
        self._skip_half(dtype)
        kp1, kp2, _, _, _ = _two_view_scene(50, 0, 0.0, seed=14)
        kp1[3] = float("nan")
        kp2[7, 0] = float("inf")
        x1, x2, t1, t2, s1, s2 = _normalize_correspondences(kp1, kp2, True)
        finite = torch.ones(50, dtype=torch.bool)
        finite[[3, 7]] = False
        # Correspondences that are not finite in both images stay non-finite and do not enter the statistics.
        assert not torch.isfinite(x1[3]).all()
        assert not torch.isfinite(x2[7]).all()
        c1, c2 = kp1[finite].mean(0), kp2[finite].mean(0)
        radius = ((kp1[finite] - c1).norm(dim=1).mean() + (kp2[finite] - c2).norm(dim=1).mean()) / 2
        # One scale for both images: sqrt(2) over the mean of the two mean radii (normalize_points adds eps = 1e-8).
        expected_scale = float((radius + 1e-8) / math.sqrt(2.0))
        self.assert_close(torch.tensor([s1, s2]), torch.tensor([expected_scale, expected_scale]), rtol=1e-12, atol=0.0)
        self.assert_close(x1[finite, :2], (kp1[finite] - c1) / s1, rtol=1e-9, atol=1e-12)
        ones = torch.ones(48, 1, dtype=kp1.dtype)
        self.assert_close((t1 @ torch.cat([kp1[finite], ones], 1).T).T, x1[finite])
        self.assert_close((t2 @ torch.cat([kp2[finite], ones], 1).T).T, x2[finite])

    @pytest.mark.parametrize("model_type", ["homography", "fundamental"])
    @pytest.mark.parametrize("loss", ["cauchy", "truncated"])
    def test_refinement_reaches_a_local_minimum(self, device, dtype, model_type, loss):
        self._skip_half(dtype)
        if device.type == "mps":
            pytest.skip("the refinement kernels run in float64, which MPS does not support")
        # Noisy inliers only, refined from a perturbed ground truth: the refit's cost is below the start's and does
        # not drop along small steps in any direction of the model's manifold.
        kp1, kp2, _, truth, _ = _scene(model_type, 80, 0, 1.0, seed=13)
        x1, x2, t1, t2, s1, s2 = _normalize_correspondences(kp1, kp2, model_type != "homography")
        if model_type == "homography":
            start = t2 @ truth @ torch.linalg.inv(t1)
            refine, scale2, x2_arg = _refine_homography_lm, (2.0 / s2) ** 2, x2[:, :2]

            def errors(models):
                return _transfer_errors(models, x1, x2[:, :2], 0.0)

        else:
            start = torch.linalg.inv(t2).mT @ truth @ torch.linalg.inv(t1)
            refine, scale2, x2_arg = _refine_fundamental_lm, (2.0 / s1) ** 2, x2

            def errors(models):
                return _sampson_errors(models, x1, x2, 0.0)

        start = start / start.norm()
        start = start + 1e-2 * torch.randn(3, 3, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
        if model_type != "homography":
            start = _rank2_projection(start[None])[0]

        def cost(models):
            r2 = errors(models)
            return (torch.log1p(r2 / scale2) if loss == "cauchy" else torch.fmin(r2, torch.tensor(scale2))).sum(1)

        refined = refine(start[None].to(device), x1.to(device), x2_arg.to(device), None, loss, scale2, 20).cpu()
        assert cost(refined) < cost(start[None])
        steps = 1e-4 * torch.randn(32, 3, 3, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
        neighbours = refined + steps * refined.norm()
        if model_type != "homography":
            neighbours = _rank2_projection(neighbours)
        assert (cost(neighbours) >= cost(refined) * (1 - 1e-9)).all()


class TestRANSACScoring(BaseTester):
    @pytest.mark.parametrize("model_type", ["homography", "fundamental", "essential"])
    @pytest.mark.parametrize("score_type", ["ransac", "msac"])
    @pytest.mark.parametrize("prosac", [False, True])
    def test_score_and_support_match_full_residuals(self, device, dtype, model_type, score_type, prosac):
        from kornia.geometry.epipolar._metrics import _sampson_from_quadratic_basis, _sampson_quadratic_basis
        from kornia.geometry.homography import _transfer_basis, _transfer_from_basis

        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("RANSAC scoring uses at least float32")
        generator = torch.Generator().manual_seed(10)
        x1 = torch.rand(137, 3, generator=generator).to(device, dtype)
        x2 = torch.rand(137, 3, generator=generator).to(device, dtype)
        x1[:, 2] = x2[:, 2] = 1
        x1[0] = float("nan")
        models = torch.rand(29, 3, 3, generator=generator).to(device, dtype)
        models[0] = float("nan")
        planar = model_type == "homography"
        basis = _transfer_basis(x1, x2[:, :2]) if planar else _sampson_quadratic_basis(x1, x2)
        residual_fn = _transfer_from_basis if planar else _sampson_from_quadratic_basis
        errors = residual_fn(models, basis)
        threshold = 0.1
        ransac = RANSAC(model_type, score_type=score_type, prosac_sampling=prosac)
        expected_support = (errors <= threshold).sum(1)
        expected_scores = ransac._lm_score(errors, threshold)
        # Force several tiles, including a tail of one model, without allocating huge fixtures.
        scores, support, masks = ransac._lm_score_models(models, basis, threshold, max_residuals=137 * 7)
        self.assert_close(scores, expected_scores)
        self.assert_close(support, expected_support.to(support.dtype), rtol=0, atol=0)
        assert scores[0] == 0 and support[0] == 0
        if prosac:
            assert masks is not None
            self.assert_close(support, masks.sum(1).to(support.dtype), rtol=0, atol=0)
        else:
            assert masks is None

    def test_msac_does_not_mutate_residuals(self, device, dtype):
        errors = torch.tensor([[0.0, 0.25, 1.0, 2.0, float("nan"), float("inf")]], device=device, dtype=dtype)
        original = errors.clone()
        score = RANSAC()._lm_score(errors, 1.0)
        self.assert_close(score, errors.new_tensor([1.75]))
        assert torch.equal(errors.isnan(), original.isnan())
        self.assert_close(errors.nan_to_num(), original.nan_to_num(), rtol=0, atol=0)


class TestRANSACDegensacOptions(BaseTester):
    @pytest.mark.parametrize(
        "model_type", ["fundamental", "fundamental_7pt", "fundamental_8pt", "homography", "essential"]
    )
    def test_default_is_off(self, model_type):
        assert RANSAC(model_type).degensac is False

    @pytest.mark.parametrize("model_type", ["fundamental", "fundamental_7pt"])
    def test_explicit_opt_in(self, model_type):
        assert RANSAC(model_type, degensac=True).degensac is True

    def test_default_is_off_with_dlt(self):
        assert RANSAC("fundamental", local_optimization="dlt").degensac is False

    @pytest.mark.parametrize(
        ("model_type", "local_optimization"),
        [
            ("homography", None),
            ("essential", None),
            ("fundamental_8pt", None),
            ("fundamental", "dlt"),
            ("homography_from_linesegments", None),
        ],
    )
    def test_true_rejects_unsupported(self, model_type, local_optimization):
        with pytest.raises(ValueError, match="degensac"):
            RANSAC(model_type, local_optimization=local_optimization, degensac=True)

    def test_rejects_non_bool(self):
        with pytest.raises(ValueError, match="degensac"):
            RANSAC("fundamental", degensac=1)

    @pytest.mark.parametrize("model_type", ["fundamental", "fundamental_8pt", "homography", "essential"])
    def test_minimal_models_omit_unused_origins(self, model_type, device, dtype):
        ransac = RANSAC(model_type)
        work = torch.float64 if dtype == torch.float64 else torch.float32
        x1 = torch.cat([torch.rand(8, ransac.minimal_sample_size, 2), torch.ones(8, ransac.minimal_sample_size, 1)], -1)
        x2 = torch.cat([torch.rand_like(x1[..., :2]), torch.ones_like(x1[..., :1])], -1)
        x1, x2 = x1.to(device, work), x2.to(device, work)
        models, origin = ransac._lm_minimal_models(x1, x2)
        assert origin is None
        tracked_models, tracked_origin = ransac._lm_minimal_models(x1, x2, track_origins=True)
        assert tracked_origin is not None and len(tracked_origin) == len(models)
        assert torch.equal(models.isnan(), tracked_models.isnan())
        self.assert_close(models.nan_to_num(), tracked_models.nan_to_num(), atol=0, rtol=0)

    def test_raw_record_setters_are_draw_order_prefix_maxima(self):
        scores = torch.tensor([0.5, 2.0, 1.0, 3.0, 3.0, -1.0, 4.0])
        assert RANSAC._raw_record_setters(scores, 1.0) == [1, 3, 6]
        assert RANSAC._raw_record_setters(scores, 5.0) == []
        assert RANSAC._raw_record_setters(scores, -1.0) == [0, 1, 3, 6]

    @pytest.mark.parametrize("model_type", ["fundamental", "fundamental_8pt", "homography", "essential"])
    def test_lm_minimal_models_report_their_sample(self, model_type, device, dtype):
        # Every model fits its own sample far better than the other fifteen, so the reported row identifies it; the
        # eight-point model is not exact on its sample after the rank-2 projection, hence argmin rather than zero.
        work = torch.float64 if dtype == torch.float64 else torch.float32
        ransac = RANSAC(model_type)
        generator = torch.Generator().manual_seed(0)
        m = ransac.minimal_sample_size
        x1 = torch.cat([torch.rand(16, m, 2, generator=generator), torch.ones(16, m, 1)], -1)
        x2 = torch.cat([torch.rand(16, m, 2, generator=generator), torch.ones(16, m, 1)], -1)
        models, rows = ransac._lm_minimal_models(x1.to(device, work), x2.to(device, work), track_origins=True)
        assert rows is not None
        assert rows.shape == (len(models),) and rows.dtype == torch.long
        assert bool((rows[1:] >= rows[:-1]).all())  # draw order
        finite = torch.isfinite(models).flatten(1).all(1)
        models, rows = models[finite].cpu().double(), rows[finite].cpu()
        s1, s2 = x1.double(), x2.double()
        if model_type == "homography":
            mapped = torch.einsum("mij,snj->msni", models, s1)
            residual = (mapped[..., :2] / mapped[..., 2:] - s2[None, ..., :2]).norm(dim=-1).amax(-1)
        else:
            residual = torch.einsum("sni,mij,snj->msn", s2, models, s1).abs().amax(-1)
        assert float((residual.argmin(1) == rows).double().mean()) >= 0.9


def _degensac_inputs(kp1, kp2, inl_th, device, work):
    """The normalized inputs ``_forward_lm`` builds for a seven-point fundamental matrix."""
    x1_host, x2_host, t1, t2, s1, _ = _normalize_correspondences(kp1.cpu().double(), kp2.cpu().double(), True)
    x1, x2 = x1_host.to(device, work), x2_host.to(device, work)
    return x1, x2, x1_host, x2_host, t1, t2, _sampson_quadratic_basis(x1, x2), (inl_th / s1) ** 2


def _degenerate_sample(labels, generator):
    """Five plane correspondences and two outliers, an H-degenerate seven-point sample."""
    plane = (labels == 0).nonzero().flatten().cpu()
    outliers = (labels == 2).nonzero().flatten().cpu()
    return torch.cat(
        [
            plane[torch.randperm(len(plane), generator=generator)[:5]],
            outliers[torch.randperm(len(outliers), generator=generator)[:2]],
        ]
    )


def _skip_half(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        pytest.skip("the scene's pixel coordinates quantize at the noise scale in half precision")


def _off_plane_error(F, clean1, clean2):
    """Median Sampson distance, in pixels, of the noise-free off-plane correspondences to F, on the host."""
    points1, points2 = clean1[None].cpu().double(), clean2[None].cpu().double()
    return float(sampson_epipolar_distance(points1, points2, F[None].cpu().double(), squared=False)[0].median())


def _explains_off_plane(F, clean1, clean2):
    """Whether F puts the noise-free off-plane correspondences within 2 px of their epipolar lines (median)."""
    return bool(F.abs().sum() > 0) and _off_plane_error(F, clean1, clean2) < 2.0


class TestRANSACDegensacRecovery(BaseTester):
    @pytest.mark.parametrize("max_lo_iters", [0, 5])
    def test_degenerate_samples_are_recovered(self, device, dtype, max_lo_iters):
        # Every sample the degeneracy test flags must come back as a model that explains the off-plane geometry;
        # a sample it does not flag must come back as None.
        _skip_half(dtype)
        work = torch.float64 if dtype == torch.float64 else torch.float32
        kp1, kp2, labels, clean1, clean2 = create_dominant_plane_scene(1000, 0.6, 0.95, 0, device=device, dtype=dtype)
        ransac = RANSAC("fundamental", inl_th=1.0, max_lo_iters=max_lo_iters, seed=0, degensac=True)
        x1, x2, x1_host, x2_host, t1, t2, basis, threshold = _degensac_inputs(kp1, kp2, 1.0, device, work)
        generator = torch.Generator().manual_seed(1)
        flagged = 0
        for _ in range(10):
            sample = _degenerate_sample(labels, generator)
            models, _ = ransac._lm_minimal_models(x1[sample.to(device)][None], x2[sample.to(device)][None])
            models = models[torch.isfinite(models).flatten(1).all(1)]
            scores, counts, _ = ransac._lm_score_models(models, basis, threshold)
            model = models[int(scores.masked_fill(counts <= 7, -1.0).argmax())]
            result = ransac._degensac_recover(
                model, sample.to(device), x1, x2, x1_host, x2_host, basis, threshold, torch.Generator().manual_seed(2)
            )
            degenerate = _h_degenerate_sample(model.cpu().double(), x1_host[sample], x2_host[sample], 3 * threshold)
            if degenerate is None:
                assert result is None
                continue
            flagged += 1
            assert result is not None
            out_models, scores, counts, masks = result
            assert masks is None
            # One model per recovery, as Chum's rFtH returns one: near-duplicates from one search would crowd the
            # eight-model pool (on the loftr_fund pair they cost test_real_dirty_7pt its margin in 2 of 20 seeds).
            assert out_models.shape == (1, 3, 3) and scores.shape == counts.shape == (1,)
            best = out_models[int(scores.argmax())].cpu().double()
            assert _explains_off_plane(t2.mT @ best @ t1, clean1.cpu(), clean2.cpu())
        assert flagged >= 5

    def test_record_checks_are_batched(self, device, dtype, monkeypatch):
        _skip_half(dtype)
        sizes = []
        original = ransac_module._h_degenerate_samples

        def check(models, *args):
            sizes.append(len(models))
            return original(models, *args)

        def scalar_check(*args):
            raise AssertionError("a record's already batched degeneracy test must not run again")

        monkeypatch.setattr(ransac_module, "_h_degenerate_samples", check)
        monkeypatch.setattr(ransac_module, "_h_degenerate_sample", scalar_check)
        kp1, kp2, _, clean1, clean2 = create_dominant_plane_scene(1000, 0.6, 0.95, 0, device=device, dtype=dtype)
        F, _ = RANSAC("fundamental", inl_th=1.0, seed=0, degensac=True)(kp1, kp2)
        assert sizes and max(sizes) > 1
        assert _explains_off_plane(F.cpu(), clean1.cpu(), clean2.cpu())

    @pytest.mark.parametrize("score_type", ["msac", "ransac"])
    def test_batched_recovery_matches_scalar_checks(self, device, dtype, score_type, monkeypatch):
        _skip_half(dtype)
        kp1, kp2, _, _, _ = create_dominant_plane_scene(1000, 0.6, 0.95, 5, device=device, dtype=dtype)
        estimator = RANSAC("fundamental", inl_th=1.0, seed=5, degensac=True, score_type=score_type)
        batched = estimator(kp1, kp2)

        def scalar_checks(models, x1, x2, threshold):
            return [_h_degenerate_sample(f, a, b, threshold) for f, a, b in zip(models, x1, x2)]

        monkeypatch.setattr(ransac_module, "_h_degenerate_samples", scalar_checks)
        scalar = estimator(kp1, kp2)
        assert torch.equal(batched[0], scalar[0]) and torch.equal(batched[1], scalar[1])

    def test_unseeded_recoveries_leave_the_global_generator(self, device, dtype):
        # The minimal samples of an unseeded call come from the global generator. Recoveries drawing from it too
        # would shift every later sample, so an unseeded call with a recovery would sample a different sequence than
        # degensac=False (on the loftr_fund pair: 9 of 20 global seeds with 2+ gross errors instead of 5).
        _skip_half(dtype)
        work = torch.float64 if dtype == torch.float64 else torch.float32
        kp1, kp2, labels, _, _ = create_dominant_plane_scene(1000, 0.6, 0.95, 0, device=device, dtype=dtype)
        ransac = RANSAC("fundamental", inl_th=1.0, degensac=True)
        x1, x2, x1_host, x2_host, _, _, basis, threshold = _degensac_inputs(kp1, kp2, 1.0, device, work)
        generator = torch.Generator().manual_seed(4)
        samples = torch.stack([_degenerate_sample(labels, generator) for _ in range(8)]).to(device)
        models, rows = ransac._lm_minimal_models(x1[samples], x2[samples], track_origins=True)
        assert rows is not None
        scores, counts, _ = ransac._lm_score_models(models, basis, threshold)
        scores = scores.masked_fill(counts <= 7, -1.0)
        state = torch.get_rng_state()
        recoveries = ransac._degensac_batch(
            models, samples[rows], scores, -1.0, x1, x2, x1_host, x2_host, basis, threshold, 0
        )
        assert recoveries  # the recovery drew
        assert torch.equal(torch.get_rng_state(), state)

    def test_unseeded_recovery_draws_follow_the_samples(self, device, dtype, monkeypatch):
        # An unseeded call seeds its private recovery generator from the sample that triggered the recovery, which
        # comes from the global generator: the draws differ between calls without advancing it. A constant seed
        # would repeat the same draws, and so the same luck, on every unseeded call.
        _skip_half(dtype)
        seeds = []
        original = RANSAC._degensac_recover

        def spy(
            self,
            model,
            sample,
            x1,
            x2,
            x1_host,
            x2_host,
            basis,
            threshold,
            generator,
            seen_planes=None,
            homography=None,
        ):
            seeds.append(generator.initial_seed())
            return original(
                self, model, sample, x1, x2, x1_host, x2_host, basis, threshold, generator, seen_planes, homography
            )

        monkeypatch.setattr(RANSAC, "_degensac_recover", spy)
        kp1, kp2, _, _, _ = create_dominant_plane_scene(1000, 0.6, 0.95, 0, device=device, dtype=dtype)
        per_call = []
        for global_seed in (0, 1):
            seeds.clear()
            torch.manual_seed(global_seed)
            RANSAC("fundamental", inl_th=1.0, degensac=True)(kp1, kp2)
            per_call.append(set(seeds))
        assert per_call[0] and per_call[1] and per_call[0] != per_call[1]

    def test_repeated_plane_is_searched_once(self, device, dtype, monkeypatch):
        # Every degenerate record setter of a dominant plane finds the same plane (the refined planes of repeated
        # recoveries have Jaccard index 0.997-1.0 on these scenes), and nine searches of it per call made DEGENSAC
        # eleven times slower than plain RANSAC; VSAC skips a model whose inliers repeat (Jaccard index 0.95).
        _skip_half(dtype)
        work = torch.float64 if dtype == torch.float64 else torch.float32
        kp1, kp2, labels, _, _ = create_dominant_plane_scene(1000, 0.6, 0.95, 0, device=device, dtype=dtype)
        ransac = RANSAC("fundamental", inl_th=1.0, seed=0, degensac=True)
        x1, x2, x1_host, x2_host, _, _, basis, threshold = _degensac_inputs(kp1, kp2, 1.0, device, work)
        generator = torch.Generator().manual_seed(4)
        samples = torch.stack([_degenerate_sample(labels, generator) for _ in range(8)]).to(device)
        models, rows = ransac._lm_minimal_models(x1[samples], x2[samples], track_origins=True)
        assert rows is not None
        scores, counts, _ = ransac._lm_score_models(models, basis, threshold)
        scores = scores.masked_fill(counts <= 7, -1.0)
        records = [
            record
            for record in RANSAC._raw_record_setters(scores.cpu(), -1.0)
            if _h_degenerate_sample(
                models[record].cpu().double(),
                x1_host[samples[rows][record].cpu()],
                x2_host[samples[rows][record].cpu()],
                3 * threshold,
            )
            is not None
        ]
        assert len(records) >= 2  # several degenerate record setters of the same plane
        calls = []
        original = ransac_module._plane_parallax_search
        monkeypatch.setattr(ransac_module, "_plane_parallax_search", lambda *a: calls.append(1) or original(*a))
        recoveries = ransac._degensac_batch(
            models, samples[rows], scores, -1.0, x1, x2, x1_host, x2_host, basis, threshold, 0
        )
        assert len(calls) == 1 and len(recoveries) == 1

    def test_dominant_plane_is_refined_once_per_call(self, device, dtype, monkeypatch):
        # A later record setter's homography is noisier than the refined plane (Jaccard index 0.03-1.0 against it at
        # N=4000), so a Jaccard test before innerH let one to five repeated refinements through per call; its inliers
        # lie inside the refined plane (98.3-100% of them in every repeat measured), which a containment test sees.
        _skip_half(dtype)
        refinements = []
        original = ransac_module._inner_homography
        monkeypatch.setattr(ransac_module, "_inner_homography", lambda *a: refinements.append(1) or original(*a))
        for seed in range(3):
            refinements.clear()
            kp1, kp2, _, _, _ = create_dominant_plane_scene(4000, 0.6, 0.95, seed, device=device, dtype=dtype)
            RANSAC("fundamental", inl_th=1.0, seed=seed, degensac=True)(kp1, kp2)
            assert len(refinements) == 1, f"seed {seed}"

    def test_dominant_plane_is_searched_at_most_twice_per_call(self, device, dtype, monkeypatch):
        _skip_half(dtype)
        calls = []
        original = ransac_module._plane_parallax_search
        monkeypatch.setattr(ransac_module, "_plane_parallax_search", lambda *a: calls.append(1) or original(*a))
        kp1, kp2, _, clean1, clean2 = create_dominant_plane_scene(1000, 0.6, 0.95, 0, device=device, dtype=dtype)
        F, _ = RANSAC("fundamental", inl_th=1.0, confidence=0.999, seed=0, degensac=True)(kp1, kp2)
        assert _explains_off_plane(F.cpu(), clean1.cpu(), clean2.cpu())
        assert 1 <= len(calls) <= 2  # one plane; nine searches before deduplication

    def _recover_once(self, max_lo_iters, score_type, scene_seed, device, dtype):
        """Recover the scene's first degenerate sample (5 plane + 2 outliers) with a fixed recovery generator."""
        work = torch.float64 if dtype == torch.float64 else torch.float32
        kp1, kp2, labels, _, _ = create_dominant_plane_scene(1000, 0.6, 0.95, scene_seed, device=device, dtype=dtype)
        x1, x2, x1_host, x2_host, _, _, basis, threshold = _degensac_inputs(kp1, kp2, 1.0, device, work)
        sample = _degenerate_sample(labels, torch.Generator().manual_seed(scene_seed)).to(device)
        ransac = RANSAC("fundamental", inl_th=1.0, score_type=score_type, max_lo_iters=max_lo_iters, degensac=True)
        models, _ = ransac._lm_minimal_models(x1[sample][None], x2[sample][None])
        models = models[torch.isfinite(models).flatten(1).all(1)]
        scores, counts, _ = ransac._lm_score_models(models, basis, threshold)
        model = models[int(scores.masked_fill(counts <= 7, -1.0).argmax())]
        generator = torch.Generator().manual_seed(999)
        return ransac._degensac_recover(model, sample, x1, x2, x1_host, x2_host, basis, threshold, generator)

    def test_refinement_never_lowers_the_recovered_score(self, device, dtype):
        # With score_type="ransac" the truncated LM can trade inliers: on this sample it took the recovered model from
        # 568 to 566. The recovery keeps the better of its raw and refined models.
        _skip_half(dtype)
        raw = self._recover_once(0, "ransac", 5, device, dtype)
        refined = self._recover_once(5, "ransac", 5, device, dtype)
        assert raw is not None and refined is not None
        assert float(refined[1][0]) >= float(raw[1][0])

    def test_worse_refinement_keeps_the_raw_model(self, device, dtype, monkeypatch):
        _skip_half(dtype)
        raw = self._recover_once(0, "msac", 0, device, dtype)
        # A "refinement" that returns a scrambled model scores far below the raw one.
        monkeypatch.setattr(RANSAC, "_lm_refine", lambda self, models, *args: models.roll(1, dims=-1))
        kept = self._recover_once(5, "msac", 0, device, dtype)
        assert raw is not None and kept is not None
        assert torch.equal(kept[0], raw[0]) and torch.equal(kept[1], raw[1])

    def test_non_degenerate_sample_returns_none(self, device, dtype):
        # Seven off-plane correspondences are not always in general position at the test's tolerance: a five-point
        # homography fit leaves two redundant constraints, and five of the first seven here fit one within 2.2 px^2
        # < 3 t. The contract is the implication: a sample the test does not flag is not recovered.
        _skip_half(dtype)
        work = torch.float64 if dtype == torch.float64 else torch.float32
        kp1, kp2, labels, _, _ = create_dominant_plane_scene(1000, 0.6, 0.5, 0, device=device, dtype=dtype)
        ransac = RANSAC("fundamental", inl_th=1.0, seed=0, degensac=True)
        x1, x2, x1_host, x2_host, _, _, basis, threshold = _degensac_inputs(kp1, kp2, 1.0, device, work)
        off_plane = (labels == 1).nonzero().flatten().cpu()
        unflagged = 0
        for group in range(6):
            sample = off_plane[7 * group : 7 * group + 7]
            models, _ = ransac._lm_minimal_models(x1[sample.to(device)][None], x2[sample.to(device)][None])
            for model in models[torch.isfinite(models).flatten(1).all(1)]:
                if _h_degenerate_sample(model.cpu().double(), x1_host[sample], x2_host[sample], 3 * threshold) is None:
                    unflagged += 1
                    recovered = ransac._degensac_recover(
                        model, sample.to(device), x1, x2, x1_host, x2_host, basis, threshold, None
                    )
                    assert recovered is None
        assert unflagged >= 3


_SCENES = {"frontal": {}, "zoom": {"zoom": 3.0}, "skew": {"skew": 0.5}}


class TestRANSACDegensac(BaseTester):
    @pytest.mark.parametrize("scene", sorted(_SCENES))
    def test_recovers_dominant_plane(self, device, dtype, scene):
        # On main (a7d177cbf) plain seven-point RANSAC fails every one of these seeds on every scene on CPU, so the
        # explicit recovery's assertions fail there. Accelerators sample differently (MPS: plain succeeds on one
        # frontal and zoomed seed), so the plain count only asserts that the scenes stay hard for it.
        _skip_half(dtype)
        plain_failures = 0
        for seed in range(5):
            kp1, kp2, labels, clean1, clean2 = create_dominant_plane_scene(
                1000, 0.6, 0.95, seed, device=device, dtype=dtype, **_SCENES[scene]
            )
            F, mask = RANSAC("fundamental", inl_th=1.0, confidence=0.999, seed=seed, degensac=True)(kp1, kp2)
            assert _explains_off_plane(F.cpu(), clean1.cpu(), clean2.cpu()), f"seed {seed}"
            off_plane = labels == 1
            assert float((mask & off_plane).sum()) >= 0.8 * float(off_plane.sum()), f"seed {seed}"
            F_plain, _ = RANSAC("fundamental", inl_th=1.0, confidence=0.999, seed=seed, degensac=False)(kp1, kp2)
            plain_failures += int(not _explains_off_plane(F_plain.cpu(), clean1.cpu(), clean2.cpu()))
        assert plain_failures >= 3

    @pytest.mark.parametrize("score_type", ["msac", "ransac"])
    def test_both_score_types(self, device, dtype, score_type):
        _skip_half(dtype)
        kp1, kp2, _, clean1, clean2 = create_dominant_plane_scene(1000, 0.6, 0.95, 0, device=device, dtype=dtype)
        F, _ = RANSAC("fundamental", inl_th=1.0, confidence=0.999, seed=0, score_type=score_type, degensac=True)(
            kp1, kp2
        )
        assert _explains_off_plane(F.cpu(), clean1.cpu(), clean2.cpu())

    def test_never_degenerate_is_bitwise_plain(self, device, dtype, monkeypatch):
        # With the test stubbed to find no degenerate sample, the record tracking, sample rows and host transfers
        # must leave the result bit for bit the same, even where recoveries would fire.
        _skip_half(dtype)
        monkeypatch.setattr(ransac_module, "_h_degenerate_samples", lambda models, *args: [None] * len(models))
        kp1, kp2, _, _, _ = create_dominant_plane_scene(1000, 0.6, 0.95, 0, device=device, dtype=dtype)
        F, mask = RANSAC("fundamental", inl_th=1.0, seed=0, degensac=True)(kp1, kp2)
        F_plain, mask_plain = RANSAC("fundamental", inl_th=1.0, seed=0, degensac=False)(kp1, kp2)
        assert torch.equal(F, F_plain) and torch.equal(mask, mask_plain)

    def test_scene_without_plane_keeps_its_accuracy(self, device, dtype, monkeypatch):
        # Without a dominant plane, record setters are still often H-degenerate at Chum's 3 t: five of seven
        # correspondences fit one homography within 0.6-2.1 t in 5 of these 6 seeds. The recoveries then join the
        # pool, so the estimate is not bitwise the plain one, but it must explain the geometry as well.
        _skip_half(dtype)
        calls = []
        original = ransac_module._plane_parallax_search
        monkeypatch.setattr(ransac_module, "_plane_parallax_search", lambda *a: calls.append(1) or original(*a))
        # plane_fraction=0 puts every inlier at a random depth: no dominant plane.
        for seed in range(6):
            kp1, kp2, _, clean1, clean2 = create_dominant_plane_scene(1000, 0.6, 0.0, seed, device=device, dtype=dtype)
            F, mask = RANSAC("fundamental", inl_th=1.0, seed=seed, degensac=True)(kp1, kp2)
            F_plain, mask_plain = RANSAC("fundamental", inl_th=1.0, seed=seed, degensac=False)(kp1, kp2)

            error, error_plain = _off_plane_error(F, clean1, clean2), _off_plane_error(F_plain, clean1, clean2)
            assert error <= error_plain + 0.01, f"seed {seed}"
            assert abs(int(mask.sum()) - int(mask_plain.sum())) <= 0.01 * len(mask), f"seed {seed}"
        assert calls  # the recovered path ran

    def test_recovered_incumbent_does_not_hide_later_records(self, device, dtype, monkeypatch):
        # The stub's first model outscores every raw model. A trigger tied to the incumbent's score would stop
        # testing after the first batch; the raw record keeps finding record setters in later batches.
        _skip_half(dtype)
        batches = []

        def stub(
            self,
            model,
            sample,
            x1,
            x2,
            x1_host,
            x2_host,
            basis,
            threshold,
            generator,
            seen_planes=None,
            homography=None,
        ):
            batches.append(generator.initial_seed())
            models = model[None].clone()
            huge = torch.full((1,), 1e9, dtype=x1.dtype, device=x1.device)
            return models, huge, torch.full((1,), float(len(x1)), dtype=x1.dtype, device=x1.device), None

        monkeypatch.setattr(RANSAC, "_degensac_recover", stub)
        kp1, kp2, _, _, _ = create_dominant_plane_scene(1000, 0.6, 0.95, 0, device=device, dtype=dtype)
        RANSAC("fundamental", inl_th=1.0, batch_size=64, max_iter=20, confidence=1.0, seed=0, degensac=True)(kp1, kp2)
        assert len(set(batches)) >= 2

    def test_seeded_calls_are_reproducible_and_private(self, device, dtype, monkeypatch):
        _skip_half(dtype)
        kp1, kp2, _, _, _ = create_dominant_plane_scene(1000, 0.6, 0.95, 0, device=device, dtype=dtype)

        def device_state():
            if device.type == "cuda":
                return torch.cuda.get_rng_state(device)
            if device.type == "mps":
                return torch.mps.get_rng_state()
            return None

        calls = []
        original = ransac_module._plane_parallax_search
        monkeypatch.setattr(ransac_module, "_plane_parallax_search", lambda *a: calls.append(1) or original(*a))
        cpu_state, accelerator_state = torch.get_rng_state(), device_state()
        first = RANSAC("fundamental", inl_th=1.0, seed=3, degensac=True)(kp1, kp2)
        second = RANSAC("fundamental", inl_th=1.0, seed=3, degensac=True)(kp1, kp2)
        assert calls  # the recovery drew
        assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
        assert torch.equal(torch.get_rng_state(), cpu_state)
        if accelerator_state is not None:
            assert torch.equal(device_state(), accelerator_state)

    # Review focus 1: non-finite correspondences.
    def test_non_finite_rows_are_skipped(self, device, dtype):
        _skip_half(dtype)
        kp1, kp2, _, clean1, clean2 = create_dominant_plane_scene(1000, 0.6, 0.95, 0, device=device, dtype=dtype)
        kp1[:5] = float("nan")
        kp2[5:10] = float("inf")
        F, mask = RANSAC("fundamental", inl_th=1.0, seed=0, degensac=True)(kp1, kp2)
        assert not bool(mask[:10].any())
        assert _explains_off_plane(F.cpu(), clean1.cpu(), clean2.cpu())

    # Review focus 2: a fully planar scene. Its off-plane set is the outliers, so the pair search runs on them and may
    # return a model consistent with the plane plus a few outliers (as Chum's code does; VSAC's DEGENSAC+ adds a
    # randomness check against that). Every model consistent with the plane is valid here: the plane must stay in.
    def test_fully_planar_scene(self, device, dtype):
        _skip_half(dtype)
        kp1, kp2, labels, _, _ = create_dominant_plane_scene(1000, 0.6, 1.0, 0, device=device, dtype=dtype)
        F, mask = RANSAC("fundamental", inl_th=1.0, seed=0, degensac=True)(kp1, kp2)
        assert F.shape == (3, 3) and bool(F.abs().sum() > 0)
        plane = labels == 0
        assert float((mask & plane).sum()) >= 0.9 * float(plane.sum())

    # Review focus 3: tiny inputs.
    @pytest.mark.parametrize("num_points", [7, 10, 20])
    def test_tiny_inputs(self, device, dtype, num_points):
        kp1, kp2, _, _, _ = create_dominant_plane_scene(num_points, 0.8, 0.9, 0, device=device, dtype=dtype)
        F, mask = RANSAC("fundamental", inl_th=1.0, seed=0, degensac=True)(kp1, kp2)
        assert F.shape == (3, 3) and mask.shape == (num_points,) and mask.dtype == torch.bool

    # Review focus 4: PROSAC's stopping rule takes the recovered incumbent's mask.
    def test_prosac(self, device, dtype, monkeypatch):
        # The raw degenerate incumbent already certifies itself after the first batch here, so the final model alone
        # cannot show which mask set the recovered incumbent's bound: spy on PROSAC's termination test instead.
        _skip_half(dtype)
        recovered, tested = [], []
        original_recover, original_prosac = RANSAC._degensac_recover, RANSAC._prosac_max_samples

        def spy_recover(self, *args):
            result = original_recover(self, *args)
            if result is not None:
                recovered.append(result[3][0].clone())
            return result

        def spy_prosac(self, inliers, num_inliers):
            tested.append(inliers.clone())
            return original_prosac(self, inliers, num_inliers)

        monkeypatch.setattr(RANSAC, "_degensac_recover", spy_recover)
        monkeypatch.setattr(RANSAC, "_prosac_max_samples", spy_prosac)
        kp1, kp2, labels, clean1, clean2 = create_dominant_plane_scene(1000, 0.6, 0.95, 0, device=device, dtype=dtype)
        order = torch.argsort(labels.cpu(), stable=True).to(device)  # inliers first, a best-first ranking
        F, _ = RANSAC("fundamental", inl_th=1.0, seed=0, prosac_sampling=True, degensac=True)(kp1[order], kp2[order])
        assert _explains_off_plane(F.cpu(), clean1.cpu(), clean2.cpu())
        # The recovered incumbent's stopping bound is PROSAC's test on its own inlier mask.
        assert recovered and any(torch.equal(mask, inliers) for mask in recovered for inliers in tested)

    # Review focus 5: half-precision correspondences run the float64 host recovery and float32 device scoring.
    def test_half_precision_smoke(self, device):
        if device.type != "cpu":
            pytest.skip("the half-precision RANSAC legs run on CPU; accelerator half kernels are covered elsewhere")
        for dtype in (torch.float16, torch.bfloat16):
            kp1, kp2, _, _, _ = create_dominant_plane_scene(500, 0.6, 0.95, 0, device=device, dtype=dtype)
            F, mask = RANSAC("fundamental", inl_th=2.0, seed=0, degensac=True)(kp1, kp2)
            assert F.shape == (3, 3) and F.dtype == dtype and mask.shape == (500,)
