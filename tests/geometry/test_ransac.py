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

from kornia.geometry import RANSAC, transform_points
from kornia.geometry.epipolar import project_to_essential, sampson_epipolar_distance

from testing.base import BaseTester
from testing.casts import dict_to


class TestRANSACHomography(BaseTester):
    def test_smoke(self, device, dtype):
        torch.random.manual_seed(0)
        points1 = torch.rand(4, 2, device=device, dtype=dtype)
        points2 = torch.rand(4, 2, device=device, dtype=dtype)
        ransac = RANSAC("homography").to(device=device, dtype=dtype)
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
    @pytest.mark.xfail(reason="might slightly and randomly imprecise due to RANSAC randomness")
    @pytest.mark.parametrize("data", ["loftr_fund"], indirect=True)
    def test_real_clean_8pt(self, device, dtype, data):
        torch.random.manual_seed(0)
        # generate input data
        data_dev = dict_to(data, device, dtype)
        pts_src = data_dev["pts0"]
        pts_dst = data_dev["pts1"]
        # compute transform from source to target
        ransac = RANSAC("fundamental", inl_th=0.5, max_iter=20, max_lo_iters=10).to(device=device, dtype=dtype)
        fundamental_matrix, _ = ransac(pts_src, pts_dst)
        gross_errors = (
            sampson_epipolar_distance(pts_src[None], pts_dst[None], fundamental_matrix[None], squared=False) > 1.0
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


class TestRansacMethods:
    def test_max_samples_by_conf(self):
        """Test max_samples_by_conf with realistic scenarios."""
        conf = 0.99

        # Test 1: Very few inliers (1 out of 1000) with sample_size=7
        # Returns 1 because n_inl <= sample_size (early exit condition)
        x = RANSAC.max_samples_by_conf(n_inl=1, num_tc=1000, sample_size=7, conf=conf)
        assert x == 1  # Early exit when n_inl <= sample_size

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
        assert x == 1  # Returns 1 when num_tc <= sample_size

        # Test 7: Edge case - too few inliers (n_inl <= sample_size)
        x = RANSAC.max_samples_by_conf(n_inl=2, num_tc=1000, sample_size=4, conf=conf)
        assert x == 1  # Returns 1 when n_inl <= sample_size

        # Test 8: Edge case - confidence at boundaries
        x = RANSAC.max_samples_by_conf(n_inl=50, num_tc=100, sample_size=4, conf=1.0)
        assert x == 1  # Returns 1 when conf >= 1.0

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

        ransac_a = RANSAC("homography", inl_th=2.0, max_iter=5, seed=1).to(device=device, dtype=dtype)
        ransac_b = RANSAC("homography", inl_th=2.0, max_iter=5, seed=2).to(device=device, dtype=dtype)

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
        before being returned, so the constraint holds regardless of which path selects the
        best model.
        """
        torch.random.manual_seed(0)
        ransac = RANSAC("essential", batch_size=32, max_iter=1, max_lo_iters=0).to(device=device, dtype=dtype)
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
