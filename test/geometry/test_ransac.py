import random

import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils
from kornia.geometry import RANSAC
from kornia.geometry.epipolar import sampson_epipolar_distance
from kornia.testing import assert_close


class TestRANSACHomography:
    def test_smoke(self, device, dtype):
        points1 = torch.rand(4, 2, device=device, dtype=dtype)
        points2 = torch.rand(4, 2, device=device, dtype=dtype)
        ransac = RANSAC('homography').to(device=device, dtype=dtype)
        H, inliers = ransac(points1, points2)
        assert H.shape == (3, 3)

    @pytest.mark.xfail(reason="might slightly and randomly imprecise due to RANSAC randomness")
    def test_dirty_points(self, device, dtype):
        # generate input data

        H = torch.eye(3, dtype=dtype, device=device)
        H[:2] = H[:2] + 0.1 * torch.rand_like(H[:2])
        H[2:, :2] = H[2:, :2] + 0.001 * torch.rand_like(H[2:, :2])

        points_src = 100.0 * torch.rand(1, 20, 2, device=device, dtype=dtype)
        points_dst = kornia.geometry.transform_points(H[None], points_src)

        # making last point an outlier
        points_dst[:, -1, :] += 800
        ransac = RANSAC('homography', inl_th=0.5, max_iter=20).to(device=device, dtype=dtype)
        # compute transform from source to target
        dst_homo_src, inliers = ransac(points_src[0], points_dst[0])

        assert_close(
            kornia.geometry.transform_points(dst_homo_src[None], points_src[:, :-1]),
            points_dst[:, :-1],
            rtol=1e-3,
            atol=1e-3)

    def test_real_clean(self, device, dtype):
        data = torch.load("data/test/loftr_outdoor_and_homography_data.pt")
        # generate input data
        homography_gt = torch.inverse(data['H_gt']).to(device, dtype)
        homography_gt = homography_gt / homography_gt[2, 2]
        pts_src = data['pts0'].to(device, dtype)
        pts_dst = data['pts1'].to(device, dtype)
        ransac = RANSAC('homography', inl_th=0.5, max_iter=20).to(device=device, dtype=dtype)
        # compute transform from source to target
        dst_homo_src, inliers = ransac(pts_src, pts_dst)

        assert_close(
            kornia.geometry.transform_points(dst_homo_src[None], pts_src[None]),
            pts_dst[None],
            rtol=1e-3,
            atol=1e-3)

    def test_real_dirty(self, device, dtype):
        data = torch.load("data/test/loftr_outdoor_and_homography_data.pt")
        # generate input data
        homography_gt = torch.inverse(data['H_gt']).to(device, dtype)
        homography_gt = homography_gt / homography_gt[2, 2]
        pts_src = data['pts0'].to(device, dtype)
        pts_dst = data['pts1'].to(device, dtype)

        kp1 = data['loftr_outdoor_tentatives0'].to(device, dtype)
        kp2 = data['loftr_outdoor_tentatives1'].to(device, dtype)

        ransac = RANSAC('homography', inl_th=3.0, max_iter=30, max_lo_iters=10).to(device=device, dtype=dtype)
        # compute transform from source to target
        dst_homo_src, inliers = ransac(kp1, kp2)

        # Reprojection error of 5px is OK
        assert_close(
            kornia.geometry.transform_points(dst_homo_src[None], pts_src[None]),
            pts_dst[None],
            rtol=5,
            atol=0.15)

    @pytest.mark.jit
    @pytest.mark.skip(reason="find_homography_dlt is using try/except block")
    def test_jit(self, device, dtype):
        points1 = torch.rand(4, 2, device=device, dtype=dtype)
        points2 = torch.rand(4, 2, device=device, dtype=dtype)
        model = RANSAC('homography').to(device=device, dtype=dtype)
        model_jit = torch.jit.script(RANSAC('homography').to(device=device,
                                                             dtype=dtype))
        assert_close(model(points1, points2)[0],
                     model_jit(points1, points2)[0],
                     rtol=1e-4,
                     atol=1e-4)


class TestRANSACFundamental:
    def test_smoke(self, device, dtype):
        points1 = torch.rand(8, 2, device=device, dtype=dtype)
        points2 = torch.rand(8, 2, device=device, dtype=dtype)
        ransac = RANSAC('fundamental').to(device=device, dtype=dtype)
        Fm, inliers = ransac(points1, points2)
        assert Fm.shape == (3, 3)

    def test_real_clean(self, device, dtype):
        data = torch.load("data/test/loftr_indoor_and_fundamental_data.pt")
        # generate input data
        F_gt = data['F_gt'].to(device, dtype)
        pts_src = data['pts0'].to(device, dtype)
        pts_dst = data['pts1'].to(device, dtype)

        ransac = RANSAC('fundamental',
                        inl_th=0.5,
                        max_iter=20,
                        max_lo_iters=10).to(device=device, dtype=dtype)
        # compute transform from source to target
        fundamental_matrix, inliers = ransac(pts_src, pts_dst)
        gross_errors = sampson_epipolar_distance(pts_src[None],
                                                 pts_dst[None],
                                                 fundamental_matrix[None],
                                                 squared=False) > 1.0
        assert gross_errors.sum().item() == 0

    @pytest.mark.xfail(reason="might fail, becuase out F-RANSAC is not yet 7pt")
    def test_real_dirty(self, device, dtype):
        data = torch.load("data/test/loftr_indoor_and_fundamental_data.pt")
        # generate input data
        F_gt = data['F_gt'].to(device, dtype)
        pts_src = data['pts0'].to(device, dtype)
        pts_dst = data['pts1'].to(device, dtype)

        kp1 = data['loftr_indoor_tentatives0'].to(device, dtype)
        kp2 = data['loftr_indoor_tentatives1'].to(device, dtype)

        ransac = RANSAC('fundamental',
                        inl_th=1.0,
                        max_iter=20,
                        max_lo_iters=10).to(device=device, dtype=dtype)
        # compute transform from source to target
        fundamental_matrix, inliers = ransac(kp1, kp2)
        gross_errors = sampson_epipolar_distance(pts_src[None],
                                                 pts_dst[None],
                                                 fundamental_matrix[None],
                                                 squared=False) > 10.0
        assert gross_errors.sum().item() < 2

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        points1 = torch.rand(8, 2, device=device, dtype=dtype)
        points2 = torch.rand(8, 2, device=device, dtype=dtype)
        model = RANSAC('fundamental').to(device=device, dtype=dtype)
        model_jit = torch.jit.script(RANSAC('fundamental').to(device=device,
                                                              dtype=dtype))
        assert_close(model(points1, points2)[0],
                     model_jit(points1, points2)[0],
                     rtol=1e-3,
                     atol=1e-3)

    @pytest.mark.skip(reason="RANSAC is random algorithm, so Jacobian is not defined")
    def test_gradcheck(self, device):
        points1 = torch.rand(8, 2, device=device, dtype=torch.float64, requires_grad=True)
        points2 = torch.rand(8, 2, device=device, dtype=torch.float64)
        model = RANSAC('fundamental').to(device=device, dtype=torch.float64)

        def gradfun(p1, p2):
            return model(p1, p2)[0]
        assert gradcheck(gradfun, (points1, points2), raise_exception=True)
