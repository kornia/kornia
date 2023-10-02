import pytest
import torch
from torch.autograd import gradcheck

import kornia.testing as utils  # test utils
from kornia.geometry import RANSAC, transform_points
from kornia.geometry.epipolar import sampson_epipolar_distance
from kornia.testing import assert_close


class TestRANSACHomography:
    def test_smoke(self, device, dtype):
        torch.random.manual_seed(0)
        points1 = torch.rand(4, 2, device=device, dtype=dtype)
        points2 = torch.rand(4, 2, device=device, dtype=dtype)
        ransac = RANSAC('homography').to(device=device, dtype=dtype)
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
        ransac = RANSAC('homography', inl_th=0.5, max_iter=20).to(device=device, dtype=dtype)
        # compute transform from source to target
        dst_homo_src, _ = ransac(points_src[0], points_dst[0])

        assert_close(transform_points(dst_homo_src[None], points_src[:, :-1]), points_dst[:, :-1], rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("data", ["loftr_homo"], indirect=True)
    def test_real_clean(self, device, dtype, data):
        # generate input data
        torch.random.manual_seed(0)
        data_dev = utils.dict_to(data, device, dtype)
        homography_gt = torch.inverse(data_dev['H_gt'])
        homography_gt = homography_gt / homography_gt[2, 2]
        pts_src = data_dev['pts0']
        pts_dst = data_dev['pts1']
        ransac = RANSAC('homography', inl_th=0.5, max_iter=20).to(device=device, dtype=dtype)
        # compute transform from source to target
        dst_homo_src, _ = ransac(pts_src, pts_dst)

        assert_close(transform_points(dst_homo_src[None], pts_src[None]), pts_dst[None], rtol=1e-2, atol=1.0)

    @pytest.mark.xfail(reason="might slightly and randomly imprecise due to RANSAC randomness")
    @pytest.mark.parametrize("data", ["loftr_homo"], indirect=True)
    def test_real_dirty(self, device, dtype, data):
        # generate input data
        torch.random.manual_seed(0)
        data_dev = utils.dict_to(data, device, dtype)
        homography_gt = torch.inverse(data_dev['H_gt'])
        homography_gt = homography_gt / homography_gt[2, 2]
        pts_src = data_dev['pts0']
        pts_dst = data_dev['pts1']

        kp1 = data_dev['loftr_outdoor_tentatives0']
        kp2 = data_dev['loftr_outdoor_tentatives1']

        ransac = RANSAC('homography', inl_th=3.0, max_iter=30, max_lo_iters=10).to(device=device, dtype=dtype)
        # compute transform from source to target
        dst_homo_src, _ = ransac(kp1, kp2)

        # Reprojection error of 5px is OK
        assert_close(transform_points(dst_homo_src[None], pts_src[None]), pts_dst[None], rtol=0.15, atol=5)

    @pytest.mark.skip(reason="find_homography_dlt is using try/except block")
    def test_jit(self, device, dtype):
        torch.random.manual_seed(0)
        points1 = torch.rand(4, 2, device=device, dtype=dtype)
        points2 = torch.rand(4, 2, device=device, dtype=dtype)
        model = RANSAC('homography').to(device=device, dtype=dtype)
        model_jit = torch.jit.script(RANSAC('homography').to(device=device, dtype=dtype))
        assert_close(model(points1, points2)[0], model_jit(points1, points2)[0], rtol=1e-4, atol=1e-4)


class TestRANSACHomographyLineSegments:
    def test_smoke(self, device, dtype):
        torch.random.manual_seed(0)
        points1 = torch.rand(4, 2, 2, device=device, dtype=dtype)
        points2 = torch.rand(4, 2, 2, device=device, dtype=dtype)
        ransac = RANSAC('homography_from_linesegments').to(device=device, dtype=dtype)
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

        ransac = RANSAC('homography_from_linesegments', inl_th=0.5, max_iter=20).to(device=device, dtype=dtype)
        # compute transform from source to target
        dst_homo_src, _ = ransac(ls1[0], ls2[0])

        assert_close(
            transform_points(dst_homo_src[None], points_src_st[:, :-1]), points_dst_st[:, :-1], rtol=1e-3, atol=1e-3
        )

    @pytest.mark.skip(reason="find_homography_dlt is using try/except block")
    def test_jit(self, device, dtype):
        torch.random.manual_seed(0)
        points1 = torch.rand(4, 2, 2, device=device, dtype=dtype)
        points2 = torch.rand(4, 2, 2, device=device, dtype=dtype)
        model = RANSAC('homography_from_linesegments').to(device=device, dtype=dtype)
        model_jit = torch.jit.script(RANSAC('homography_from_linesegments').to(device=device, dtype=dtype))
        assert_close(model(points1, points2)[0], model_jit(points1, points2)[0], rtol=1e-4, atol=1e-4)


class TestRANSACFundamental:
    def test_smoke(self, device, dtype):
        torch.random.manual_seed(0)
        points1 = torch.rand(8, 2, device=device, dtype=dtype)
        points2 = torch.rand(8, 2, device=device, dtype=dtype)
        ransac = RANSAC('fundamental').to(device=device, dtype=dtype)
        Fm, _ = ransac(points1, points2)
        assert Fm.shape == (3, 3)

    @pytest.mark.xfail(reason="might slightly and randomly imprecise due to RANSAC randomness")
    @pytest.mark.parametrize("data", ["loftr_fund"], indirect=True)
    def test_real_clean(self, device, dtype, data):
        torch.random.manual_seed(0)
        # generate input data
        data_dev = utils.dict_to(data, device, dtype)
        pts_src = data_dev['pts0']
        pts_dst = data_dev['pts1']
        # compute transform from source to target
        ransac = RANSAC('fundamental', inl_th=0.5, max_iter=20, max_lo_iters=10).to(device=device, dtype=dtype)
        fundamental_matrix, _ = ransac(pts_src, pts_dst)
        gross_errors = (
            sampson_epipolar_distance(pts_src[None], pts_dst[None], fundamental_matrix[None], squared=False) > 1.0
        )
        assert gross_errors.sum().item() == 0

    @pytest.mark.parametrize("data", ["loftr_fund"], indirect=True)
    def test_real_clean_7pt(self, device, dtype, data):
        torch.random.manual_seed(0)
        # generate input data
        data_dev = utils.dict_to(data, device, dtype)
        pts_src = data_dev['pts0']
        pts_dst = data_dev['pts1']
        # compute transform from source to target
        ransac = RANSAC('fundamental_7pt', inl_th=1.0, max_iter=100, max_lo_iters=10).to(device=device, dtype=dtype)
        fundamental_matrix, _ = ransac(pts_src, pts_dst)
        gross_errors = (
            sampson_epipolar_distance(pts_src[None], pts_dst[None], fundamental_matrix[None], squared=False) > 1.0
        )
        assert gross_errors.sum().item() == 0

    @pytest.mark.xfail(reason="might fail, because this F-RANSAC is not 7pt")
    @pytest.mark.parametrize("data", ["loftr_fund"], indirect=True)
    def test_real_dirty(self, device, dtype, data):
        torch.random.manual_seed(0)
        # generate input data
        data_dev = utils.dict_to(data, device, dtype)
        pts_src = data_dev['pts0']
        pts_dst = data_dev['pts1']

        kp1 = data_dev['loftr_indoor_tentatives0']
        kp2 = data_dev['loftr_indoor_tentatives1']

        ransac = RANSAC('fundamental', inl_th=1.0, max_iter=20, max_lo_iters=10).to(device=device, dtype=dtype)
        # compute transform from source to target
        fundamental_matrix, _ = ransac(kp1, kp2)
        gross_errors = (
            sampson_epipolar_distance(pts_src[None], pts_dst[None], fundamental_matrix[None], squared=False) > 10.0
        )
        assert gross_errors.sum().item() < 2

    @pytest.mark.parametrize("data", ["loftr_fund"], indirect=True)
    def test_real_dirty(self, device, dtype, data):
        torch.random.manual_seed(0)
        # generate input data
        data_dev = utils.dict_to(data, device, dtype)
        pts_src = data_dev['pts0']
        pts_dst = data_dev['pts1']

        kp1 = data_dev['loftr_indoor_tentatives0']
        kp2 = data_dev['loftr_indoor_tentatives1']

        ransac = RANSAC('fundamental_7pt', inl_th=1.0, max_iter=20, max_lo_iters=10).to(device=device, dtype=dtype)
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
        model = RANSAC('fundamental').to(device=device, dtype=dtype)
        model_jit = torch.jit.script(model)
        assert_close(model(points1, points2)[0], model_jit(points1, points2)[0], rtol=1e-3, atol=1e-3)

    @pytest.mark.skip(reason="RANSAC is random algorithm, so Jacobian is not defined")
    def test_gradcheck(self, device):
        torch.random.manual_seed(0)
        points1 = torch.rand(8, 2, device=device, dtype=torch.float64, requires_grad=True)
        points2 = torch.rand(8, 2, device=device, dtype=torch.float64)
        model = RANSAC('fundamental').to(device=device, dtype=torch.float64)

        def gradfun(p1, p2):
            return model(p1, p2)[0]

        assert gradcheck(gradfun, (points1, points2), raise_exception=True)
