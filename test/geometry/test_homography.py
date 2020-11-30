import pytest

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose

import kornia
import kornia.testing as utils  # test utils
from kornia.geometry.homography import find_homography_dlt, find_homography_dlt_iterated


class TestFindHomographyDLT:
    def test_smoke(self, device, dtype):
        points1 = torch.rand(1, 4, 2, device=device, dtype=dtype)
        points2 = torch.rand(1, 4, 2, device=device, dtype=dtype)
        weights = torch.ones(1, 4, device=device, dtype=dtype)
        H = find_homography_dlt(points1, points2, weights)
        assert H.shape == (1, 3, 3)

    @pytest.mark.parametrize(
        "batch_size, num_points", [(1, 4), (2, 5), (3, 6)],
    )
    def test_shape(self, batch_size, num_points, device, dtype):
        B, N = batch_size, num_points
        points1 = torch.rand(B, N, 2, device=device, dtype=dtype)
        points2 = torch.rand(B, N, 2, device=device, dtype=dtype)
        weights = torch.ones(B, N, device=device, dtype=dtype)
        H = find_homography_dlt(points1, points2, weights)
        assert H.shape == (B, 3, 3)

    @pytest.mark.parametrize("batch_size, num_points", [(1, 4), (2, 5), (3, 6)],)
    def test_shape_noweights(self, batch_size, num_points, device, dtype):
        B, N = batch_size, num_points
        points1 = torch.rand(B, N, 2, device=device, dtype=dtype)
        points2 = torch.rand(B, N, 2, device=device, dtype=dtype)
        H = find_homography_dlt(points1, points2, None)
        assert H.shape == (B, 3, 3)

    @pytest.mark.parametrize("batch_size, num_points", [(1, 4), (2, 5), (3, 6)], )
    def test_points_noweights(self, batch_size, num_points, device, dtype):
        B, N = batch_size, num_points
        points1 = torch.rand(B, N, 2, device=device, dtype=dtype)
        points2 = torch.rand(B, N, 2, device=device, dtype=dtype)
        weights = torch.ones(B, N, device=device, dtype=dtype)
        H_noweights = find_homography_dlt(points1, points2, None)
        H_withweights = find_homography_dlt(points1, points2, weights)
        assert H_noweights.shape == (B, 3, 3) and H_withweights.shape == (B, 3, 3)
        assert_allclose(H_noweights, H_withweights)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_clean_points(self, batch_size, device, dtype):
        # generate input data
        points_src = torch.rand(batch_size, 10, 2, device=device, dtype=dtype)
        H = kornia.eye_like(3, points_src)
        H = H * 0.3 * torch.rand_like(H)
        H = H / H[:, 2:3, 2:3]

        points_dst = kornia.transform_points(H, points_src)
        weights = torch.ones(batch_size, 10, device=device, dtype=dtype)

        # compute transform from source to target
        dst_homo_src = find_homography_dlt(points_src, points_dst, weights)

        assert_allclose(
            kornia.transform_points(dst_homo_src, points_src), points_dst, rtol=1e-3, atol=1e-4)

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        points_src = torch.rand(1, 10, 2, device=device, dtype=torch.float64, requires_grad=True)
        points_dst = torch.rand_like(points_src)
        weights = torch.ones_like(points_src)[..., 0]
        assert gradcheck(find_homography_dlt, (points_src, points_dst, weights),
                         rtol=1e-1, atol=1e-5, raise_exception=True)


class TestFindHomographyDLTIter:
    def test_smoke(self, device, dtype):
        points1 = torch.rand(1, 4, 2, device=device, dtype=dtype)
        points2 = torch.rand(1, 4, 2, device=device, dtype=dtype)
        weights = torch.ones(1, 4, device=device, dtype=dtype)
        H = find_homography_dlt_iterated(points1, points2, weights, 5)
        assert H.shape == (1, 3, 3)

    @pytest.mark.parametrize(
        "batch_size, num_points", [(1, 4), (2, 5), (3, 6)],
    )
    def test_shape(self, batch_size, num_points, device, dtype):
        B, N = batch_size, num_points
        points1 = torch.rand(B, N, 2, device=device, dtype=dtype)
        points2 = torch.rand(B, N, 2, device=device, dtype=dtype)
        weights = torch.ones(B, N, device=device, dtype=dtype)
        H = find_homography_dlt_iterated(points1, points2, weights, 5)
        assert H.shape == (B, 3, 3)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_clean_points(self, batch_size, device, dtype):
        # generate input data
        points_src = torch.rand(batch_size, 10, 2, device=device, dtype=dtype)
        H = kornia.eye_like(3, points_src)
        H = H * 0.3 * torch.rand_like(H)
        H = H / H[:, 2:3, 2:3]

        points_dst = kornia.transform_points(H, points_src)
        weights = torch.ones(batch_size, 10, device=device, dtype=dtype)

        # compute transform from source to target
        dst_homo_src = find_homography_dlt_iterated(points_src, points_dst, weights, 10)

        assert_allclose(
            kornia.transform_points(dst_homo_src, points_src), points_dst, rtol=1e-3, atol=1e-4)

    # TODO(ducha-aiki): remove once #795 is fixed
    @pytest.mark.xfail
    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        points_src = torch.rand(1, 10, 2, device=device, dtype=torch.float64, requires_grad=True)
        points_dst = torch.rand_like(points_src)
        weights = torch.ones_like(points_src)[..., 0]
        assert gradcheck(find_homography_dlt_iterated, (points_src, points_dst, weights),
                         rtol=1e-3, atol=1e-4, raise_exception=True)

    # TODO(ducha-aiki): remove once #795 is fixed
    @pytest.mark.xfail
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_dirty_points_and_gradcheck(self, batch_size, device, dtype):
        # generate input data
        points_src = torch.rand(batch_size, 10, 2, device=device, dtype=dtype)
        H = kornia.eye_like(3, points_src)
        H = H * 0.3 * torch.rand_like(H)
        H = H / H[:, 2:3, 2:3]

        points_src = 100. * torch.rand(batch_size, 20, 2, device=device, dtype=dtype)
        points_dst = kornia.transform_points(H, points_src)

        # making last point an outlier
        points_dst[:, -1, :] += 20

        weights = torch.ones(batch_size, 20, device=device, dtype=dtype)

        # compute transform from source to target
        dst_homo_src = find_homography_dlt_iterated(points_src, points_dst, weights, 0.5, 10)

        assert_allclose(
            kornia.transform_points(dst_homo_src, points_src[:, :-1]),
            points_dst[:, :-1], rtol=1e-3, atol=1e-3)
