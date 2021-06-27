import random

import pytest
import torch
from torch.autograd import gradcheck

import kornia
from kornia.geometry.homography import find_homography_dlt, find_homography_dlt_iterated
from kornia.testing import assert_close


class TestFindHomographyDLT:
    def test_smoke(self, device, dtype):
        points1 = torch.rand(1, 4, 2, device=device, dtype=dtype)
        points2 = torch.rand(1, 4, 2, device=device, dtype=dtype)
        weights = torch.ones(1, 4, device=device, dtype=dtype)
        H = find_homography_dlt(points1, points2, weights)
        assert H.shape == (1, 3, 3)

    @pytest.mark.parametrize("batch_size, num_points", [(1, 4), (2, 5), (3, 6)])
    def test_shape(self, batch_size, num_points, device, dtype):
        B, N = batch_size, num_points
        points1 = torch.rand(B, N, 2, device=device, dtype=dtype)
        points2 = torch.rand(B, N, 2, device=device, dtype=dtype)
        weights = torch.ones(B, N, device=device, dtype=dtype)
        H = find_homography_dlt(points1, points2, weights)
        assert H.shape == (B, 3, 3)

    @pytest.mark.parametrize("batch_size, num_points", [(1, 4), (2, 5), (3, 6)])
    def test_shape_noweights(self, batch_size, num_points, device, dtype):
        B, N = batch_size, num_points
        points1 = torch.rand(B, N, 2, device=device, dtype=dtype)
        points2 = torch.rand(B, N, 2, device=device, dtype=dtype)
        H = find_homography_dlt(points1, points2, None)
        assert H.shape == (B, 3, 3)

    @pytest.mark.parametrize("batch_size, num_points", [(1, 4), (2, 5), (3, 6)])
    def test_points_noweights(self, batch_size, num_points, device, dtype):
        B, N = batch_size, num_points
        points1 = torch.rand(B, N, 2, device=device, dtype=dtype)
        points2 = torch.rand(B, N, 2, device=device, dtype=dtype)
        weights = torch.ones(B, N, device=device, dtype=dtype)
        H_noweights = find_homography_dlt(points1, points2, None)
        H_withweights = find_homography_dlt(points1, points2, weights)
        assert H_noweights.shape == (B, 3, 3) and H_withweights.shape == (B, 3, 3)
        assert_close(H_noweights, H_withweights, rtol=1e-3, atol=1e-4)

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

        assert_close(kornia.transform_points(dst_homo_src, points_src), points_dst, rtol=1e-3, atol=1e-4)

    @pytest.mark.grad
    @pytest.mark.skipif(torch.__version__ < '1.7', reason="pytorch bug of incopatible types: #33546 fixed in v1.7")
    def test_gradcheck(self, device):
        # Save initial seed
        initial_seed = torch.random.initial_seed()
        max_number_of_checks = 10

        # Test gradients for a max_number_of_checks times
        current_seed = initial_seed
        for i in range(max_number_of_checks):
            torch.manual_seed(current_seed)
            points_src = torch.rand(1, 10, 2, device=device, dtype=torch.float64, requires_grad=True)
            points_dst = torch.rand_like(points_src)
            weights = torch.ones_like(points_src)[..., 0]
            try:
                gradcheck(
                    find_homography_dlt, (points_src, points_dst, weights), rtol=1e-6, atol=1e-6, raise_exception=True
                )

            # Gradcheck failed
            except RuntimeError:

                # All iterations failed
                if i == max_number_of_checks - 1:
                    assert gradcheck(
                        find_homography_dlt,
                        (points_src, points_dst, weights),
                        rtol=1e-6,
                        atol=1e-6,
                        raise_exception=True,
                    )
                # Next iteration
                else:
                    current_seed = random.randrange(0xFFFFFFFFFFFFFFFF)
                    continue

            # Gradcheck succeed
            torch.manual_seed(initial_seed)
            return


class TestFindHomographyDLTIter:
    def test_smoke(self, device, dtype):
        points1 = torch.rand(1, 4, 2, device=device, dtype=dtype)
        points2 = torch.rand(1, 4, 2, device=device, dtype=dtype)
        weights = torch.ones(1, 4, device=device, dtype=dtype)
        H = find_homography_dlt_iterated(points1, points2, weights, 5)
        assert H.shape == (1, 3, 3)

    @pytest.mark.parametrize("batch_size, num_points", [(1, 4), (2, 5), (3, 6)])
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

        assert_close(kornia.transform_points(dst_homo_src, points_src), points_dst, rtol=1e-3, atol=1e-4)

    @pytest.mark.grad
    @pytest.mark.skipif(torch.__version__ < '1.7', reason="pytorch bug of incopatible types: #33546 fixed in v1.7")
    def test_gradcheck(self, device):

        # Save initial seed
        initial_seed = torch.random.initial_seed()
        max_number_of_checks = 10

        # Test gradients for a max_number_of_checks times
        current_seed = initial_seed
        for i in range(max_number_of_checks):
            torch.manual_seed(current_seed)
            points_src = torch.rand(1, 10, 2, device=device, dtype=torch.float64, requires_grad=True)
            points_dst = torch.rand_like(points_src)
            weights = torch.ones_like(points_src)[..., 0]
            try:
                gradcheck(
                    find_homography_dlt_iterated,
                    (points_src, points_dst, weights),
                    rtol=1e-6,
                    atol=1e-6,
                    raise_exception=True,
                )

            # Gradcheck failed
            except RuntimeError:

                # All iterations failed
                if i == max_number_of_checks - 1:
                    assert gradcheck(
                        find_homography_dlt_iterated,
                        (points_src, points_dst, weights),
                        rtol=1e-6,
                        atol=1e-6,
                        raise_exception=True,
                    )
                # Next iteration
                else:
                    current_seed = random.randrange(0xFFFFFFFFFFFFFFFF)
                    continue

            # Gradcheck succeed
            torch.manual_seed(initial_seed)
            return

    @pytest.mark.grad
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_dirty_points_and_gradcheck(self, batch_size, device, dtype):
        # generate input data
        points_src = torch.rand(batch_size, 10, 2, device=device, dtype=dtype)
        H = kornia.eye_like(3, points_src)
        H = H * 0.3 * torch.rand_like(H)
        H = H / H[:, 2:3, 2:3]

        points_src = 100.0 * torch.rand(batch_size, 20, 2, device=device, dtype=dtype)
        points_dst = kornia.transform_points(H, points_src)

        # making last point an outlier
        points_dst[:, -1, :] += 20

        weights = torch.ones(batch_size, 20, device=device, dtype=dtype)

        # compute transform from source to target
        dst_homo_src = find_homography_dlt_iterated(points_src, points_dst, weights, 0.5, 10)

        assert_close(
            kornia.transform_points(dst_homo_src, points_src[:, :-1]), points_dst[:, :-1], rtol=1e-3, atol=1e-3
        )
