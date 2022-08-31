import random

import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils
from kornia.geometry.homography import (
    find_homography_dlt,
    find_homography_dlt_iterated,
    find_homography_lines_dlt,
    line_segment_transfer_error_one_way,
    oneway_transfer_error,
    sample_is_valid_for_homography,
    symmetric_transfer_error,
)
from kornia.testing import assert_close


class TestSampleValidation:
    def test_good(self, device, dtype):
        pts1 = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], device=device, dtype=dtype)[None]
        mask = sample_is_valid_for_homography(pts1, pts1)
        expected = torch.tensor([True], device=device, dtype=torch.bool)
        assert torch.equal(mask, expected)

    def test_bad(self, device, dtype):
        pts1 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], device=device, dtype=dtype)[None]

        pts2 = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], device=device, dtype=dtype)[None]
        mask = sample_is_valid_for_homography(pts1, pts2)
        expected = torch.tensor([False], device=device, dtype=torch.bool)
        assert torch.equal(mask, expected)

    def test_batch(self, device, dtype):
        batch_size = 5
        pts1 = torch.rand(batch_size, 4, 2, device=device, dtype=dtype)
        pts2 = torch.rand(batch_size, 4, 2, device=device, dtype=dtype)
        mask = sample_is_valid_for_homography(pts1, pts2)
        assert mask.shape == torch.Size([batch_size])


class TestOneWayError:
    def test_smoke(self, device, dtype):
        pts1 = torch.rand(1, 6, 2, device=device, dtype=dtype)
        pts2 = torch.rand(1, 6, 2, device=device, dtype=dtype)
        H = utils.create_random_homography(1, 3).type_as(pts1).to(device)
        assert oneway_transfer_error(pts1, pts2, H).shape == (1, 6)

    def test_batch(self, device, dtype):
        batch_size = 5
        pts1 = torch.rand(batch_size, 3, 2, device=device, dtype=dtype)
        pts2 = torch.rand(batch_size, 3, 2, device=device, dtype=dtype)
        H = utils.create_random_homography(1, 3).type_as(pts1).to(device)
        assert oneway_transfer_error(pts1, pts2, H).shape == (batch_size, 3)

    def test_gradcheck(self, device):
        # generate input data
        batch_size, num_points, num_dims = 2, 3, 2
        points1 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64, requires_grad=True)
        points2 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64)
        H = utils.create_random_homography(batch_size, 3).type_as(points1).to(device)
        assert gradcheck(oneway_transfer_error, (points1, points2, H), raise_exception=True)

    def test_shift(self, device, dtype):
        pts1 = torch.zeros(3, 2, device=device, dtype=dtype)[None]
        pts2 = torch.tensor([[1.0, 0.0], [2.0, 0.0], [2.0, 2.0]], device=device, dtype=dtype)[None]
        H = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype, device=device)[None]
        expected = torch.tensor([0.0, 1.0, 5.0], device=device, dtype=dtype)[None]
        assert_close(oneway_transfer_error(pts1, pts2, H), expected, atol=1e-4, rtol=1e-4)


class TestLineSegmentOneWayError:
    def test_smoke(self, device, dtype):
        ls1 = torch.rand(1, 6, 2, 2, device=device, dtype=dtype)
        ls2 = torch.rand(1, 6, 2, 2, device=device, dtype=dtype)
        H = utils.create_random_homography(1, 3).type_as(ls1).to(device)
        assert line_segment_transfer_error_one_way(ls1, ls2, H).shape == (1, 6)

    def test_batch(self, device, dtype):
        batch_size = 5
        ls1 = torch.rand(batch_size, 3, 2, 2, device=device, dtype=dtype)
        ls2 = torch.rand(batch_size, 3, 2, 2, device=device, dtype=dtype)
        H = utils.create_random_homography(1, 3).type_as(ls1).to(device)
        assert line_segment_transfer_error_one_way(ls1, ls2, H).shape == (batch_size, 3)

    def test_gradcheck(self, device):
        # generate input data
        batch_size, num_points, num_dims = 2, 3, 2
        ls1 = torch.rand(batch_size, num_points, num_dims, 2, device=device, dtype=torch.float64, requires_grad=True)
        ls2 = torch.rand(batch_size, num_points, num_dims, 2, device=device, dtype=torch.float64)
        H = utils.create_random_homography(batch_size, 3).type_as(ls1).to(device)
        assert gradcheck(line_segment_transfer_error_one_way, (ls1, ls2, H), raise_exception=True)

    def test_shift(self, device, dtype):
        pts1 = torch.zeros(3, 2, device=device, dtype=dtype)[None]
        pts1_end = torch.ones(3, 2, device=device, dtype=dtype)[None]
        ls1 = torch.stack([pts1, pts1_end], dim=2)

        pts2 = torch.tensor([[1.0, 0.0], [2.0, 0.0], [2.0, 2.0]], device=device, dtype=dtype)[None]
        pts2_end = pts2 + torch.ones(3, 2, device=device, dtype=dtype)[None]
        ls2 = torch.stack([pts2, pts2_end], dim=2)
        H = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype, device=device)[None]
        expected = torch.tensor([0.0, 1.0, 1.0], device=device, dtype=dtype)[None]
        assert_close(line_segment_transfer_error_one_way(ls1, ls2, H), expected, atol=1e-4, rtol=1e-4)


class TestSymmetricTransferError:
    def test_smoke(self, device, dtype):
        pts1 = torch.rand(1, 6, 2, device=device, dtype=dtype)
        pts2 = torch.rand(1, 6, 2, device=device, dtype=dtype)
        H = utils.create_random_homography(1, 3).type_as(pts1).to(device)
        assert symmetric_transfer_error(pts1, pts2, H).shape == (1, 6)

    def test_batch(self, device, dtype):
        batch_size = 5
        pts1 = torch.rand(batch_size, 3, 2, device=device, dtype=dtype)
        pts2 = torch.rand(batch_size, 3, 2, device=device, dtype=dtype)
        H = utils.create_random_homography(1, 3).type_as(pts1).to(device)
        assert symmetric_transfer_error(pts1, pts2, H).shape == (batch_size, 3)

    def test_gradcheck(self, device):
        # generate input data
        batch_size, num_points, num_dims = 2, 3, 2
        points1 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64, requires_grad=True)
        points2 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64)
        H = utils.create_random_homography(batch_size, 3).type_as(points1).to(device)
        assert gradcheck(symmetric_transfer_error, (points1, points2, H), raise_exception=True)

    def test_shift(self, device, dtype):
        pts1 = torch.zeros(3, 2, device=device, dtype=dtype)[None]
        pts2 = torch.tensor([[1.0, 0.0], [2.0, 0.0], [2.0, 2.0]], device=device, dtype=dtype)[None]
        H = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype, device=device)[None]
        expected = torch.tensor([0.0, 2.0, 10.0], device=device, dtype=dtype)[None]
        assert_close(symmetric_transfer_error(pts1, pts2, H), expected, atol=1e-4, rtol=1e-4)


class TestFindHomographyDLT:
    def test_smoke(self, device, dtype):
        points1 = torch.rand(1, 4, 2, device=device, dtype=dtype)
        points2 = torch.rand(1, 4, 2, device=device, dtype=dtype)
        weights = torch.ones(1, 4, device=device, dtype=dtype)
        H = find_homography_dlt(points1, points2, weights)
        assert H.shape == (1, 3, 3)

    def test_nocrash(self, device, dtype):
        points1 = torch.rand(1, 4, 2, device=device, dtype=dtype)
        points2 = torch.rand(1, 4, 2, device=device, dtype=dtype)
        weights = torch.ones(1, 4, device=device, dtype=dtype)
        points1[0, 0, 0] = float('nan')
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

        points_dst = kornia.geometry.transform_points(H, points_src)
        weights = torch.ones(batch_size, 10, device=device, dtype=dtype)

        # compute transform from source to target
        dst_homo_src = find_homography_dlt(points_src, points_dst, weights)

        assert_close(kornia.geometry.transform_points(dst_homo_src, points_src), points_dst, rtol=1e-3, atol=1e-4)

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


class TestFindHomographyFromLinesDLT:
    def test_smoke(self, device, dtype):
        points1st = torch.rand(1, 4, 2, device=device, dtype=dtype)
        points1end = torch.rand(1, 4, 2, device=device, dtype=dtype)
        points2st = torch.rand(1, 4, 2, device=device, dtype=dtype)
        points2end = torch.rand(1, 4, 2, device=device, dtype=dtype)
        weights = torch.ones(1, 4, device=device, dtype=dtype)
        ls1 = torch.stack([points1st, points1end], dim=2)
        ls2 = torch.stack([points2st, points2end], dim=2)
        H = find_homography_lines_dlt(ls1, ls2, weights)
        assert H.shape == (1, 3, 3)

    def test_smoke2(self, device, dtype):
        points1st = torch.rand(4, 2, device=device, dtype=dtype)
        points1end = torch.rand(4, 2, device=device, dtype=dtype)
        points2st = torch.rand(4, 2, device=device, dtype=dtype)
        points2end = torch.rand(4, 2, device=device, dtype=dtype)
        ls1 = torch.stack([points1st, points1end], dim=1)
        ls2 = torch.stack([points2st, points2end], dim=1)
        H = find_homography_lines_dlt(ls1, ls2, None)
        assert H.shape == (1, 3, 3)

    def test_nocrash(self, device, dtype):
        points1st = torch.rand(1, 4, 2, device=device, dtype=dtype)
        points1end = torch.rand(1, 4, 2, device=device, dtype=dtype)
        points2st = torch.rand(1, 4, 2, device=device, dtype=dtype)
        points2end = torch.rand(1, 4, 2, device=device, dtype=dtype)
        weights = torch.ones(1, 4, device=device, dtype=dtype)
        points1st[0, 0, 0] = float('nan')
        ls1 = torch.stack([points1st, points1end], dim=2)
        ls2 = torch.stack([points2st, points2end], dim=2)
        H = find_homography_lines_dlt(ls1, ls2, weights)
        assert H.shape == (1, 3, 3)

    @pytest.mark.parametrize("batch_size, num_points", [(1, 4), (2, 5), (3, 6)])
    def test_shape(self, batch_size, num_points, device, dtype):
        B, N = batch_size, num_points
        points1st = torch.rand(B, N, 2, device=device, dtype=dtype)
        points1end = torch.rand(B, N, 2, device=device, dtype=dtype)
        points2st = torch.rand(B, N, 2, device=device, dtype=dtype)
        points2end = torch.rand(B, N, 2, device=device, dtype=dtype)
        weights = torch.ones(B, N, device=device, dtype=dtype)
        ls1 = torch.stack([points1st, points1end], dim=2)
        ls2 = torch.stack([points2st, points2end], dim=2)
        H = find_homography_lines_dlt(ls1, ls2, weights)
        assert H.shape == (B, 3, 3)

    @pytest.mark.parametrize("batch_size, num_points", [(1, 4), (2, 5), (3, 6)])
    def test_shape_noweights(self, batch_size, num_points, device, dtype):
        B, N = batch_size, num_points
        points1st = torch.rand(B, N, 2, device=device, dtype=dtype)
        points1end = torch.rand(B, N, 2, device=device, dtype=dtype)
        points2st = torch.rand(B, N, 2, device=device, dtype=dtype)
        points2end = torch.rand(B, N, 2, device=device, dtype=dtype)
        ls1 = torch.stack([points1st, points1end], dim=2)
        ls2 = torch.stack([points2st, points2end], dim=2)
        H = find_homography_lines_dlt(ls1, ls2, None)
        assert H.shape == (B, 3, 3)

    @pytest.mark.parametrize("batch_size, num_points", [(1, 4), (2, 5), (3, 6)])
    def test_points_noweights(self, batch_size, num_points, device, dtype):
        B, N = batch_size, num_points
        points1st = torch.rand(B, N, 2, device=device, dtype=dtype)
        points1end = torch.rand(B, N, 2, device=device, dtype=dtype)
        points2st = torch.rand(B, N, 2, device=device, dtype=dtype)
        points2end = torch.rand(B, N, 2, device=device, dtype=dtype)
        weights = torch.ones(B, N, device=device, dtype=dtype)
        ls1 = torch.stack([points1st, points1end], dim=2)
        ls2 = torch.stack([points2st, points2end], dim=2)
        H_noweights = find_homography_lines_dlt(ls1, ls2, None)
        H_withweights = find_homography_lines_dlt(ls1, ls2, weights)
        assert H_noweights.shape == (B, 3, 3) and H_withweights.shape == (B, 3, 3)
        assert_close(H_noweights, H_withweights, rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_clean_points(self, batch_size, device, dtype):
        # generate input data
        points_src_st = torch.rand(batch_size, 10, 2, device=device, dtype=dtype)
        points_src_end = torch.rand(batch_size, 10, 2, device=device, dtype=dtype)

        H = kornia.eye_like(3, points_src_st)
        H = H * 0.3 * torch.rand_like(H)
        H = H / H[:, 2:3, 2:3]
        points_dst_st = kornia.geometry.transform_points(H, points_src_st)
        points_dst_end = kornia.geometry.transform_points(H, points_src_end)

        ls1 = torch.stack([points_src_st, points_src_end], axis=2)
        ls2 = torch.stack([points_dst_st, points_dst_end], axis=2)
        # compute transform from source to target
        dst_homo_src = find_homography_lines_dlt(ls1, ls2, None)

        assert_close(kornia.geometry.transform_points(dst_homo_src, points_src_st), points_dst_st, rtol=1e-3, atol=1e-4)

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
            points_src_st = torch.rand(1, 10, 2, device=device, dtype=torch.float64, requires_grad=True)
            points_src_end = torch.rand(1, 10, 2, device=device, dtype=torch.float64, requires_grad=True)

            points_dst_st = torch.rand_like(points_src_st)
            points_dst_end = torch.rand_like(points_src_end)
            weights = torch.ones_like(points_src_st)[..., 0]
            ls1 = torch.stack([points_src_st, points_src_end], axis=2)
            ls2 = torch.stack([points_dst_st, points_dst_end], axis=2)
            try:
                gradcheck(find_homography_lines_dlt, (ls1, ls2, weights), rtol=1e-6, atol=1e-6, raise_exception=True)

            # Gradcheck failed
            except RuntimeError:

                # All iterations failed
                if i == max_number_of_checks - 1:
                    assert gradcheck(
                        find_homography_lines_dlt, (ls1, ls2, weights), rtol=1e-6, atol=1e-6, raise_exception=True
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

        points_dst = kornia.geometry.transform_points(H, points_src)
        weights = torch.ones(batch_size, 10, device=device, dtype=dtype)

        # compute transform from source to target
        dst_homo_src = find_homography_dlt_iterated(points_src, points_dst, weights, 10)

        assert_close(kornia.geometry.transform_points(dst_homo_src, points_src), points_dst, rtol=1e-3, atol=1e-4)

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
        H = H * (1 + torch.rand_like(H))
        H = H / H[:, 2:3, 2:3]

        points_src = 100.0 * torch.rand(batch_size, 20, 2, device=device, dtype=dtype)
        points_dst = kornia.geometry.transform_points(H, points_src)

        # making last point an outlier
        points_dst[:, -1, :] += 20

        weights = torch.ones(batch_size, 20, device=device, dtype=dtype)

        # compute transform from source to target
        dst_homo_src = find_homography_dlt_iterated(points_src, points_dst, weights, 0.5, 10)

        assert_close(
            kornia.geometry.transform_points(dst_homo_src, points_src[:, :-1]), points_dst[:, :-1], rtol=1e-3, atol=1e-3
        )
