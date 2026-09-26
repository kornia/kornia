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

import sys

import pytest
import torch

import kornia
from kornia.core._compat import torch_version_le
from kornia.geometry.homography import (
    find_homography_dlt,
    find_homography_dlt_iterated,
    find_homography_lines_dlt,
    find_homography_lines_dlt_iterated,
    line_segment_transfer_error_one_way,
    oneway_transfer_error,
    sample_is_valid_for_homography,
    symmetric_transfer_error,
)

from testing.base import BaseTester
from testing.geometry.create import create_random_homography


class TestSampleValidation(BaseTester):
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


class TestOneWayError(BaseTester):
    def test_smoke(self, device, dtype):
        pts1 = torch.rand(1, 6, 2, device=device, dtype=dtype)
        pts2 = torch.rand(1, 6, 2, device=device, dtype=dtype)
        H = create_random_homography(pts1, 3)
        assert oneway_transfer_error(pts1, pts2, H).shape == (1, 6)

    def test_batch(self, device, dtype):
        batch_size = 5
        pts1 = torch.rand(batch_size, 3, 2, device=device, dtype=dtype)
        pts2 = torch.rand(batch_size, 3, 2, device=device, dtype=dtype)
        H = create_random_homography(pts1, 3)
        assert oneway_transfer_error(pts1, pts2, H).shape == (batch_size, 3)

    def test_gradcheck(self, device):
        # generate input data
        batch_size, num_points, num_dims = 2, 3, 2
        points1 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64, requires_grad=True)
        points2 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64)
        H = create_random_homography(points1, 3)
        self.gradcheck(oneway_transfer_error, (points1, points2, H))

    def test_shift(self, device, dtype):
        pts1 = torch.zeros(3, 2, device=device, dtype=dtype)[None]
        pts2 = torch.tensor([[1.0, 0.0], [2.0, 0.0], [2.0, 2.0]], device=device, dtype=dtype)[None]
        H = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype, device=device)[None]
        expected = torch.tensor([0.0, 1.0, 5.0], device=device, dtype=dtype)[None]
        self.assert_close(oneway_transfer_error(pts1, pts2, H), expected, atol=1e-4, rtol=1e-4)


class TestLineSegmentOneWayError(BaseTester):
    def test_smoke(self, device, dtype):
        ls1 = torch.rand(1, 6, 2, 2, device=device, dtype=dtype)
        ls2 = torch.rand(1, 6, 2, 2, device=device, dtype=dtype)
        H = create_random_homography(ls1, 3)
        assert line_segment_transfer_error_one_way(ls1, ls2, H).shape == (1, 6)

    def test_batch(self, device, dtype):
        batch_size = 5
        ls1 = torch.rand(batch_size, 3, 2, 2, device=device, dtype=dtype)
        ls2 = torch.rand(batch_size, 3, 2, 2, device=device, dtype=dtype)
        H = create_random_homography(ls1, 3)
        assert line_segment_transfer_error_one_way(ls1, ls2, H).shape == (batch_size, 3)

    def test_gradcheck(self, device):
        # generate input data
        batch_size, num_points, num_dims = 2, 3, 2
        ls1 = torch.rand(batch_size, num_points, num_dims, 2, device=device, dtype=torch.float64, requires_grad=True)
        ls2 = torch.rand(batch_size, num_points, num_dims, 2, device=device, dtype=torch.float64)
        H = create_random_homography(ls1, 3)
        self.gradcheck(line_segment_transfer_error_one_way, (ls1, ls2, H))

    def test_shift(self, device, dtype):
        pts1 = torch.zeros(3, 2, device=device, dtype=dtype)[None]
        pts1_end = torch.ones(3, 2, device=device, dtype=dtype)[None]
        ls1 = torch.stack([pts1, pts1_end], dim=2)

        pts2 = torch.tensor([[1.0, 0.0], [2.0, 0.0], [2.0, 2.0]], device=device, dtype=dtype)[None]
        pts2_end = pts2 + torch.ones(3, 2, device=device, dtype=dtype)[None]
        ls2 = torch.stack([pts2, pts2_end], dim=2)
        H = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype, device=device)[None]
        expected = torch.tensor([0.0, 1.0, 1.0], device=device, dtype=dtype)[None]
        self.assert_close(line_segment_transfer_error_one_way(ls1, ls2, H), expected, atol=1e-4, rtol=1e-4)


class TestSymmetricTransferError(BaseTester):
    def test_smoke(self, device, dtype):
        pts1 = torch.rand(1, 6, 2, device=device, dtype=dtype)
        pts2 = torch.rand(1, 6, 2, device=device, dtype=dtype)
        H = create_random_homography(pts1, 3)
        assert symmetric_transfer_error(pts1, pts2, H).shape == (1, 6)

    def test_batch(self, device, dtype):
        batch_size = 5
        pts1 = torch.rand(batch_size, 3, 2, device=device, dtype=dtype)
        pts2 = torch.rand(batch_size, 3, 2, device=device, dtype=dtype)
        H = create_random_homography(pts1, 3)
        assert symmetric_transfer_error(pts1, pts2, H).shape == (batch_size, 3)

    def test_gradcheck(self, device):
        # generate input data
        batch_size, num_points, num_dims = 2, 3, 2
        points1 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64, requires_grad=True)
        points2 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64)
        H = create_random_homography(points1, 3)
        self.gradcheck(symmetric_transfer_error, (points1, points2, H))

    def test_shift(self, device, dtype):
        pts1 = torch.zeros(3, 2, device=device, dtype=dtype)[None]
        pts2 = torch.tensor([[1.0, 0.0], [2.0, 0.0], [2.0, 2.0]], device=device, dtype=dtype)[None]
        H = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype, device=device)[None]
        expected = torch.tensor([0.0, 2.0, 10.0], device=device, dtype=dtype)[None]
        self.assert_close(symmetric_transfer_error(pts1, pts2, H), expected, atol=1e-4, rtol=1e-4)

    def test_singular(self, device, dtype):
        max_num = torch.finfo(dtype).max
        pts1 = torch.rand(2, 5, 2, device=device, dtype=dtype, requires_grad=True)
        pts2 = torch.rand(2, 5, 2, device=device, dtype=dtype)
        H = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(2, 1, 1)
        # Make second homography in the batch singular
        H[1] = 0.0
        H[1, 0, 0] = 1.0

        for squared in (True, False):
            err = symmetric_transfer_error(pts1, pts2, H, squared=squared)
            assert not torch.isnan(err).any()
            assert torch.isfinite(err[0]).all()
            expected_singular = torch.full_like(err[1], max_num)
            self.assert_close(err[1], expected_singular)

        # Check gradient finiteness across mixed batch
        err = symmetric_transfer_error(pts1, pts2, H)
        err[0].sum().backward()
        assert pts1.grad is not None
        assert not torch.isnan(pts1.grad).any()
        assert torch.isfinite(pts1.grad[0]).all()
        assert (pts1.grad[1] == 0.0).all()

    def test_singular_gradient_wrt_homography(self, device, dtype):
        pts1 = torch.rand(2, 5, 2, device=device, dtype=dtype)
        pts2 = torch.rand(2, 5, 2, device=device, dtype=dtype)
        H = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(2, 1, 1)
        H[1] = 0.0
        H[1, 0, 0] = 1.0
        H.requires_grad_(True)

        symmetric_transfer_error(pts1, pts2, H)[0].sum().backward()
        assert H.grad is not None
        assert torch.isfinite(H.grad).all()
        assert (H.grad[1] == 0.0).all()

    def test_singular_gradient_through_estimated_homography(self, device, dtype):
        # ``find_homography_dlt_iterated`` differentiates the error through an estimated ``H``, so a
        # singular intermediate must not poison the gradient of the correspondences that produced it.
        pts1 = torch.rand(2, 5, 2, device=device, dtype=dtype, requires_grad=True)
        pts2 = torch.rand(2, 5, 2, device=device, dtype=dtype)
        eye = torch.eye(3, device=device, dtype=dtype)
        zeros = torch.zeros(3, 3, device=device, dtype=dtype)
        H = torch.stack([eye * pts1[0, 0, 0], zeros * pts1[1, 0, 0]])

        symmetric_transfer_error(pts1, pts2, H).sum().backward()
        assert pts1.grad is not None
        assert torch.isfinite(pts1.grad).all()


class TestFindHomographyDLT(BaseTester):
    def test_smoke(self, device, dtype):
        points1 = torch.rand(1, 4, 2, device=device, dtype=dtype)
        points2 = torch.rand(1, 4, 2, device=device, dtype=dtype)
        weights = torch.ones(1, 4, device=device, dtype=dtype)
        H = find_homography_dlt(points1, points2, weights)
        assert H.shape == (1, 3, 3)

    # A NaN in a minimal sample used to make torch.linalg.qr spin forever on CUDA (#4770). The
    # thread method aborts the session instead of letting a stuck kernel stall the whole run.
    @pytest.mark.timeout(120, method="thread")
    def test_nocrash(self, device, dtype):
        points1 = torch.rand(1, 4, 2, device=device, dtype=dtype)
        points2 = torch.rand(1, 4, 2, device=device, dtype=dtype)
        weights = torch.ones(1, 4, device=device, dtype=dtype)
        points1[0, 0, 0] = float("nan")
        H = find_homography_dlt(points1, points2, weights)
        assert H.shape == (1, 3, 3)
        # Reading the values synchronizes the device, so a hang surfaces in this test (#4770).
        assert H.isnan().all().item()

    @pytest.mark.timeout(120, method="thread")
    def test_nocrash_lu(self, device, dtype):
        points1 = torch.rand(1, 4, 2, device=device, dtype=dtype)
        points2 = torch.rand(1, 4, 2, device=device, dtype=dtype)
        weights = torch.ones(1, 4, device=device, dtype=dtype)
        points1[0, 0, 0] = float("nan")
        H = find_homography_dlt(points1, points2, weights, "lu")
        assert H.shape == (1, 3, 3)
        assert H.isnan().all().item()

    @pytest.mark.timeout(120, method="thread")
    def test_nonfinite_sample_leaves_batch_intact(self, device, dtype):
        points1 = torch.rand(3, 4, 2, device=device, dtype=dtype)
        points2 = torch.rand(3, 4, 2, device=device, dtype=dtype)
        weights = torch.ones(3, 4, device=device, dtype=dtype)
        expected = find_homography_dlt(points1[:1], points2[:1], weights[:1])
        points1[1, 0, 0] = float("nan")
        weights[2, 1] = float("inf")
        H = find_homography_dlt(points1, points2, weights)
        assert H[1:].isnan().all().item()
        self.assert_close(H[:1], expected)

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
        assert H_noweights.shape == (B, 3, 3)
        assert H_withweights.shape == (B, 3, 3)
        self.assert_close(H_noweights, H_withweights, rtol=1e-3, atol=1e-4)

    def test_scaled_fixed_points(self, device, dtype):
        points1 = torch.tensor([[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]], device=device, dtype=dtype)
        points2 = points1 * 100
        for weights in (None, torch.ones(1, 4, device=device, dtype=dtype)):
            H = find_homography_dlt(points1, points2, weights, "lu")
            assert torch.isfinite(H).all()
            self.assert_close(kornia.geometry.transform_points(H, points1), points2, rtol=1e-4, atol=1e-4)

    def test_projective_fixed_points(self, device):
        # Guards the adaptive gauge itself rather than the Windows/torch-2.14 NaN: for this
        # configuration the normalized-frame null vector has last component ~-1.3e-16, so a
        # fixed h33=1 gauge is singular here. It passes against the pre-gauge implementation,
        # which reached a finite (if badly scaled) answer on every platform but Windows.
        dtype = torch.float32 if device.type == "mps" else torch.float64
        points1 = torch.tensor([[[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]], device=device, dtype=dtype)
        points2 = torch.tensor([[[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]]], device=device, dtype=dtype)

        H = find_homography_dlt(points1, points2, None, "lu")

        assert torch.isfinite(H).all()
        self.assert_close(kornia.geometry.transform_points(H, points1), points2, rtol=1e-4, atol=1e-4)

    def test_zero_weight_minimal_lu(self, device, dtype):
        # A zero weight zeroes two rows of the design matrix, leaving the retained 8x8 system
        # singular. The result must stay finite -- a NaN homography propagates silently through
        # RANSAC verification -- and a healthy batch entry must be unaffected by a degenerate one.
        points1 = torch.tensor([[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]], device=device, dtype=dtype)
        points2 = points1 * 2.0 + 0.1

        for weights in ([1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]):
            H = find_homography_dlt(points1, points2, torch.tensor([weights], device=device, dtype=dtype), "lu")
            assert torch.isfinite(H).all()

        healthy = find_homography_dlt(points1, points2, torch.ones(1, 4, device=device, dtype=dtype), "lu")
        mixed = find_homography_dlt(
            points1.repeat(2, 1, 1),
            points2.repeat(2, 1, 1),
            torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0]], device=device, dtype=dtype),
            "lu",
        )
        assert torch.isfinite(mixed).all()
        self.assert_close(mixed[0], healthy[0])

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_clean_points_svd(self, batch_size, device, dtype):
        # generate input data
        points_src = torch.rand(batch_size, 10, 2, device=device, dtype=dtype)
        H = kornia.core.ops.eye_like(3, points_src)
        H = H * 0.3 * torch.rand_like(H)
        H = H / H[:, 2:3, 2:3]

        points_dst = kornia.geometry.transform_points(H, points_src)
        weights = torch.ones(batch_size, 10, device=device, dtype=dtype)

        # compute transform from source to target
        dst_homo_src = find_homography_dlt(points_src, points_dst, weights, "svd")
        rtol = 1e-3
        atol = 1e-4
        if dtype not in (torch.float32, torch.float64):
            rtol = 3e-3
            atol = 1e-3
        elif device.type == "cuda" and dtype == torch.float32:
            atol = 2e-3
        self.assert_close(kornia.geometry.transform_points(dst_homo_src, points_src), points_dst, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_clean_points_lu(self, batch_size, device, dtype):
        # generate input data
        points_src = torch.rand(batch_size, 10, 2, device=device, dtype=dtype)
        H = kornia.core.ops.eye_like(3, points_src)
        H = H * 0.3 * torch.rand_like(H)
        H = H / H[:, 2:3, 2:3]

        points_dst = kornia.geometry.transform_points(H, points_src)
        weights = torch.ones(batch_size, 10, device=device, dtype=dtype)

        # compute transform from source to target
        dst_homo_src = find_homography_dlt(points_src, points_dst, weights, "lu")
        rtol = 1e-3
        atol = 1e-4
        if dtype not in (torch.float32, torch.float64):
            rtol = 3e-3
            atol = 1e-3
        elif device.type == "cuda" and dtype == torch.float32:
            atol = 2e-3
        self.assert_close(kornia.geometry.transform_points(dst_homo_src, points_src), points_dst, rtol=rtol, atol=atol)

    def test_gradcheck(self, device):
        points_src = torch.rand(1, 10, 2, device=device, dtype=torch.float64, requires_grad=True)
        points_dst = torch.rand_like(points_src)
        weights = torch.ones_like(points_src)[..., 0]

        self.gradcheck(find_homography_dlt, (points_src, points_dst, weights), rtol=1e-6, atol=1e-6)

    def test_gradcheck_lu(self, device):
        points_src = torch.rand(1, 10, 2, device=device, dtype=torch.float64, requires_grad=True)

        points_dst = torch.rand_like(points_src)
        weights = torch.ones_like(points_src)[..., 0]
        self.gradcheck(find_homography_dlt, (points_src, points_dst, weights, "lu"), rtol=1e-6, atol=1e-6)

    def test_gradcheck_lu_minimal(self, device):
        points_src = torch.tensor(
            [[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]],
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        points_dst = torch.tensor(
            [[[0.1, 0.2], [0.2, 1.4], [1.3, 0.1], [1.1, 1.2]]], device=device, dtype=torch.float64
        )

        self.gradcheck(find_homography_dlt, (points_src, points_dst, None, "lu"), rtol=1e-6, atol=1e-6)


class TestFindHomographyFromLinesDLT(BaseTester):
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
        points1st[0, 0, 0] = float("nan")
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
        assert H_noweights.shape == (B, 3, 3)
        assert H_withweights.shape == (B, 3, 3)
        # On CUDA with float32, A^T A and A^T diag(1) A use different cuBLAS call sequences,
        # so TF32 rounding accumulates differently.  Use the same relaxed tolerance as
        # test_clean_points does for this device+dtype combination.
        rtol, atol = 1e-3, 1e-4
        if device.type == "cuda" and dtype == torch.float32:
            rtol, atol = 5e-3, 5e-3
        self.assert_close(H_noweights, H_withweights, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_clean_points(self, batch_size, device, dtype):
        # generate input data
        points_src_st = torch.rand(batch_size, 10, 2, device=device, dtype=dtype)
        points_src_end = torch.rand(batch_size, 10, 2, device=device, dtype=dtype)

        H = kornia.core.ops.eye_like(3, points_src_st)
        H = H * 0.3 * torch.rand_like(H)
        H = H / H[:, 2:3, 2:3]
        points_dst_st = kornia.geometry.transform_points(H, points_src_st)
        points_dst_end = kornia.geometry.transform_points(H, points_src_end)

        ls1 = torch.stack([points_src_st, points_src_end], axis=2)
        ls2 = torch.stack([points_dst_st, points_dst_end], axis=2)
        # compute transform from source to target
        dst_homo_src = find_homography_lines_dlt(ls1, ls2, None)
        rtol = 1e-3
        atol = 1e-4
        if dtype not in (torch.float32, torch.float64):
            rtol = 5e-3
            atol = 1e-3
        elif device.type == "cuda" and dtype == torch.float32:
            atol = 5e-3
        self.assert_close(
            kornia.geometry.transform_points(dst_homo_src, points_src_st), points_dst_st, rtol=rtol, atol=atol
        )

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_clean_points_iter(self, batch_size, device, dtype):
        # generate input data
        points_src_st = torch.rand(batch_size, 10, 2, device=device, dtype=dtype)
        points_src_end = torch.rand(batch_size, 10, 2, device=device, dtype=dtype)

        H = kornia.core.ops.eye_like(3, points_src_st)
        H = H * 0.3 * torch.rand_like(H)
        H = H / H[:, 2:3, 2:3]
        points_dst_st = kornia.geometry.transform_points(H, points_src_st)
        points_dst_end = kornia.geometry.transform_points(H, points_src_end)

        ls1 = torch.stack([points_src_st, points_src_end], axis=2)
        ls2 = torch.stack([points_dst_st, points_dst_end], axis=2)
        # compute transform from source to target
        dst_homo_src = find_homography_lines_dlt_iterated(ls1, ls2, None, 5)
        rtol = 1e-3
        atol = 1e-4
        if dtype not in (torch.float32, torch.float64):
            rtol = 5e-3
            atol = 1e-3
        elif device.type == "cuda" and dtype == torch.float32:
            atol = 5e-3
        self.assert_close(
            kornia.geometry.transform_points(dst_homo_src, points_src_st), points_dst_st, rtol=rtol, atol=atol
        )

    def test_gradcheck(self, device):
        points_src_st = torch.rand(1, 10, 2, device=device, dtype=torch.float64, requires_grad=True)
        points_src_end = torch.rand(1, 10, 2, device=device, dtype=torch.float64, requires_grad=True)

        points_dst_st = torch.rand_like(points_src_st)
        points_dst_end = torch.rand_like(points_src_end)
        weights = torch.ones_like(points_src_st)[..., 0]
        ls1 = torch.stack([points_src_st, points_src_end], axis=2)
        ls2 = torch.stack([points_dst_st, points_dst_end], axis=2)

        self.gradcheck(find_homography_lines_dlt, (ls1, ls2, weights), rtol=1e-6, atol=1e-6)


class TestFindHomographyDLTIter(BaseTester):
    def test_smoke(self, device, dtype):
        points1 = torch.rand(1, 4, 2, device=device, dtype=dtype)
        points2 = torch.rand(1, 4, 2, device=device, dtype=dtype)
        weights = torch.ones(1, 4, device=device, dtype=dtype)
        H = find_homography_dlt_iterated(points1, points2, weights, 5)
        assert H.shape == (1, 3, 3)

    @pytest.mark.parametrize("batch_size, num_points", [(1, 4), (2, 5), (3, 6)])
    @pytest.mark.skipif(
        sys.platform == "darwin" and torch_version_le(1, 9, 1), reason="Known bug in torch 1.9.1 on macos"
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
        H = kornia.core.ops.eye_like(3, points_src)
        H = H * 0.3 * torch.rand_like(H)
        H = H / H[:, 2:3, 2:3]

        points_dst = kornia.geometry.transform_points(H, points_src)
        weights = torch.ones(batch_size, 10, device=device, dtype=dtype)

        # compute transform from source to target
        dst_homo_src = find_homography_dlt_iterated(points_src, points_dst, weights, 10)

        atol = 2e-3 if (device.type == "cuda" and dtype == torch.float32) else 1e-4
        self.assert_close(kornia.geometry.transform_points(dst_homo_src, points_src), points_dst, rtol=1e-3, atol=atol)

    def test_gradcheck(self, device):
        torch.manual_seed(0)
        points_src = torch.rand(1, 10, 2, device=device, dtype=torch.float64, requires_grad=True)
        points_dst = torch.rand_like(points_src)
        weights = torch.ones_like(points_src)[..., 0]
        self.gradcheck(find_homography_dlt_iterated, (points_src, points_dst, weights), rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_dirty_points_and_gradcheck(self, batch_size, device, dtype):
        # generate input data
        points_src = torch.rand(batch_size, 10, 2, device=device, dtype=dtype)
        H = kornia.core.ops.eye_like(3, points_src)
        H = H * (1 + torch.rand_like(H))
        H = H / H[:, 2:3, 2:3]

        points_src = 100.0 * torch.rand(batch_size, 20, 2, device=device, dtype=dtype)
        points_dst = kornia.geometry.transform_points(H, points_src)

        # making last point an outlier
        points_dst[:, -1, :] += 20

        weights = torch.ones(batch_size, 20, device=device, dtype=dtype)

        # compute transform from source to target
        dst_homo_src = find_homography_dlt_iterated(points_src, points_dst, weights, 0.5, 10)

        self.assert_close(
            kornia.geometry.transform_points(dst_homo_src, points_src[:, :-1]), points_dst[:, :-1], rtol=1e-3, atol=1e-3
        )


# Planar fixture for the convention pins: twelve generic image-1 points, no two sharing a coordinate, and a fixed
# homography _H_TRUE (a mild projective warp) whose entries are all distinct and non-zero, so swapping the images,
# transposing or inverting H changes every quantity derived from it. The pins compute p2 = _H_TRUE(p1) themselves.
_PLANAR_P1 = [
    [318.8, 279.9], [194.8, 137.7], [316.9, 316.9], [362.2, 154.2], [153.2, 90.1], [325.8, 259.0],
    [204.8, 129.7], [447.6, 170.0], [294.1, 216.2], [158.5, 95.9], [258.3, 329.9], [242.0, 87.6],
]  # fmt: skip
_H_TRUE = [
    [1.034685, -0.04160598, -63.61482],
    [0.1039753, 1.135451, -62.87412],
    [2.851519e-04, 1.404178e-04, 1.0],
]
_HALF_DLT = (
    "the homography DLTs form the normalised points, the design and normal matrices and the denormalisation in the "
    "input dtype, so in float16/bfloat16 an exact fit misses by up to several pixels"
)
_F16_LU = (
    "find_homography_dlt's solver='lu' path casts its unnormalised float32 solution back to float16, where the "
    "denormalisation can overflow to inf/NaN depending on rounding (the float16 manifest carries test_clean_points_lu "
    "for the same reason, #4153)"
)
_HALF_POINTS = (
    "kornia evaluates the transfer errors in the input dtype, and the fixture's pixel coordinates carry up to 2 px "
    "of rounding in bfloat16 (0.25 px in float16), the size of the offsets these pins measure"
)
_HALF_LINES = (
    "line_segment_transfer_error_one_way builds the unnormalised image-2 line in the input dtype; its constant term "
    "is of order (pixel coordinate)**2, which overflows float16 and which bfloat16 cannot resolve"
)


def _skip_half(dtype: torch.dtype, reason: str) -> None:
    if dtype in (torch.float16, torch.bfloat16):
        pytest.skip(reason)


def _planar(device, dtype):
    """Twelve exact planar matches ``p2 = H(p1)``, generated in float64 on the CPU and cast, and ``H``."""
    p1 = torch.tensor([_PLANAR_P1], dtype=torch.float64)
    H = torch.tensor([_H_TRUE], dtype=torch.float64)
    p2 = kornia.geometry.transform_points(H, p1)
    return p1.to(device, dtype), p2.to(device, dtype), H.to(device, dtype)


def _inverse(H: torch.Tensor) -> torch.Tensor:
    """H^-1 scaled so that its [2, 2] entry is 1, computed in float64 on the CPU."""
    H_inv = torch.linalg.inv(H.cpu().double())
    return (H_inv / H_inv[..., 2:, 2:]).to(H.device, H.dtype)


def _transfer_max(H: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    return (kornia.geometry.transform_points(H, src) - dst).abs().max()


def _unit_normal(seg: torch.Tensor) -> torch.Tensor:
    """Unit normal of the segment ``seg = [start, end]``, shape ``(..., 2, 2)`` -> ``(..., 2)``."""
    d = seg[..., 1, :] - seg[..., 0, :]
    return torch.stack([-d[..., 1], d[..., 0]], -1) / d.norm(dim=-1, keepdim=True)


class TestConventionHomography(BaseTester):
    def test_convention_find_homography_dlt_maps_p1_to_p2(self, device, dtype):
        if dtype == torch.float16:
            pytest.skip(_F16_LU)
        p1, p2, H_true = _planar(device, dtype)
        H = find_homography_dlt(p1, p2)
        # points2 ~ H @ points1 (OpenCV's findHomography(src, dst) order). The control: applying H to points2 misses
        # points1 by about 195 px.
        assert _transfer_max(H, p1, p2) < 0.05 * _transfer_max(H, p2, p1)
        self.assert_close(H[..., 2, 2], torch.ones_like(H[..., 2, 2]))
        if dtype in (torch.float32, torch.float64):  # half precision resolves the direction but not the entries
            self.assert_close(H, H_true, rtol=1e-4, atol=1e-4)
            # Relabelling the images returns the inverse.
            self.assert_close(find_homography_dlt(p2, p1), _inverse(H_true), rtol=1e-4, atol=1e-4)
        # Four or more correspondences: three are rejected before any solve.
        with pytest.raises(Exception):
            find_homography_dlt(p1[:, :3], p2[:, :3])

    def test_convention_find_homography_dlt_lu_equals_svd(self, device, dtype):
        _skip_half(dtype, _HALF_DLT)
        p1, p2, _ = _planar(device, dtype)
        tol = 1e-8 if dtype == torch.float64 else 5e-2  # pixels
        # Twelve points take the normal equations; four take solver="lu"'s QR-gauge path.
        for n in (12, 4):
            H_lu = find_homography_dlt(p1[:, :n], p2[:, :n], solver="lu")
            H_svd = find_homography_dlt(p1[:, :n], p2[:, :n], solver="svd")
            assert _transfer_max(H_svd, p1, p2) < tol
            diff = (kornia.geometry.transform_points(H_lu, p1) - kornia.geometry.transform_points(H_svd, p1)).abs()
            assert diff.max() < tol

    @pytest.mark.parametrize("solver", ["lu", "svd"])
    def test_convention_find_homography_dlt_zero_weight_drops_point(self, solver, device, dtype):
        _skip_half(dtype, _HALF_DLT)
        p1, p2, _ = _planar(device, dtype)
        p2_out = p2.clone()
        p2_out[0, 5] += torch.tensor([40.0, -25.0], device=device, dtype=dtype)  # one correspondence off by 47 px
        clean = torch.arange(12, device=device) != 5
        tol = 1e-8 if dtype == torch.float64 else 1e-2

        def clean_error(H):
            return _transfer_max(H, p1[:, clean], p2[:, clean])

        # On exact data: unweighted, the outlier pulls H by about 12 px on the clean points, and a weight of 0 removes
        # its equations (on noisy data it still moves H through the point normalisation, #4890).
        assert clean_error(find_homography_dlt(p1, p2_out, None, solver)) > 5.0
        w_zero = torch.ones(1, 12, device=device, dtype=dtype)
        w_zero[0, 5] = 0.0
        assert clean_error(find_homography_dlt(p1, p2_out, w_zero, solver)) < tol
        # Only relative weights matter: w and 5 w give the same H, while changing one weight relative to the
        # others moves it by pixels.
        w = torch.tensor([[1.0, 0.5, 2.0, 1.5, 0.75, 0.3, 1.25, 1.0, 0.5, 2.0, 1.5, 1.0]], device=device, dtype=dtype)
        w_other = w.clone()
        w_other[0, 5] = 0.9
        mapped = kornia.geometry.transform_points(find_homography_dlt(p1, p2_out, w, solver), p1)
        assert _transfer_max(find_homography_dlt(p1, p2_out, 5.0 * w, solver), p1, mapped) < tol
        assert _transfer_max(find_homography_dlt(p1, p2_out, w_other, solver), p1, mapped) > 1.0

    def test_convention_find_homography_lines_dlt_direction_and_layout(self, device, dtype):
        _skip_half(dtype, _HALF_DLT)
        p1, p2, _ = _planar(device, dtype)
        # Six segments (B, N, 2, 2) = [start, end] x (x, y), whose endpoints are point correspondences.
        ls1, ls2 = p1.reshape(1, 6, 2, 2), p2.reshape(1, 6, 2, 2)
        tol = 1e-8 if dtype == torch.float64 else 1e-2
        H = find_homography_lines_dlt(ls1, ls2)
        # Same direction as find_homography_dlt: H maps image-1 points onto image 2.
        assert _transfer_max(H, p1, p2) < tol
        assert _transfer_max(H, p2, p1) > 100.0
        self.assert_close(H[..., 2, 2], torch.ones_like(H[..., 2, 2]))
        mapped = kornia.geometry.transform_points(H, p1)
        # Unbatched (N, 2, 2) input is B = 1.
        H_unbatched = find_homography_lines_dlt(ls1[0], ls2[0])
        assert H_unbatched.shape == (1, 3, 3)
        assert _transfer_max(H_unbatched, p1, mapped) < tol
        # The control for the layout: the same numbers read as (x, y) x [start, end] fit a different map.
        assert _transfer_max(find_homography_lines_dlt(ls1.transpose(-2, -1), ls2.transpose(-2, -1)), p1, p2) > 100.0
        # weights has one entry per segment.
        assert (
            _transfer_max(find_homography_lines_dlt(ls1, ls2, torch.ones(1, 6, device=device, dtype=dtype)), p1, mapped)
            < tol
        )

    @pytest.mark.parametrize("error_fn", ["oneway", "symmetric", "line"])
    def test_convention_transfer_errors_argument_order_and_squared(self, error_fn, device, dtype):
        _skip_half(dtype, _HALF_LINES if error_fn == "line" else _HALF_POINTS)
        p1, p2, H = _planar(device, dtype)
        others = torch.arange(12, device=device) != 5
        if error_fn == "line":
            ls1, ls2 = p1.reshape(1, 6, 2, 2), p2.reshape(1, 6, 2, 2)
            # (ls1, ls2, H) with H mapping image 1 to image 2; the swapped call is far off.
            exact = line_segment_transfer_error_one_way(ls1, ls2, H)
            assert exact.max() < 1e-3 * line_segment_transfer_error_one_way(ls2, ls1, H).min()
            # squared=False is the default here, unlike the two point errors.
            ls2_off = ls2.clone()
            ls2_off[0, 2] += 3.0 * _unit_normal(ls2[0, 2])  # segment 2 moved 3 px off its line in image 2
            default = line_segment_transfer_error_one_way(ls1, ls2_off, H)
            unsquared = line_segment_transfer_error_one_way(ls1, ls2_off, H, squared=False)
            squared = line_segment_transfer_error_one_way(ls1, ls2_off, H, squared=True)
            self.assert_close(default, unsquared)
            self.assert_close(squared, unsquared**2)
            assert (squared - unsquared)[0, 2] > 1.0
            return
        p2_off = p2.clone()
        p2_off[0, 5] += torch.tensor([3.0, -4.0], device=device, dtype=dtype)  # match 5 moved 5 px in image 2
        if error_fn == "oneway":
            # (pts1, pts2, H) measures in image 2 between H(pts1) and pts2; squared=True is the default.
            default = oneway_transfer_error(p1, p2_off, H)
            self.assert_close(default[0, 5], torch.tensor(25.0, device=device, dtype=dtype))
            self.assert_close(oneway_transfer_error(p1, p2_off, H, squared=False)[0, 5], default.new_tensor(5.0))
            assert default[0, others].max() < 1e-6
            assert oneway_transfer_error(p2_off, p1, H).min() > 1e3
            return
        # The squared symmetric error is the image-2 error of H plus the image-1 error of H^-1, squared=True by
        # default, and squared=False is the square root of that sum (not the sum of the two distances).
        default = symmetric_transfer_error(p1, p2_off, H)
        there_and_back = oneway_transfer_error(p1, p2_off, H) + oneway_transfer_error(p2_off, p1, _inverse(H))
        self.assert_close(default, there_and_back)
        unsquared = symmetric_transfer_error(p1, p2_off, H, squared=False)
        self.assert_close(unsquared[0, 5], default[0, 5].sqrt())
        assert default[0, 5] > 50.0 and default[0, others].max() < 1e-6
        assert symmetric_transfer_error(p2_off, p1, H).min() > 1e3

    def test_convention_sample_is_valid_for_homography_rejects_reflection(self, device, dtype):
        def valid(pts1, pts2):
            t = torch.tensor([pts1], device=device, dtype=dtype), torch.tensor([pts2], device=device, dtype=dtype)
            return bool(sample_is_valid_for_homography(*t)[0])

        view1 = [[10.0, 20.0], [70.0, 30.0], [60.0, 90.0], [15.0, 75.0]]
        view2 = [[12.0, 18.0], [80.0, 35.0], [66.0, 99.0], [11.0, 70.0]]
        mirrored = [[100.0 - x, y] for x, y in view2]
        # The first four points must keep the orientation of all four triples across the views.
        assert valid(view1, view2)
        assert valid(view2, view1)
        assert not valid(view1, mirrored)
        # Collinearity is not checked: a triple collinear in both views passes (collinear in one view only fails),
        # and so does a sample that repeats a point.
        collinear1 = [[10.0, 20.0], [30.0, 50.0], [50.0, 80.0], [15.0, 75.0]]
        collinear2 = [[12.0, 18.0], [32.0, 28.0], [52.0, 38.0], [11.0, 70.0]]
        assert valid(collinear1, collinear2)
        assert not valid(collinear1, view2)
        assert valid([*view1[:3], view1[0]], [*view2[:3], view2[0]])
        # Points after the fourth are ignored.
        assert valid([*view1, [1000.0, -7.0]], [*view2, [-5.0, 400.0]])
        mask = sample_is_valid_for_homography(
            torch.tensor([view1] * 3, device=device, dtype=dtype), torch.tensor([view2] * 3, device=device, dtype=dtype)
        )
        assert mask.dtype == torch.bool and mask.shape == (3,)

    @pytest.mark.parametrize("model", ["points", "lines"])
    def test_convention_find_homography_dlt_iterated_n_iter_counts_solves(self, model, device, dtype, monkeypatch):
        if model == "points" and dtype == torch.float16:
            pytest.skip(_F16_LU)
        p1, p2, _ = _planar(device, dtype)
        if model == "points":
            # The point polisher builds the DLT system once and solves it per iteration: count the solves.
            name, iterated, args = "_homography_from_dlt_system", find_homography_dlt_iterated, (p1, p2)
            plain = find_homography_dlt
        else:
            name = "find_homography_lines_dlt"
            iterated, args = find_homography_lines_dlt_iterated, (p1.reshape(1, 6, 2, 2), p2.reshape(1, 6, 2, 2))
            plain = getattr(kornia.geometry.homography, name)
        solve = getattr(kornia.geometry.homography, name)
        weights = torch.ones(1, args[0].shape[1], device=device, dtype=dtype)
        calls = []

        def spy(*a, **k):
            calls.append(1)
            return solve(*a, **k)

        monkeypatch.setattr(kornia.geometry.homography, name, spy)
        # n_iter counts the solves, the initial one included, so n_iter=1 is the plain solver.
        H1 = iterated(*args, weights, n_iter=1)
        assert len(calls) == 1
        assert torch.equal(H1, plain(*args, weights))
        calls.clear()
        H = iterated(*args, weights, n_iter=3)
        assert len(calls) == 3
        if dtype in (torch.float32, torch.float64):
            # Same direction as the plain solver; half precision does not hold it through the re-weighted solves on
            # this fixture.
            assert _transfer_max(H, p1, p2) < 1e-2
            assert _transfer_max(H, p2, p1) > 100.0

    def test_wart_find_homography_lines_dlt_endpoint_pairing_4866(self, device, dtype):
        _skip_half(dtype, _HALF_LINES)
        p1, p2, H_true = _planar(device, dtype)
        ls1, ls2 = p1.reshape(1, 6, 2, 2), p2.reshape(1, 6, 2, 2)
        # Slide every image-2 endpoint along its segment's true line: H_true still maps each image-1 segment onto its
        # image-2 line, but the endpoints are no longer point correspondences.
        d = ls2[:, :, 1] - ls2[:, :, 0]
        slid = torch.stack([ls2[:, :, 0] + 0.3 * d, ls2[:, :, 1] + 0.25 * d], dim=2)
        assert line_segment_transfer_error_one_way(ls1, slid, H_true).max() < 1e-2
        # #4866: the equations of segment i use flattened points i and N + i, so the estimate misses by thousands
        # of pixels here. Once the endpoints are paired per segment it recovers H_true.
        assert _transfer_max(find_homography_lines_dlt(ls1, slid), p1, p2) > 100.0
        # For the same reason a zero weight does not remove its segment: the outlier still moves the estimate.
        ls2_out = ls2.clone()
        ls2_out[0, 3] += torch.tensor([[35.0, -20.0], [-15.0, 30.0]], device=device, dtype=dtype)
        weights = torch.ones(1, 6, device=device, dtype=dtype)
        weights[0, 3] = 0.0
        clean = (torch.arange(12, device=device) != 6) & (torch.arange(12, device=device) != 7)
        assert _transfer_max(find_homography_lines_dlt(ls1, ls2_out, weights), p1[:, clean], p2[:, clean]) > 10.0

    def test_wart_line_segment_error_scales_with_length_4867(self, device, dtype):
        _skip_half(dtype, _HALF_LINES)
        p1, _, H = _planar(device, dtype)
        short = p1[:, [1, 7]].reshape(1, 1, 2, 2)
        mid = short.mean(dim=2, keepdim=True)
        errors, lengths = [], []
        for seg1 in (short, mid + 2.0 * (short - mid)):  # the same line, the segment twice as long
            seg2 = kornia.geometry.transform_points(H, seg1.reshape(1, 2, 2)).reshape(1, 1, 2, 2)
            offset = seg2 + 3.0 * _unit_normal(seg2)[..., None, :]  # moved 3 px off its line in image 2
            errors.append(line_segment_transfer_error_one_way(seg1, offset, H))
            lengths.append((seg2[..., 1, :] - seg2[..., 0, :]).norm(dim=-1))
        # #4867: the image-2 line is not normalised, so the error is 3 px times the image-2 segment length (223 and
        # 447 px here), not 3 px. Once the line is normalised both errors are 3.
        for error, length in zip(errors, lengths):
            self.assert_close(error / length, torch.full_like(error, 3.0))
        assert errors[1] > 1.9 * errors[0]

    @pytest.mark.parametrize("model", ["points", "lines"])
    def test_wart_find_homography_dlt_iterated_weight_unsquared_4870(self, model, device, dtype, monkeypatch):
        if model == "points" and dtype == torch.float16:
            pytest.skip(_F16_LU)
        if model == "lines":
            _skip_half(dtype, _HALF_LINES)
        p1, p2, _ = _planar(device, dtype)
        if model == "points":
            p2_off = p2.clone()
            p2_off[0, 5] += torch.tensor([6.0, -4.0], device=device, dtype=dtype)
            name, iterated, args, k = "find_homography_dlt", find_homography_dlt_iterated, (p1, p2_off), 5

            def error(H):
                return symmetric_transfer_error(p1, p2_off, H, squared=False)

        else:
            ls1, ls2 = p1.reshape(1, 6, 2, 2), p2.reshape(1, 6, 2, 2).clone()
            ls2[0, 2] += 0.05 * _unit_normal(ls2[0, 2])
            name, iterated, args, k = "find_homography_lines_dlt", find_homography_lines_dlt_iterated, (ls1, ls2), 2

            def error(H):
                return line_segment_transfer_error_one_way(ls1, ls2, H, squared=False)

        plain = getattr(kornia.geometry.homography, name)
        weights = torch.ones(1, args[0].shape[1], device=device, dtype=dtype)
        # soft_inl_th is set from the moved correspondence's first-solve error e_k, so that its weight under the
        # linear kernel is exp(-1/2) whatever the error's scale.
        sigma = float(error(plain(*args, weights))[0, k].sqrt())
        calls = []

        if model == "points":
            # The point polisher solves the prebuilt DLT system; its weights are the second argument.
            name = "_homography_from_dlt_system"
            solve = kornia.geometry.homography._homography_from_dlt_system

            def spy(system, w, *rest):
                H = solve(system, w, *rest)
                calls.append((w, H))
                return H

        else:

            def spy(a, b, w=None, *rest):
                H = plain(a, b, w, *rest)
                calls.append((w, H))
                return H

        monkeypatch.setattr(kornia.geometry.homography, name, spy)
        iterated(*args, weights, soft_inl_th=sigma, n_iter=2)
        assert len(calls) == 2
        e = error(calls[0][1])
        # #4870: the second solve weights each correspondence by exp(-e / (2 sigma^2)) with the unsquared error e,
        # which for the moved correspondence is far from the Gaussian exp(-e^2 / (2 sigma^2)).
        self.assert_close(calls[1][0], torch.exp(-e / (2.0 * sigma**2)))
        assert (calls[1][0] - torch.exp(-(e**2) / (2.0 * sigma**2)))[0, k].abs() > 0.1

    def test_wart_transfer_error_exact_match_is_sqrt_eps_4881(self, device, dtype):
        _skip_half(
            dtype,
            "eps=1e-8 is below float16's subnormal range, and in bfloat16 the fixture's matches are 2 px from exact",
        )
        p1, p2, H = _planar(device, dtype)
        # #4881: eps is added inside the square root, so an exact match scores sqrt(1e-8) = 1e-4, not 0 ...
        assert oneway_transfer_error(p1, p2, H, squared=False).min() > 5e-5
        assert symmetric_transfer_error(p1, p2, H, squared=False).min() > 5e-5
        if dtype == torch.float64:
            assert oneway_transfer_error(p1, p2, H, squared=False, eps=0.0).max() < 1e-9
        # ... and to the projective denominator, so the error depends on the scale of H: 1e-8 H scores tens of
        # pixels on the same exact matches.
        assert oneway_transfer_error(p1, p2, 1e-8 * H, squared=False).min() > 1.0

    @pytest.mark.parametrize("model", ["points", "lines"])
    def test_wart_find_homography_dlt_h22_eps_divisor_4874(self, model, device, dtype):
        _skip_half(dtype, _HALF_DLT)
        p1, p2, _ = _planar(device, dtype)
        # The same fixture with the true H[2, 2] set to 1e-6 (a mild warp whose origin maps far out), exact matches.
        H_small = torch.tensor([_H_TRUE], dtype=torch.float64)
        H_small[0, 2, 2] = 1e-6
        p2_small = kornia.geometry.transform_points(H_small, p1.cpu().double()).to(device, dtype)

        def fit(dst):
            if model == "points":
                return find_homography_dlt(p1, dst, solver="svd")
            return find_homography_lines_dlt(p1.reshape(1, 6, 2, 2), dst.reshape(1, 6, 2, 2))

        # With the true H[2, 2] = 1 the returned entry is 1.
        assert (fit(p2)[0, 2, 2] - 1.0).abs() < 1e-4
        # #4874: H is divided by H[2, 2] + 1e-8, and the unnormalised entry here is a few 1e-6, so the returned
        # H[2, 2] misses 1 by about 2e-3. Once the divisor is H[2, 2] itself, it is 1.
        assert (fit(p2_small)[0, 2, 2] - 1.0).abs() > 1e-3

    @pytest.mark.parametrize("solver", ["lu", "svd"])
    def test_wart_find_homography_dlt_zero_weight_moves_normalisation_4890(self, solver, device, dtype):
        _skip_half(dtype, _HALF_DLT)
        p1, p2, _ = _planar(device, dtype)
        # 3-px noise on the twelve matches, generated by
        #   g = torch.Generator().manual_seed(0)
        #   torch.round(3 * torch.randn(1, 12, 2, generator=g, dtype=torch.float64), decimals=1)
        noise = [
            [-6.9, -1.1], [-3.2, 3.0], [-2.7, -3.8], [-1.9, -2.6], [0.7, 2.2], [-4.0, 3.9],
            [-4.5, -2.1], [1.6, -2.9], [2.0, 3.1], [0.5, -2.2], [-2.2, 0.3], [-5.2, 3.1],
        ]  # fmt: skip
        p2 = p2 + torch.tensor([noise], device=device, dtype=dtype)
        # A far correspondence with weight 0, appended to the twelve.
        far1 = torch.cat([p1, torch.tensor([[[2000.0, 1500.0]]], device=device, dtype=dtype)], 1)
        far2 = torch.cat([p2, torch.tensor([[[-800.0, 2400.0]]], device=device, dtype=dtype)], 1)
        weights = torch.ones(1, 13, device=device, dtype=dtype)
        weights[0, 12] = 0.0
        H_weighted = find_homography_dlt(far1, far2, weights, solver)
        H_dropped = find_homography_dlt(p1, p2, None, solver)
        # #4890: the zero-weight correspondence leaves the equations but still enters the Hartley normalisation, so
        # on noisy data it moves H (by 0.08 px with solver="lu" and 0.25 px with "svd" here). Once the normalisation
        # uses the weights, the two estimates agree to roundoff.
        assert _transfer_max(H_weighted, p1, kornia.geometry.transform_points(H_dropped, p1)) > 1e-2
