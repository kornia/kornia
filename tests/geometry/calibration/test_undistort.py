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

import kornia.geometry.calibration.undistort as undistort_module
from kornia.geometry.calibration.distort import distort_points
from kornia.geometry.calibration.undistort import undistort_image, undistort_points
from kornia.geometry.grid import create_meshgrid
from kornia.geometry.transform import remap

from testing.base import BaseTester


def _k_asymmetric(device, dtype):
    """``fx = fy = 100``, ``cx = 4``, ``cy = 3`` -- ``cx != cy`` so a transposed reading changes the literals."""
    return torch.tensor([[[100.0, 0.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)


def _k_short_focal(device, dtype, fx=3.0, fy=2.0, cx=3.0, cy=2.0):
    """A camera whose distortion actually bites on a 5 x 7 image.

    Every intrinsic differs from every other, and the focal lengths are short enough that the 5 x 7 pixel grid
    spans a normalized radius of about 1.4, so the radial polynomial displaces the map by 0.74 px instead of the
    3.5e-04 px that ``fx = 100`` would give -- a resampling pin on that map would pass for the identity.
    """
    return torch.tensor([[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)


def _ramp_image(batch, channels, height, width, device, dtype):
    """A deterministic, non-constant image.

    ``% 17`` on a ramp of length ``B * C * H * W`` makes neighbouring pixels differ in every direction, so a
    resampling pin cannot pass by accident on a flat image; ``H != W`` keeps a row/column swap visible.
    """
    numel = batch * channels * height * width
    ramp = torch.arange(numel, device=device, dtype=torch.float32) % 17.0
    return (ramp / 17.0).reshape(batch, channels, height, width).to(dtype)


class TestUndistortPoints(BaseTester):
    def test_smoke(self, device, dtype):
        points = torch.rand(1, 2, device=device, dtype=dtype)
        K = torch.rand(3, 3, device=device, dtype=dtype)
        distCoeff = torch.rand(4, device=device, dtype=dtype)
        pointsu = undistort_points(points, K, distCoeff)
        assert points.shape == pointsu.shape

        new_K = torch.rand(3, 3, device=device, dtype=dtype)
        pointsu = undistort_points(points, K, distCoeff, new_K)
        assert points.shape == pointsu.shape

    def test_smoke_batch(self, device, dtype):
        points = torch.rand(1, 1, 2, device=device, dtype=dtype)
        K = torch.rand(1, 3, 3, device=device, dtype=dtype)
        distCoeff = torch.rand(1, 4, device=device, dtype=dtype)
        pointsu = undistort_points(points, K, distCoeff)
        assert points.shape == pointsu.shape

        new_K = torch.rand(1, 3, 3, device=device, dtype=dtype)
        pointsu = undistort_points(points, K, distCoeff, new_K)
        assert points.shape == pointsu.shape

    def test_tilt_multi_axis_batch(self, device, dtype):
        num_points = 5
        points = torch.rand(2, 3, num_points, 2, device=device, dtype=dtype)
        K = torch.eye(3, device=device, dtype=dtype).expand(2, 3, 3, 3).clone()
        dist = torch.zeros(2, 3, 14, device=device, dtype=dtype)
        dist[..., 12] = 0.01
        dist[..., 13] = -0.02

        actual = undistort_points(points, K, dist)
        expected = torch.stack(
            [undistort_points(p, k, d) for p, k, d in zip(points.flatten(0, -3), K.flatten(0, -3), dist.flatten(0, -2))]
        ).reshape(2, 3, num_points, 2)

        assert actual.shape == points.shape
        self.assert_close(actual, expected)

    def test_export_multi_axis_batch(self, monkeypatch, device, dtype):
        points = torch.rand(2, 3, 5, 2, device=device, dtype=dtype)
        K = torch.eye(3, device=device, dtype=dtype).expand(2, 3, 3, 3).clone()
        dist = torch.tensor([0.01, -0.02, 0.001, -0.001], device=device, dtype=dtype).expand(2, 3, 4).clone()
        expected = undistort_points(points, K, dist)

        monkeypatch.setattr(undistort_module, "is_exporting", lambda: True)
        actual = undistort_points(points, K, dist)

        assert actual.shape == points.shape
        self.assert_close(actual, expected)

    def test_export_unbatched(self, monkeypatch, device, dtype):
        points = torch.rand(5, 2, device=device, dtype=dtype)
        K = torch.eye(3, device=device, dtype=dtype)
        dist = torch.tensor([0.01, -0.02, 0.001, -0.001], device=device, dtype=dtype)
        expected = undistort_points(points, K, dist)

        monkeypatch.setattr(undistort_module, "is_exporting", lambda: True)
        actual = undistort_points(points, K, dist)

        assert actual.shape == points.shape
        self.assert_close(actual, expected)

    @pytest.mark.parametrize(
        "batch_size, num_points, num_distcoeff", [(1, 3, 4), (2, 4, 5), (3, 5, 8), (4, 6, 12), (5, 7, 14)]
    )
    def test_shape(self, batch_size, num_points, num_distcoeff, device, dtype):
        B, N, Ndist = batch_size, num_points, num_distcoeff

        points = torch.rand(B, N, 2, device=device, dtype=dtype)
        K = torch.rand(B, 3, 3, device=device, dtype=dtype)
        distCoeff = torch.rand(B, Ndist, device=device, dtype=dtype)

        pointsu = undistort_points(points, K, distCoeff)
        assert pointsu.shape == (B, N, 2)

        new_K = torch.rand(B, 3, 3, device=device, dtype=dtype)
        pointsu = undistort_points(points, K, distCoeff, new_K)
        assert pointsu.shape == (B, N, 2)

    @pytest.mark.device_agnostic
    def test_convention_out_of_radius_fixed_point_is_a_two_cycle_4285(self):
        dtype = torch.float64
        K = torch.tensor([[[100.0, 0.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]]], dtype=dtype)
        dist = torch.tensor([[0.5, 0.0, 0.0, 0.0]], dtype=dtype)
        points = torch.tensor([[[304.0, 3.0]]], dtype=dtype)
        distorted = distort_points(points, K, dist)

        residuals = {}
        for num_iters in (5, 6, 7, 8):
            recovered = undistort_points(distorted, K, dist, num_iters=num_iters)
            residuals[num_iters] = (recovered - points).abs().max()

        self.assert_close(residuals[5], residuals[7])
        self.assert_close(residuals[6], residuals[8])
        assert residuals[6] > residuals[5] > 1

    def test_opencv_five_coeff(self, device, dtype):
        # Test using 5 distortion coefficients
        pts = torch.tensor(
            [[1028.0374, 788.7520], [1025.1218, 716.8726], [1022.1792, 645.1857]], device=device, dtype=dtype
        )

        K = torch.tensor(
            [[1.7315e03, 0.0000e00, 6.2289e02], [0.0000e00, 1.7320e03, 5.3537e02], [0.0000e00, 0.0000e00, 1.0000e00]],
            device=device,
            dtype=dtype,
        )

        dist = torch.tensor([-0.1007, 0.2650, -0.0018, 0.0007, -0.2597], device=device, dtype=dtype)

        # Expected output generated with OpenCV:
        # import cv2
        # ptsu_expected = cv2.undistortPoints(pts.numpy().reshape(-1,1,2), K.numpy(),
        #                               dist1.numpy(), None, None, K.numpy()).reshape(-1,2)
        ptsu_expected = torch.tensor(
            [[1030.5992, 790.65533], [1027.3059, 718.10020], [1024.0700, 645.90600]], device=device, dtype=dtype
        )
        ptsu = undistort_points(pts, K, dist)
        self.assert_close(ptsu, ptsu_expected, rtol=1e-4, atol=1e-4)

        new_K = K * 2
        new_K[2, 2] = 1
        # Expected output generated with OpenCV:
        # import cv2
        # ptsu_expected = cv2.undistortPoints(pts.numpy().reshape(-1,1,2), K.numpy(),
        #                                    dist.numpy(), None, None, new_K.numpy()).reshape(-1,2)
        print(ptsu_expected)
        ptsu_expected = 2 * torch.tensor(
            [[1030.5992, 790.65533], [1027.3059, 718.10020], [1024.0700, 645.90600]], device=device, dtype=dtype
        )
        print(ptsu_expected)
        ptsu = undistort_points(pts, K, dist, new_K)
        self.assert_close(ptsu, ptsu_expected, rtol=1e-4, atol=1e-4)

    def test_opencv_all_coeff(self, device, dtype):
        # Test using 14 distortion coefficients
        pts = torch.tensor(
            [[1028.0374, 788.7520], [1025.1218, 716.8726], [1022.1792, 645.1857]], device=device, dtype=dtype
        )

        K = torch.tensor(
            [[1.7315e03, 0.0000e00, 6.2289e02], [0.0000e00, 1.7320e03, 5.3537e02], [0.0000e00, 0.0000e00, 1.0000e00]],
            device=device,
            dtype=dtype,
        )

        dist = torch.tensor(
            [
                -5.6388e-02,
                2.3881e-01,
                8.3374e-02,
                2.0710e-03,
                7.1349e00,
                5.6335e-02,
                -3.1738e-01,
                4.9981e00,
                -4.0287e-03,
                -2.8246e-02,
                -8.6064e-02,
                1.5543e-02,
                -1.7322e-01,
                2.3154e-03,
            ],
            device=device,
            dtype=dtype,
        )

        # Expected output generated with OpenCV:
        # import cv2
        # ptsu_expected = cv2.undistortPoints(pts.numpy().reshape(-1,1,2), K.numpy(),
        #                               dist2.numpy(), None, None, K.numpy()).reshape(-1,2)
        ptsu_expected = torch.tensor(
            [[1030.8245, 786.3807], [1027.5505, 715.0732], [1024.2753, 644.0319]], device=device, dtype=dtype
        )
        ptsu = undistort_points(pts, K, dist)
        self.assert_close(ptsu, ptsu_expected, rtol=1e-4, atol=1e-4)

        new_K = K * 2
        new_K[2, 2] = 1
        # Expected output generated with OpenCV:
        # import cv2
        # ptsu_expected = cv2.undistortPoints(pts.numpy().reshape(-1,1,2), K.numpy(),
        #                                    dist.numpy(), None, None, new_K.numpy()).reshape(-1,2)
        print(ptsu_expected)
        ptsu_expected = 2 * torch.tensor(
            [[1030.8245, 786.3807], [1027.5505, 715.0732], [1024.2753, 644.0319]], device=device, dtype=dtype
        )
        print(ptsu_expected)
        ptsu = undistort_points(pts, K, dist, new_K)
        self.assert_close(ptsu, ptsu_expected, rtol=1e-4, atol=1e-4)

    def test_opencv_stereo(self, device, dtype):
        # Udistort stereo points with data given in two batches using 14 distortion coefficients
        pts = torch.tensor(
            [
                [[1028.0374, 788.7520], [1025.1218, 716.8726], [1022.1792, 645.1857]],
                [[345.9135, 847.9113], [344.0880, 773.9890], [342.2381, 700.3029]],
            ],
            device=device,
            dtype=dtype,
        )

        K = torch.tensor(
            [
                [
                    [3.3197e03, 0.0000e00, 6.1813e02],
                    [0.0000e00, 3.3309e03, 5.2281e02],
                    [0.0000e00, 0.0000e00, 1.0000e00],
                ],
                [
                    [1.9206e03, 0.0000e00, 6.1395e02],
                    [0.0000e00, 1.9265e03, 7.7164e02],
                    [0.0000e00, 0.0000e00, 1.0000e00],
                ],
            ],
            device=device,
            dtype=dtype,
        )

        dist = torch.tensor(
            [
                [
                    -5.6388e-02,
                    2.3881e-01,
                    8.3374e-02,
                    2.0710e-03,
                    7.1349e00,
                    5.6335e-02,
                    -3.1738e-01,
                    4.9981e00,
                    -4.0287e-03,
                    -2.8246e-02,
                    -8.6064e-02,
                    1.5543e-02,
                    -1.7322e-01,
                    2.3154e-03,
                ],
                [
                    1.4050e-03,
                    -3.0691e00,
                    -1.0209e-01,
                    -2.3687e-02,
                    -1.7082e02,
                    4.3593e-03,
                    -3.1904e00,
                    -1.7050e02,
                    1.7854e-02,
                    1.8999e-02,
                    9.9122e-02,
                    3.6675e-02,
                    3.0816e-03,
                    -5.7133e-02,
                ],
            ],
            device=device,
            dtype=dtype,
        )

        # Expected output generated with OpenCV:
        # import cv2
        # ptsu_expected1 = cv2.undistortPoints(pts[0].numpy().reshape(-1,1,2), K[0].numpy(),
        #                               dist[0].numpy(), None, None, K[0].numpy()).reshape(-1,2)
        # ptsu_expected2 = cv2.undistortPoints(pts[1].numpy().reshape(-1,1,2), K[1].numpy(),
        #                               dist[1].numpy(), None, None, K[1].numpy()).reshape(-1,2)
        ptsu_expected1 = torch.tensor(
            [[1029.3234, 785.4813], [1026.1599, 714.3689], [1023.02045, 643.5359]], device=device, dtype=dtype
        )

        ptsu_expected2 = torch.tensor(
            [[344.04456, 848.7696], [344.27606, 774.1254], [344.47018, 700.8522]], device=device, dtype=dtype
        )

        ptsu = undistort_points(pts, K, dist)
        self.assert_close(ptsu[0], ptsu_expected1, rtol=1e-4, atol=1e-4)
        self.assert_close(ptsu[1], ptsu_expected2, rtol=1e-4, atol=1e-4)

    def test_convention_undistort_points_inverts_distort_points(self, device, dtype):
        # Convention pin: undistort_points is the
        # ITERATIVE inverse of distort_points (a 5-step fixed point by default), so the round trip closes to the
        # dtype tolerance on these points with moderate coefficients. Both
        # functions take PIXEL points and a (3, 3) K -- contrast the affine and Kannala-Brandt pairs in
        # tests/geometry/camera/test_distortion.py, which take normalized z = 1 points and a flat parameter
        # vector. The pin asserts closure at assert_close's dtype tolerance and states no error bound; the
        # five-step float64 residual (4.62e-05) passes because atol plus rtol * abs(points) allows it.
        # Raising the count to 50 improves float32/float64 accuracy; float16 stays at its rounding floor.
        # Outside the valid radius the iteration can cycle (#4285), pinned separately below.
        # The two pixels sit half a focal length off the principal point in both directions and on both sides of
        # it, so the forward map actually moves them by 2.775 px -- a round trip pin on near-axis points would
        # pass with undistort_points replaced by the identity.
        # Snippet used to generate expected: (undistort_points(distort_points(pts, K, d), K, d) - pts).abs().max()
        # executed 2026-09-06 on commit c0b50ad7 (torch 2.14.0), differenced in the working dtype -> cpu
        # float32 3.81e-05, float64 4.62e-05, float16 1.56e-02, bfloat16 0.0; mps float32 3.81e-05,
        # float16 1.56e-02. The forward displacement (distorted - points).abs().max() is 2.775 px on cpu float32.
        points = torch.tensor([[[54.0, 53.0], [-16.0, 23.0]]], device=device, dtype=dtype)
        K = _k_asymmetric(device, dtype)
        dist = torch.tensor([[0.1, 0.01, 0.001, 0.001]], device=device, dtype=dtype)
        distorted = distort_points(points, K, dist)
        assert not torch.allclose(distorted.float(), points.float())
        self.assert_close(undistort_points(distorted, K, dist), points)
        self.assert_close(undistort_points(distorted, K, dist, num_iters=50), points)

    def test_wart_distort_undistort_round_trip_breaks_with_tilt_4276(self, device, dtype):
        # Wart pin for kornia#4276: distort_points and undistort_points stop being
        # inverses as soon as the 13th and 14th coefficients (taux, tauy) are non-zero, because distort_points
        # applies tilt_projection's FORWARD branch (Pz @ R.T) while undistort_points applies the return_inverse
        # branch (inv(Pz @ R)) -- see test_wart_tilt_projection_forward_is_pz_times_r_transpose_4276 in
        # test_distort.py. With taux = 0.1, tauy = 0.2 the round trip misses by tens of pixels; with the same
        # 14-coefficient vector and both tilt angles zero it closes within the dtype tolerance.
        # taux != tauy so a symmetric tilt cannot mask the defect.
        # The two 14-coefficient vectors below share the SAME radial and tangential part, so the only difference
        # between the two arms is the tilt: with tau = 0 the round trip on genuinely distorted points closes, and
        # with tau != 0 it misses by tens of pixels.
        # Snippet used to generate expected: (undistort_points(distort_points(pts, K, d14), K, d14) - pts)
        # .abs().max() executed 2026-09-06 on commit c0b50ad7 (torch 2.14.0), tau = (0.1, 0.2) -> cpu
        # float32 47.765, float64 47.765, float16 47.812, bfloat16 47.5; mps float32 47.765, float16 47.75. The
        # same vector with tau = 0 gives 3.81e-05 (cpu float32) / 0.0 (cpu bfloat16).
        # Pins the CURRENT behavior; NOT a contract; delete when #4276 is repaired.
        points = torch.tensor([[[54.0, 53.0], [-16.0, 23.0]]], device=device, dtype=dtype)
        K = _k_asymmetric(device, dtype)
        radial = torch.tensor([[0.1, 0.01, 0.001, 0.001]], device=device, dtype=dtype)
        zero_tilt = torch.cat([radial, torch.zeros(1, 10, device=device, dtype=dtype)], -1)
        tilted = torch.cat([zero_tilt[:, :12], torch.tensor([[0.1, 0.2]], device=device, dtype=dtype)], -1)
        assert not torch.allclose(
            undistort_points(distort_points(points, K, tilted), K, tilted).float(), points.float()
        )
        self.assert_close(undistort_points(distort_points(points, K, zero_tilt), K, zero_tilt), points)

    def test_convention_new_K_denormalizes_and_K_normalizes(self, device, dtype):
        # The two intrinsics play the mirror image of their
        # roles in distort_points -- here K maps the incoming distorted pixel onto the normalized z = 1 plane and
        # new_K maps the undistorted normalized point back to pixels, which is why the same new_K that shrinks
        # the forward answer enlarges this one. Every intrinsic is distinct (K: fx 2, fy 3, cx 1, cy 1; new_K:
        # fx 5, fy 7, cx 2, cy 4), so swapping the two arguments, or fx with fy, changes both literals:
        # (10, 10) normalizes under K to ((10-1)/2, (10-1)/3) = (4.5, 3.0) and denormalizes under new_K to
        # (5*4.5+2, 7*3+4) = (24.5, 25.0). The sibling pin in test_distort.py fixes the forward direction.
        # Snippet used to generate expected: undistort_points([[[10., 10.]]], K, zeros(1, 4), new_K) executed
        # 2026-09-06 on commit c0b50ad7 (torch 2.14.0, cpu float32 and float64) -> [[[24.5, 25.0]]]; the
        # same call on mps float32 gives the same value. The second case, the mirror of test_distort.py's
        # diag(2, 2, 1) probe, doubles instead of halving: (1, 2) under K = eye(3) and new_K = diag(2, 2, 1)
        # comes back as (2.0, 4.0).
        K = torch.tensor([[[2.0, 0.0, 1.0], [0.0, 3.0, 1.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        new_K = torch.tensor([[[5.0, 0.0, 2.0], [0.0, 7.0, 4.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        zero_dist = torch.zeros(1, 4, device=device, dtype=dtype)
        points = torch.tensor([[[10.0, 10.0]]], device=device, dtype=dtype)
        self.assert_close(
            undistort_points(points, K, zero_dist, new_K),
            torch.tensor([[[24.5, 25.0]]], device=device, dtype=dtype),
        )
        self.assert_close(
            undistort_points(
                torch.tensor([[[1.0, 2.0]]], device=device, dtype=dtype),
                torch.eye(3, device=device, dtype=dtype)[None],
                zero_dist,
                torch.tensor([[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype),
            ),
            torch.tensor([[[2.0, 4.0]]], device=device, dtype=dtype),
        )

    def test_convention_more_iterations_shrink_the_round_trip_residual(self, device, dtype):
        # Inside the convergence region, more iterations reduce the residual down to the rounding floor.
        # The fx = fy = 100 camera above already passes assert_close after five steps, although 50 steps
        # improve its float32/float64 residual. Halving fy here makes the five-step error exceed that tolerance.
        # The first assertion is the non-triviality guard: the five-step answer is OUTSIDE the tolerance that
        # assert_close would use on these points, so the strict decrease below is not a decrease between three
        # already-converged answers.
        # Snippet used to generate expected: (undistort_points(distort_points(pts, K, d), K, d, num_iters=n)
        # - pts).abs().max() for n in (5, 10, 50), executed 2026-09-06 on commit c0b50ad7 (torch 2.14.0),
        # differenced in the working dtype -> cpu float64 1.355131e-02, 2.410766e-05, 1.421085e-14; cpu float32
        # 1.355362e-02, 3.051758e-05, 1.907349e-06; mps float32 the same three float32 values. float32 has
        # reached its own rounding floor by 50 steps while float64 is still closing, which is the scoping the
        # Convention block states. The forward map displaces these points by 7.306 px, so an identity
        # undistort_points would not pass. In float16 and bfloat16 the residual is pinned at the dtype's own
        # quantum from the first step (1.5625e-02 and 0.5), so the three counts are indistinguishable there and
        # those cells are skipped.
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("half precision quantizes the residual, so the three step counts are indistinguishable")
        points = torch.tensor([[[54.0, 53.0], [-16.0, 23.0]]], device=device, dtype=dtype)
        K = torch.tensor([[[100.0, 0.0, 4.0], [0.0, 50.0, 3.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        dist = torch.tensor([[0.1, 0.01, 0.001, 0.001]], device=device, dtype=dtype)
        distorted = distort_points(points, K, dist)
        with pytest.raises(AssertionError):
            self.assert_close(undistort_points(distorted, K, dist, num_iters=5), points)
        residuals = [(undistort_points(distorted, K, dist, num_iters=n) - points).abs().max() for n in (5, 10, 50)]
        assert residuals[0] > residuals[1]
        assert residuals[1] > residuals[2]
        self.assert_close(undistort_points(distorted, K, dist, num_iters=50), points)

    def test_wart_fixed_point_cycles_outside_valid_radius_4285(self, device, dtype):
        # The iteration alternates between two inaccurate values; a larger even count is worse
        # than an odd count, rather than the residual increasing monotonically. Delete after #4285.
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("probe requires float32/float64 to avoid half-precision overflow")
        points = torch.tensor([[[304.0, 3.0]]], device=device, dtype=dtype)
        K = _k_asymmetric(device, dtype)
        dist = torch.tensor([[0.5, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
        distorted = distort_points(points, K, dist)
        residuals = [(undistort_points(distorted, K, dist, num_iters=n) - points).abs().max() for n in (5, 6, 7, 8)]
        self.assert_close(residuals[0], residuals[2], atol=1e-3, rtol=1e-5)
        self.assert_close(residuals[1], residuals[3], atol=1e-3, rtol=1e-5)
        assert residuals[1] > residuals[0] > 1

    def test_gradcheck(self, device):
        points = torch.rand(1, 8, 2, device=device, dtype=torch.float64, requires_grad=True)
        K = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        new_K = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        distCoeff = torch.rand(1, 4, device=device, dtype=torch.float64)

        self.gradcheck(undistort_points, (points, K, distCoeff, new_K), requires_grad=(True, False, False, False))

    def test_dynamo(self, device, dtype, torch_optimizer):
        points = torch.rand(1, 1, 2, device=device, dtype=dtype)
        K = torch.rand(1, 3, 3, device=device, dtype=dtype)
        new_K = torch.rand(1, 3, 3, device=device, dtype=dtype)
        distCoeff = torch.rand(1, 4, device=device, dtype=dtype)
        inputs = (points, K, distCoeff, new_K)

        op = undistort_points
        op_optimized = torch_optimizer(op)
        self.assert_close(op(*inputs), op_optimized(*inputs))


class TestUndistortImage(BaseTester):
    def test_shape(self, device, dtype):
        im = torch.rand(1, 3, 5, 5, device=device, dtype=dtype)
        K = torch.rand(3, 3, device=device, dtype=dtype)
        distCoeff = torch.rand(4, device=device, dtype=dtype)

        imu = undistort_image(im, K, distCoeff)
        assert imu.shape == (1, 3, 5, 5)

    def test_shape_minimum_dims(self, device, dtype):
        im = torch.rand(3, 5, 5, device=device, dtype=dtype)
        K = torch.rand(3, 3, device=device, dtype=dtype)
        distCoeff = torch.rand(4, device=device, dtype=dtype)

        imu = undistort_image(im, K, distCoeff)
        assert imu.shape == (3, 5, 5)

    def test_shape_extra_dims(self, device, dtype):
        im = torch.rand(1, 1, 3, 5, 5, device=device, dtype=dtype).tile(3, 2, 1, 1, 1)
        K = torch.rand(1, 1, 3, 3, device=device, dtype=dtype).tile(3, 2, 1, 1)
        distCoeff = torch.rand(1, 1, 4, device=device, dtype=dtype).tile(3, 2, 1)

        imu = undistort_image(im, K, distCoeff)
        assert imu.shape == (3, 2, 3, 5, 5)
        self.assert_close(imu[0], imu[1])

    def test_tilt_multi_axis_batch(self, device, dtype):
        image = torch.rand(2, 3, 1, 5, 6, device=device, dtype=dtype)
        K = torch.tensor([[3.0, 0.0, 3.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
        K = K.expand(2, 3, 3, 3).clone()
        dist = torch.zeros(2, 3, 14, device=device, dtype=dtype)
        dist[..., 12] = 0.01
        dist[..., 13] = 0.02

        actual = undistort_image(image, K, dist)
        expected = torch.stack(
            [torch.stack([undistort_image(image[i, j], K[i, j], dist[i, j]) for j in range(3)]) for i in range(2)]
        )

        assert actual.shape == image.shape
        self.assert_close(actual, expected)

    def test_exception(self, device, dtype):
        with pytest.raises(ValueError):
            im = torch.rand(5, 5, device=device, dtype=dtype)
            K = torch.rand(3, 3, device=device, dtype=dtype)
            distCoeff = torch.rand(4, device=device, dtype=dtype)
            undistort_image(im, K, distCoeff)

        with pytest.raises(ValueError):
            im = torch.rand(3, 5, 5, device=device, dtype=dtype)
            K = torch.rand(4, 4, device=device, dtype=dtype)
            distCoeff = torch.rand(4, device=device, dtype=dtype)
            undistort_image(im, K, distCoeff)

        with pytest.raises(ValueError):
            im = torch.rand(3, 5, 5, device=device, dtype=dtype)
            K = torch.rand(3, 3, device=device, dtype=dtype)
            distCoeff = torch.rand(6, device=device, dtype=dtype)
            undistort_image(im, K, distCoeff)

        with pytest.raises(ValueError):
            im = torch.randint(0, 256, (3, 5, 5), device=device, dtype=torch.uint8)
            K = torch.rand(3, 3, device=device, dtype=dtype)
            distCoeff = torch.rand(4, device=device, dtype=dtype)
            undistort_image(im, K, distCoeff)

        with pytest.raises(ValueError):
            im = torch.rand(1, 1, 3, 5, 5, device=device, dtype=dtype)
            K = torch.rand(1, 3, 3, device=device, dtype=dtype)
            distCoeff = torch.rand(1, 4, device=device, dtype=dtype)
            undistort_image(im, K, distCoeff)

    def test_opencv(self, device, dtype):
        im = torch.tensor(
            [
                [
                    [
                        [116, 75, 230, 5, 32],
                        [9, 182, 97, 213, 3],
                        [91, 10, 33, 141, 230],
                        [229, 63, 221, 244, 61],
                        [19, 137, 23, 59, 227],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        K = torch.tensor([[2, 0, 2], [0, 2, 2], [0, 0, 1]], device=device, dtype=dtype)

        dist = torch.tensor([0.2290, 0.9565, 0.0083, 0.0475], device=device, dtype=dtype)

        # Expected output generated with OpenCV:
        # import cv2
        # imu_expected = cv2.undistort(np.uint8(im[0,0].numpy()), K.numpy(), dist.numpy())
        imu_expected = torch.tensor(
            [[[[0, 0, 0, 0, 0], [0, 124, 112, 82, 0], [0, 13, 33, 158, 0], [0, 108, 197, 150, 0], [0, 0, 0, 0, 0]]]],
            device=device,
            dtype=dtype,
        )

        imu = undistort_image(im / 255.0, K, dist)
        self.assert_close(imu, imu_expected / 255.0, rtol=1e-2, atol=1e-2)

    def test_convention_zero_coefficients_are_close_but_not_byte_identical(self, device, dtype):
        # Convention pin: with every coefficient zero the
        # DISTORTION MAP is exact -- distort_points reproduces the pixel grid bit for bit (Y3-03: 0.0) -- but
        # undistort_image still routes the image through remap's bilinear sampler, so the returned image is NOT
        # byte-identical to the input; it is only equal within the dtype tolerance. This refutes the plausible
        # "zero coefficients give back the same image" reading, and it is why the pin uses assert_close where the
        # point-level pin in test_distort.py uses torch.equal. No residual bound is asserted.
        # Snippet used to generate expected: (torch.equal(undistort_image(img, K, zeros(1, 4)), img),
        # (undistort_image(...) - img).abs().max()) executed 2026-09-06 on commit c0b50ad7 (torch 2.14.0)
        # -> cpu float32 (False, 1.19e-07), float64 (False, 2.09e-16), float16 (False, 9.19e-04), bfloat16
        # (False, 7.81e-03); mps float32 (False, 1.19e-07), float16 (False, 1.84e-03). Only the mps float16 cell
        # exceeds its atol, so the skip is conditioned on the device and the cpu float16 leg still runs.
        if dtype == torch.float16 and device.type == "mps":
            pytest.skip("mps float16 only: the remap round trip leaves a 1.84e-03 residual, outside the atol")
        image = _ramp_image(1, 3, 5, 7, device, dtype)
        K = _k_short_focal(device, dtype)
        out = undistort_image(image, K, torch.zeros(1, 4, device=device, dtype=dtype))
        assert not torch.equal(out, image)
        self.assert_close(out, image)

    def test_convention_accepts_both_batch_conventions(self, device, dtype):
        # Convention pin: undistort_image accepts the batched form
        # (B, C, H, W) + (B, 3, 3) + (B, n) AND the legacy unbatched form (1, C, H, W) + (3, 3) + (n,), which the
        # source keeps "to avoid a breaking change". The legacy relaxation is special-cased to B = 1 only: the
        # same unbatched K and dist with a B = 2 image raise ValueError rather than broadcasting. The batch is
        # built from the SAME image repeated twice with different intrinsics (fx 3/5, fy 2/4, cx 3/2, cy 2/1) and
        # different coefficients, so the two output rows can only differ if each element used its own K and dist.
        # Snippet used to generate expected: shapes from undistort_image on each form, plus
        # (out[0] - out[1]).abs().max(), executed 2026-09-06 on commit c0b50ad7 (torch 2.14.0) ->
        # (2, 3, 5, 7), (1, 3, 5, 7), ValueError("Input shape is invalid. Input batch dimensions should match."),
        # and a row-to-row difference of 0.284 (cpu float32) / 0.283 (mps float32) on an image in [0, 1).
        dist = torch.tensor([[0.1, 0.01, 0.001, 0.001]], device=device, dtype=dtype)
        K = _k_short_focal(device, dtype)
        K_batch = torch.cat([K, _k_short_focal(device, dtype, fx=5.0, fy=4.0, cx=2.0, cy=1.0)])
        image = _ramp_image(1, 3, 5, 7, device, dtype)
        batched = undistort_image(torch.cat([image, image]), K_batch, torch.cat([dist, dist * 2]))
        assert batched.shape == (2, 3, 5, 7)
        assert not torch.allclose(batched[0].float(), batched[1].float())
        legacy = undistort_image(image, K[0], dist[0])
        assert legacy.shape == (1, 3, 5, 7)
        self.assert_close(legacy, batched[:1])
        with pytest.raises(ValueError, match="Input batch dimensions should match"):
            undistort_image(torch.cat([image, image]), K[0], dist[0])

    def test_convention_resamples_with_align_corners_true(self, device, dtype):
        # Convention pin: undistort_image is exactly
        # ``remap(image, mapx, mapy, align_corners=True)`` over the map that distort_points produces on the
        # create_meshgrid pixel grid -- the align_corners choice is BAKED IN and the function exposes no way to
        # change it (a documented window item; the same baked flag appears in warp_frame_depth and DepthWarper).
        # The align_corners=False arm is asserted to DIFFER on the same map, so the pin discriminates rather than
        # passing for both settings; the coefficients are non-trivial for the same reason.
        # Snippet used to generate expected: torch.equal(undistort_image(img, K, dist), remap(img, mapx, mapy,
        # align_corners=True)) executed 2026-09-06 on commit c0b50ad7 (torch 2.14.0) -> True on cpu for
        # float32/float64/float16/bfloat16 and on mps for float32/float16; the align_corners=False output is
        # visibly different on every one of those cells, which the pin asserts without a bound. On this
        # short-focal camera the map deviates from the pixel grid by 0.738 px and the undistorted image differs
        # from the input by 0.652 (cpu float32), so the pin is not comparing two copies of an unresampled image.
        image = _ramp_image(1, 3, 5, 7, device, dtype)
        K = _k_short_focal(device, dtype)
        dist = torch.tensor([[0.1, 0.01, 0.001, 0.001]], device=device, dtype=dtype)
        grid = create_meshgrid(5, 7, False, device, dtype).reshape(-1, 2)
        distorted = distort_points(grid, K, dist)
        assert not torch.allclose(distorted.float(), grid.float())
        mapx = distorted[..., 0].reshape(1, 5, 7)
        mapy = distorted[..., 1].reshape(1, 5, 7)
        out = undistort_image(image, K, dist)
        assert not torch.allclose(out.float(), image.float())
        self.assert_close(out, remap(image, mapx, mapy, align_corners=True), atol=0.0, rtol=0.0)
        assert not torch.allclose(out.float(), remap(image, mapx, mapy, align_corners=False).float())

    def test_gradcheck(self, device):
        im = torch.rand(1, 1, 15, 15, device=device, dtype=torch.float64, requires_grad=True)
        K = torch.rand(3, 3, device=device, dtype=torch.float64)
        distCoeff = torch.rand(4, device=device, dtype=torch.float64)

        self.gradcheck(undistort_image, (im, K, distCoeff), requires_grad=(True, False, False))

    @pytest.mark.xfail(reason="Some times this seems to random fail")
    def test_dynamo(self, device, dtype, torch_optimizer):
        # TODO: check if `undistort_image` fully support dynamo
        im = torch.rand(1, 3, 5, 5, device=device, dtype=dtype)
        K = torch.rand(3, 3, device=device, dtype=dtype)
        distCoeff = torch.rand(4, device=device, dtype=dtype)
        inputs = (im, K, distCoeff)

        op = undistort_image
        op_optimized = torch_optimizer(op)
        self.assert_close(op(*inputs), op_optimized(*inputs))
