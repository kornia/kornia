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

        # Forward distortion of the independent OpenCV values must recover the input (#4276).
        self.assert_close(distort_points(ptsu_expected, K, dist), pts, rtol=1e-4, atol=1e-4)

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
        # undistort_points is the iterative (5-step fixed point by default) inverse of distort_points on pixel
        # points and a (3, 3) K; the round trip closes to the dtype tolerance with moderate coefficients. The
        # points sit half a focal length off the principal point, so the forward map moves them (2.775 px) and an
        # identity undistort_points would fail. Outside the valid radius the iteration cycles (#4285, below).
        points = torch.tensor([[[54.0, 53.0], [-16.0, 23.0]]], device=device, dtype=dtype)
        K = _k_asymmetric(device, dtype)
        dist = torch.tensor([[0.1, 0.01, 0.001, 0.001]], device=device, dtype=dtype)
        distorted = distort_points(points, K, dist)
        assert not torch.allclose(distorted.float(), points.float())
        self.assert_close(undistort_points(distorted, K, dist), points)
        self.assert_close(undistort_points(distorted, K, dist, num_iters=50), points)

    def test_convention_distort_undistort_round_trip_with_tilt_4276(self, device, dtype):
        # Unequal tilt angles plus non-zero radial/tangential coefficients expose #4276.
        points = torch.tensor([[[54.0, 53.0], [-16.0, 23.0]]], device=device, dtype=dtype)
        K = _k_asymmetric(device, dtype)
        radial = torch.tensor([[0.1, 0.01, 0.001, 0.001]], device=device, dtype=dtype)
        zero_tilt = torch.cat([radial, torch.zeros(1, 10, device=device, dtype=dtype)], -1)
        tilted = torch.cat([zero_tilt[:, :12], torch.tensor([[0.1, 0.2]], device=device, dtype=dtype)], -1)
        # float16 accumulates two ulps through forward tilt and iterative inversion;
        # increasing num_iters from 5 to 50 leaves the same 0.0625-pixel rounding floor.
        rtol = 2 * torch.finfo(dtype).eps if dtype == torch.float16 else None
        atol = 0.0 if dtype == torch.float16 else None
        self.assert_close(undistort_points(distort_points(points, K, tilted), K, tilted), points, rtol=rtol, atol=atol)
        self.assert_close(undistort_points(distort_points(points, K, zero_tilt), K, zero_tilt), points)

    def test_convention_new_K_denormalizes_and_K_normalizes(self, device, dtype):
        # The mirror image of distort_points: K maps the incoming pixel onto the normalized z = 1 plane and new_K
        # maps it back to pixels. Every intrinsic is distinct, so a swap changes the literal: (10, 10) normalizes
        # under K to ((10-1)/2, (10-1)/3) = (4.5, 3.0) and denormalizes under new_K to (5*4.5+2, 7*3+4).
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
        # Inside the convergence region, more iterations shrink the residual. Halving fy puts the five-step
        # answer outside assert_close's tolerance, so the decrease is not between already-converged answers.
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
        # Wart pin for #4285: outside the valid radius the fixed-point iteration silently alternates between two
        # inaccurate values (odd and even counts agree, even is worse) instead of converging or reporting it.
        # Delete when #4285 is repaired.
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

    def test_convention_zero_coefficients_are_a_no_op_to_tolerance(self, device, dtype):
        # With every coefficient zero the image still goes through remap's bilinear sampler, so it comes back
        # equal to the input within the dtype tolerance.
        if dtype == torch.float16 and device.type == "mps":
            pytest.skip("mps float16: the remap round trip residual exceeds the float16 atol")
        image = _ramp_image(1, 3, 5, 7, device, dtype)
        K = _k_short_focal(device, dtype)
        out = undistort_image(image, K, torch.zeros(1, 4, device=device, dtype=dtype))
        self.assert_close(out, image)

    def test_convention_accepts_both_batch_conventions(self, device, dtype):
        # Batched (B, C, H, W) + (B, 3, 3) + (B, n) and the legacy unbatched (1, C, H, W) + (3, 3) + (n,) are both
        # accepted; the legacy form is B = 1 only (a B = 2 image with an unbatched K raises, no broadcasting).
        # The same image with two different K and dist must give two different rows.
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
        # undistort_image is remap(image, mapx, mapy, align_corners=True) over the map distort_points produces
        # on the create_meshgrid pixel grid; align_corners is fixed and not exposed. The align_corners=False arm
        # must differ, so the pin discriminates between the two settings.
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
