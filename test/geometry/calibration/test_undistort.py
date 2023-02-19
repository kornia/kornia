import pytest
import torch

from kornia.geometry.calibration.undistort import undistort_image, undistort_points
from kornia.testing import assert_close, gradcheck


class TestUndistortPoints:
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
        assert_close(ptsu, ptsu_expected, rtol=1e-4, atol=1e-4)

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
        assert_close(ptsu, ptsu_expected, rtol=1e-4, atol=1e-4)

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
        assert_close(ptsu, ptsu_expected, rtol=1e-4, atol=1e-4)

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
        assert_close(ptsu, ptsu_expected, rtol=1e-4, atol=1e-4)

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
        assert_close(ptsu[0], ptsu_expected1, rtol=1e-4, atol=1e-4)
        assert_close(ptsu[1], ptsu_expected2, rtol=1e-4, atol=1e-4)

    def test_gradcheck(self, device):
        points = torch.rand(1, 8, 2, device=device, dtype=torch.float64, requires_grad=True)
        K = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        new_K = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        distCoeff = torch.rand(1, 4, device=device, dtype=torch.float64)

        gradcheck(undistort_points, (points, K, distCoeff, new_K))

    def test_dynamo(self, device, dtype, torch_optimizer):
        points = torch.rand(1, 1, 2, device=device, dtype=dtype)
        K = torch.rand(1, 3, 3, device=device, dtype=dtype)
        new_K = torch.rand(1, 3, 3, device=device, dtype=dtype)
        distCoeff = torch.rand(1, 4, device=device, dtype=dtype)
        inputs = (points, K, distCoeff, new_K)

        op = undistort_points
        op_optimized = torch_optimizer(op)
        assert_close(op(*inputs), op_optimized(*inputs))


class TestUndistortImage:
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
        assert_close(imu[0], imu[1])

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
        assert_close(imu, imu_expected / 255.0, rtol=1e-2, atol=1e-2)

    def test_gradcheck(self, device):
        im = torch.rand(1, 1, 15, 15, device=device, dtype=torch.float64, requires_grad=True)
        K = torch.rand(3, 3, device=device, dtype=torch.float64)
        distCoeff = torch.rand(4, device=device, dtype=torch.float64)

        gradcheck(undistort_image, (im, K, distCoeff))

    def test_jit(self, device, dtype):
        im = torch.rand(1, 3, 5, 5, device=device, dtype=dtype)
        K = torch.rand(3, 3, device=device, dtype=dtype)
        distCoeff = torch.rand(4, device=device, dtype=dtype)
        inputs = (im, K, distCoeff)

        op = undistort_image
        op_jit = torch.jit.script(op)
        assert_close(op(*inputs), op_jit(*inputs))
