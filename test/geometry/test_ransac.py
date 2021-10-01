import random

import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils
from kornia.geometry import RANSAC
from kornia.testing import assert_close


class TestRANSACHomography:
    def test_smoke(self, device, dtype):
        points1 = torch.rand(4, 2, device=device, dtype=dtype)
        points2 = torch.rand(4, 2, device=device, dtype=dtype)
        ransac = RANSAC('homography').to(device=device, dtype=dtype)
        H, inliers = ransac(points1, points2)
        assert H.shape == (3, 3)

    def test_dirty_points(self, device, dtype):
        # generate input data

        H = torch.eye(3, dtype=dtype, device=device)
        H = H * (1 + 0.1 * torch.rand_like(H))
        H = H / H[2:3, 2:3]

        points_src = 100.0 * torch.rand(1, 20, 2, device=device, dtype=dtype)
        points_dst = kornia.transform_points(H[None], points_src)

        # making last point an outlier
        points_dst[:, -1, :] += 100
        ransac = RANSAC('homography', inl_th=2.0).to(device=device, dtype=dtype)
        # compute transform from source to target
        dst_homo_src, inliers = ransac(points_src[0], points_dst[0])

        assert_close(
            kornia.transform_points(dst_homo_src[None], points_src[:, :-1]),
            points_dst[:, :-1],
            rtol=1e-3,
            atol=1e-3)


class TestRANSACFundamental:
    def test_smoke(self, device, dtype):
        points1 = torch.rand(8, 2, device=device, dtype=dtype)
        points2 = torch.rand(8, 2, device=device, dtype=dtype)
        ransac = RANSAC('fundamental').to(device=device, dtype=dtype)
        Fm, inliers = ransac(points1, points2)
        assert Fm.shape == (3, 3)

    def test_dirty_points(self, device, dtype):
        points1 = torch.tensor(
            [
                [
                    [0.8569, 0.5982],
                    [0.0059, 0.9649],
                    [0.1968, 0.8846],
                    [0.6084, 0.3467],
                    [0.9633, 0.5274],
                    [0.8941, 0.8939],
                    [0.0863, 0.5133],
                    [0.2645, 0.8882]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        points2 = torch.tensor(
            [
                [
                    [0.0928, 0.3013],
                    [0.0989, 0.9649],
                    [0.0341, 0.4827],
                    [0.8294, 0.4469],
                    [0.2230, 0.2998],
                    [0.1722, 0.8182],
                    [0.5264, 0.8869],
                    [0.8908, 0.1233]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        # generated with OpenCV using above points
        # import cv2
        # Fm_expected, _ = cv2.findFundamentalMat(
        #   points1.detach().numpy().reshape(-1, 1, 2),
        #   points2.detach().numpy().reshape(-1, 1, 2), cv2.FM_8POINT)

        Fm_expected = torch.tensor(
            [
                [
                    [0.2019, 0.6860, -0.6610],
                    [0.5520, 0.8154, -0.8044],
                    [-0.5002, -1.0254, 1.]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        ransac = RANSAC('fundamental',
                        max_iter=1,
                        inl_th=1.0).to(device=device, dtype=dtype)
        F_mat, inliers = ransac(points1[0], points2[0])
        assert_close(F_mat, Fm_expected[0], rtol=1e-3, atol=1e-3)
