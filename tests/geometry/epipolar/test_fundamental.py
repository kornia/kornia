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

from typing import Dict

import pytest
import torch

import kornia.geometry.epipolar as epi

from testing.base import BaseTester
from testing.geometry.create import create_random_fundamental_matrix, generate_two_view_random_scene


class TestNormalizePoints(BaseTester):
    def test_smoke(self, device, dtype):
        points = torch.rand(1, 1, 2, device=device, dtype=dtype)
        output = epi.normalize_points(points)
        assert len(output) == 2
        assert output[0].shape == (1, 1, 2)
        assert output[1].shape == (1, 3, 3)

    @pytest.mark.parametrize("batch_size, num_points", [(1, 2), (2, 3), (3, 2)])
    def test_shape(self, batch_size, num_points, device, dtype):
        B, N = batch_size, num_points
        points = torch.rand(B, N, 2, device=device, dtype=dtype)
        output = epi.normalize_points(points)
        assert output[0].shape == (B, N, 2)
        assert output[1].shape == (B, 3, 3)

    def test_mean_std(self, device, dtype):
        points = torch.tensor([[[0.0, 0.0], [0.0, 2.0], [1.0, 1.0], [1.0, 3.0]]], device=device, dtype=dtype)

        points_norm, _ = epi.normalize_points(points)
        points_std, points_mean = torch.std_mean(points_norm, dim=1)

        self.assert_close(points_mean, torch.zeros_like(points_mean))
        assert (points_std < 2.0).all()

    def test_gradcheck(self, device):
        points = torch.rand(2, 3, 2, device=device, requires_grad=True, dtype=torch.float64)
        self.gradcheck(epi.normalize_points, (points,))


class TestNormalizeTransformation(BaseTester):
    def test_smoke(self, device, dtype):
        trans = torch.rand(2, 2, device=device, dtype=dtype)
        trans_norm = epi.normalize_transformation(trans)
        assert trans_norm.shape == (2, 2)

    @pytest.mark.parametrize("batch_size, rows, cols", [(1, 2, 2), (2, 3, 3), (3, 4, 4), (2, 1, 2)])
    def test_shape(self, batch_size, rows, cols, device, dtype):
        B, N, M = batch_size, rows, cols
        trans = torch.rand(B, N, M, device=device, dtype=dtype)
        trans_norm = epi.normalize_transformation(trans)
        assert trans_norm.shape == (B, N, M)

    def test_check_last_val(self, device, dtype):
        trans = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 2.0, 0.0], [0.5, 0.0, 0.5]]], device=device, dtype=dtype)

        trans_expected = torch.tensor([[[0.0, 0.0, 2.0], [0.0, 4.0, 0.0], [1.0, 0.0, 1.0]]], device=device, dtype=dtype)

        trans_norm = epi.normalize_transformation(trans)
        self.assert_close(trans_norm, trans_expected, atol=1e-4, rtol=1e-4)

    def test_check_corner_case(self, device, dtype):
        trans = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 2.0, 0.0], [0.5, 0.0, 0.0]]], device=device, dtype=dtype)

        trans_expected = trans.clone()

        trans_norm = epi.normalize_transformation(trans)
        self.assert_close(trans_norm, trans_expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        trans = torch.rand(2, 3, 3, device=device, requires_grad=True, dtype=torch.float64)
        self.gradcheck(epi.normalize_transformation, (trans,))


class TestFindFundamental(BaseTester):
    def test_smoke(self, device, dtype):
        points1 = torch.rand(1, 8, 2, device=device, dtype=dtype)
        points2 = torch.rand(1, 8, 2, device=device, dtype=dtype)
        weights = torch.ones(1, 8, device=device, dtype=dtype)
        F_mat = epi.find_fundamental(points1, points2, weights)
        assert F_mat.shape == (1, 3, 3)

    @pytest.mark.parametrize("batch_size, num_points", [(1, 8), (2, 9), (3, 10)])
    def test_shape(self, batch_size, num_points, device, dtype):
        B, N = batch_size, num_points
        points1 = torch.rand(B, N, 2, device=device, dtype=dtype)
        points2 = torch.rand(B, N, 2, device=device, dtype=dtype)
        weights = torch.ones(B, N, device=device, dtype=dtype)
        F_mat = epi.find_fundamental(points1, points2, weights)
        assert F_mat.shape == (B, 3, 3)

    @pytest.mark.parametrize("batch_size, num_points", [(1, 8), (2, 8), (3, 10)])
    def test_shape_noweights(self, batch_size, num_points, device, dtype):
        B, N = batch_size, num_points
        points1 = torch.rand(B, N, 2, device=device, dtype=dtype)
        points2 = torch.rand(B, N, 2, device=device, dtype=dtype)
        weights = None
        F_mat = epi.find_fundamental(points1, points2, weights)
        assert F_mat.shape == (B, 3, 3)

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_shape_7point(self, batch_size, device, dtype):
        B = batch_size
        points1 = torch.rand(B, 7, 2, device=device, dtype=dtype)
        points2 = torch.rand(B, 7, 2, device=device, dtype=dtype)
        torch.ones(B, 7, device=device, dtype=dtype)
        F_mat = epi.find_fundamental(points1, points2, method="7POINT")
        assert F_mat.shape == (B, 3, 3, 3)

    def test_opencv_svd(self, device, dtype):
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
                    [0.2645, 0.8882],
                    [0.2411, 0.3045],
                    [0.8199, 0.4107],
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
                    [0.8908, 0.1233],
                    [0.2338, 0.7663],
                    [0.4466, 0.5696],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        weights = torch.ones(1, 10, device=device, dtype=dtype)

        # generated with OpenCV using above points
        # import cv2
        # Fm_expected, _ = cv2.findFundamentalMat(
        #   points1.detach().numpy().reshape(-1, 1, 2),
        #   points2.detach().numpy().reshape(-1, 1, 2), cv2.FM_8POINT)

        Fm_expected = torch.tensor(
            [
                [
                    [-0.47408533, 0.22033807, -0.00346677],
                    [0.54935973, 1.31080955, -1.25028275],
                    [-0.36690215, -1.08143769, 1.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        F_mat = epi.find_fundamental(points1, points2, weights)
        self.assert_close(F_mat, Fm_expected, rtol=1e-4, atol=1e-4)

    def test_7point_opencv(self, device, dtype):
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
                ]
            ],
            device=device,
            dtype=dtype,
        )

        # generated with OpenCV using above points
        # Fm_expected shape is 9x3
        # import cv2
        # Fm_expected, _ = cv2.findFundamentalMat(
        #   points1.detach().numpy().reshape(-1, 1, 2),
        #   points2.detach().numpy().reshape(-1, 1, 2), cv2.FM_7POINT)

        Fm_expected = torch.tensor(
            [
                [
                    [
                        [-2.87490907, 5.41934672, 0.73871396],
                        [0.34010174, 3.70371623, -4.65517276],
                        [-0.1809933, -0.56577107, 1.0],
                    ],
                    [
                        [0.14465888, 0.68711702, -0.65570944],
                        [0.53424758, 0.7988479, -0.75446946],
                        [-0.48201197, -1.05375511, 1.0],
                    ],
                    [
                        [-0.0901827, 1.05515785, -0.54726062],
                        [0.51914823, 1.02476892, -1.05783979],
                        [-0.45860077, -1.01580301, 1.0],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        # We need this voodoo, because the order of the solutions is not guaranteed by the algorithm.
        F_mat = epi.find_fundamental(points1, points2, method="7POINT")
        ordering = []
        for expected in Fm_expected[0]:
            min_diff = float("inf")
            for i, estimated in enumerate(F_mat[0]):
                diff = (expected - estimated).abs().sum()
                if diff < min_diff:
                    min_diff = diff
                    min_index = i
            ordering.append(min_index)
        F_mat[0] = F_mat[0][ordering]
        self.assert_close(F_mat, Fm_expected, rtol=1e-3, atol=1e-3)

    def test_synthetic_sampson_7point(self, device, dtype):
        scene: Dict[str, torch.Tensor] = generate_two_view_random_scene(device, dtype)
        x1 = scene["x1"][:, :7, :]
        x2 = scene["x2"][:, :7, :]
        F_est = epi.find_fundamental(x1, x2, None, "7POINT")
        for i in range(3):
            F = F_est[0][i].unsqueeze(0)
            if torch.all(F != 0):
                error = epi.sampson_epipolar_distance(x1, x2, F)
                self.assert_close(error, torch.zeros((F.shape[0], 7), device=device, dtype=dtype), atol=1e-4, rtol=1e-4)

    @pytest.mark.xfail()
    def test_epipolar_constraint_7point(self, device, dtype):
        scene: Dict[str, torch.Tensor] = generate_two_view_random_scene(device, dtype)
        x1 = scene["x1"][:, :7, :]
        x2 = scene["x2"][:, :7, :]
        F_est = epi.find_fundamental(x1, x2, None, "7POINT")
        for i in range(3):
            F = F_est[0][i].unsqueeze(0)
            if torch.all(F != 0):
                distance = epi.symmetrical_epipolar_distance(x1, x2, F)
                mean_error = distance.mean()
                self.assert_close(mean_error, torch.tensor(0.0, device=device, dtype=dtype), atol=1e-4, rtol=1e-4)

    def test_synthetic_sampson(self, device, dtype):
        scene: Dict[str, torch.Tensor] = generate_two_view_random_scene(device, dtype)

        x1 = scene["x1"]
        x2 = scene["x2"]

        weights = torch.ones_like(x1)[..., 0]
        F_est = epi.find_fundamental(x1, x2, weights)

        error = epi.sampson_epipolar_distance(x1, x2, F_est)
        self.assert_close(error, torch.zeros((x1.shape[:2]), device=device, dtype=dtype), atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        points1 = torch.rand(1, 10, 2, device=device, dtype=torch.float64, requires_grad=True)
        points2 = torch.rand(1, 10, 2, device=device, dtype=torch.float64)
        weights = torch.ones(1, 10, device=device, dtype=torch.float64)
        self.gradcheck(epi.find_fundamental, (points1, points2, weights))


class TestComputeCorrespondEpilines(BaseTester):
    def test_smoke(self, device, dtype):
        point = torch.rand(1, 1, 2, device=device, dtype=dtype)
        F_mat = torch.rand(1, 3, 3, device=device, dtype=dtype)
        lines = epi.compute_correspond_epilines(point, F_mat)
        assert lines.shape == (1, 1, 3)

    @pytest.mark.parametrize("batch_size, num_points", [(1, 2), (2, 3), (3, 2)])
    def test_shape(self, batch_size, num_points, device, dtype):
        B, N = batch_size, num_points
        point = torch.rand(B, N, 2, device=device, dtype=dtype)
        F_mat = torch.rand(B, 3, 3, device=device, dtype=dtype)
        lines = epi.compute_correspond_epilines(point, F_mat)
        assert lines.shape == (B, N, 3)

    @pytest.mark.parametrize(
        "batch_size, num_frames, num_points",
        [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2), (2, 1, 1), (2, 1, 2), (2, 2, 1), (2, 2, 2)],
    )
    def test_volumetric(self, batch_size, num_frames, num_points, device, dtype):
        B, T, N = batch_size, num_frames, num_points
        point = torch.rand(B, T, N, 2, device=device, dtype=dtype)
        F_mat = torch.rand(B, T, 3, 3, device=device, dtype=dtype)

        lines_T_hops = torch.zeros(B, T, N, 3, device=device, dtype=dtype)
        for i in range(T):
            lines_T_hops[:, i, ...] = epi.compute_correspond_epilines(point[:, i, ...], F_mat[:, i, ...])
        lines_one_hop = epi.compute_correspond_epilines(point, F_mat)

        self.assert_close(lines_T_hops, lines_one_hop, atol=2e-7, rtol=2e-7)

    def test_opencv(self, device, dtype):
        point = torch.rand(1, 2, 2, device=device, dtype=dtype)
        F_mat = torch.rand(1, 3, 3, device=device, dtype=dtype)

        point = torch.tensor([[[0.9794, 0.7994], [0.8163, 0.8500]]], device=device, dtype=dtype)

        F_mat = torch.tensor(
            [[[0.1185, 0.4438, 0.9869], [0.5670, 0.9447, 0.4100], [0.1546, 0.2554, 0.4485]]], device=device, dtype=dtype
        )

        # generated with OpenCV using above points
        # import cv2
        # lines_expected = cv2.computeCorrespondEpilines(
        #    point.detach().numpy().reshape(-1, 1, 2), 0,
        #    F_mat.detach().numpy()[0]).transpose(1, 0, 2)

        lines_expected = torch.tensor(
            [[[0.64643687, 0.7629675, 0.35658622], [0.65710586, 0.7537983, 0.35616538]]], device=device, dtype=dtype
        )

        lines_est = epi.compute_correspond_epilines(point, F_mat)
        self.assert_close(lines_est, lines_expected, rtol=1e-4, atol=1e-4)

    def test_gradcheck(self, device):
        point = torch.rand(1, 4, 2, device=device, dtype=torch.float64, requires_grad=True)
        F_mat = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        self.gradcheck(epi.compute_correspond_epilines, (point, F_mat), requires_grad=(True, False))


class TestFundamentlFromEssential(BaseTester):
    def test_smoke(self, device, dtype):
        E_mat = torch.rand(1, 3, 3, device=device, dtype=dtype)
        K1 = torch.rand(1, 3, 3, device=device, dtype=dtype)
        K2 = torch.rand(1, 3, 3, device=device, dtype=dtype)
        F_mat = epi.fundamental_from_essential(E_mat, K1, K2)
        assert F_mat.shape == (1, 3, 3)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 7])
    def test_shape(self, batch_size, device, dtype):
        B: int = batch_size
        E_mat = torch.rand(B, 3, 3, device=device, dtype=dtype)
        K1 = torch.rand(B, 3, 3, device=device, dtype=dtype)
        K2 = torch.rand(1, 3, 3, device=device, dtype=dtype)  # check broadcasting
        F_mat = epi.fundamental_from_essential(E_mat, K1, K2)
        assert F_mat.shape == (B, 3, 3)

    def test_shape_large(self, device, dtype):
        E_mat = torch.rand(1, 2, 3, 3, device=device, dtype=dtype)
        K1 = torch.rand(1, 2, 3, 3, device=device, dtype=dtype)
        K2 = torch.rand(1, 1, 3, 3, device=device, dtype=dtype)  # check broadcasting
        F_mat = epi.fundamental_from_essential(E_mat, K1, K2)
        assert F_mat.shape == (1, 2, 3, 3)

    def test_from_to_essential(self, device, dtype):
        scene = generate_two_view_random_scene(device, dtype)

        F_mat = scene["F"]
        E_mat = epi.essential_from_fundamental(F_mat, scene["K1"], scene["K2"])
        F_hat = epi.fundamental_from_essential(E_mat, scene["K1"], scene["K2"])

        F_mat_norm = epi.normalize_transformation(F_mat)
        F_hat_norm = epi.normalize_transformation(F_hat)
        self.assert_close(F_mat_norm, F_hat_norm, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        E_mat = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        K1 = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        K2 = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        self.gradcheck(epi.fundamental_from_essential, (E_mat, K1, K2), requires_grad=(True, False, False))


class TestFundamentalFromProjections(BaseTester):
    def test_smoke(self, device, dtype):
        P1 = torch.rand(1, 3, 4, device=device, dtype=dtype)
        P2 = torch.rand(1, 3, 4, device=device, dtype=dtype)
        F_mat = epi.fundamental_from_projections(P1, P2)
        assert F_mat.shape == (1, 3, 3)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 7])
    def test_shape(self, batch_size, device, dtype):
        B: int = batch_size
        P1 = torch.rand(B, 3, 4, device=device, dtype=dtype)
        P2 = torch.rand(B, 3, 4, device=device, dtype=dtype)
        F_mat = epi.fundamental_from_projections(P1, P2)
        assert F_mat.shape == (B, 3, 3)

    def test_shape_large(self, device, dtype):
        P1 = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        P2 = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        F_mat = epi.fundamental_from_projections(P1, P2)
        assert F_mat.shape == (1, 2, 3, 3)

    def test_from_to_projections(self, device, dtype):
        P1 = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]]], device=device, dtype=dtype
        )

        P2 = torch.tensor(
            [[[1.0, 1.0, 1.0, 3.0], [0.0, 2.0, 0.0, 3.0], [0.0, 1.0, 1.0, 0.0]]], device=device, dtype=dtype
        )

        F_mat = epi.fundamental_from_projections(P1, P2)
        P_mat = epi.projections_from_fundamental(F_mat)
        F_hat = epi.fundamental_from_projections(P_mat[..., 0], P_mat[..., 1])

        F_mat_norm = epi.normalize_transformation(F_mat)
        F_hat_norm = epi.normalize_transformation(F_hat)
        self.assert_close(F_mat_norm, F_hat_norm, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        P1 = torch.rand(1, 3, 4, device=device, dtype=torch.float64, requires_grad=True)
        P2 = torch.rand(1, 3, 4, device=device, dtype=torch.float64)
        self.gradcheck(epi.fundamental_from_projections, (P1, P2), requires_grad=(True, False))

    def test_batch_support_check(self, device, dtype):
        P1_batch = torch.tensor(
            [
                [
                    [9.4692e02, -9.6658e02, 6.0862e02, -2.3076e05],
                    [-2.1829e02, 5.4163e02, 1.3445e03, -6.4387e05],
                    [-6.0675e-01, -6.9807e-01, 3.8021e-01, 3.8896e02],
                ],
                [
                    [9.4692e02, -9.6658e02, 6.0862e02, -2.3076e05],
                    [-2.1829e02, 5.4163e02, 1.3445e03, -6.4387e05],
                    [-6.0675e-01, -6.9807e-01, 3.8021e-01, 3.8896e02],
                ],
            ],
            device=device,
            dtype=dtype,
        )
        P1 = torch.tensor(
            [
                [
                    [9.4692e02, -9.6658e02, 6.0862e02, -2.3076e05],
                    [-2.1829e02, 5.4163e02, 1.3445e03, -6.4387e05],
                    [-6.0675e-01, -6.9807e-01, 3.8021e-01, 3.8896e02],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        P2_batch = torch.tensor(
            [
                [
                    [1.1518e03, -7.5822e02, 5.4764e02, -1.9764e05],
                    [-2.1548e02, 5.3102e02, 1.3492e03, -6.4731e05],
                    [-4.3727e-01, -7.8632e-01, 4.3646e-01, 3.4515e02],
                ],
                [
                    [9.9595e02, -8.6464e02, 6.7959e02, -2.7517e05],
                    [-8.1716e01, 7.7826e02, 1.2395e03, -5.8137e05],
                    [-5.7090e-01, -6.0416e-01, 5.5594e-01, 2.8111e02],
                ],
            ],
            device=device,
            dtype=dtype,
        )
        P2 = torch.tensor(
            [
                [
                    [1.1518e03, -7.5822e02, 5.4764e02, -1.9764e05],
                    [-2.1548e02, 5.3102e02, 1.3492e03, -6.4731e05],
                    [-4.3727e-01, -7.8632e-01, 4.3646e-01, 3.4515e02],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        F_batch = epi.fundamental_from_projections(P1_batch, P2_batch)
        F = epi.fundamental_from_projections(P1, P2)
        self.assert_close(F_batch[0], F[0])


class TestPerpendicular(BaseTester):
    def test_shape(self, device, dtype):
        lines = torch.rand(2, 4, 3, device=device, dtype=dtype)
        points = torch.rand(2, 4, 2, device=device, dtype=dtype)
        perp = epi.get_perpendicular(lines, points)
        assert perp.shape == (2, 4, 3)

    def test_result(self, device, dtype):
        points = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], device=device, dtype=dtype)

        lines = torch.tensor([[[1.0, -1.0, 0.0], [0.0, 1.0, 1.0]]], device=device, dtype=dtype)
        perp = epi.get_perpendicular(lines, points)
        expected = torch.tensor([[[1.0, 1.0, -1.0], [-1.0, 0.0, 0.0]]], device=device, dtype=dtype)
        self.assert_close(perp, expected)

    def test_gradcheck(self, device):
        pt = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        line = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        self.gradcheck(epi.get_perpendicular, (pt, line), requires_grad=(True, False))


class TestGetClosestPointOnEpipolarLine(BaseTester):
    def test_shape(self, device, dtype):
        pts1 = torch.rand(2, 4, 2, device=device, dtype=dtype)
        pts2 = torch.rand(2, 4, 2, device=device, dtype=dtype)
        Fm = create_random_fundamental_matrix(1, device=device, dtype=dtype)
        perp = epi.get_closest_point_on_epipolar_line(pts1, pts2, Fm)
        assert perp.shape == (2, 4, 2)

    def test_shift(self, device, dtype):
        pts1 = torch.zeros(3, 2, device=device, dtype=dtype)[None]
        pts2 = torch.tensor([[2, 4.0], [2, 1], [2, 2.0]], device=device, dtype=dtype)[None]
        Fm = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=dtype, device=device)[None]
        cp = epi.get_closest_point_on_epipolar_line(pts1, pts2, Fm)
        expected = torch.tensor([[[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]]], device=device, dtype=dtype)
        self.assert_close(cp, expected)

    def test_gradcheck(self, device):
        pts1 = torch.rand(2, 4, 2, device=device, dtype=torch.float64, requires_grad=True)
        pts2 = torch.rand(2, 4, 2, device=device, dtype=torch.float64)
        Fm = create_random_fundamental_matrix(1, dtype=torch.float64, device=device)
        self.gradcheck(epi.get_closest_point_on_epipolar_line, (pts1, pts2, Fm), requires_grad=(True, False, False))


_NO_HALF_EIGH = "find_fundamental calls torch.linalg.eigh, which has no float16/bfloat16 kernel"
_HALF_PIXEL_F = (
    "a pixel-unit F spans eight decades (entries down to ~1e-8): float16 flushes the small entries to zero and "
    "bfloat16's 8-bit mantissa cannot resolve the epipolar residual, which kornia evaluates in the input dtype"
)
# Fixed pixel offsets added to x2 where a pin needs inexact matches.
_NOISE = [
    [1.5, -2.0], [-2.5, 1.0], [0.5, 3.0], [-1.0, -1.5], [2.0, 0.5], [-3.0, 2.5],
    [1.0, -0.5], [-0.5, -3.0], [2.5, 1.5], [-2.0, -2.5], [3.0, -1.0], [-1.5, 2.0],
]  # fmt: skip


def _skip_half(dtype: torch.dtype, reason: str) -> None:
    if dtype in (torch.float16, torch.bfloat16):
        pytest.skip(reason)


def _hom(p: torch.Tensor) -> torch.Tensor:
    return torch.cat([p, torch.ones_like(p[..., :1])], -1)


def _epipolar_residual(F: torch.Tensor, pts1: torch.Tensor, pts2: torch.Tensor) -> torch.Tensor:
    """|pts2^T F pts1| per correspondence."""
    return (_hom(pts2) * (_hom(pts1) @ F.transpose(-2, -1))).sum(-1).abs()


def _pixel_F(scene: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Ground-truth F of the two-view fixture in closed form, scaled so that F[2, 2] = 1."""
    eye = torch.eye(3, device=scene["R"].device, dtype=scene["R"].dtype)[None]
    E = epi.essential_from_Rt(eye, torch.zeros_like(scene["t"]), scene["R"], scene["t"])
    F = epi.fundamental_from_essential(E, scene["K1"], scene["K2"])
    return F / F[..., 2:, 2:]


class TestConventionFundamental(BaseTester):
    def test_convention_find_fundamental_acts_x2_F_x1(self, two_view, device, dtype):
        _skip_half(dtype, _NO_HALF_EIGH)
        x1, x2 = two_view["x1"], two_view["x2"]
        F = epi.find_fundamental(x1, x2, torch.ones_like(x1[..., 0]))
        # x2^T F x1 = 0 for points1 from the first image and points2 from the second (OpenCV's findFundamentalMat
        # order). The swapped product is the control: on this fixture it is of order 1.
        assert _epipolar_residual(F, x1, x2).max() < 1e-3 * _epipolar_residual(F, x2, x1).max()
        # Relabelling the images returns the transpose.
        self.assert_close(epi.find_fundamental(x2, x1), F.transpose(-2, -1), rtol=1e-4, atol=1e-4)
        # The result is scaled so that F[2, 2] = 1.
        self.assert_close(F[..., 2, 2], torch.ones_like(F[..., 2, 2]))

    def test_convention_find_fundamental_7point_candidates(self, two_view, device, dtype):
        _skip_half(dtype, _NO_HALF_EIGH)
        # A 7-point sample whose cubic has three real roots, so all three candidates are genuine solutions.
        idx = [0, 1, 2, 3, 4, 5, 6]
        x1, x2 = two_view["x1"][:, idx], two_view["x2"][:, idx]
        F = epi.find_fundamental(x1, x2, method="7POINT")
        assert F.shape == (1, 3, 3, 3)
        for k in range(3):
            Fk = F[:, k]
            assert _epipolar_residual(Fk, x1, x2).max() < 1e-3 * _epipolar_residual(Fk, x2, x1).max()
            sv = torch.linalg.svdvals(Fk.cpu().double())
            assert sv[..., 2] < 1e-8 * sv[..., 0]  # rank 2
        self.assert_close(F[..., 2, 2], torch.ones_like(F[..., 2, 2]))
        # The candidate order carries no meaning: exactly one candidate fits all twelve points, and which index it
        # has changes with the dtype on this sample, so it is selected by residual, never by position.
        fits = [bool(_epipolar_residual(F[:, k], two_view["x1"], two_view["x2"]).max() < 1e-2) for k in range(3)]
        assert sum(fits) == 1
        # Relabelling the images returns the transposed candidates, as a set.
        Fs = epi.find_fundamental(x2, x1, method="7POINT").transpose(-2, -1)
        dist = (F[0, :, None] - Fs[0, None]).abs().amax(dim=(-2, -1))
        assert dist.amin(dim=1).max() < 1e-3

    def test_convention_fundamental_from_essential_K_sides(self, two_view, device, dtype):
        _skip_half(dtype, _HALF_PIXEL_F)
        K1, K2, x1, x2 = two_view["K1"], two_view["K2"], two_view["x1"], two_view["x2"]
        eye = torch.eye(3, device=device, dtype=dtype)[None]
        E = epi.essential_from_Rt(eye, torch.zeros_like(two_view["t"]), two_view["R"], two_view["t"])
        F = epi.fundamental_from_essential(E, K1, K2)
        # F = K2^-T E K1^-1: K1 is the camera of points1, so x2^T F x1 = 0 in pixels.
        self.assert_close(K2.transpose(-2, -1) @ F @ K1, E)
        F_swapped = epi.fundamental_from_essential(E, K2, K1)
        resid = _epipolar_residual(F / F.norm(), x1, x2)
        assert resid.max() < 1e-3 * _epipolar_residual(F_swapped / F_swapped.norm(), x1, x2).max()

    def test_convention_epilines_of_image1_points_lie_in_image2(self, two_view, device, dtype):
        _skip_half(dtype, _HALF_PIXEL_F)
        x1, x2 = two_view["x1"], two_view["x2"]
        F = _pixel_F(two_view)
        # Lines F x1 of first-image points live in the second image, scaled to a^2 + b^2 = 1.
        lines = epi.compute_correspond_epilines(x1, F)
        self.assert_close(lines[..., :2].norm(dim=-1), torch.ones_like(lines[..., 0]))
        on, off = (_hom(x2) * lines).sum(-1).abs(), (_hom(x1) * lines).sum(-1).abs()
        assert on.max() < 1e-3 * off.max()
        # For second-image points pass F transposed: the lines live in the first image.
        lines1 = epi.compute_correspond_epilines(x2, F.transpose(-2, -1))
        self.assert_close(lines1[..., :2].norm(dim=-1), torch.ones_like(lines1[..., 0]))
        on1, off1 = (_hom(x1) * lines1).sum(-1).abs(), (_hom(x2) * lines1).sum(-1).abs()
        assert on1.max() < 1e-3 * off1.max()

    def test_convention_normalize_points_hartley(self, two_view, device, dtype):
        # Hartley normalisation: translate to zero mean and scale isotropically to mean distance sqrt(2); the
        # returned T maps the input onto the output. The fixture's spread differs in x and y.
        points = two_view["x1"]
        points_norm, T = epi.normalize_points(points)
        mean_atol = {torch.bfloat16: 3e-2, torch.float16: 3e-3}.get(dtype, 1e-5)
        self.assert_close(points_norm.mean(dim=1), torch.zeros_like(points_norm[:, 0]), rtol=0.0, atol=mean_atol)
        mean_dist = points_norm.norm(dim=-1).mean(dim=-1)
        self.assert_close(mean_dist, torch.full_like(mean_dist, 2.0**0.5))
        self.assert_close((_hom(points) @ T.transpose(-2, -1))[..., :2], points_norm, low_tolerance=True)
        assert T[0, 0, 0] == T[0, 1, 1]
        assert T[0, 0, 1] == 0 and T[0, 1, 0] == 0

    def test_convention_get_closest_point_on_epipolar_line_in_image2(self, two_view, device, dtype):
        _skip_half(dtype, _HALF_PIXEL_F)
        x1 = two_view["x1"]
        x2 = two_view["x2"] + torch.tensor([_NOISE], device=device, dtype=dtype)
        F = _pixel_F(two_view)
        # The result lies in the second image, on the epiline F x1, at the foot of the perpendicular from x2.
        closest = epi.get_closest_point_on_epipolar_line(x1, x2, F)
        lines = epi.compute_correspond_epilines(x1, F)
        assert (_hom(closest) * lines).sum(-1).abs().max() < 1e-3
        self.assert_close((closest - x2).norm(dim=-1), epi.left_to_right_epipolar_distance(x1, x2, F))
        # Control: with the arguments swapped the point is off that line by pixels.
        swapped = epi.get_closest_point_on_epipolar_line(x2, x1, F)
        assert (_hom(swapped) * lines).sum(-1).abs().max() > 1.0

    def test_wart_run_7point_padded_roots_returned_4862(self, two_view, device, dtype):
        _skip_half(dtype, _NO_HALF_EIGH)
        # #4862: a 7-point sample whose cubic has one real root. The solver pads the two missing roots with 0.0 and the
        # validity mask never fires, so candidates 1 and 2 are the same rank-3 matrix instead of being zeroed.
        # Any fix (zeros, NaN, or fewer candidates) makes these assertions fail.
        idx = [0, 1, 2, 3, 4, 6, 10]
        F = epi.find_fundamental(two_view["x1"][:, idx], two_view["x2"][:, idx], method="7POINT")
        assert F.shape == (1, 3, 3, 3)
        assert torch.equal(F[:, 1], F[:, 2])
        sv = torch.linalg.svdvals(F[0].cpu().double())
        assert sv[0, 2] < 1e-8 * sv[0, 0]  # candidate 0 is a genuine rank-2 solution
        assert (sv[1:, 2] > 1e-7 * sv[1:, 0]).all()  # the padded candidates are rank 3

    def test_wart_normalize_transformation_eps_divisor_4874(self, device, dtype):
        # #4874: the divisor is M[2, 2] + eps, so the last entry is 1 only to eps / |M[2, 2]|.
        M = torch.tensor([[[2.0, 0.5, 3.0], [-1.0, 4.0, 0.25], [0.75, -2.0, 0.1]]], device=device, dtype=dtype)
        out = epi.normalize_transformation(M, eps=1e-3)
        assert (out[0, 2, 2] - 1.0).abs() > 5e-3
        if dtype != torch.float16:  # float16 rounds 1e-6 + 1e-8 back to 1e-6
            M[0, 2, 2] = 1e-6
            assert (epi.normalize_transformation(M)[0, 2, 2] - 1.0).abs() > 5e-3

    def test_wart_find_fundamental_zero_weight_changes_result_4875(self, two_view, device, dtype):
        _skip_half(dtype, _NO_HALF_EIGH)
        # #4875: a correspondence with weight 0 leaves the linear system but still enters the Hartley
        # normalisation, so a far outlier with weight 0 moves the estimate. Once fixed, the two estimates agree.
        x1 = two_view["x1"]
        x2 = two_view["x2"] + torch.tensor([_NOISE], device=device, dtype=dtype)
        outlier1 = torch.tensor([[[2000.0, -1500.0]]], device=device, dtype=dtype)
        outlier2 = torch.tensor([[[-1800.0, 2500.0]]], device=device, dtype=dtype)
        weights = torch.ones(1, 13, device=device, dtype=dtype)
        weights[0, 12] = 0.0
        F_weighted = epi.find_fundamental(torch.cat([x1, outlier1], 1), torch.cat([x2, outlier2], 1), weights)
        F_dropped = epi.find_fundamental(x1, x2)
        err_weighted = epi.sampson_epipolar_distance(x1, x2, F_weighted).mean()
        err_dropped = epi.sampson_epipolar_distance(x1, x2, F_dropped).mean()
        assert err_weighted > 2.0 * err_dropped

    def test_wart_fundamental_from_projections_float16_overflow_4877(self, two_view, device, dtype):
        if dtype != torch.float16:
            pytest.skip("the overflow is float16's: bfloat16, float32 and float64 hold pixel-unit 4x4 determinants")
        # #4877: the 4x4 determinants of pixel-unit projection matrices exceed float16's range, so F holds inf.
        F = epi.fundamental_from_projections(two_view["P1"], two_view["P2"])
        assert F.dtype == torch.float16
        assert torch.isinf(F).any()
