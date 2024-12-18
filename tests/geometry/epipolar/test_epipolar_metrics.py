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

import torch

import kornia.geometry.epipolar as epi

from testing.base import BaseTester
from testing.geometry.create import create_random_fundamental_matrix


class TestSymmetricalEpipolarDistance(BaseTester):
    def test_smoke(self, device, dtype):
        pts1 = torch.rand(1, 4, 3, device=device, dtype=dtype)
        pts2 = torch.rand(1, 4, 3, device=device, dtype=dtype)
        Fm = create_random_fundamental_matrix(1, dtype=dtype, device=device)
        assert epi.symmetrical_epipolar_distance(pts1, pts2, Fm).shape == (1, 4)

    def test_batch(self, device, dtype):
        batch_size = 5
        pts1 = torch.rand(batch_size, 4, 3, device=device, dtype=dtype)
        pts2 = torch.rand(batch_size, 4, 3, device=device, dtype=dtype)
        Fm = create_random_fundamental_matrix(1, dtype=dtype, device=device)
        assert epi.symmetrical_epipolar_distance(pts1, pts2, Fm).shape == (5, 4)

    def test_frames(self, device, dtype):
        batch_size, num_frames, num_points = 5, 3, 4
        pts1 = torch.rand(batch_size, num_frames, num_points, 3, device=device, dtype=dtype)
        pts2 = torch.rand(batch_size, num_frames, num_points, 3, device=device, dtype=dtype)
        Fm = torch.stack(
            [create_random_fundamental_matrix(1, dtype=dtype, device=device) for _ in range(num_frames)], dim=1
        )
        dist_frame_by_frame = torch.stack(
            [
                epi.symmetrical_epipolar_distance(pts1[:, t, ...], pts2[:, t, ...], Fm[:, t, ...])
                for t in range(num_frames)
            ],
            dim=1,
        )
        dist_all_frames = epi.symmetrical_epipolar_distance(pts1, pts2, Fm)
        self.assert_close(dist_frame_by_frame, dist_all_frames, atol=1e-6, rtol=1e-6)

    def test_gradcheck(self, device):
        # generate input data
        batch_size, num_points, num_dims = 2, 3, 2
        points1 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64, requires_grad=True)
        points2 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64)
        Fm = create_random_fundamental_matrix(batch_size, dtype=torch.float64, device=device)

        self.gradcheck(epi.symmetrical_epipolar_distance, (points1, points2, Fm), requires_grad=(True, False, False))

    def test_shift(self, device, dtype):
        pts1 = torch.zeros(3, 2, device=device, dtype=dtype)[None]
        pts2 = torch.tensor([[2, 0.0], [2, 1], [2, 2.0]], device=device, dtype=dtype)[None]
        Fm = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=dtype, device=device)[None]
        expected = torch.tensor([0.0, 2.0, 8.0], device=device, dtype=dtype)[None]
        self.assert_close(epi.symmetrical_epipolar_distance(pts1, pts2, Fm), expected, atol=1e-4, rtol=1e-4)


class TestSampsonEpipolarDistance(BaseTester):
    def test_smoke(self, device, dtype):
        pts1 = torch.rand(1, 4, 3, device=device, dtype=dtype)
        pts2 = torch.rand(1, 4, 3, device=device, dtype=dtype)
        Fm = create_random_fundamental_matrix(1, dtype=dtype, device=device)

        assert epi.sampson_epipolar_distance(pts1, pts2, Fm).shape == (1, 4)

    def test_batch(self, device, dtype):
        batch_size = 5
        pts1 = torch.rand(batch_size, 4, 3, device=device, dtype=dtype)
        pts2 = torch.rand(batch_size, 4, 3, device=device, dtype=dtype)
        Fm = create_random_fundamental_matrix(1, dtype=dtype, device=device)
        assert epi.sampson_epipolar_distance(pts1, pts2, Fm).shape == (5, 4)

    def test_frames(self, device, dtype):
        batch_size, num_frames, num_points = 5, 3, 4
        pts1 = torch.rand(batch_size, num_frames, num_points, 3, device=device, dtype=dtype)
        pts2 = torch.rand(batch_size, num_frames, num_points, 3, device=device, dtype=dtype)
        Fm = torch.stack(
            [create_random_fundamental_matrix(1, dtype=dtype, device=device) for _ in range(num_frames)], dim=1
        )
        dist_frame_by_frame = torch.stack(
            [epi.sampson_epipolar_distance(pts1[:, t, ...], pts2[:, t, ...], Fm[:, t, ...]) for t in range(num_frames)],
            dim=1,
        )
        dist_all_frames = epi.sampson_epipolar_distance(pts1, pts2, Fm)
        self.assert_close(dist_frame_by_frame, dist_all_frames, atol=1e-6, rtol=1e-6)

    def test_shift(self, device, dtype):
        pts1 = torch.zeros(3, 2, device=device, dtype=dtype)[None]
        pts2 = torch.tensor([[2, 0.0], [2, 1], [2, 2.0]], device=device, dtype=dtype)[None]
        Fm = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=dtype, device=device)[None]
        expected = torch.tensor([0.0, 0.5, 2.0], device=device, dtype=dtype)[None]
        self.assert_close(epi.sampson_epipolar_distance(pts1, pts2, Fm), expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        # generate input data
        batch_size, num_points, num_dims = 2, 3, 2
        points1 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64, requires_grad=True)
        points2 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64)
        Fm = create_random_fundamental_matrix(batch_size, dtype=torch.float64, device=device)

        self.gradcheck(epi.sampson_epipolar_distance, (points1, points2, Fm), requires_grad=(True, False, False))


class TestLeftToRightEpipolarDistance(BaseTester):
    def test_smoke(self, device, dtype):
        pts1 = torch.rand(1, 4, 3, device=device, dtype=dtype)
        pts2 = torch.rand(1, 4, 3, device=device, dtype=dtype)
        Fm = create_random_fundamental_matrix(1, dtype=dtype, device=device)

        assert epi.left_to_right_epipolar_distance(pts1, pts2, Fm).shape == (1, 4)

    def test_batch(self, device, dtype):
        batch_size = 5
        pts1 = torch.rand(batch_size, 4, 3, device=device, dtype=dtype)
        pts2 = torch.rand(batch_size, 4, 3, device=device, dtype=dtype)
        Fm = create_random_fundamental_matrix(1, dtype=dtype, device=device)
        assert epi.left_to_right_epipolar_distance(pts1, pts2, Fm).shape == (5, 4)

    def test_frames(self, device, dtype):
        batch_size, num_frames, num_points = 5, 3, 4
        pts1 = torch.rand(batch_size, num_frames, num_points, 3, device=device, dtype=dtype)
        pts2 = torch.rand(batch_size, num_frames, num_points, 3, device=device, dtype=dtype)
        Fm = torch.stack(
            [create_random_fundamental_matrix(1, dtype=dtype, device=device) for _ in range(num_frames)], dim=1
        )
        dist_frame_by_frame = torch.stack(
            [
                epi.left_to_right_epipolar_distance(pts1[:, t, ...], pts2[:, t, ...], Fm[:, t, ...])
                for t in range(num_frames)
            ],
            dim=1,
        )
        dist_all_frames = epi.left_to_right_epipolar_distance(pts1, pts2, Fm)
        self.assert_close(dist_frame_by_frame, dist_all_frames, atol=1e-6, rtol=1e-6)

    def test_shift(self, device, dtype):
        pts1 = torch.zeros(3, 2, device=device, dtype=dtype)[None]
        pts2 = torch.tensor([[2, 0.0], [2, 1], [2, 2.0]], device=device, dtype=dtype)[None]
        Fm = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=dtype, device=device)[None]
        expected = torch.tensor([0.0, 1.0, 2.0], device=device, dtype=dtype)[None]
        self.assert_close(epi.left_to_right_epipolar_distance(pts1, pts2, Fm), expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        # generate input data
        batch_size, num_points, num_dims = 2, 3, 2
        points1 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64, requires_grad=True)
        points2 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64)
        Fm = create_random_fundamental_matrix(batch_size, dtype=torch.float64, device=device)

        self.gradcheck(epi.left_to_right_epipolar_distance, (points1, points2, Fm), requires_grad=(True, False, False))


class TestRightToLeftEpipolarDistance(BaseTester):
    def test_smoke(self, device, dtype):
        pts1 = torch.rand(1, 4, 3, device=device, dtype=dtype)
        pts2 = torch.rand(1, 4, 3, device=device, dtype=dtype)
        Fm = create_random_fundamental_matrix(1, dtype=dtype, device=device)

        assert epi.right_to_left_epipolar_distance(pts1, pts2, Fm).shape == (1, 4)

    def test_batch(self, device, dtype):
        batch_size = 5
        pts1 = torch.rand(batch_size, 4, 3, device=device, dtype=dtype)
        pts2 = torch.rand(batch_size, 4, 3, device=device, dtype=dtype)
        Fm = create_random_fundamental_matrix(1, dtype=dtype, device=device)
        assert epi.right_to_left_epipolar_distance(pts1, pts2, Fm).shape == (5, 4)

    def test_frames(self, device, dtype):
        batch_size, num_frames, num_points = 5, 3, 4
        pts1 = torch.rand(batch_size, num_frames, num_points, 3, device=device, dtype=dtype)
        pts2 = torch.rand(batch_size, num_frames, num_points, 3, device=device, dtype=dtype)
        Fm = torch.stack(
            [create_random_fundamental_matrix(1, dtype=dtype, device=device) for _ in range(num_frames)], dim=1
        )
        dist_frame_by_frame = torch.stack(
            [
                epi.right_to_left_epipolar_distance(pts1[:, t, ...], pts2[:, t, ...], Fm[:, t, ...])
                for t in range(num_frames)
            ],
            dim=1,
        )
        dist_all_frames = epi.right_to_left_epipolar_distance(pts1, pts2, Fm)
        self.assert_close(dist_frame_by_frame, dist_all_frames, atol=1e-6, rtol=1e-6)

    def test_shift(self, device, dtype):
        pts1 = torch.zeros(3, 2, device=device, dtype=dtype)[None]
        pts2 = torch.tensor([[2, 0.0], [2, 1], [2, 2.0]], device=device, dtype=dtype)[None]
        Fm = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=dtype, device=device)[None]
        expected = torch.tensor([0.0, 1.0, 2.0], device=device, dtype=dtype)[None]
        self.assert_close(epi.right_to_left_epipolar_distance(pts1, pts2, Fm), expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        # generate input data
        batch_size, num_points, num_dims = 2, 3, 2
        points1 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64, requires_grad=True)
        points2 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64)
        Fm = create_random_fundamental_matrix(batch_size, dtype=torch.float64, device=device)

        self.gradcheck(epi.right_to_left_epipolar_distance, (points1, points2, Fm), requires_grad=(True, False, False))
