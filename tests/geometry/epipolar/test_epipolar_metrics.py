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


_HALF_PIXEL_F = (
    "a pixel-unit F spans eight decades (entries down to ~1e-8): float16 flushes the small entries to zero and "
    "bfloat16's 8-bit mantissa cannot resolve the epipolar residual, which kornia evaluates in the input dtype"
)
# Fixed pixel offsets added to x2 so that the distances are non-zero.
_NOISE = [
    [1.5, -2.0], [-2.5, 1.0], [0.5, 3.0], [-1.0, -1.5], [2.0, 0.5], [-3.0, 2.5],
    [1.0, -0.5], [-0.5, -3.0], [2.5, 1.5], [-2.0, -2.5], [3.0, -1.0], [-1.5, 2.0],
]  # fmt: skip


def _hom(p: torch.Tensor) -> torch.Tensor:
    return torch.cat([p, torch.ones_like(p[..., :1])], -1)


def _pixel_F(scene) -> torch.Tensor:
    """Ground-truth F of the two-view fixture in closed form, scaled to unit Frobenius norm."""
    eye = torch.eye(3, device=scene["R"].device, dtype=scene["R"].dtype)[None]
    E = epi.essential_from_Rt(eye, torch.zeros_like(scene["t"]), scene["R"], scene["t"])
    F = epi.fundamental_from_essential(E, scene["K1"], scene["K2"])
    return F / F.norm(dim=(-2, -1), keepdim=True)


def _cpu64(x: torch.Tensor) -> torch.Tensor:
    """Reference arithmetic in float64 on the CPU (MPS has no float64)."""
    return x.detach().cpu().double()


class TestConventionEpipolarMetrics(BaseTester):
    @pytest.mark.parametrize("metric", ["sampson", "symmetrical", "left_to_right", "right_to_left"])
    def test_convention_metrics_argument_order_and_squared(self, metric, two_view, device, dtype):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip(_HALF_PIXEL_F)
        x1 = two_view["x1"]
        x2 = two_view["x2"] + torch.tensor([_NOISE], device=device, dtype=dtype)
        F = _pixel_F(two_view)
        # Reference in float64 from the point-to-epiline geometry, in pixels: pts1 are first-image points, pts2
        # second-image points, and Fm follows x2^T F x1 = 0.
        F64, p1, p2 = _cpu64(F), _cpu64(x1), _cpu64(x2)
        lines_in_2 = _hom(p1) @ F64.transpose(-2, -1)  # epilines of pts1, in the second image
        lines_in_1 = _hom(p2) @ F64  # epilines of pts2, in the first image
        algebraic = (_hom(p2) * lines_in_2).sum(-1)
        d2 = algebraic.abs() / lines_in_2[..., :2].norm(dim=-1)  # pts2 to the epiline of pts1
        d1 = algebraic.abs() / lines_in_1[..., :2].norm(dim=-1)  # pts1 to the epiline of pts2
        expected = {
            "sampson": algebraic**2 / (lines_in_2[..., :2].pow(2).sum(-1) + lines_in_1[..., :2].pow(2).sum(-1)),
            "symmetrical": d1**2 + d2**2,  # the sum of the two squared distances, not their mean
            "left_to_right": d2,
            "right_to_left": d1,
        }[metric]
        fn = getattr(epi, f"{metric}_epipolar_distance")
        self.assert_close(_cpu64(fn(x1, x2, F)), expected, rtol=1e-3, atol=0.0)
        # Control: the points swapped against the same F give a different value on every correspondence.
        assert ((_cpu64(fn(x2, x1, F)) - expected).abs() / expected).min() > 0.5
        # Relabelling the images (swap the points, transpose F) keeps Sampson and the symmetric distance and swaps
        # the two one-way distances.
        relabelled = {"left_to_right": "right_to_left", "right_to_left": "left_to_right"}.get(metric, metric)
        relabelled_fn = getattr(epi, f"{relabelled}_epipolar_distance")
        self.assert_close(_cpu64(relabelled_fn(x2, x1, F.transpose(-2, -1))), expected, rtol=1e-3, atol=0.0)
        if metric in ("sampson", "symmetrical"):
            # squared=True is the default; squared=False returns the root, in pixels. The one-way distances are
            # unsquared and have no squared argument.
            self.assert_close(_cpu64(fn(x1, x2, F, squared=False)), expected.sqrt(), rtol=1e-3, atol=0.0)

    def test_wart_metrics_eps_scale_dependence_4881(self, two_view, device, dtype):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip(_HALF_PIXEL_F)
        x1 = two_view["x1"]
        x2 = two_view["x2"] + torch.tensor([_NOISE], device=device, dtype=dtype)
        F = _pixel_F(two_view)
        # #4881: eps is added inside the denominators, so the distance depends on the scale of F: the same F at
        # ||F|| = 1e-3 scores much lower than at ||F|| = 1. Once fixed the two agree.
        for fn in (epi.sampson_epipolar_distance, epi.symmetrical_epipolar_distance):
            unit, small = fn(x1, x2, F), fn(x1, x2, 1e-3 * F)
            assert ((unit - small).abs() / unit).min() > 0.5
            # squared=False returns sqrt(d^2 + eps), so an exact match scores about sqrt(eps) = 1e-4, not 0.
            exact = fn(x1, two_view["x2"], F, squared=False)
            assert (exact > 0.9e-4).all()
