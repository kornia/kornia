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

from functools import partial

import pytest
import torch

import kornia.geometry.epipolar as epi
from kornia.geometry.epipolar._metrics import (
    _sampson_epipolar_distance_manual_impl_,
    _sampson_epipolar_distance_matmul_impl_,
    _sampson_epipolar_distance_shared_impl_,
    _sampson_errors,
    _sampson_from_quadratic_basis,
    _sampson_quadratic_basis,
)
from kornia.geometry.epipolar.fundamental import _rank2_projection

from testing.base import BaseTester
from testing.geometry.create import create_random_fundamental_matrix
from testing.two_view import two_view_scene


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

    def test_homogeneous_weight_4935(self, device, dtype):
        # #4935: the measured point went to point_line_distance with its weight ignored, so (10, 14, 2) scored 6.08
        # where the same point (5, 7) scored 13.72.
        Fm = torch.tensor([[[0.0, -0.02, 0.3], [0.02, 0.0, -0.9], [-0.3, 0.9, 0.1]]], device=device, dtype=dtype)
        pts1 = torch.tensor([[[10.0, 20.0]]], device=device, dtype=dtype)
        pts2 = torch.tensor([[[5.0, 7.0]]], device=device, dtype=dtype)
        pts2_weighted = torch.tensor([[[10.0, 14.0, 2.0]]], device=device, dtype=dtype)
        expected = epi.left_to_right_epipolar_distance(pts1, pts2, Fm)
        self.assert_close(expected, torch.tensor([[13.717871]], device=device, dtype=dtype))
        self.assert_close(epi.left_to_right_epipolar_distance(pts1, pts2_weighted, Fm), expected)


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

    def test_homogeneous_weight_4935(self, device, dtype):
        # #4935: the measured point went to point_line_distance with its weight ignored, so (20, 40, 2) scored 29.54
        # where the same point (10, 20) scored 11.89.
        Fm = torch.tensor([[[0.0, -0.02, 0.3], [0.02, 0.0, -0.9], [-0.3, 0.9, 0.1]]], device=device, dtype=dtype)
        pts1 = torch.tensor([[[10.0, 20.0]]], device=device, dtype=dtype)
        pts1_weighted = torch.tensor([[[20.0, 40.0, 2.0]]], device=device, dtype=dtype)
        pts2 = torch.tensor([[[5.0, 7.0]]], device=device, dtype=dtype)
        expected = epi.right_to_left_epipolar_distance(pts1, pts2, Fm)
        self.assert_close(epi.right_to_left_epipolar_distance(pts1_weighted, pts2, Fm), expected)


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


def _pixel_F_unit_norm(scene) -> torch.Tensor:
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
    def test_convention_metrics_argument_order_and_squared(self, metric, device, dtype):
        two_view = two_view_scene(device, dtype)
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip(_HALF_PIXEL_F)
        x1 = two_view["x1"]
        x2 = two_view["x2"] + torch.tensor([_NOISE], device=device, dtype=dtype)
        F = _pixel_F_unit_norm(two_view)
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
        if metric == "sampson":
            # A 3-vector point is used as given: w = 1 matches the 2-vector input, w = 2 is not dehomogenised and
            # changes every value.
            self.assert_close(fn(_hom(x1), _hom(x2), F), fn(x1, x2, F))
            assert ((fn(2.0 * _hom(x1), _hom(x2), F) - fn(x1, x2, F)).abs() / fn(x1, x2, F)).min() > 0.25

    def test_wart_metrics_eps_scale_dependence_4881(self, device, dtype):
        two_view = two_view_scene(device, dtype)
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip(_HALF_PIXEL_F)
        x1 = two_view["x1"]
        x2 = two_view["x2"] + torch.tensor([_NOISE], device=device, dtype=dtype)
        F = _pixel_F_unit_norm(two_view)
        # #4881: eps is added inside the denominators, so the distance depends on the scale of F: the same F at
        # ||F|| = 1e-3 scores much lower than at ||F|| = 1. Once fixed the two agree.
        # Select Sampson's manual path on every device; its CUDA matmul path omits denominator eps.
        for fn in (
            partial(epi.sampson_epipolar_distance, use_matmul_at_less_than_points=0),
            epi.symmetrical_epipolar_distance,
        ):
            unit, small = fn(x1, x2, F), fn(x1, x2, 1e-3 * F)
            assert ((unit - small).abs() / unit).min() > 0.5
            # squared=False returns sqrt(d^2 + eps), so an exact match scores about sqrt(eps) = 1e-4, not 0.
            exact = fn(x1, two_view["x2"], F, squared=False)
            assert (exact > 0.9e-4).all()
        # The one-way distances go through point_line_distance, which adds eps to the line norm: at ||F|| = 1e-6 the
        # value moves by a few percent.
        for fn in (epi.left_to_right_epipolar_distance, epi.right_to_left_epipolar_distance):
            unit, tiny = fn(x1, x2, F), fn(x1, x2, 1e-6 * F)
            assert ((unit - tiny).abs() / unit).min() > 1e-2


def _near_epipole_scene(n: int, seed: int):
    """Pixel correspondences 0.01-1000 px from both epipoles, which lie inside a 1920x1080 image, and their F."""
    generator = torch.Generator().manual_seed(seed)
    f64 = torch.float64
    size = torch.tensor([1920.0, 1080.0], dtype=f64)
    e1 = torch.cat([torch.rand(2, generator=generator, dtype=f64) * size, torch.ones(1, dtype=f64)])
    e2 = torch.cat([torch.rand(2, generator=generator, dtype=f64) * size, torch.ones(1, dtype=f64)])
    eye = torch.eye(3, dtype=f64)
    A = torch.randn(3, 3, generator=generator, dtype=f64)
    F = (eye - e2[:, None] * e2[None] / (e2 @ e2)) @ A @ (eye - e1[:, None] * e1[None] / (e1 @ e1))
    offsets = torch.logspace(-2, 3, n, dtype=f64)[:, None]
    x1 = e1[:2] + torch.randn(n, 2, generator=generator, dtype=f64) * offsets
    x2 = e2[:2] + torch.randn(n, 2, generator=generator, dtype=f64) * offsets
    return x1, x2, F / F[2, 2]


class TestSampsonSharedPoints(BaseTester):
    """One point set scored against many fundamental matrices: two matrix products instead of per-model broadcasting."""

    @pytest.mark.parametrize("squared", [True, False])
    def test_public_dispatch_and_eps_4881(self, device, dtype, squared):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("eps = 1e-8 is below half precision's resolution")
        # 256 matrices x 300 points reach the shared path on every device. At 1e-4 of unit scale, eps moves the
        # distances by tens of percent, so the comparison tells the CPU rule (eps in the denominator) from the CUDA
        # matmul one (no eps in the denominator, #4881).
        generator = torch.Generator().manual_seed(5)
        pts1 = torch.rand(1, 300, 2, generator=generator).to(device, dtype)
        pts2 = torch.rand(1, 300, 2, generator=generator).to(device, dtype)
        Fm = 1e-4 * create_random_fundamental_matrix(256, dtype=dtype, device=device)
        out = epi.sampson_epipolar_distance(pts1, pts2, Fm, squared=squared)
        matmul = device.type == "cuda"
        shared = _sampson_epipolar_distance_shared_impl_(pts1, pts2, Fm, squared, 1e-8, 0.0 if matmul else 1e-8)
        assert torch.equal(out, shared)
        reference = _sampson_epipolar_distance_matmul_impl_ if matmul else _sampson_epipolar_distance_manual_impl_
        expected = reference(pts1, pts2, Fm, squared, 1e-8)
        # Roundoff on the near-zero distances aside, eps in or out of the denominator is a 20-60% difference.
        self.assert_close(out, expected, rtol=1e-3, atol=1e-3 * float(expected.abs().median()))

    def test_single_matrix_without_batch_dimension(self, device, dtype):
        if device.type != "cuda":
            pytest.skip("the manual implementation, used on CPU, has always required a batch dimension on Fm")
        # The CUDA matmul implementation broadcasts a single (3, 3) matrix; the shared-point dispatch must not reject
        # it.
        pts1 = torch.rand(1, 9, 2, device=device, dtype=dtype)
        pts2 = torch.rand(1, 9, 2, device=device, dtype=dtype)
        Fm = create_random_fundamental_matrix(1, dtype=dtype, device=device)
        out = epi.sampson_epipolar_distance(pts1, pts2, Fm[0])
        assert out.shape == (1, 9)
        self.assert_close(out, epi.sampson_epipolar_distance(pts1, pts2, Fm))

    @pytest.mark.parametrize("squared", [True, False])
    def test_matches_the_other_paths(self, device, dtype, squared):
        pts1 = torch.rand(1, 20, 2, device=device, dtype=dtype)
        pts2 = torch.rand(1, 20, 2, device=device, dtype=dtype)
        # 4000 models x 20 points: the GEMM path on every device.
        Fm = create_random_fundamental_matrix(4000, dtype=dtype, device=device)
        out = epi.sampson_epipolar_distance(pts1, pts2, Fm, squared=squared)
        other = (
            _sampson_epipolar_distance_matmul_impl_
            if device.type == "cuda"
            else _sampson_epipolar_distance_manual_impl_
        )
        work = torch.promote_types(dtype, torch.float32)
        expected = other(pts1.to(work), pts2.to(work), Fm.to(work), squared, 1e-8).to(dtype)
        assert out.dtype == dtype
        self.assert_close(out, expected)

    def test_shapes(self, device, dtype):
        pts1 = torch.rand(1, 1, 7, 2, device=device, dtype=dtype)
        pts2 = torch.cat(
            [torch.rand(1, 7, 2, device=device, dtype=dtype), torch.ones(1, 7, 1, device=device, dtype=dtype)], -1
        )
        Fm = create_random_fundamental_matrix(6, dtype=dtype, device=device).reshape(2, 3, 3, 3)
        assert _sampson_epipolar_distance_shared_impl_(pts1, pts2, Fm, True, 1e-8, 1e-8).shape == (2, 3, 7)
        assert _sampson_epipolar_distance_shared_impl_(pts1[0, 0], pts2[0], Fm[0], True, 1e-8, 1e-8).shape == (3, 7)
        empty = torch.zeros(1, 0, 2, device=device, dtype=dtype)
        assert _sampson_epipolar_distance_shared_impl_(empty, empty, Fm[0], True, 1e-8, 1e-8).shape == (3, 0)

    def test_homogeneous_points_are_used_as_given(self, device, dtype):
        pts1 = torch.rand(1, 9, 2, device=device, dtype=dtype)
        pts2 = torch.rand(1, 9, 2, device=device, dtype=dtype)
        Fm = create_random_fundamental_matrix(3, dtype=dtype, device=device)
        scaled = 2.0 * _hom(pts1)
        out = _sampson_epipolar_distance_shared_impl_(scaled, pts2, Fm, True, 1e-8, 1e-8)
        work = torch.promote_types(dtype, torch.float32)
        expected = _sampson_epipolar_distance_manual_impl_(scaled.to(work), pts2.to(work), Fm.to(work), True, 1e-8)
        self.assert_close(out, expected.to(dtype))

    def test_non_contiguous_and_mixed_dtype_models(self, device, dtype):
        pts1 = torch.rand(1, 11, 2, device=device, dtype=torch.float64)
        pts2 = torch.rand(1, 11, 2, device=device, dtype=torch.float64)
        Fm = create_random_fundamental_matrix(4, dtype=dtype, device=device)
        out = _sampson_epipolar_distance_shared_impl_(pts1, pts2, Fm.mT, True, 1e-8, 1e-8)
        assert out.dtype == torch.promote_types(torch.float64, dtype)
        self.assert_close(
            out, _sampson_epipolar_distance_shared_impl_(pts1, pts2, Fm.mT.contiguous(), True, 1e-8, 1e-8)
        )
        one = Fm[:1].expand(5, 3, 3)
        expanded = _sampson_epipolar_distance_shared_impl_(pts1, pts2, one, True, 1e-8, 1e-8)
        self.assert_close(expanded, expanded[:1].expand_as(expanded))

    def test_residual_order_regression(self, device, dtype):
        if dtype != torch.float32:
            pytest.skip("a float32 cancellation case (half cannot hold 1000.1 or 2e6; float64 is the reference)")
        # PR #5031 review: a numerator expanded into monomials of the coordinates cancels here (0.0521 against a
        # float64 value of 0.0403); forming x2 . (F x1) keeps the manual path's 0.0319.
        F = torch.tensor(
            [[1.0, 0.0, -1000.0], [0.0, 1.0, -1000.0], [-1000.0, -1000.0, 2e6]], device=device, dtype=dtype
        )
        x1 = torch.tensor([[[1000.1, 1000.2]]], device=device, dtype=dtype)
        x2 = torch.tensor([[[1000.3, 1000.4]]], device=device, dtype=dtype)
        models = F.expand(64, 3, 3)
        shared = _sampson_epipolar_distance_shared_impl_(x1, x2, models, True, 0.0, 0.0)[:, 0].double().cpu()
        manual = _sampson_epipolar_distance_manual_impl_(x1, x2, F[None], True, 0.0)[0, 0].double().cpu()
        reference = _sampson_epipolar_distance_manual_impl_(x1.double(), x2.double(), F[None].double(), True, 0.0)
        reference = reference[0, 0].cpu()
        self.assert_close(shared, manual.expand_as(shared), rtol=1e-4, atol=0.0)
        assert (shared - reference).abs().max() <= 2 * (manual - reference).abs()

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_near_epipoles_in_pixels(self, device, dtype, seed):
        if dtype != torch.float32:
            pytest.skip("the float32 cancellation case")
        x1, x2, F = _near_epipole_scene(2000, seed=seed)
        perturb = 1 + 1e-4 * torch.randn(7, 3, 3, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
        models = torch.cat([F[None], F[None] * perturb])
        reference = _sampson_epipolar_distance_manual_impl_(x1[None], x2[None], models, True, 0.0)
        cast = lambda t: t.to(device, dtype)  # noqa: E731
        shared = _sampson_epipolar_distance_shared_impl_(cast(x1[None]), cast(x2[None]), cast(models), True, 0.0, 0.0)
        manual = _sampson_epipolar_distance_manual_impl_(cast(x1[None]), cast(x2[None]), cast(models), True, 0.0)
        shared, manual = shared.double().cpu(), manual.double().cpu()
        assert torch.isfinite(shared).all()
        assert (shared >= 0).all()
        # Within 0.1 px of an epipole both float32 computations are rounding noise (the maximum error of either is
        # an accident of summation order), so compare the error distributions: the tail and the number of
        # distances off by more than 1 px^2. An expanded monomial residual fails this on seeds 1 and 3.
        small = reference < 100
        error_shared = (shared - reference).abs()[small]
        error_manual = (manual - reference).abs()[small]
        assert error_shared.quantile(0.99) <= 1.25 * error_manual.quantile(0.99)
        assert (error_shared > 1).sum() <= 1.25 * (error_manual > 1).sum() + 2

    def test_ransac_quadratic_form_near_epipoles(self, device, dtype):
        if dtype != torch.float32:
            pytest.skip("RANSAC scores in float32 on the device")
        x1, x2, F = _near_epipole_scene(2000, seed=0)
        n1, t1 = epi.normalize_points(x1[None])
        n2, t2 = epi.normalize_points(x2[None])
        h1, h2 = _hom(n1[0]), _hom(n2[0])
        Fn = torch.linalg.inv(t2[0]).mT @ F @ torch.linalg.inv(t1[0])
        Fn = Fn / Fn.norm()
        generator = torch.Generator().manual_seed(2)
        perturbed = Fn + 1e-3 * torch.randn(7, 3, 3, generator=generator, dtype=torch.float64)
        models = torch.cat([Fn[None], _rank2_projection(perturbed)])
        models = models / models.flatten(1).norm(dim=1)[:, None, None]
        reference = _sampson_errors(models, h1, h2, 0.0)
        cast = lambda t: t.to(device, torch.float32)  # noqa: E731
        errors = _sampson_from_quadratic_basis(cast(models), _sampson_quadratic_basis(cast(h1), cast(h2)))
        errors = errors.double().cpu()
        assert not errors.isnan().any()
        assert (errors >= 0).all()
        scale = float(t2[0, 0, 0])
        decisions = 0
        flips = 0
        for px in (0.5, 1.0, 2.0, 4.0):
            threshold = (px * scale) ** 2
            flips += int(((errors <= threshold) != (reference <= threshold)).sum())
            decisions += errors.numel()
        # Measured 1.2e-5 of the decisions over 40 such scenes (PR #5031 review).
        assert flips <= 1e-4 * decisions

    def test_gradcheck(self, device):
        pts1 = torch.rand(1, 5, 2, device=device, dtype=torch.float64)
        pts2 = torch.rand(1, 5, 2, device=device, dtype=torch.float64)
        Fm = create_random_fundamental_matrix(3, dtype=torch.float64, device=device)
        self.gradcheck(
            lambda a, b, F: _sampson_epipolar_distance_shared_impl_(a, b, F, True, 1e-8, 1e-8), (pts1, pts2, Fm)
        )
