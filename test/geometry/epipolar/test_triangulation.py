from __future__ import annotations

from typing import Dict

import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.geometry.epipolar as epi
from kornia.testing import assert_close


class TestTriangulation:
    def test_smoke(self, device, dtype):
        P1 = torch.rand(1, 3, 4, device=device, dtype=dtype)
        P2 = torch.rand(1, 3, 4, device=device, dtype=dtype)
        points1 = torch.rand(1, 1, 2, device=device, dtype=dtype)
        points2 = torch.rand(1, 1, 2, device=device, dtype=dtype)
        points3d = epi.triangulate_points(P1, P2, points1, points2)
        assert points3d.shape == (1, 1, 3)

    @pytest.mark.parametrize("batch_size, num_points", [(1, 3), (2, 4), (3, 5)])
    def test_shape(self, batch_size, num_points, device, dtype):
        B, N = batch_size, num_points
        P1 = torch.rand(B, 3, 4, device=device, dtype=dtype)
        P2 = torch.rand(1, 3, 4, device=device, dtype=dtype)
        points1 = torch.rand(1, N, 2, device=device, dtype=dtype)
        points2 = torch.rand(B, N, 2, device=device, dtype=dtype)
        points3d = epi.triangulate_points(P1, P2, points1, points2)
        assert points3d.shape == (B, N, 3)

    @pytest.mark.xfail
    def test_two_view(self, device, dtype):
        num_views: int = 2
        num_points: int = 10
        scene: dict[str, torch.Tensor] = epi.generate_scene(num_views, num_points)

        P1 = scene['P'][0:1]
        P2 = scene['P'][1:2]
        x1 = scene['points2d'][0:1]
        x2 = scene['points2d'][1:2]

        X = epi.triangulate_points(P1, P2, x1, x2)
        x_reprojected = kornia.geometry.transform_points(scene['P'], X.expand(num_views, -1, -1))

        assert_close(scene['points3d'], X, rtol=1e-4, atol=1e-4)
        assert_close(scene['points2d'], x_reprojected, rtol=1e-4, atol=1e-4)

    def test_gradcheck(self, device):
        points1 = torch.rand(1, 8, 2, device=device, dtype=torch.float64, requires_grad=True)
        points2 = torch.rand(1, 8, 2, device=device, dtype=torch.float64)
        P1 = kornia.eye_like(3, points1)
        P1 = torch.nn.functional.pad(P1, [0, 1])
        P2 = kornia.eye_like(3, points2)
        P2 = torch.nn.functional.pad(P2, [0, 1])
        assert gradcheck(epi.triangulate_points, (P1, P2, points1, points2), raise_exception=True)
