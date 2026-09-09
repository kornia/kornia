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

from kornia.geometry.camera import (
    distort_points_affine,
    distort_points_kannala_brandt,
    undistort_points_kannala_brandt,
)
from kornia.geometry.vector import Vector2
from kornia.sensors.camera.distortion_model import AffineTransform, KannalaBrandtK3Transform

from testing.base import BaseTester


class TestAffineTransform(BaseTester):
    @pytest.mark.skip(reason="Unnecessary test")
    def test_smoke(self, device, dtype):
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_gradcheck(self, device):
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_module(self, device, dtype):
        pass

    def test_distort(self, device, dtype):
        distortion = AffineTransform()
        points = torch.tensor([[1.0, 1.0], [1.0, 5.0], [2.0, 4.0], [3.0, 9.0]], device=device, dtype=dtype)
        params = torch.tensor([[328.0, 328.0, 150.0, 150.0]], device=device, dtype=dtype)
        expected = torch.tensor(
            [[478.0, 478.0], [478.0, 1790.0], [806.0, 1462.0], [1134.0, 3102.0]], device=device, dtype=dtype
        )
        self.assert_close(distortion.distort(params, Vector2(points)).data, expected)

    def test_undistort(self, device, dtype):
        distortion = AffineTransform()
        points = torch.tensor(
            [[478.0, 478.0], [478.0, 1790.0], [806.0, 1462.0], [1134.0, 3102.0]], device=device, dtype=dtype
        )
        params = torch.tensor([[328.0, 328.0, 150.0, 150.0]], device=device, dtype=dtype)
        expected = torch.tensor([[1.0, 1.0], [1.0, 5.0], [2.0, 4.0], [3.0, 9.0]], device=device, dtype=dtype)
        self.assert_close(distortion.undistort(params, Vector2(points)).data, expected)

    def test_convention_distort_is_the_affine_map_and_undistort_is_its_exact_inverse(self, device, dtype):
        # Convention pin (audit label 5d-sc-32): ``AffineTransform`` is the pinhole "distortion" -- it is the
        # map from NORMALIZED (z = 1 plane) coordinates to PIXELS, u = fx * x + cx and v = fy * y + cy, with
        # ``params`` laid out as (fx, fy, cx, cy).  It agrees byte-for-byte with
        # ``kornia.geometry.camera.distort_points_affine`` (the duplication-ledger row "geometry.camera /
        # sensors.camera", KEEP SEPARATE, kornia#4274), and ``undistort`` is its closed-form inverse -- one
        # subtraction and one division per axis, no iteration.  The "exact inverse" in this method's name is
        # scoped to the literals pinned below: on THESE representable points the round trip returns the input
        # bit-for-bit, so atol = rtol = 0 is the right assertion here rather than a rounded one.  It is NOT
        # bit-exact in general: over 2000 random float32 draws 1456 differ from the input, by up to 3.9e-06.
        # What the docstring claims, and what this pin checks, is the algebraic inverse, not float exactness.
        # The round trip is non-trivial -- the distorted point [[54.0, 15.5]] is not the input [[0.5, 0.25]]
        # -- and fx = 100 != fy = 50, cx = 4 != cy = 3 with an off-axis point, so swapping either pair
        # changes both components.
        # Snippet used to generate expected: AffineTransform().distort(tensor([100., 50., 4., 3.]),
        # Vector2(tensor([[0.5, 0.25]]))).data and the undistort of that executed 2026-09-06 on this worktree
        # (torch 2.14.0) -> [[54.0, 15.5]] and [[0.5, 0.25]], both torch.equal against distort_points_affine
        # and against the input, on cpu for float32, float64, float16 and bfloat16 and on mps for float32 and
        # float16.  With the audit's symmetric fy = 100 the distorted point is [[54.0, 28.0]].
        # Snippet used for the "not in general" figures: 2000 iterations of params = rand(4) * 200 + 1,
        # points = (rand(1, 2) - 0.5) * 4, both float32 from torch.Generator().manual_seed(0), comparing
        # torch.equal(undistort(params, distort(params, points)).data, points) -- executed 2026-09-06 on this
        # worktree (torch 2.14.0, cpu) -> 544 bit-exact, 1456 differing, max abs error 3.934e-06.
        transform = AffineTransform()
        points = torch.tensor([[0.5, 0.25]], device=device, dtype=dtype)
        params = torch.tensor([100.0, 50.0, 4.0, 3.0], device=device, dtype=dtype)
        distorted = transform.distort(params, Vector2(points))
        self.assert_close(distorted.data, torch.tensor([[54.0, 15.5]], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        assert not torch.equal(distorted.data, points)
        assert torch.equal(distorted.data, distort_points_affine(points, params))
        assert torch.equal(transform.undistort(params, distorted).data, points)
        square = torch.tensor([100.0, 100.0, 4.0, 3.0], device=device, dtype=dtype)
        self.assert_close(
            transform.distort(square, Vector2(points)).data,
            torch.tensor([[54.0, 28.0]], device=device, dtype=dtype),
            atol=0.0,
            rtol=0.0,
        )



class TestKannalaBrandtK3Transform(BaseTester):
    def test_distort_matches_geometry(self, device, dtype):
        distortion = KannalaBrandtK3Transform()
        points = torch.tensor(
            [[0.1, 0.2], [-0.3, 0.25]],
            device=device,
            dtype=dtype,
        )
        params = torch.tensor(
            [300.0, 320.0, 160.0, 120.0, 0.01, -0.001, 0.0001, -0.00001],
            device=device,
            dtype=dtype,
        )

        expected = distort_points_kannala_brandt(points, params)
        actual = distortion.distort(params, Vector2(points)).data

        self.assert_close(actual, expected, atol=0.0, rtol=0.0)

    def test_undistort_matches_geometry(self, device, dtype):
        distortion = KannalaBrandtK3Transform()
        normalized = torch.tensor(
            [[0.1, 0.2], [-0.3, 0.25]],
            device=device,
            dtype=dtype,
        )
        params = torch.tensor(
            [300.0, 320.0, 160.0, 120.0, 0.01, -0.001, 0.0001, -0.00001],
            device=device,
            dtype=dtype,
        )
        distorted = distort_points_kannala_brandt(normalized, params)

        expected = undistort_points_kannala_brandt(distorted, params)
        actual = distortion.undistort(params, Vector2(distorted)).data

        self.assert_close(actual, expected, atol=0.0, rtol=0.0)
