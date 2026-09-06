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

from kornia.geometry.camera import distort_points_affine
from kornia.geometry.vector import Vector2
from kornia.sensors.camera.distortion_model import AffineTransform, BrownConradyTransform, KannalaBrandtK3Transform

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
        # sensors.camera", KEEP SEPARATE, kornia#4274), and ``undistort`` is its exact inverse: the round trip
        # returns the input bit-for-bit, not merely within tolerance, so atol = rtol = 0 is the right
        # assertion rather than a rounded one.  The round trip is non-trivial -- the distorted point
        # [[54.0, 15.5]] is not the input [[0.5, 0.25]] -- and fx = 100 != fy = 50, cx = 4 != cy = 3 with an
        # off-axis point, so swapping either pair changes both components.
        # Snippet used to generate expected: AffineTransform().distort(tensor([100., 50., 4., 3.]),
        # Vector2(tensor([[0.5, 0.25]]))).data and the undistort of that executed 2026-09-06 on this worktree
        # (torch 2.14.0) -> [[54.0, 15.5]] and [[0.5, 0.25]], both torch.equal against distort_points_affine
        # and against the input, on cpu for float32, float64, float16 and bfloat16 and on mps for float32 and
        # float16.  With the audit's symmetric fy = 100 the distorted point is [[54.0, 28.0]].
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


class TestUnimplementedDistortions(BaseTester):
    def test_wart_brown_conrady_and_kannala_brandt_transforms_are_placeholders_4284(self, device, dtype):
        # Wart pin for kornia#4284 (audit labels 5d-sc-24, 5d-sc-25): two of the three distortion models in
        # ``kornia.sensors.camera.distortion_model`` are bare ``raise NotImplementedError`` placeholders with
        # an EMPTY message, in both directions -- ``distort`` and ``undistort``.  They are what makes
        # ``CameraModel(..., BROWN_CONRADY, ...)`` and ``CameraModel(..., KANNALA_BRANDT_K3, ...)``
        # unusable in BOTH directions (pinned at the model level in
        # tests/sensors/camera/test_camera_model.py, whose comment records the measured raise sites --
        # distortion_model.py:108/128 and :153/171 for these four calls).  Working equivalents already exist
        # next door as ``kornia.geometry.calibration.distort_points`` and
        # ``kornia.geometry.camera.distort_points_kannala_brandt``.  Parameter vectors of the documented
        # lengths (12 and 8) are used, so the raise is not a shape rejection in disguise.  The empty message
        # is asserted rather than described, because #4284's Expected asks at minimum for a message naming
        # the model: a message-only partial fix must flip this pin.
        # Snippet used to generate expected: BrownConradyTransform().distort(ones(12), Vector2([[0.5,
        # 0.25]])) and the three sibling calls executed 2026-09-06 on this worktree (torch 2.14.0) ->
        # NotImplementedError('') for all four, on cpu for float32, float64, float16 and bfloat16 and on mps
        # for float32 and float16.
        # Pins the CURRENT behaviour; NOT a contract; delete when #4284 is repaired.
        points = Vector2(torch.tensor([[0.5, 0.25]], device=device, dtype=dtype))
        for transform, length in ((BrownConradyTransform(), 12), (KannalaBrandtK3Transform(), 8)):
            params = torch.ones(length, device=device, dtype=dtype)
            for call in (transform.distort, transform.undistort):
                with pytest.raises(NotImplementedError) as raised:
                    call(params, points)
                assert str(raised.value) == ""
        affine_params = torch.ones(4, device=device, dtype=dtype)
        assert isinstance(AffineTransform().distort(affine_params, points), Vector2)
