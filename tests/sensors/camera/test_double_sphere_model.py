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

from kornia.geometry.vector import Vector2, Vector3
from kornia.image import ImageSize
from kornia.sensors.camera import CameraModel, CameraModelType, DoubleSphereModel

from testing.base import BaseTester


class TestDoubleSphereModel(BaseTester):
    _PARAMS = (300.0, 320.0, 160.0, 120.0, 0.5, 0.4)

    def test_camera_model_builds_double_sphere(self, device, dtype):
        params = torch.tensor(self._PARAMS, device=device, dtype=dtype)
        cam = CameraModel(ImageSize(240, 320), CameraModelType.DOUBLE_SPHERE, params)
        self.assert_close(cam.params, params)
        self.assert_close(
            cam.matrix(),
            torch.tensor([[300.0, 0.0, 160.0], [0.0, 320.0, 120.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype),
        )

    def test_project_and_range_unproject_round_trip(self, device, dtype):
        params = torch.tensor(self._PARAMS, device=device, dtype=dtype)
        cam = DoubleSphereModel(ImageSize(240, 320), params)
        points = torch.tensor([[0.3, -0.2, 2.0]], device=device, dtype=dtype)
        expected_pixels = torch.tensor([[189.7549202367, 98.8409456095]], device=device, dtype=dtype)

        pixels = cam.project(Vector3(points))
        distance = torch.linalg.vector_norm(points, dim=-1)
        unit_rays = cam.unproject_rays(pixels)
        reconstructed = cam.unproject(pixels, distance)

        assert isinstance(pixels, Vector2)
        assert isinstance(unit_rays, Vector3)
        self.assert_close(pixels.data, expected_pixels)
        self.assert_close(torch.linalg.vector_norm(unit_rays.data, dim=-1), torch.ones(1, device=device, dtype=dtype))
        self.assert_close(reconstructed.data, points)

    def test_unproject_accepts_negative_z_rays(self, device, dtype):
        params = torch.tensor(self._PARAMS, device=device, dtype=dtype)
        cam = CameraModel(ImageSize(240, 320), CameraModelType.DOUBLE_SPHERE, params)
        pixels = Vector2(torch.tensor([[910.0, 120.0]], device=device, dtype=dtype))
        points = cam.unproject(pixels, torch.ones(1, device=device, dtype=dtype))
        expected = torch.tensor([[0.8660254038, 0.0, -0.5]], device=device, dtype=dtype)
        self.assert_close(points.data, expected)

    def test_scale_changes_intrinsics_and_preserves_shape_parameters(self, device, dtype):
        params = torch.tensor(self._PARAMS, device=device, dtype=dtype)
        cam = DoubleSphereModel(ImageSize(240, 320), params)
        scaled = cam.scale(2.0)
        expected = torch.tensor([600.0, 640.0, 320.0, 240.0, 0.5, 0.4], device=device, dtype=dtype)
        self.assert_close(scaled.params, expected)
        assert scaled.image_size == ImageSize(480, 640)

    @pytest.mark.parametrize("shape", [(5,), (1, 1, 6)])
    def test_params_shape_is_validated(self, device, dtype, shape):
        with pytest.raises(ValueError, match="params must be of shape"):
            DoubleSphereModel(ImageSize(240, 320), torch.ones(*shape, device=device, dtype=dtype))
