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

from kornia.geometry.camera import project_points_double_sphere, unproject_points_double_sphere

from testing.base import BaseTester


class TestDoubleSphereProjection(BaseTester):
    _PARAMS = (300.0, 320.0, 160.0, 120.0, 0.5, 0.4)

    def test_convention_projection_matches_paper_equation(self, device, dtype):
        points = torch.tensor([[0.3, -0.2, 2.0]], device=device, dtype=dtype)
        params = torch.tensor(self._PARAMS, device=device, dtype=dtype)
        # Independent double-precision evaluation of Eqs. 40-42 in Usenko et al. (2018).
        expected = torch.tensor([[189.7549202367, 98.8409456095]], device=device, dtype=dtype)
        self.assert_close(project_points_double_sphere(points, params), expected)

    def test_convention_alpha_zero_xi_zero_reduces_to_pinhole(self, device, dtype):
        points = torch.tensor([[0.3, -0.2, 2.0]], device=device, dtype=dtype)
        params = torch.tensor([300.0, 320.0, 160.0, 120.0, 0.0, 0.0], device=device, dtype=dtype)
        expected = torch.tensor([[205.0, 88.0]], device=device, dtype=dtype)
        self.assert_close(project_points_double_sphere(points, params), expected, atol=0.0, rtol=0.0)

    def test_convention_project_and_unproject_cover_rays_beyond_180_degrees(self, device, dtype):
        # At 120 degrees the ray has negative z. With these DS parameters it is inside the model's valid domain
        # (the complete projection domain extends to about 147.4 degrees from +z).
        points = torch.tensor([[0.8660254037844386, 0.0, -0.5]], device=device, dtype=dtype)
        params = torch.tensor(self._PARAMS, device=device, dtype=dtype)
        expected_pixels = torch.tensor([[910.0, 120.0]], device=device, dtype=dtype)
        expected_ray = torch.tensor([[0.8660254038, 0.0, -0.5]], device=device, dtype=dtype)

        pixels = project_points_double_sphere(points, params)
        rays = unproject_points_double_sphere(pixels, params)

        self.assert_close(pixels, expected_pixels)
        self.assert_close(rays, expected_ray)
        self.assert_close(torch.linalg.vector_norm(rays, dim=-1), torch.ones(1, device=device, dtype=dtype))

    def test_convention_principal_point_unprojects_to_optical_axis(self, device, dtype):
        params = torch.tensor(self._PARAMS, device=device, dtype=dtype)
        pixels = torch.tensor([[160.0, 120.0]], device=device, dtype=dtype)
        expected = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype)
        self.assert_close(unproject_points_double_sphere(pixels, params), expected, atol=0.0, rtol=0.0)

    def test_convention_unprojection_matches_paper_equation(self, device, dtype):
        pixels = torch.tensor([[220.0, 100.0]], device=device, dtype=dtype)
        params = torch.tensor(self._PARAMS, device=device, dtype=dtype)
        # Independent double-precision evaluation of Eqs. 46-50 in Usenko et al. (2018).
        expected = torch.tensor([[0.2929158483, -0.0915362026, 0.9517465153]], device=device, dtype=dtype)
        self.assert_close(unproject_points_double_sphere(pixels, params), expected)

    def test_convention_per_camera_params_broadcast_over_point_cloud(self, device, dtype):
        points = torch.tensor([[[1.0, 0.0, 1.0]], [[1.0, 0.0, 1.0]]], device=device, dtype=dtype)
        params = torch.tensor(
            [[100.0, 100.0, 10.0, 20.0, 0.0, 0.0], [200.0, 200.0, 20.0, 30.0, 0.0, 0.0]],
            device=device,
            dtype=dtype,
        )[:, None]
        expected = torch.tensor([[[110.0, 20.0]], [[220.0, 30.0]]], device=device, dtype=dtype)
        self.assert_close(project_points_double_sphere(points, params), expected, atol=0.0, rtol=0.0)

    def test_gradcheck_projection_and_unprojection(self, device):
        params = torch.tensor(self._PARAMS, device=device, dtype=torch.float64, requires_grad=True)
        point = torch.tensor([[0.3, -0.2, 2.0]], device=device, dtype=torch.float64, requires_grad=True)
        pixels = torch.tensor([[189.75, 98.84]], device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(project_points_double_sphere, (point, params))
        self.gradcheck(unproject_points_double_sphere, (pixels, params))

    def test_projection_and_unprojection_compile(self, device, dtype, torch_optimizer):
        params = torch.tensor(self._PARAMS, device=device, dtype=dtype)
        point = torch.tensor([[0.3, -0.2, 2.0]], device=device, dtype=dtype)
        pixels = project_points_double_sphere(point, params)
        compiled_project = torch_optimizer(project_points_double_sphere)
        compiled_unproject = torch_optimizer(unproject_points_double_sphere)
        self.assert_close(compiled_project(point, params), pixels)
        self.assert_close(compiled_unproject(pixels, params), unproject_points_double_sphere(pixels, params))

    def test_parameter_shape_is_checked(self, device, dtype):
        points = torch.ones(1, 3, device=device, dtype=dtype)
        with pytest.raises(ValueError):
            project_points_double_sphere(points, torch.ones(5, device=device, dtype=dtype))
