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

from kornia.geometry.linalg import point_line_distance

from testing.base import BaseTester


class TestPointLineDistancePrecision(BaseTester):
    def test_convention_pixel_coordinates(self, device, dtype):
        # Distance from (512, 0) to the diagonal through (0, 0), (512, 512): 512 / sqrt(2).
        point = torch.tensor([[512.0, 0.0]], device=device, dtype=dtype)
        # Cross product of homogeneous endpoints (0, 0, 1) and (512, 512, 1).
        line = torch.tensor([[-512.0, 512.0, 0.0]], device=device, dtype=dtype)
        expected = torch.tensor([362.03867196751236], device=device, dtype=dtype)
        self.assert_close(point_line_distance(point, line), expected)
        self.assert_close(point_line_distance(point, line / 512.0), expected)
        homogeneous = torch.cat([point, torch.ones_like(point[..., :1])], -1)
        self.assert_close(point_line_distance(homogeneous, line), expected)

    def test_convention_pixel_coordinate_gradients(self, device, dtype):
        point = torch.tensor([[512.0, 0.0]], device=device, dtype=dtype, requires_grad=True)
        line = torch.tensor([[-512.0, 512.0, 0.0]], device=device, dtype=dtype, requires_grad=True)
        # Differentiating |l.p| / ||l[:2]|| by the quotient rule gives these scalar literals.
        direction = torch.tensor([[0.7071067811865476, -0.7071067811865476]], device=device, dtype=dtype)
        line_derivative = torch.tensor(
            [[-0.3535533905932738, -0.3535533905932738, -0.0013810679320049757]],
            device=device,
            dtype=dtype,
        )
        point_grad, line_grad = torch.autograd.grad(point_line_distance(point, line).sum(), (point, line))
        self.assert_close(point_grad, direction)
        self.assert_close(line_grad, line_derivative)

    def test_singular_conventions_remain_finite_in_backward(self, device, dtype):
        point = torch.tensor([[512.0, 0.0]], device=device, dtype=dtype, requires_grad=True)
        line = torch.zeros(1, 3, device=device, dtype=dtype, requires_grad=True)
        distance = point_line_distance(point, line)
        assert distance.isposinf().all()
        gradient = torch.autograd.grad(distance.sum(), (point, line))
        assert all(torch.isfinite(value).all() for value in gradient)

    def test_dynamo(self, device, dtype, torch_optimizer):
        point = torch.tensor([[512.0, 0.0]], device=device, dtype=dtype)
        line = torch.tensor([[-512.0, 512.0, 0.0]], device=device, dtype=dtype)
        expected = torch.tensor([362.03867196751236], device=device, dtype=dtype)
        self.assert_close(torch_optimizer(point_line_distance)(point, line), expected)
