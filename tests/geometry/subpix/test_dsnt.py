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

import kornia

from testing.base import BaseTester


class TestRenderGaussian2d(BaseTester):
    @pytest.fixture()
    def gaussian(self, device, dtype):
        # For a standard gaussian on 5 points [-1, -0.5, 0, 0.5, 1] with std=0.25
        # The equation is exp( -x^2 / (2 * std^2) ) -> exp( -x^2 * 8 )
        # x=0   -> exp(0)  = 1.0
        # x=0.5 -> exp(-2) ≈ 0.135335
        # x=1.0 -> exp(-8) ≈ 0.000335

        vec = torch.tensor([0.00033546, 0.13533528, 1.00000000, 0.13533528, 0.00033546], device=device, dtype=dtype)

        # Create 2D from 1D (Outer Product)
        grid = vec.unsqueeze(1) * vec.unsqueeze(0)

        # Normalize sum to 1
        return grid / grid.sum()

    def test_normalized_coordinates(self, gaussian, device, dtype):
        mean = torch.tensor([0.0, 0.0], dtype=dtype, device=device)
        std = torch.tensor([0.25, 0.25], dtype=dtype, device=device)

        actual = kornia.geometry.subpix.render_gaussian2d(mean.view(1, 2), std.view(1, 2), (5, 5), True)

        self.assert_close(actual[0], gaussian, rtol=1e-5, atol=1e-5)

    def test_pixel_coordinates(self, gaussian, device, dtype):
        mean = torch.tensor([2.0, 2.0], dtype=dtype, device=device)
        std = torch.tensor([0.5, 0.5], dtype=dtype, device=device)

        actual = kornia.geometry.subpix.render_gaussian2d(mean.view(1, 2), std.view(1, 2), (5, 5), False)

        self.assert_close(actual[0], gaussian, rtol=1e-5, atol=1e-5)

    def test_dynamo(self, device, dtype, torch_optimizer):
        mean = torch.tensor([0.0, 0.0], dtype=dtype, device=device)
        std = torch.tensor([0.25, 0.25], dtype=dtype, device=device)

        op = kornia.geometry.subpix.render_gaussian2d
        op_optimized = torch_optimizer(op)

        res_orig = op(mean.view(1, 2), std.view(1, 2), (5, 5), True)
        res_opt = op_optimized(mean.view(1, 2), std.view(1, 2), (5, 5), True)

        self.assert_close(res_orig, res_opt)


class TestSpatialSoftmax2d(BaseTester):
    @pytest.fixture(params=[torch.ones(1, 1, 5, 7), torch.randn(2, 3, 16, 16)])
    def input(self, request, device, dtype):
        return request.param.to(device, dtype)

    def test_forward(self, input):
        actual = kornia.geometry.subpix.spatial_softmax2d(input)
        assert actual.lt(0).sum().item() == 0, "expected no negative values"
        sums = actual.sum(-1).sum(-1)
        self.assert_close(sums, torch.ones_like(sums))

    def test_dynamo(self, input, torch_optimizer):
        op = kornia.geometry.subpix.spatial_softmax2d
        op_optimized = torch_optimizer(op)

        self.assert_close(op(input), op_optimized(input))


class TestSpatialExpectation2d(BaseTester):
    @pytest.fixture(
        params=[
            (
                torch.tensor([[[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]]]),
                torch.tensor([[[1.0, -1.0]]]),
                torch.tensor([[[2.0, 0.0]]]),
            )
        ]
    )
    def example(self, request, device, dtype):
        input, expected_norm, expected_px = request.param
        return input.to(device, dtype), expected_norm.to(device, dtype), expected_px.to(device, dtype)

    def test_forward(self, example):
        input, expected_norm, expected_px = example
        actual_norm = kornia.geometry.subpix.spatial_expectation2d(input, True)
        self.assert_close(actual_norm, expected_norm)
        actual_px = kornia.geometry.subpix.spatial_expectation2d(input, False)
        self.assert_close(actual_px, expected_px)

    @pytest.mark.skip("After the op be optimized the results are not the same")
    def test_dynamo(self, dtype, device, torch_optimizer):
        data = torch.tensor([[[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]]], device=device, dtype=dtype)
        op = kornia.geometry.subpix.spatial_expectation2d
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data, True), op_optimized(data, True))
