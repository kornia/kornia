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
from torch.nn.functional import mse_loss

import kornia
from kornia.geometry.subpix.spatial_soft_argmax import (
    _get_center_kernel2d,
    _get_center_kernel3d,
    iterative_quad_interp3d,
)

from testing.base import BaseTester


class TestCenterKernel2d(BaseTester):
    def test_smoke(self, device, dtype):
        kernel = _get_center_kernel2d(3, 4, device=device).to(dtype=dtype)
        assert kernel.shape == (2, 2, 3, 4)

    def test_odd(self, device, dtype):
        kernel = _get_center_kernel2d(3, 3, device=device).to(dtype=dtype)
        expected = torch.tensor(
            [
                [
                    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ],
                [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                ],
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(kernel, expected, atol=1e-4, rtol=1e-4)

    def test_even(self, device, dtype):
        kernel = _get_center_kernel2d(2, 2, device=device).to(dtype=dtype)
        expected = torch.ones(2, 2, 2, 2, device=device, dtype=dtype) * 0.25
        expected[0, 1] = 0
        expected[1, 0] = 0
        self.assert_close(kernel, expected, atol=1e-4, rtol=1e-4)


class TestCenterKernel3d(BaseTester):
    def test_smoke(self, device, dtype):
        kernel = _get_center_kernel3d(6, 3, 4, device=device).to(dtype=dtype)
        assert kernel.shape == (3, 3, 6, 3, 4)

    def test_odd(self, device, dtype):
        kernel = _get_center_kernel3d(3, 5, 7, device=device).to(dtype=dtype)
        expected = torch.zeros(3, 3, 3, 5, 7, device=device, dtype=dtype)
        expected[0, 0, 1, 2, 3] = 1.0
        expected[1, 1, 1, 2, 3] = 1.0
        expected[2, 2, 1, 2, 3] = 1.0
        self.assert_close(kernel, expected, atol=1e-4, rtol=1e-4)

    def test_even(self, device, dtype):
        kernel = _get_center_kernel3d(2, 4, 3, device=device).to(dtype=dtype)
        expected = torch.zeros(3, 3, 2, 4, 3, device=device, dtype=dtype)
        expected[0, 0, :, 1:3, 1] = 0.25
        expected[1, 1, :, 1:3, 1] = 0.25
        expected[2, 2, :, 1:3, 1] = 0.25
        self.assert_close(kernel, expected, atol=1e-4, rtol=1e-4)


class TestSpatialSoftArgmax2d(BaseTester):
    def test_smoke(self, device, dtype):
        sample = torch.zeros(1, 1, 2, 3, device=device, dtype=dtype)
        m = kornia.geometry.subpix.SpatialSoftArgmax2d()
        assert m(sample).shape == (1, 1, 2)

    def test_smoke_batch(self, device, dtype):
        sample = torch.zeros(2, 1, 2, 3, device=device, dtype=dtype)
        m = kornia.geometry.subpix.SpatialSoftArgmax2d()
        assert m(sample).shape == (2, 1, 2)

    def test_top_left_normalized(self, device, dtype):
        sample = torch.zeros(1, 1, 2, 3, device=device, dtype=dtype)
        sample[..., 0, 0] = 1e16

        coord = kornia.geometry.subpix.spatial_soft_argmax2d(sample, normalized_coordinates=True)
        self.assert_close(coord[..., 0].item(), -1.0, atol=1e-4, rtol=1e-4)
        self.assert_close(coord[..., 1].item(), -1.0, atol=1e-4, rtol=1e-4)

    def test_top_left(self, device, dtype):
        sample = torch.zeros(1, 1, 2, 3, device=device, dtype=dtype)
        sample[..., 0, 0] = 1e16

        coord = kornia.geometry.subpix.spatial_soft_argmax2d(sample, normalized_coordinates=False)
        self.assert_close(coord[..., 0].item(), 0.0, atol=1e-4, rtol=1e-4)
        self.assert_close(coord[..., 1].item(), 0.0, atol=1e-4, rtol=1e-4)

    def test_bottom_right_normalized(self, device, dtype):
        sample = torch.zeros(1, 1, 2, 3, device=device, dtype=dtype)
        sample[..., -1, -1] = 1e16

        coord = kornia.geometry.subpix.spatial_soft_argmax2d(sample, normalized_coordinates=True)
        self.assert_close(coord[..., 0].item(), 1.0, atol=1e-4, rtol=1e-4)
        self.assert_close(coord[..., 1].item(), 1.0, atol=1e-4, rtol=1e-4)

    def test_bottom_right(self, device, dtype):
        sample = torch.zeros(1, 1, 2, 3, device=device, dtype=dtype)
        sample[..., -1, -1] = 1e16

        coord = kornia.geometry.subpix.spatial_soft_argmax2d(sample, normalized_coordinates=False)
        self.assert_close(coord[..., 0].item(), 2.0, atol=1e-4, rtol=1e-4)
        self.assert_close(coord[..., 1].item(), 1.0, atol=1e-4, rtol=1e-4)

    def test_batch2_n2(self, device, dtype):
        sample = torch.zeros(2, 2, 2, 3, device=device, dtype=dtype)
        sample[0, 0, 0, 0] = 1e16  # top-left
        sample[0, 1, 0, -1] = 1e16  # top-right
        sample[1, 0, -1, 0] = 1e16  # bottom-left
        sample[1, 1, -1, -1] = 1e16  # bottom-right

        coord = kornia.geometry.subpix.spatial_soft_argmax2d(sample)
        self.assert_close(coord[0, 0, 0].item(), -1.0, atol=1e-4, rtol=1e-4)  # top-left
        self.assert_close(coord[0, 0, 1].item(), -1.0, atol=1e-4, rtol=1e-4)
        self.assert_close(coord[0, 1, 0].item(), 1.0, atol=1e-4, rtol=1e-4)  # top-right
        self.assert_close(coord[0, 1, 1].item(), -1.0, atol=1e-4, rtol=1e-4)
        self.assert_close(coord[1, 0, 0].item(), -1.0, atol=1e-4, rtol=1e-4)  # bottom-left
        self.assert_close(coord[1, 0, 1].item(), 1.0, atol=1e-4, rtol=1e-4)
        self.assert_close(coord[1, 1, 0].item(), 1.0, atol=1e-4, rtol=1e-4)  # bottom-right
        self.assert_close(coord[1, 1, 1].item(), 1.0, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        sample = torch.rand(2, 3, 3, 2, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.subpix.spatial_soft_argmax2d, (sample))

    def test_end_to_end(self, device, dtype):
        sample = torch.full((1, 2, 7, 7), 1.0, requires_grad=True, device=device, dtype=dtype)
        target = torch.as_tensor([[[0.0, 0.0], [1.0, 1.0]]], device=device, dtype=dtype)
        std = torch.tensor([1.0, 1.0], device=device, dtype=dtype)

        hm = kornia.geometry.subpix.spatial_softmax2d(sample)
        self.assert_close(
            hm.sum(-1).sum(-1), torch.tensor([[1.0, 1.0]], device=device, dtype=dtype), atol=1e-4, rtol=1e-4
        )

        pred = kornia.geometry.subpix.spatial_expectation2d(hm)
        self.assert_close(
            pred, torch.as_tensor([[[0.0, 0.0], [0.0, 0.0]]], device=device, dtype=dtype), atol=1e-4, rtol=1e-4
        )

        loss1 = mse_loss(pred, target, size_average=None, reduce=None, reduction="none").mean(-1, keepdim=False)
        expected_loss1 = torch.as_tensor([[0.0, 1.0]], device=device, dtype=dtype)
        self.assert_close(loss1, expected_loss1, atol=1e-4, rtol=1e-4)

        target_hm = kornia.geometry.subpix.render_gaussian2d(target, std, sample.shape[-2:]).contiguous()

        loss2 = kornia.losses.js_div_loss_2d(hm, target_hm, reduction="none")
        expected_loss2 = torch.as_tensor([[0.0087, 0.0818]], device=device, dtype=dtype)
        self.assert_close(loss2, expected_loss2, rtol=0, atol=1e-3)

        loss = (loss1 + loss2).mean()
        loss.backward()

    def test_dynamo(self, device, dtype, torch_optimizer):
        data = torch.rand((2, 3, 7, 7), dtype=dtype, device=device)
        op = kornia.geometry.subpix.spatial_soft_argmax2d
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data), op_optimized(data))


class TestConvSoftArgmax2d(BaseTester):
    def test_smoke(self, device, dtype):
        sample = torch.zeros(1, 1, 3, 3, device=device, dtype=dtype)
        m = kornia.geometry.subpix.ConvSoftArgmax2d((3, 3))
        assert m(sample).shape == (1, 1, 2, 3, 3)

    def test_smoke_batch(self, device, dtype):
        sample = torch.zeros(2, 5, 3, 3, device=device, dtype=dtype)
        m = kornia.geometry.subpix.ConvSoftArgmax2d()
        assert m(sample).shape == (2, 5, 2, 3, 3)

    def test_smoke_with_val(self, device, dtype):
        sample = torch.zeros(1, 1, 3, 3, device=device, dtype=dtype)
        m = kornia.geometry.subpix.ConvSoftArgmax2d((3, 3), output_value=True)
        coords, val = m(sample)
        assert coords.shape == (1, 1, 2, 3, 3)
        assert val.shape == (1, 1, 3, 3)

    def test_smoke_batch_with_val(self, device, dtype):
        sample = torch.zeros(2, 5, 3, 3, device=device, dtype=dtype)
        m = kornia.geometry.subpix.ConvSoftArgmax2d((3, 3), output_value=True)
        coords, val = m(sample)
        assert coords.shape == (2, 5, 2, 3, 3)
        assert val.shape == (2, 5, 3, 3)

    def test_gradcheck(self, device):
        sample = torch.rand(2, 3, 5, 5, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.subpix.conv_soft_argmax2d, (sample), nondet_tol=1e-8)

    def test_cold_diag(self, device, dtype):
        sample = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        softargmax = kornia.geometry.subpix.ConvSoftArgmax2d(
            (3, 3), (2, 2), (0, 0), temperature=0.05, normalized_coordinates=False, output_value=True
        )
        expected_val = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], device=device, dtype=dtype)
        expected_coord = torch.tensor(
            [[[[[1.0, 3.0], [1.0, 3.0]], [[1.0, 1.0], [3.0, 3.0]]]]], device=device, dtype=dtype
        )
        coords, val = softargmax(sample)
        self.assert_close(val, expected_val, atol=1e-4, rtol=1e-4)
        self.assert_close(coords, expected_coord, atol=1e-4, rtol=1e-4)

    def test_hot_diag(self, device, dtype):
        sample = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        softargmax = kornia.geometry.subpix.ConvSoftArgmax2d(
            (3, 3), (2, 2), (0, 0), temperature=10.0, normalized_coordinates=False, output_value=True
        )
        expected_val = torch.tensor([[[[0.1214, 0.0], [0.0, 0.1214]]]], device=device, dtype=dtype)
        expected_coord = torch.tensor(
            [[[[[1.0, 3.0], [1.0, 3.0]], [[1.0, 1.0], [3.0, 3.0]]]]], device=device, dtype=dtype
        )
        coords, val = softargmax(sample)
        self.assert_close(val, expected_val, atol=1e-4, rtol=1e-4)
        self.assert_close(coords, expected_coord, atol=1e-4, rtol=1e-4)

    def test_cold_diag_norm(self, device, dtype):
        sample = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        softargmax = kornia.geometry.subpix.ConvSoftArgmax2d(
            (3, 3), (2, 2), (0, 0), temperature=0.05, normalized_coordinates=True, output_value=True
        )
        expected_val = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], device=device, dtype=dtype)
        expected_coord = torch.tensor(
            [[[[[-0.5, 0.5], [-0.5, 0.5]], [[-0.5, -0.5], [0.5, 0.5]]]]], device=device, dtype=dtype
        )
        coords, val = softargmax(sample)
        self.assert_close(val, expected_val, atol=1e-4, rtol=1e-4)
        self.assert_close(coords, expected_coord, atol=1e-4, rtol=1e-4)

    def test_hot_diag_norm(self, device, dtype):
        sample = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        softargmax = kornia.geometry.subpix.ConvSoftArgmax2d(
            (3, 3), (2, 2), (0, 0), temperature=10.0, normalized_coordinates=True, output_value=True
        )
        expected_val = torch.tensor([[[[0.1214, 0.0], [0.0, 0.1214]]]], device=device, dtype=dtype)
        expected_coord = torch.tensor(
            [[[[[-0.5, 0.5], [-0.5, 0.5]], [[-0.5, -0.5], [0.5, 0.5]]]]], device=device, dtype=dtype
        )
        coords, val = softargmax(sample)
        self.assert_close(val, expected_val, atol=1e-4, rtol=1e-4)
        self.assert_close(coords, expected_coord, atol=1e-4, rtol=1e-4)


class TestConvSoftArgmax3d(BaseTester):
    def test_smoke(self, device, dtype):
        sample = torch.zeros(1, 1, 3, 3, 3, device=device, dtype=dtype)
        m = kornia.geometry.subpix.ConvSoftArgmax3d((3, 3, 3), output_value=False)
        assert m(sample).shape == (1, 1, 3, 3, 3, 3)

    def test_smoke_with_val(self, device, dtype):
        sample = torch.zeros(1, 1, 3, 3, 3, device=device, dtype=dtype)
        m = kornia.geometry.subpix.ConvSoftArgmax3d((3, 3, 3), output_value=True)
        coords, val = m(sample)
        assert coords.shape == (1, 1, 3, 3, 3, 3)
        assert val.shape == (1, 1, 3, 3, 3)

    def test_gradcheck(self, device):
        sample = torch.rand(1, 2, 3, 5, 5, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.subpix.conv_soft_argmax3d, (sample), nondet_tol=1e-8)

    def test_cold_diag(self, device, dtype):
        sample = torch.tensor(
            [
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        softargmax = kornia.geometry.subpix.ConvSoftArgmax3d(
            (1, 3, 3), (1, 2, 2), (0, 0, 0), temperature=0.05, normalized_coordinates=False, output_value=True
        )
        expected_val = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]], device=device, dtype=dtype)
        expected_coord = torch.tensor(
            [[[[[[0.0, 0.0], [0.0, 0.0]]], [[[1.0, 3.0], [1.0, 3.0]]], [[[1.0, 1.0], [3.0, 3.0]]]]]],
            device=device,
            dtype=dtype,
        )
        coords, val = softargmax(sample)
        self.assert_close(val, expected_val, atol=1e-4, rtol=1e-4)
        self.assert_close(coords, expected_coord, atol=1e-4, rtol=1e-4)

    def test_hot_diag(self, device, dtype):
        sample = torch.tensor(
            [
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        softargmax = kornia.geometry.subpix.ConvSoftArgmax3d(
            (1, 3, 3), (1, 2, 2), (0, 0, 0), temperature=10.0, normalized_coordinates=False, output_value=True
        )
        expected_val = torch.tensor([[[[[0.1214, 0.0], [0.0, 0.1214]]]]], device=device, dtype=dtype)
        expected_coord = torch.tensor(
            [[[[[[0.0, 0.0], [0.0, 0.0]]], [[[1.0, 3.0], [1.0, 3.0]]], [[[1.0, 1.0], [3.0, 3.0]]]]]],
            device=device,
            dtype=dtype,
        )
        coords, val = softargmax(sample)
        self.assert_close(val, expected_val, atol=1e-4, rtol=1e-4)
        self.assert_close(coords, expected_coord, atol=1e-4, rtol=1e-4)

    def test_cold_diag_norm(self, device, dtype):
        sample = torch.tensor(
            [
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        softargmax = kornia.geometry.subpix.ConvSoftArgmax3d(
            (1, 3, 3), (1, 2, 2), (0, 0, 0), temperature=0.05, normalized_coordinates=True, output_value=True
        )
        expected_val = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]], device=device, dtype=dtype)
        expected_coord = torch.tensor(
            [[[[[[-1.0, -1.0], [-1.0, -1.0]]], [[[-0.5, 0.5], [-0.5, 0.5]]], [[[-0.5, -0.5], [0.5, 0.5]]]]]],
            device=device,
            dtype=dtype,
        )
        coords, val = softargmax(sample)
        self.assert_close(val, expected_val, atol=1e-4, rtol=1e-4)
        self.assert_close(coords, expected_coord, atol=1e-4, rtol=1e-4)

    def test_hot_diag_norm(self, device, dtype):
        sample = torch.tensor(
            [
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        softargmax = kornia.geometry.subpix.ConvSoftArgmax3d(
            (1, 3, 3), (1, 2, 2), (0, 0, 0), temperature=10.0, normalized_coordinates=True, output_value=True
        )
        expected_val = torch.tensor([[[[[0.1214, 0.0], [0.0, 0.1214]]]]], device=device, dtype=dtype)
        expected_coord = torch.tensor(
            [[[[[[-1.0, -1.0], [-1.0, -1.0]]], [[[-0.5, 0.5], [-0.5, 0.5]]], [[[-0.5, -0.5], [0.5, 0.5]]]]]],
            device=device,
            dtype=dtype,
        )
        coords, val = softargmax(sample)
        self.assert_close(val, expected_val, atol=1e-4, rtol=1e-4)
        self.assert_close(coords, expected_coord, atol=1e-4, rtol=1e-4)


class TestConvQuadInterp3d(BaseTester):
    def test_smoke(self, device, dtype):
        sample = torch.randn(2, 3, 3, 4, 4, device=device, dtype=dtype)
        nms = kornia.geometry.ConvQuadInterp3d(1)
        coord, val = nms(sample)
        assert coord.shape == (2, 3, 3, 3, 4, 4)
        assert val.shape == (2, 3, 3, 4, 4)

    def test_gradcheck(self, device):
        sample = torch.rand(1, 1, 3, 5, 5, device=device, dtype=torch.float64)
        sample[0, 0, 1, 2, 2] += 20.0
        self.gradcheck(kornia.geometry.ConvQuadInterp3d(strict_maxima_bonus=0), (sample), atol=1e-3, rtol=1e-3)

    def test_diag(self, device, dtype):
        sample = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0, 0],
                        [0.0, 0.0, 0.0, 0, 0.0],
                        [0.0, 0, 0.0, 0, 0.0],
                        [0.0, 0.0, 0, 0, 0.0],
                        [0.0, 0.0, 0.0, 0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0, 0],
                        [0.0, 0.0, 1, 0, 0.0],
                        [0.0, 1, 1.2, 1.1, 0.0],
                        [0.0, 0.0, 1.0, 0, 0.0],
                        [0.0, 0.0, 0.0, 0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0, 0],
                        [0.0, 0.0, 0.0, 0, 0.0],
                        [0.0, 0, 0.0, 0, 0.0],
                        [0.0, 0.0, 0, 0, 0.0],
                        [0.0, 0.0, 0.0, 0, 0.0],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        sample = kornia.filters.gaussian_blur2d(sample, (5, 5), (0.5, 0.5)).unsqueeze(0)
        softargmax = kornia.geometry.ConvQuadInterp3d(10)
        expected_val = torch.tensor(
            [
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0, 0],
                            [0.0, 0.0, 0.0, 0, 0.0],
                            [0.0, 0, 0.0, 0, 0.0],
                            [0.0, 0.0, 0, 0, 0.0],
                            [0.0, 0.0, 0.0, 0, 0.0],
                        ],
                        [
                            [2.2504e-04, 2.3146e-02, 1.6808e-01, 2.3188e-02, 2.3628e-04],
                            [2.3146e-02, 1.8118e-01, 7.4338e-01, 1.8955e-01, 2.5413e-02],
                            [1.6807e-01, 7.4227e-01, 1.1086e01, 8.0414e-01, 1.8482e-01],
                            [2.3146e-02, 1.8118e-01, 7.4338e-01, 1.8955e-01, 2.5413e-02],
                            [2.2504e-04, 2.3146e-02, 1.6808e-01, 2.3188e-02, 2.3628e-04],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0, 0],
                            [0.0, 0.0, 0.0, 0, 0.0],
                            [0.0, 0, 0.0, 0, 0.0],
                            [0.0, 0.0, 0, 0, 0.0],
                            [0.0, 0.0, 0.0, 0, 0.0],
                        ],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        expected_coord = torch.tensor(
            [
                [
                    [
                        [
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ],
                            [
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                            ],
                            [
                                [2.0, 2.0, 2.0, 2.0, 2.0],
                                [2.0, 2.0, 2.0, 2.0, 2.0],
                                [2.0, 2.0, 2.0, 2.0, 2.0],
                                [2.0, 2.0, 2.0, 2.0, 2.0],
                                [2.0, 2.0, 2.0, 2.0, 2.0],
                            ],
                        ],
                        [
                            [
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                            ],
                            [
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                                [0.0, 1.0, 2.0495, 3.0, 4.0],
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                            ],
                            [
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                            ],
                        ],
                        [
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                [2.0, 2.0, 2.0, 2.0, 2.0],
                                [3.0, 3.0, 3.0, 3.0, 3.0],
                                [4.0, 4.0, 4.0, 4.0, 4.0],
                            ],
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                [2.0, 2.0, 2.0, 2.0, 2.0],
                                [3.0, 3.0, 3.0, 3.0, 3.0],
                                [4.0, 4.0, 4.0, 4.0, 4.0],
                            ],
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                [2.0, 2.0, 2.0, 2.0, 2.0],
                                [3.0, 3.0, 3.0, 3.0, 3.0],
                                [4.0, 4.0, 4.0, 4.0, 4.0],
                            ],
                        ],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        coords, val = softargmax(sample)
        self.assert_close(val, expected_val, atol=1e-4, rtol=1e-4)
        self.assert_close(coords, expected_coord, atol=1e-4, rtol=1e-4)


class TestIterativeQuadInterp3d(BaseTester):
    def test_smoke(self, device, dtype):
        sample = torch.randn(2, 3, 3, 4, 4, device=device, dtype=dtype)
        op = kornia.geometry.IterativeQuadInterp3d(n_iters=3, strict_maxima_bonus=1)
        coord, val = op(sample)
        assert coord.shape == (2, 3, 3, 3, 4, 4)
        assert val.shape == (2, 3, 3, 4, 4)

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            iterative_quad_interp3d("not_a_tensor")
        with pytest.raises(ValueError):
            iterative_quad_interp3d(torch.randn(3, 4, 4, device=device, dtype=dtype))

    def test_cardinality(self, device, dtype):
        for B, C, D, H, W in [(1, 1, 3, 5, 5), (2, 4, 3, 8, 6)]:
            sample = torch.randn(B, C, D, H, W, device=device, dtype=dtype)
            coord, val = iterative_quad_interp3d(sample)
            assert coord.shape == (B, C, 3, D, H, W)
            assert val.shape == (B, C, D, H, W)

    def test_gradcheck(self, device):
        sample = torch.rand(1, 1, 3, 5, 5, device=device, dtype=torch.float64)
        sample[0, 0, 1, 2, 2] += 20.0
        self.gradcheck(
            kornia.geometry.IterativeQuadInterp3d(strict_maxima_bonus=0, n_iters=1),
            (sample,),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_dynamo(self, device, dtype, torch_optimizer):
        sample = torch.rand(1, 1, 3, 5, 5, device=device, dtype=dtype)
        sample[0, 0, 1, 2, 2] += 20.0
        op = kornia.geometry.IterativeQuadInterp3d(strict_maxima_bonus=0, n_iters=1)
        op_opt = torch_optimizer(op)
        self.assert_close(op(sample)[0], op_opt(sample)[0])
        self.assert_close(op(sample)[1], op_opt(sample)[1])

    def test_peak_at_center(self, device, dtype):
        # A clear peak at scale=1, h=2, w=2 should return coords close to (1, 2, 2).
        sample = torch.zeros(1, 1, 3, 5, 5, device=device, dtype=dtype)
        sample[0, 0, 1, 2, 2] = 10.0
        coord, _val = iterative_quad_interp3d(sample, strict_maxima_bonus=0)
        # coords_max layout: dim2 = [scale, x(width), y(height)]
        assert coord[0, 0, 0, 1, 2, 2].item() == pytest.approx(1.0, abs=1e-3)  # scale
        assert coord[0, 0, 1, 1, 2, 2].item() == pytest.approx(2.0, abs=1e-3)  # x
        assert coord[0, 0, 2, 1, 2, 2].item() == pytest.approx(2.0, abs=1e-3)  # y

    def test_subpixel_shift(self, device, dtype):
        # Peak shifted slightly — subpixel offset should be non-zero.
        # Use D=5 so the center scale index 2 has valid ±1 neighbours.
        sample = torch.zeros(1, 1, 5, 7, 7, device=device, dtype=dtype)
        # Place a Gaussian-shaped peak slightly off-center (center at d=2, h=3, w=3).
        for dd in range(-1, 2):
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    dist2 = (dd - 0.2) ** 2 + (dh - 0.1) ** 2 + (dw + 0.15) ** 2
                    sample[0, 0, 2 + dd, 3 + dh, 3 + dw] += float(torch.exp(torch.tensor(-dist2 * 4)))
        coord, _ = iterative_quad_interp3d(sample, strict_maxima_bonus=0)
        # The refined x (width) coord at the integer peak (d=2, h=3, w=3) should shift toward -0.15.
        x_coord = coord[0, 0, 1, 2, 3, 3].item()
        assert x_coord < 3.0  # shift in negative x direction

    def test_no_keypoints(self, device, dtype):
        # Flat input — no NMS maxima, output should equal input coords/values.
        sample = torch.ones(1, 1, 3, 4, 4, device=device, dtype=dtype)
        coord, val = iterative_quad_interp3d(sample, strict_maxima_bonus=0)
        assert coord.shape == (1, 1, 3, 3, 4, 4)
        assert val.shape == (1, 1, 3, 4, 4)

    def test_convergence_within_iters(self, device, dtype):
        # With a clear symmetric peak a single iteration should converge.
        sample = torch.zeros(1, 1, 3, 5, 5, device=device, dtype=dtype)
        sample[0, 0, 1, 2, 2] = 5.0
        coord1, _ = iterative_quad_interp3d(sample, n_iters=1, strict_maxima_bonus=0)
        coord5, _ = iterative_quad_interp3d(sample, n_iters=5, strict_maxima_bonus=0)
        self.assert_close(coord1, coord5, atol=1e-5, rtol=1e-5)
