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
    conv_quad_interp3d,
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


class TestConvQuadInterp3dModule(BaseTester):
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


class TestConvQuadInterp3d(BaseTester):
    def test_smoke(self, device, dtype):
        sample = torch.randn(2, 3, 3, 4, 4, device=device, dtype=dtype)
        op = kornia.geometry.subpix.ConvQuadInterp3d(n_iters=3, strict_maxima_bonus=1)
        coord, val = op(sample)
        assert coord.shape == (2, 3, 3, 3, 4, 4)
        assert val.shape == (2, 3, 3, 4, 4)

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            conv_quad_interp3d("not_a_tensor")
        with pytest.raises(ValueError):
            conv_quad_interp3d(torch.randn(3, 4, 4, device=device, dtype=dtype))

    def test_cardinality(self, device, dtype):
        for B, C, D, H, W in [(1, 1, 3, 5, 5), (2, 4, 3, 8, 6)]:
            sample = torch.randn(B, C, D, H, W, device=device, dtype=dtype)
            coord, val = conv_quad_interp3d(sample)
            assert coord.shape == (B, C, 3, D, H, W)
            assert val.shape == (B, C, D, H, W)

    def test_gradcheck(self, device):
        sample = torch.rand(1, 1, 3, 5, 5, device=device, dtype=torch.float64)
        sample[0, 0, 1, 2, 2] += 20.0
        self.gradcheck(
            kornia.geometry.subpix.ConvQuadInterp3d(strict_maxima_bonus=0, n_iters=1),
            (sample,),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_dynamo(self, device, dtype, torch_optimizer):
        sample = torch.rand(1, 1, 3, 5, 5, device=device, dtype=dtype)
        sample[0, 0, 1, 2, 2] += 20.0
        op = kornia.geometry.subpix.ConvQuadInterp3d(strict_maxima_bonus=0, n_iters=1)
        op_opt = torch_optimizer(op)
        self.assert_close(op(sample)[0], op_opt(sample)[0])
        self.assert_close(op(sample)[1], op_opt(sample)[1])

    def test_peak_at_center(self, device, dtype):
        # A clear peak at scale=1, h=2, w=2 should return coords close to (1, 2, 2).
        sample = torch.zeros(1, 1, 3, 5, 5, device=device, dtype=dtype)
        sample[0, 0, 1, 2, 2] = 10.0
        coord, _val = conv_quad_interp3d(sample, strict_maxima_bonus=0)
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
        coord, _ = conv_quad_interp3d(sample, strict_maxima_bonus=0)
        # The refined x (width) coord at the integer peak (d=2, h=3, w=3) should shift toward -0.15.
        x_coord = coord[0, 0, 1, 2, 3, 3].item()
        assert x_coord < 3.0  # shift in negative x direction

    def test_no_keypoints(self, device, dtype):
        # Flat input — no NMS maxima, output should equal input coords/values.
        sample = torch.ones(1, 1, 3, 4, 4, device=device, dtype=dtype)
        coord, val = conv_quad_interp3d(sample, strict_maxima_bonus=0)
        assert coord.shape == (1, 1, 3, 3, 4, 4)
        assert val.shape == (1, 1, 3, 4, 4)

    def test_convergence_within_iters(self, device, dtype):
        # With a clear symmetric peak a single iteration should converge.
        sample = torch.zeros(1, 1, 3, 5, 5, device=device, dtype=dtype)
        sample[0, 0, 1, 2, 2] = 5.0
        coord1, _ = conv_quad_interp3d(sample, n_iters=1, strict_maxima_bonus=0)
        coord5, _ = conv_quad_interp3d(sample, n_iters=5, strict_maxima_bonus=0)
        self.assert_close(coord1, coord5, atol=1e-5, rtol=1e-5)


class TestAdaptiveQuadInterp3d(BaseTester):
    def test_smoke(self, device, dtype):
        for mode in ("patch", "conv", "auto"):
            x = torch.randn(1, 1, 3, 8, 8, device=device, dtype=dtype)
            coords, vals = kornia.geometry.subpix.AdaptiveQuadInterp3d(mode=mode)(x)
            assert coords.shape == (1, 1, 3, 3, 8, 8)
            assert vals.shape == (1, 1, 3, 8, 8)

    def test_invalid_mode(self, device, dtype):
        with pytest.raises(ValueError, match="mode must be one of"):
            kornia.geometry.subpix.AdaptiveQuadInterp3d(mode="bogus")

    def test_patch_conv_agree(self, device, dtype):
        """patch and conv backends must produce numerically identical results."""
        torch.manual_seed(7)
        x = torch.randn(1, 1, 5, 16, 16, device=device, dtype=dtype)
        coords_p, vals_p = kornia.geometry.subpix.AdaptiveQuadInterp3d(mode="patch", strict_maxima_bonus=0)(x)
        coords_c, vals_c = kornia.geometry.subpix.AdaptiveQuadInterp3d(mode="conv", strict_maxima_bonus=0)(x)
        from kornia.geometry.subpix import nms3d

        mask = nms3d(x, (3, 3, 3), True)
        b, c, d, h, w = torch.where(mask)
        self.assert_close(coords_p[b, c, :, d, h, w], coords_c[b, c, :, d, h, w], atol=1e-5, rtol=1e-5)
        self.assert_close(vals_p[b, c, d, h, w], vals_c[b, c, d, h, w], atol=1e-5, rtol=1e-5)

    def test_auto_dispatches(self, device, dtype):
        """auto mode must use conv on CUDA, patch on CPU (verified via result equality)."""
        x = torch.randn(1, 1, 3, 8, 8, device=device, dtype=dtype)
        auto = kornia.geometry.subpix.AdaptiveQuadInterp3d(mode="auto", strict_maxima_bonus=0)
        expected_mode = "conv" if x.is_cuda else "patch"
        ref = kornia.geometry.subpix.AdaptiveQuadInterp3d(mode=expected_mode, strict_maxima_bonus=0)
        self.assert_close(auto(x)[0], ref(x)[0], atol=1e-5, rtol=1e-5)
        self.assert_close(auto(x)[1], ref(x)[1], atol=1e-5, rtol=1e-5)

    def test_gradcheck(self, device):
        x = torch.zeros(1, 1, 3, 5, 5, device=device, dtype=torch.float64)
        x[0, 0, 1, 2, 2] = 5.0
        self.gradcheck(
            kornia.geometry.subpix.AdaptiveQuadInterp3d(mode="patch", strict_maxima_bonus=0),
            (x,),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_dynamo(self, device, dtype, torch_optimizer):
        x = torch.rand(1, 1, 3, 5, 5, device=device, dtype=dtype)
        x[0, 0, 1, 2, 2] += 20.0
        op = kornia.geometry.subpix.AdaptiveQuadInterp3d(mode="patch", strict_maxima_bonus=0)
        op_opt = torch_optimizer(op)
        self.assert_close(op(x)[0], op_opt(x)[0])

    def test_conv_matches_iterative(self, device, dtype):
        """conv_quad_interp3d and iterative_quad_interp3d must give identical results at NMS maxima.

        Regression test for the c000_safe normalisation bug: dividing the Hessian by |centre|
        before the det-threshold check made conv accept poorly-conditioned positions that
        iterative correctly rejected, causing up to ~1 px coordinate divergence.
        """
        from kornia.geometry.subpix.nms import nms3d_minmax
        from kornia.geometry.subpix.spatial_soft_argmax import conv_quad_interp3d, iterative_quad_interp3d

        torch.manual_seed(7)
        B, C, D, H, W = 1, 1, 5, 32, 32
        # Synthetic DoG-like response: mix of positive and negative small-amplitude blobs
        x = torch.zeros(B, C, D, H, W, device=device, dtype=dtype)
        for d0, h0, w0, sign, amp in [(2, 8, 10, 1, 0.008), (2, 22, 16, -1, 0.005), (3, 14, 24, 1, 0.012)]:
            dd = torch.arange(D, device=device, dtype=dtype) - d0
            dh = torch.arange(H, device=device, dtype=dtype) - h0
            dw = torch.arange(W, device=device, dtype=dtype) - w0
            x[0, 0] += (
                sign
                * amp
                * (
                    torch.exp(-0.5 * dd**2).view(D, 1, 1)
                    * torch.exp(-0.5 * (dh / 2.5) ** 2).view(1, H, 1)
                    * torch.exp(-0.5 * (dw / 2.5) ** 2).view(1, 1, W)
                )
            )

        max_mask, _ = nms3d_minmax(x)
        coord_conv, _ = conv_quad_interp3d(
            x, n_iters=5, strict_maxima_bonus=0.0, precomputed_nms_mask=max_mask, dilation_radius=3
        )
        coord_iter, _ = iterative_quad_interp3d(x, n_iters=5, strict_maxima_bonus=0.0)

        d_idx, h_idx, w_idx = torch.where(max_mask.view(D, H, W))
        assert len(d_idx) > 0, "No NMS maxima found — check the synthetic input"
        diff = (coord_conv[0, 0, :, d_idx, h_idx, w_idx] - coord_iter[0, 0, :, d_idx, h_idx, w_idx]).abs().max()
        self.assert_close(diff, torch.zeros_like(diff), atol=1e-5, rtol=0)
