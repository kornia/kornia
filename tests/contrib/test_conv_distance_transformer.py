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


class TestConvDistanceTransform(BaseTester):
    @pytest.mark.parametrize("kernel_size", [3, 5, 7])
    def test_smoke(self, kernel_size, device, dtype):
        sample1 = torch.rand(1, 3, 100, 100, device=device, dtype=dtype)
        sample2 = torch.rand(1, 1, 100, 100, device=device, dtype=dtype)
        distance_transformer = kornia.contrib.DistanceTransform(kernel_size)

        output1 = distance_transformer(sample1)
        output2 = kornia.contrib.distance_transform(sample2, kernel_size)

        assert isinstance(output1, torch.Tensor)
        assert isinstance(output2, torch.Tensor)
        assert output1.shape == sample1.shape

    def test_module(self, device, dtype):
        B, C, H, W = 1, 1, 99, 100
        sample1 = torch.rand(B, C, H, W, device=device, dtype=dtype)
        distance_transformer = kornia.contrib.DistanceTransform().to(device, dtype)

        output1 = distance_transformer(sample1)
        output2 = kornia.contrib.distance_transform(sample1)
        self.assert_close(output1, output2)

    def test_exception(self, device, dtype):
        B, C, H, W = 1, 1, 224, 224
        sample1 = torch.rand(B, C, H, W, device=device, dtype=dtype)
        sample2 = torch.rand(C, H, W, device=device, dtype=dtype)

        # Non-odd kernel size
        with pytest.raises(ValueError):
            ConvDT = kornia.contrib.DistanceTransform(6)
            ConvDT.forward(sample1)

        with pytest.raises(ValueError):
            kornia.contrib.distance_transform(sample1, 4)

        # Invalid input dimensions
        with pytest.raises(ValueError):
            kornia.contrib.distance_transform(sample2)

        # Invalid input type
        with pytest.raises(TypeError):
            kornia.contrib.distance_transform(None)

    def test_value(self, device, dtype):
        B, C, H, W = 1, 1, 4, 4
        kernel_size = 7
        h = 0.35
        sample1 = torch.zeros(B, C, H, W, device=device, dtype=dtype)
        sample1[:, :, 1, 1] = 1.0
        expected_output1 = torch.tensor(
            [
                [
                    [
                        [1.4142135382, 1.0000000000, 1.4142135382, 2.2360680103],
                        [1.0000000000, 0.0000000000, 1.0000000000, 2.0000000000],
                        [1.4142135382, 1.0000000000, 1.4142135382, 2.2360680103],
                        [2.2360680103, 2.0000000000, 2.2360680103, 2.8284270763],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        output1 = kornia.contrib.distance_transform(sample1, kernel_size, h)
        self.assert_close(expected_output1, output1)

    def test_gradcheck(self, device):
        B, C, H, W = 1, 1, 32, 32
        sample1 = torch.ones(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(kornia.contrib.distance_transform, (sample1))

    def test_loss_grad(self, device, dtype):
        B, C, H, W = 1, 1, 32, 32
        sample1 = torch.rand(B, C, H, W, device=device, dtype=dtype, requires_grad=True)
        sample2 = torch.rand(B, C, H, W, device=device, dtype=dtype, requires_grad=True)
        tiny_module = torch.nn.Conv2d(1, 1, (3, 3), (1, 1), (1, 1)).to(device=device, dtype=dtype)
        sample1 = kornia.contrib.distance_transform(tiny_module(sample1))
        sample2 = kornia.contrib.distance_transform(sample2)
        loss = torch.nn.functional.mse_loss(sample1, sample2)
        loss.backward()

    def test_offset_parenthesis_fix(self, device, dtype):
        img = torch.zeros(1, 1, 8, 4, device=device, dtype=dtype)
        img[0, 0, 1, :] = 1.0
        out = kornia.contrib.distance_transform(img, kernel_size=3, h=0.01)
        expected = torch.tensor(
            [
                [0.9998, 0.9998, 0.9998, 0.9998],
                [0.0000, 0.0000, 0.0000, 0.0000],
                [0.9998, 0.9998, 0.9998, 0.9998],
                [1.9998, 1.9998, 1.9998, 1.9998],
                [2.9998, 2.9998, 2.9998, 2.9998],
                [3.9998, 3.9998, 3.9998, 3.9998],
                [4.9998, 4.9998, 4.9998, 4.9998],
                [5.9998, 5.9998, 5.9998, 5.9998],
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(out[0, 0], expected, rtol=1e-3, atol=1e-3)
