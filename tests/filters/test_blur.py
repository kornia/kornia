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

from kornia.filters import BoxBlur, box_blur

from testing.base import BaseTester


class TestBoxBlur(BaseTester):
    @pytest.mark.parametrize("kernel_size", [5, (3, 5)])
    def test_smoke(self, kernel_size, device, dtype):
        data = torch.rand(1, 1, 10, 10, device=device, dtype=dtype)

        bb = BoxBlur(kernel_size, "reflect")
        actual = bb(data)
        assert actual.shape == (1, 1, 10, 10)

    @pytest.mark.parametrize("kernel_size", [5, (3, 5)])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_separable(self, batch_size, kernel_size, device, dtype):
        data = torch.randn(batch_size, 3, 10, 10, device=device, dtype=dtype)
        out1 = box_blur(data, kernel_size, separable=False)
        out2 = box_blur(data, kernel_size, separable=True)
        self.assert_close(out1, out2)

    def test_exception(self):
        data = torch.rand(1, 1, 3, 3)

        with pytest.raises(Exception) as errinfo:
            box_blur(data, (1,))
        assert "2D Kernel size should have a length of 2." in str(errinfo)

    @pytest.mark.parametrize("kernel_size", [(3, 3), 5, (5, 7)])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_cardinality(self, batch_size, kernel_size, device, dtype):
        inp = torch.zeros(batch_size, 3, 4, 4, device=device, dtype=dtype)
        blur = BoxBlur(kernel_size)
        actual = blur(inp)
        expected = (batch_size, 3, 4, 4)
        assert actual.shape == expected

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_kernel_3x3(self, batch_size, device, dtype):
        inp = torch.tensor(
            [
                [
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0, 2.0, 2.0],
                        [2.0, 2.0, 2.0, 2.0, 2.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        ).repeat(batch_size, 1, 1, 1)

        kernel_size = (3, 3)
        actual = box_blur(inp, kernel_size)
        expected = torch.tensor(35.0 * batch_size, device=device, dtype=dtype)

        self.assert_close(actual.sum(), expected)

    @pytest.mark.parametrize("batch_size", [None, 1, 3])
    def test_kernel_5x5(self, batch_size, device, dtype):
        inp = torch.tensor(
            [
                [
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0, 2.0, 2.0],
                        [2.0, 2.0, 2.0, 2.0, 2.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        if batch_size:
            inp = inp.repeat(batch_size, 1, 1, 1)

        kernel_size = (5, 5)

        actual = box_blur(inp, kernel_size)
        expected = inp.sum((1, 2, 3)) / torch.mul(*kernel_size)

        self.assert_close(actual[:, 0, 2, 2], expected)

    def test_kernel_3x1(self, device, dtype):
        inp = torch.arange(16, device=device, dtype=dtype).view(1, 1, 4, 4)

        ky, kx = 3, 1
        actual = box_blur(inp, (ky, kx))

        self.assert_close(actual[0, 0, 0, 0], torch.tensor((4 + 0 + 4) / 3, device=device, dtype=dtype))
        self.assert_close(actual[0, 0, 1, 0], torch.tensor((0 + 4 + 8) / 3, device=device, dtype=dtype))

    @pytest.mark.parametrize("separable", [False, True])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_noncontiguous(self, batch_size, separable, device, dtype):
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        actual = box_blur(inp, 3, separable=separable)

        assert actual.is_contiguous()

    @pytest.mark.parametrize("kernel_size", [(3, 3), 5, (5, 7)])
    def test_gradcheck(self, kernel_size, device):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        fast_mode = "cpu" in str(device)  # Disable fast mode for GPU
        self.gradcheck(box_blur, (img, kernel_size), fast_mode=fast_mode)

    @pytest.mark.parametrize("kernel_size", [(3, 3), 5, (5, 7)])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_module(self, kernel_size, batch_size, device, dtype):
        op = box_blur
        op_module = BoxBlur

        img = torch.rand(batch_size, 3, 4, 5, device=device, dtype=dtype)
        actual = op_module(kernel_size)(img)
        expected = op(img, kernel_size)

        self.assert_close(actual, expected)

    @pytest.mark.parametrize("separable", [False, True])
    @pytest.mark.parametrize("kernel_size", [5, (5, 7)])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_dynamo(self, batch_size, kernel_size, separable, device, dtype, torch_optimizer):
        data = torch.ones(batch_size, 3, 10, 10, device=device, dtype=dtype)
        op = BoxBlur(kernel_size, separable=separable)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data), op_optimized(data))
