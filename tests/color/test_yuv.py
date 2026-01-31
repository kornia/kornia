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
from torch.autograd import gradcheck

import kornia
from kornia.core.exceptions import ShapeError

from testing.base import BaseTester


class TestRgbToYuv(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 3, 4, 5
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert isinstance(kornia.color.rgb_to_yuv(img)[0], torch.Tensor)
        assert isinstance(kornia.color.rgb_to_yuv(img)[1], torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        assert kornia.color.rgb_to_yuv(img).shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises((TypeError, AttributeError)):
            kornia.color.rgb_to_yuv([0.0])

        with pytest.raises(ShapeError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            kornia.color.rgb_to_yuv(img)

        with pytest.raises(ShapeError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            kornia.color.rgb_to_yuv(img)

    # ✅ Implemented missing unit test
    def test_unit(self, device, dtype):
        rgb = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 1.0, 1.0],  # White
                [0.0, 0.0, 0.0],  # Black
            ],
            device=device,
            dtype=dtype,
        ).view(5, 3, 1, 1)

        yuv = kornia.color.rgb_to_yuv(rgb)

        expected = torch.tensor(
            [
                [0.299, -0.14713, 0.615],
                [0.587, -0.28886, -0.51499],
                [0.114, 0.436, -0.10001],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            device=device,
            dtype=dtype,
        ).view(5, 3, 1, 1)

        self.assert_close(yuv, expected, atol=1e-3)

    # ✅ Improved accuracy
    def test_forth_and_back(self, device, dtype):
        data = torch.rand(3, 4, 5, device=device, dtype=dtype)
        yuv = kornia.color.rgb_to_yuv
        rgb = kornia.color.yuv_to_rgb

        data_out = rgb(yuv(data))
        self.assert_close(data_out, data, atol=1e-3)

    @pytest.mark.grad()
    def test_gradcheck(self, device, dtype):
        img = torch.rand(2, 3, 4, 4, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.rgb_to_yuv, (img,), raise_exception=True, fast_mode=True)

    @pytest.mark.jit()
    def test_jit(self, device, dtype):
        img = torch.ones(2, 3, 4, 4, device=device, dtype=dtype)
        op = kornia.color.rgb_to_yuv
        op_jit = torch.jit.script(op)
        self.assert_close(op(img), op_jit(img))

    def test_module(self, device, dtype):
        img = torch.ones(2, 3, 4, 4, device=device, dtype=dtype)
        ops = kornia.color.RgbToYuv().to(device, dtype)
        self.assert_close(ops(img), kornia.color.rgb_to_yuv(img))
