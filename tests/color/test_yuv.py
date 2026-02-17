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
        img = torch.rand(3, 4, 5, device=device, dtype=dtype)
        out = kornia.color.rgb_to_yuv(img)
        assert isinstance(out, torch.Tensor)

    @pytest.mark.parametrize(
        "shape",
        [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1)],
    )
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        out = kornia.color.rgb_to_yuv(img)
        assert out.shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises((TypeError, AttributeError)):
            kornia.color.rgb_to_yuv([0.0])

        with pytest.raises(ShapeError):
            kornia.color.rgb_to_yuv(torch.ones(1, 1, device=device, dtype=dtype))

        with pytest.raises(ShapeError):
            kornia.color.rgb_to_yuv(torch.ones(2, 1, 1, device=device, dtype=dtype))

    def test_unit_invariants(self, device, dtype):
        rgb = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # red
                [0.0, 1.0, 0.0],  # green
                [0.0, 0.0, 1.0],  # blue
                [1.0, 1.0, 1.0],  # white
                [0.0, 0.0, 0.0],  # black
            ],
            device=device,
            dtype=dtype,
        ).view(5, 3, 1, 1)

        yuv = kornia.color.rgb_to_yuv(rgb)

        # shape preserved
        assert yuv.shape == rgb.shape

        Y = yuv[:, 0, 0, 0]

        # basic luminance ordering invariants
        assert Y[3] > Y[4]  # white > black
        assert Y[1] > Y[2]  # green generally brighter than blue

        # neutral colors have near-zero chroma
        self.assert_close(yuv[3, 1:], torch.zeros_like(yuv[3, 1:]), atol=1e-4, rtol=1e-4)
        self.assert_close(yuv[4], torch.zeros_like(yuv[4]), atol=1e-4, rtol=1e-4)

    def test_round_trip_rgb_yuv_rgb(self, device, dtype):
        rgb = torch.rand(3, 4, 5, device=device, dtype=dtype)

        yuv = kornia.color.rgb_to_yuv(rgb)
        rgb_back = kornia.color.yuv_to_rgb(yuv)

        atol = 1e-3 if dtype in (torch.float32, torch.float64) else 1e-2

        self.assert_close(rgb_back, rgb, atol=atol, rtol=1e-3)

    @pytest.mark.grad()
    def test_gradcheck(self, device, dtype):
        img = torch.rand(2, 3, 4, 4, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            kornia.color.rgb_to_yuv,
            (img,),
            raise_exception=True,
            fast_mode=True,
        )

    @pytest.mark.jit()
    def test_jit(self, device, dtype):
        img = torch.ones(2, 3, 4, 4, device=device, dtype=dtype)
        op = kornia.color.rgb_to_yuv
        op_jit = torch.jit.script(op)
        self.assert_close(op(img), op_jit(img))

    def test_module(self, device, dtype):
        img = torch.ones(2, 3, 4, 4, device=device, dtype=dtype)
        module = kornia.color.RgbToYuv().to(device, dtype)
        self.assert_close(module(img), kornia.color.rgb_to_yuv(img))
