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

from kornia.color import ApplyColorMap, ColorMap, ColorMapType, apply_colormap

from testing.base import BaseTester, assert_close


def test_autumn(device, dtype):
    cm = ColorMap(base="autumn", num_colors=64, device=device, dtype=dtype)
    colors = cm.colors

    actual = colors[..., 0]
    expected = torch.tensor([1, 0, 0], device=device, dtype=dtype)
    assert_close(actual, expected)

    actual = colors[..., 32]
    expected = torch.tensor([1.0, 0.5079365079365079, 0.0], device=device, dtype=dtype)
    assert_close(actual, expected)

    actual = colors[..., -1]
    expected = torch.tensor([1, 1, 0], device=device, dtype=dtype)
    assert_close(actual, expected)


class TestApplyColorMap(BaseTester):
    def test_smoke(self, device, dtype):
        input_tensor = torch.tensor([[[0, 1, 2], [15, 25, 33], [128, 158, 188]]], device=device, dtype=dtype)
        expected_tensor = torch.tensor(
            [
                [
                    [
                        [1.0000000000, 1.0000000000, 1.0000000000],
                        [1.0000000000, 1.0000000000, 1.0000000000],
                        [1.0000000000, 1.0000000000, 1.0000000000],
                    ],
                    [
                        [0.0000000000, 0.0158730168, 0.0158730168],
                        [0.0634920672, 0.1111111119, 0.1428571492],
                        [0.5079365373, 0.6190476418, 0.7301587462],
                    ],
                    [
                        [0.0000000000, 0.0000000000, 0.0000000000],
                        [0.0000000000, 0.0000000000, 0.0000000000],
                        [0.0000000000, 0.0000000000, 0.0000000000],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        cm = ColorMap(base="autumn", device=device, dtype=dtype)
        actual = apply_colormap(input_tensor, cm)

        self.assert_close(actual, expected_tensor)

    def test_exception(self, device, dtype):
        cm = ColorMap(base="autumn", device=device, dtype=dtype)
        with pytest.raises(Exception):
            apply_colormap(torch.rand(size=(3, 3), dtype=dtype, device=device), cm)

        with pytest.raises(Exception):
            apply_colormap(torch.rand(size=(3), dtype=dtype, device=device), cm)

        with pytest.raises(Exception):
            apply_colormap(torch.rand(size=(3), dtype=dtype, device=device).item(), cm)

    @pytest.mark.parametrize("shape", [(2, 1, 3, 3), (1, 3, 3, 3), (1, 3, 3)])
    @pytest.mark.parametrize("cmap_base", ColorMapType)
    def test_cardinality(self, shape, device, dtype, cmap_base):
        cm = ColorMap(base=cmap_base, num_colors=256, device=device, dtype=dtype)
        input_tensor = torch.randint(0, 256, shape, device=device, dtype=dtype)
        actual = apply_colormap(input_tensor, cm)

        if len(shape) == 4:
            expected_shape = (shape[-4], shape[-3] * 3, shape[-2], shape[-1])
        else:
            expected_shape = (1, shape[-3] * 3, shape[-2], shape[-1])

        assert actual.shape == expected_shape

    @pytest.mark.skip(reason="jacobian mismatch")
    def test_gradcheck(self, device):
        # TODO: implement differentiability
        cm = ColorMap(base="autumn", device=device, dtype=torch.float64)
        input_tensor = torch.randint(0, 63, (1, 2, 1), device=device, dtype=torch.float64)

        self.gradcheck(apply_colormap, (input_tensor, cm))

    def test_dynamo(self, device, dtype, torch_optimizer):
        op = apply_colormap
        op_script = torch_optimizer(op)

        cm = ColorMap(base="autumn", device=device, dtype=dtype)
        img = torch.ones(1, 3, 3, device=device, dtype=dtype)

        self.assert_close(op(img, cm), op_script(img, cm))

    def test_module(self, device, dtype):
        op = apply_colormap
        cm = ColorMap(base="autumn", device=device, dtype=dtype)
        op_module = ApplyColorMap(colormap=cm)

        img = torch.ones(1, 3, 3, device=device, dtype=dtype)

        self.assert_close(op(img, colormap=cm), op_module(img))
