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


class TestColorMap(BaseTester):
    @pytest.mark.applies_to(
        when=lambda p: not (p["device_name"] == "mps" and p["default_dtype"] == torch.float64),
        reason="MPS does not support float64.",
    )
    @pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
    def test_integer_base_uses_default_float_dtype(self, device, default_dtype):
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(default_dtype)
            actual = ColorMap(base=[[0, 0, 0], [1, 1, 1]], num_colors=3, device=device)
            expected = torch.tensor([[0.0, 0.5, 1.0]] * 3, device=device)

            assert actual.colors.dtype == default_dtype
            self.assert_close(actual.colors, expected)
        finally:
            torch.set_default_dtype(previous_dtype)

    def test_integer_base_respects_explicit_dtype(self, device, dtype):
        actual = ColorMap(base=[[0, 0, 0], [1, 1, 1]], num_colors=3, device=device, dtype=dtype)
        expected = torch.tensor([[0.0, 0.5, 1.0]] * 3, device=device, dtype=dtype)

        assert actual.colors.dtype == dtype
        self.assert_close(actual.colors, expected)

    @pytest.mark.parametrize("num_colors", [2, 4, 5, 8, 256])
    def test_ramp_resamples_evenly_6556(self, device, dtype, num_colors):
        actual = ColorMap(base=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], num_colors=num_colors, device=device, dtype=dtype)
        ramp = torch.arange(num_colors, device=device, dtype=dtype) / (num_colors - 1)
        self.assert_close(actual.colors, ramp.expand(3, -1))

    @pytest.mark.parametrize("name", ["viridis", "jet"])
    def test_one_color_map_holds_first_color_6556(self, device, dtype, name):
        base = ColorMap(base=name, device=device, dtype=dtype).colors
        actual = ColorMap(base=name, num_colors=1, device=device, dtype=dtype).colors
        self.assert_close(actual, base[:, :1], atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("num_colors", [8, 63, 65, 256])
    @pytest.mark.parametrize("name", ["viridis", "jet", "autumn"])
    def test_resampled_map_keeps_end_colors_6556(self, device, dtype, name, num_colors):
        base = ColorMap(base=name, device=device, dtype=dtype).colors
        actual = ColorMap(base=name, num_colors=num_colors, device=device, dtype=dtype).colors
        assert actual.shape == (3, num_colors)
        self.assert_close(actual[:, 0], base[:, 0], atol=0.0, rtol=0.0)
        self.assert_close(actual[:, -1], base[:, -1], atol=0.0, rtol=0.0)
        assert not torch.equal(actual[:, 0], actual[:, 1])
        assert not torch.equal(actual[:, -2], actual[:, -1])


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
                        # The green of autumn is i / 63, at i = floor(v * 64 / 255) for the byte value v.
                        [0.0 / 63, 0.0 / 63, 0.0 / 63],
                        [3.0 / 63, 6.0 / 63, 8.0 / 63],
                        [32.0 / 63, 39.0 / 63, 47.0 / 63],
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

    def test_upper_endpoint_selects_last_color(self, device, dtype):
        cm = ColorMap(
            base=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            num_colors=4,
            device=device,
            dtype=dtype,
        )
        input_tensor = torch.ones(1, 1, 1, device=device, dtype=dtype)

        actual = apply_colormap(input_tensor, cm)
        expected = cm.colors[..., -1].reshape(1, 3, 1, 1)

        self.assert_close(actual, expected)

    def test_uint8_selects_color_by_value(self, device, dtype):
        # With 256 colors every byte value v selects palette color v, including 255 (the last color).
        cm = ColorMap(base=[[i / 255.0] * 3 for i in range(256)], num_colors=256, device=device, dtype=dtype)
        input_tensor = torch.arange(256, device=device).to(torch.uint8).reshape(1, 1, 1, 256)

        actual = apply_colormap(input_tensor, cm)

        self.assert_close(actual, cm.colors.reshape(1, 3, 1, 256), rtol=0.0, atol=0.0)

    @pytest.mark.parametrize(
        "num_colors, values, expected",
        [
            (1, [0.0, 0.5, 1.0], [0, 0, 0]),
            (2, [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0], [0, 0, 0, 1, 1, 1, 1]),
            (3, [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0], [0, 0, 0, 1, 2, 2, 2]),
            (4, [0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0], [0, 0, 1, 2, 3, 3, 3]),
            (64, [0.0, 0.0078125, 0.015625, 0.5, 0.984375, 0.9921875, 1.0], [0, 0, 1, 32, 63, 63, 63]),
        ],
    )
    def test_convention_apply_colormap_6622_equal_bins(self, num_colors, values, expected, device, dtype):
        # Expected is matplotlib's lookup, written out: floor(x * N), with x = 1 in the last bin. Values on a bin edge
        # are exact in float16 and bfloat16; the others sit well inside a bin.
        cm = ColorMap(base=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], num_colors=num_colors, device=device, dtype=dtype)
        input_tensor = torch.tensor(values, device=device, dtype=dtype).reshape(1, 1, -1)

        actual = apply_colormap(input_tensor, cm)

        self.assert_close(actual[0, 0, 0], cm.colors[0, expected], rtol=0.0, atol=0.0)

    def test_uint8_selects_equal_bins_6622(self, device, dtype):
        # floor(v * 16 / 255): each of the 16 colors covers 15 or 16 byte values, including the first and the last.
        cm = ColorMap(base=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], num_colors=16, device=device, dtype=dtype)
        input_tensor = torch.tensor([[[0, 15, 16, 31, 32, 127, 128, 239, 240, 255]]], device=device, dtype=torch.uint8)

        actual = apply_colormap(input_tensor, cm)

        expected = [0, 0, 1, 1, 2, 7, 8, 14, 15, 15]
        self.assert_close(actual[0, 0, 0], cm.colors[0, expected], rtol=0.0, atol=0.0)

    def test_outside_the_range_selects_the_ends_6622(self, device, dtype):
        # The rule stops at the ends, and a NaN stays on the last entry, as it did before the equal bins.
        cm = ColorMap(base=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], num_colors=4, device=device, dtype=dtype)
        input_tensor = torch.tensor([[[-0.5, 300.0, float("inf"), float("nan")]]], device=device, dtype=dtype)

        actual = apply_colormap(input_tensor, cm)

        self.assert_close(actual[0, 0, 0], cm.colors[0, [0, 3, 3, 3]], rtol=0.0, atol=0.0)

    def test_uint8_dark_values_use_byte_range(self, device, dtype):
        cm = ColorMap(base=[[i / 255.0] * 3 for i in range(256)], num_colors=256, device=device, dtype=dtype)
        image = torch.tensor([[[[0, 1]]]], device=device, dtype=torch.uint8)

        actual = apply_colormap(image, cm)

        self.assert_close(actual[0, :, 0, 1], cm.colors[:, 1], rtol=0.0, atol=0.0)

    def test_float_channels_choose_ranges_independently(self, device, dtype):
        cm = ColorMap(base="autumn", num_colors=256, device=device, dtype=dtype)
        image = torch.tensor([[[[0.5]], [[0.0]]]], device=device, dtype=dtype)
        image_with_bright_channel = image.clone()
        image_with_bright_channel[:, 1] = 200.0

        actual = apply_colormap(image_with_bright_channel, cm)
        expected = apply_colormap(image, cm)

        self.assert_close(actual[:, :3], expected[:, :3])

    def test_float_samples_choose_ranges_independently(self, device, dtype):
        cm = ColorMap(base="autumn", num_colors=256, device=device, dtype=dtype)
        image = torch.tensor([[[[0.5]]]], device=device, dtype=dtype)
        paired = torch.cat([image, torch.full_like(image, 200.0)])

        self.assert_close(apply_colormap(paired, cm)[:1], apply_colormap(image, cm))

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

    @pytest.mark.parametrize(
        "shape,expected_shape",
        [
            ((0, 1, 4, 5), (0, 3, 4, 5)),
            ((0, 2, 4, 5), (0, 6, 4, 5)),
            ((2, 0, 4, 5), (2, 0, 4, 5)),
            ((0, 4, 5), (1, 0, 4, 5)),
            ((2, 1, 0, 5), (2, 3, 0, 5)),
            ((1, 2, 0, 0), (1, 6, 0, 0)),
        ],
    )
    @pytest.mark.parametrize("integer_input", [False, True])
    def test_empty_input(self, shape, expected_shape, integer_input, device, dtype):
        # An empty tensor raised in reshape(B, C, -1), and a float one without pixels in amax.
        cm = ColorMap(base="viridis", device=device, dtype=dtype)
        input_tensor = torch.zeros(shape, device=device, dtype=torch.uint8 if integer_input else dtype)
        for op in (lambda x: apply_colormap(x, cm), ApplyColorMap(colormap=cm)):
            actual = op(input_tensor)
            assert actual.shape == expected_shape
            assert actual.device == input_tensor.device
            assert actual.dtype == dtype

    def test_preserves_input_shape(self, device):
        cm = ColorMap(
            base="autumn",
            num_colors=256,
            device=device,
            dtype=torch.float32,
        )
        input_tensor = torch.zeros(1, 2, 3, device=device, dtype=torch.float32)
        original = input_tensor.clone()

        apply_colormap(input_tensor, cm)

        assert input_tensor.shape == original.shape
        self.assert_close(input_tensor, original)

    def test_preserves_float_values(self, device):
        cm = ColorMap(
            base="autumn",
            num_colors=256,
            device=device,
            dtype=torch.float32,
        )
        input_tensor = torch.tensor(
            [[[0.0, 255.0]]],
            device=device,
            dtype=torch.float32,
        )
        original = input_tensor.clone()

        apply_colormap(input_tensor, cm)

        self.assert_close(input_tensor, original)

    def test_accepts_leaf_tensor_requires_grad(self, device):
        cm = ColorMap(
            base="autumn",
            num_colors=256,
            device=device,
            dtype=torch.float32,
        )
        input_tensor = torch.ones(
            1,
            1,
            2,
            3,
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )

        output = apply_colormap(input_tensor, cm)

        assert output.shape == (1, 3, 2, 3)
        assert input_tensor.requires_grad

    def test_dynamo(self, device, dtype, torch_optimizer):
        op = apply_colormap
        op_script = torch_optimizer(op)

        cm = ColorMap(base="autumn", device=device, dtype=dtype)
        img = torch.arange(9, device=device, dtype=dtype).view(1, 3, 3) / 8

        self.assert_close(op(img, cm), op_script(img, cm))

    def test_module(self, device, dtype):
        op = apply_colormap
        cm = ColorMap(base="autumn", device=device, dtype=dtype)
        op_module = ApplyColorMap(colormap=cm)

        img = torch.ones(1, 3, 3, device=device, dtype=dtype)

        self.assert_close(op(img, colormap=cm), op_module(img))
        assert "colors" in dict(op_module.named_buffers())
        assert op_module.colors is cm.colors

    def test_module_palette_follows_to(self, device):
        # The palette is a buffer, so converting or moving the module converts or moves it.
        module = ApplyColorMap(ColorMap(base="autumn")).to(device=device, dtype=torch.float16)
        expected = ApplyColorMap(ColorMap(base="autumn", device=device, dtype=torch.float16))
        img = torch.tensor([[[[0, 64, 128, 255]]]], device=device, dtype=torch.uint8)

        actual = module(img)

        assert actual.dtype == torch.float16
        self.assert_close(actual, expected(img))

    def test_module_state_dict_round_trip(self, device, dtype):
        source = ApplyColorMap(ColorMap(base="winter", device=device, dtype=dtype))
        target = ApplyColorMap(ColorMap(base="autumn", device=device, dtype=dtype))
        img = torch.tensor([[[[0, 64, 128, 255]]]], device=device, dtype=torch.uint8)

        target.load_state_dict(source.state_dict())

        self.assert_close(target(img), source(img))

    def test_module_loads_state_dict_without_palette(self, device, dtype):
        # State dicts saved before the palette was a buffer have no "colors" key: they still load strictly,
        # also when the module is nested, and keep the palette the module was built with.
        model = torch.nn.Sequential(ApplyColorMap(ColorMap(base="autumn", device=device, dtype=dtype)))
        img = torch.tensor([[[[0, 64, 128, 255]]]], device=device, dtype=torch.uint8)
        expected = model(img)

        model.load_state_dict({})

        self.assert_close(model(img), expected)
