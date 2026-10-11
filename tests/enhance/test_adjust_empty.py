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

from math import prod

import pytest
import torch

from kornia import enhance

from testing.base import BaseTester

_FUNCTIONS = ["equalize", "equalize3d", "posterize", "sharpness"]


def _arguments(name):
    if name == "posterize":
        return (4,)
    if name == "sharpness":
        return (0.5,)
    return ()


def _spatial_shape(name):
    return (3, 5, 7) if name == "equalize3d" else (5, 7)


def _input(shape, device, dtype):
    return (torch.arange(prod(shape)).remainder(128).to(device=device, dtype=dtype) / 128).reshape(shape)


def _weights(input):
    return (torch.arange(input.numel()).remainder(17).to(input) + 1).reshape(input.shape)


def _equalize_reference(input):
    # Independent per-plane histogram: bincount instead of the batched scatter_add implementation.
    count_dtype = torch.float64 if input.dtype == torch.float64 else torch.float32
    planes = input.reshape(input.shape[0] * input.shape[1], -1)
    outputs = []
    for plane in planes:
        scaled = plane * 255.0
        indices = (scaled * (256.0 / 255.0)).floor().long().clamp(0, 255)
        hist = torch.bincount(indices, minlength=256).to(count_dtype)
        last_count = hist[hist.nonzero()[-1, 0]]
        step = torch.div(hist.sum() - last_count, 255, rounding_mode="trunc")
        lut = torch.div(
            hist.cumsum(0) + torch.div(step, 2, rounding_mode="trunc"), step.clamp(min=1), rounding_mode="trunc"
        )
        lut = torch.cat([torch.zeros_like(lut[:1]), lut[:-1]]).clamp(0, 255).to(input.dtype)
        outputs.append(torch.where(step == 0, scaled, lut[scaled.long()]) / 255.0)
    return torch.stack(outputs).reshape(input.shape)


class TestEmptyIntensity(BaseTester):
    @pytest.mark.parametrize("name", _FUNCTIONS)
    @pytest.mark.parametrize("leading", [(0,), (2, 0), (0, 2), (2, 1, 0)])
    @pytest.mark.parametrize("transpose", [False, True])
    def test_empty_shape_and_backward_4429(self, name, leading, transpose, device, dtype):
        image = torch.empty((*leading, 3, *_spatial_shape(name)), device=device, dtype=dtype)
        if transpose:
            image = image.transpose(-1, -2)
        image.requires_grad_()
        output = getattr(enhance, name)(image, *_arguments(name))
        assert output.shape == image.shape
        assert output.device == image.device
        assert output.dtype == image.dtype
        assert output.numel() == 0
        output.sum().backward()
        assert image.grad is not None
        assert image.grad.shape == image.shape

    @pytest.mark.parametrize("name", _FUNCTIONS)
    @pytest.mark.parametrize("leading", [None, (), (1,), (2,), (2, 1, 2)])
    @pytest.mark.parametrize("transpose", [False, True])
    def test_nonempty_rank_interpretation_4429(self, name, leading, transpose, device, dtype):
        spatial = _spatial_shape(name)
        shape = spatial if leading is None else (*leading, 3, *spatial)
        image = _input(shape, device, dtype)
        if transpose:
            image = image.transpose(-1, -2)
        channels = 1 if leading is None else 3
        canonical = image.reshape(-1, channels, *image.shape[-len(spatial) :])
        fn = getattr(enhance, name)
        expected = fn(canonical, *_arguments(name)).reshape(image.shape)
        output = fn(image, *_arguments(name))
        assert output.shape == image.shape
        self.assert_close(output, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("name", _FUNCTIONS)
    def test_invalid_planes_4429(self, name, device, dtype):
        shape = (0, 3, *_spatial_shape(name))
        for axis in range(1, len(shape)):
            invalid = list(shape)
            invalid[axis] = 0
            with pytest.raises(ValueError):
                getattr(enhance, name)(torch.empty(invalid, device=device, dtype=dtype), *_arguments(name))
        # In a minimum-rank tensor, the first dimension is spatial, not an empty batch.
        invalid = (0, *_spatial_shape(name)[1:])
        with pytest.raises(ValueError):
            getattr(enhance, name)(torch.empty(invalid, device=device, dtype=dtype), *_arguments(name))

    @pytest.mark.parametrize("name", _FUNCTIONS)
    def test_invalid_rank_and_type_4429(self, name, device, dtype):
        for shape in [(), (3,)]:
            with pytest.raises(ValueError):
                getattr(enhance, name)(torch.ones(shape, device=device, dtype=dtype), *_arguments(name))
        with pytest.raises(TypeError):
            getattr(enhance, name)([1.0], *_arguments(name))

    @pytest.mark.parametrize("batch", [0, 2])
    @pytest.mark.parametrize("form", ["scalar", "one", "sample", "channel"])
    def test_posterize_parameter_forms_4429(self, batch, form, device, dtype):
        image = _input((2, batch, 3, 5, 7), device, dtype).requires_grad_()
        count = 2 * batch
        if form == "scalar":
            bits = 4
            mask = torch.zeros((count, 3, 1, 1), device=device, dtype=dtype)
        elif form == "one":
            bits = torch.tensor([8])
            mask = torch.ones((count, 3, 1, 1), device=device, dtype=dtype)
        elif form == "sample":
            bits = torch.arange(count).remainder(3) * 4
            mask = (bits == 8).to(image).reshape(count, 1, 1, 1).expand(count, 3, 1, 1)
        else:
            bits = (torch.arange(count * 3).remainder(3) * 4).reshape(count, 3)
            mask = (bits == 8).to(image).reshape(count, 3, 1, 1)
        output = enhance.posterize(image, bits)
        assert output.shape == image.shape
        assert output.device == image.device
        assert output.dtype == dtype
        weights = _weights(image)
        (output * weights).sum().backward()
        assert image.grad is not None
        expected_grad = (mask.expand(count, 3, 5, 7) * weights.reshape(count, 3, 5, 7)).reshape(image.shape)
        self.assert_close(image.grad, expected_grad, rtol=0, atol=0)
        if batch:
            canonical = image.detach().reshape(count, 3, 5, 7)
            expected = []
            for index in range(count):
                per_image_bits = bits if form in ("scalar", "one") else bits[index : index + 1]
                expected.append(enhance.posterize(canonical[index : index + 1], per_image_bits))
            self.assert_close(output, torch.cat(expected).reshape(image.shape), rtol=0, atol=0)

    @pytest.mark.parametrize("batch", [0, 2])
    @pytest.mark.parametrize("form", ["scalar", "sample"])
    def test_sharpness_parameter_forms_4429(self, batch, form, device, dtype):
        image = _input((2, batch, 3, 5, 7), device, dtype).requires_grad_()
        count = 2 * batch
        factor = 0.5 if form == "scalar" else torch.linspace(0.2, 1.5, count)
        output = enhance.sharpness(image, factor)
        assert output.shape == image.shape
        assert output.device == image.device
        assert output.dtype == dtype
        (output * _weights(image)).sum().backward()
        assert image.grad is not None
        assert image.grad.shape == image.shape
        if batch:
            canonical = image.detach().reshape(count, 3, 5, 7)
            expected = [
                enhance.sharpness(canonical[index : index + 1], factor if form == "scalar" else factor[index])
                for index in range(count)
            ]
            self.assert_close(output, torch.cat(expected).reshape(image.shape))

    def test_empty_parameters_are_validated_4429(self, device, dtype):
        image = torch.empty(0, 3, 6, 8, device=device, dtype=dtype)
        with pytest.raises(TypeError, match="bits type"):
            enhance.posterize(image, 4.0)
        for bits in [torch.tensor([4, 8]), torch.empty(0, 2, dtype=torch.long)]:
            with pytest.raises(AssertionError, match="equal between bits and input"):
                enhance.posterize(image, bits)
        for factor in [torch.tensor([0.5]), torch.empty(0, 1)]:
            with pytest.raises(AssertionError, match="batch size shall match"):
                enhance.sharpness(image, factor)

    @pytest.mark.parametrize("spatial", [(2, 8), (6, 2)])
    def test_small_empty_sharpness_is_rejected_4429(self, spatial, device, dtype):
        with pytest.raises(RuntimeError, match="Kernel size"):
            enhance.sharpness(torch.empty(0, 3, *spatial, device=device, dtype=dtype), 0.5)

    @pytest.mark.parametrize("volume", [False, True])
    @pytest.mark.parametrize("plane_size", [255, 256, 429])
    def test_equalize_independent_planes_4429(self, volume, plane_size, device, dtype):
        spatial = {255: (3, 5, 17), 256: (2, 8, 16), 429: (3, 11, 13)}[plane_size]
        if not volume:
            spatial = {255: (5, 51), 256: (16, 16), 429: (13, 33)}[plane_size]
        indices = torch.arange(plane_size)
        levels = torch.tensor([0.0, 0.125, 0.375, 0.875])
        channel0 = levels[indices.remainder(4)]
        channel1 = levels[(indices * indices + 3 * indices + 1).remainder(4)]
        image = torch.stack([channel0, channel1]).reshape(1, 2, *spatial).to(device=device, dtype=dtype)
        image.requires_grad_()
        reference_input = image.detach().clone().requires_grad_()
        fn = enhance.equalize3d if volume else enhance.equalize
        output = fn(image)
        expected = _equalize_reference(reference_input)
        self.assert_close(output, expected, rtol=0, atol=0)
        weights = _weights(image)
        (output * weights).sum().backward()
        (expected * weights).sum().backward()
        self.assert_close(image.grad, reference_input.grad, rtol=0, atol=0)
        if plane_size == 429:
            assert not torch.equal(output, image)
            if volume:
                per_slice = torch.stack([enhance.equalize(image[:, :, depth]) for depth in range(spatial[0])], 2)
                assert not torch.equal(output, per_slice)

    @pytest.mark.parametrize("name", _FUNCTIONS)
    @pytest.mark.parametrize("batch", [0, 2])
    def test_dynamo_shape_and_values_4429(self, name, batch, device, dtype, torch_optimizer):
        image = _input((2, batch, 3, *_spatial_shape(name)), device, dtype)
        fn = getattr(enhance, name)
        expected = fn(image, *_arguments(name))
        optimized = torch_optimizer(fn)
        output = optimized(image, *_arguments(name))
        assert output.shape == image.shape
        assert output.device == image.device
        assert output.dtype == dtype
        self.assert_close(output, expected)
