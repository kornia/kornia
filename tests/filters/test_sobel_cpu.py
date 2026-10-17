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

import importlib
import warnings

import pytest
import torch
import torch.nn.functional as F

from kornia.filters import sobel, spatial_gradient
from kornia.filters.sobel import _spatial_gradient_cpu_eligible, _spatial_gradient_sobel_cpu

from testing.base import BaseTester

sobel_module = importlib.import_module("kornia.filters.sobel")


def _require_native_cpu(device: torch.device, dtype: torch.dtype) -> None:
    if device.type != "cpu" or dtype not in (torch.float32, torch.float64):
        pytest.skip("the CPU slice Sobel path supports native CPU float32 and float64 only")


def _cpu_tolerance(dtype: torch.dtype) -> float:
    return 5e-7 if dtype == torch.float32 else 1e-12


def _conv_reference(image: torch.Tensor, normalized: bool) -> torch.Tensor:
    """The first-order Sobel derivative as one replicate-padded convolution, the path the helper replaces."""
    scale = 0.125 if normalized else 1.0
    kernel = scale * torch.tensor(
        [
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        ],
        device=image.device,
        dtype=image.dtype,
    )
    b, c, h, w = image.shape
    padded = F.pad(image.reshape(b * c, 1, h, w), [1, 1, 1, 1], "replicate")
    return F.conv2d(padded, kernel[:, None]).reshape(b, c, 2, h, w)


def _forbid_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("the slice path must not run here")

    monkeypatch.setattr(sobel_module, "_spatial_gradient_sobel_cpu", fail)


class TestSpatialGradientSobelCpu(BaseTester):
    @pytest.mark.parametrize("normalized", [True, False])
    def test_helper_matches_convolution(self, normalized, device, dtype):
        _require_native_cpu(device, dtype)
        image = torch.rand(2, 3, 257, 256, device=device, dtype=dtype)

        tolerance = _cpu_tolerance(dtype)
        self.assert_close(
            _spatial_gradient_sobel_cpu(image, normalized),
            _conv_reference(image, normalized),
            rtol=tolerance,
            atol=tolerance,
        )

    @pytest.mark.parametrize("normalized", [True, False])
    def test_helper_preserves_nonfinite_pattern(self, normalized, device, dtype):
        # The zero-weight centre taps of the convolution still multiply their pixels, so 0 * inf and 0 * nan make
        # NaN there; the slices must keep exactly the convolution's NaN and inf positions.
        _require_native_cpu(device, dtype)
        generator = torch.Generator().manual_seed(0)
        image = torch.rand(1, 2, 256, 256, generator=generator, dtype=dtype)
        flat = image.view(-1)
        positions = torch.randperm(flat.numel(), generator=generator)[:40]
        flat[positions[:14]] = float("nan")
        flat[positions[14:27]] = float("inf")
        flat[positions[27:]] = -float("inf")
        image = image.to(device)

        actual = _spatial_gradient_sobel_cpu(image, normalized)
        expected = _conv_reference(image, normalized)

        assert torch.equal(actual.isnan(), expected.isnan())
        assert torch.equal(actual.isinf(), expected.isinf())
        assert torch.equal(actual.isinf() & (actual > 0), expected.isinf() & (expected > 0))
        tolerance = _cpu_tolerance(dtype)
        self.assert_close(
            actual.nan_to_num(posinf=0.0, neginf=0.0),
            expected.nan_to_num(posinf=0.0, neginf=0.0),
            rtol=tolerance,
            atol=tolerance,
        )

    def test_public_large_path_dispatches_without_reverse_mode_gradients(self, monkeypatch, device, dtype):
        _require_native_cpu(device, dtype)
        calls = []

        def counting(image, normalized):
            calls.append(image.shape)
            return _spatial_gradient_sobel_cpu(image, normalized)

        monkeypatch.setattr(sobel_module, "_spatial_gradient_sobel_cpu", counting)
        image = torch.rand(2, 3, 256, 256, device=device, dtype=dtype)

        actual = spatial_gradient(image)

        assert len(calls) == 1
        tolerance = _cpu_tolerance(dtype)
        self.assert_close(actual, _conv_reference(image, True), rtol=tolerance, atol=tolerance)

    def test_public_reverse_mode_gradients_use_convolution(self, monkeypatch, device, dtype):
        _require_native_cpu(device, dtype)
        _forbid_helper(monkeypatch)
        image = torch.rand(2, 3, 256, 256, device=device, dtype=dtype, requires_grad=True)
        reference = image.detach().clone().requires_grad_()
        weights = torch.rand(2, 3, 2, 256, 256, device=device, dtype=dtype)

        (spatial_gradient(image) * weights).sum().backward()
        (_conv_reference(reference, True) * weights).sum().backward()

        self.assert_close(image.grad, reference.grad)

    def test_small_input_uses_convolution(self, monkeypatch, device, dtype):
        _require_native_cpu(device, dtype)
        _forbid_helper(monkeypatch)
        image = torch.rand(1, 1, 3, 3, device=device, dtype=dtype)

        assert not _spatial_gradient_cpu_eligible(image)
        self.assert_close(spatial_gradient(image), _conv_reference(image, True))

    def test_noncontiguous_and_channels_last_match_contiguous(self, device, dtype):
        _require_native_cpu(device, dtype)
        image = torch.rand(2, 3, 256, 256, device=device, dtype=dtype)
        transposed = image.transpose(-1, -2).contiguous().transpose(-1, -2)
        channels_last = image.contiguous(memory_format=torch.channels_last)
        assert not transposed.is_contiguous()

        expected = spatial_gradient(image)
        tolerance = _cpu_tolerance(dtype)
        for variant in (transposed, channels_last):
            self.assert_close(spatial_gradient(variant), expected, rtol=tolerance, atol=tolerance)

    @pytest.mark.parametrize(
        "shape, eligible",
        [
            ((1, 3, 127, 128), False),
            ((1, 3, 128, 128), True),
            ((1, 1, 128, 128), False),
            ((1, 1, 255, 257), True),
            ((2, 3, 256, 256), True),
            ((1, 1, 1, 70000), True),
        ],
    )
    def test_threshold_and_odd_shapes(self, shape, eligible, device, dtype):
        # The slices beat the convolution from 128 x 128 images of 3 channels on (float64 128^2: 0.88x the time).
        _require_native_cpu(device, dtype)
        image = torch.rand(*shape, device=device, dtype=dtype)
        assert _spatial_gradient_cpu_eligible(image) == eligible

        tolerance = _cpu_tolerance(dtype)
        self.assert_close(spatial_gradient(image), _conv_reference(image, True), rtol=tolerance, atol=tolerance)

    def test_helper_vmap_has_no_user_warning(self, device, dtype):
        _require_native_cpu(device, dtype)
        image = torch.rand(2, 3, 7, 9, device=device, dtype=dtype)

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            actual = torch.vmap(lambda x: _spatial_gradient_sobel_cpu(x[None], True)[0])(image)
        assert not any(issubclass(warning.category, UserWarning) for warning in recorded)
        tolerance = _cpu_tolerance(dtype)
        self.assert_close(actual, _conv_reference(image, True), rtol=tolerance, atol=tolerance)

    def test_helper_gradcheck_small_input(self, device):
        if device.type != "cpu":
            pytest.skip("the CPU slice Sobel path runs on the CPU only")
        image = torch.rand(1, 1, 7, 9, device=device, dtype=torch.float64)
        self.gradcheck(_spatial_gradient_sobel_cpu, (image, True))

    @pytest.mark.parametrize("normalized", [True, False])
    def test_sobel_magnitude_large_path_matches(self, normalized, device, dtype):
        _require_native_cpu(device, dtype)
        generator = torch.Generator().manual_seed(0)
        image = torch.rand(2, 3, 256, 256, generator=generator, dtype=dtype)
        image.view(-1)[torch.randperm(image.numel(), generator=generator)[:10]] = float("nan")
        image = image.to(device)
        assert _spatial_gradient_cpu_eligible(image)

        actual = sobel(image, normalized=normalized)
        edges = _conv_reference(image, normalized)
        expected = torch.sqrt(edges[:, :, 0] * edges[:, :, 0] + edges[:, :, 1] * edges[:, :, 1] + 1e-6)

        assert torch.equal(actual.isnan(), expected.isnan())
        tolerance = _cpu_tolerance(dtype)
        self.assert_close(actual.nan_to_num(), expected.nan_to_num(), rtol=tolerance, atol=tolerance)
