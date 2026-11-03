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

from kornia.core.exceptions import BaseError
from kornia.filters import SauvolaThreshold, sauvola_threshold

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, dynamo_is_available


class TestSauvolaThreshold(BaseTester):
    @pytest.mark.parametrize("window_size", [1, 3, (3, 5)])
    def test_smoke(self, device, dtype, window_size):
        image = torch.rand(2, 3, 7, 9, device=device, dtype=dtype).transpose(-1, -2)
        mask, threshold = sauvola_threshold(image, window_size)
        assert mask.shape == threshold.shape == image.shape
        assert mask.dtype == torch.bool
        assert threshold.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
        assert mask.device == threshold.device == image.device
        self.assert_close(mask, image > threshold)
        other = SauvolaThreshold(window_size)(image)
        self.assert_close(other[0], mask)
        self.assert_close(other[1], threshold)

    def test_scikit_image_reference(self, device, dtype):
        # scikit-image 0.26.0: threshold_sauvola(np.array(values, dtype=np.float64), 3, k=0.2, r=0.5)
        # Hardcoded float64 output keeps the correctness test independent of optional libraries.
        values = [
            [0, 0.1, 0.9, 0.9, 1],
            [0, 0.2, 0.8, 0.7, 1],
            [0.1, 0.1, 0.9, 1, 0.9],
            [0.2, 0, 0.7, 0.8, 1],
            [0, 0.1, 1, 0.9, 0.8],
        ]
        expected = [
            [0.092777287839, 0.314945691072, 0.543454278809, 0.733362951997, 0.720904271308],
            [0.082666666667, 0.327152719599, 0.586185142947, 0.753941125497, 0.761569219382],
            [0.083265986324, 0.311991284565, 0.543588635548, 0.733362951997, 0.745312766437],
            [0.064177678012, 0.327963903785, 0.585604657759, 0.746446506311, 0.749393876913],
            [0.055510657549, 0.303560163312, 0.528974038364, 0.724325077190, 0.721638548139],
        ]
        # Fixed float32 inputs keep this reference check available on devices without float64.
        image = torch.tensor(values, device=device, dtype=torch.float32)[None, None]
        mask, actual = sauvola_threshold(image, 3)
        expected = torch.tensor(expected, dtype=torch.float64)[None, None]
        self.assert_close(actual.cpu().double(), expected, atol=1e-6, rtol=1e-6)
        self.assert_close(mask.cpu(), image.cpu().double() > expected)

    @pytest.mark.parametrize("window_size", [3, (3, 5)])
    def test_window_statistics(self, device, dtype, window_size):
        image = torch.arange(63, dtype=torch.float64).reshape(1, 1, 7, 9).div(62).to(device=device, dtype=dtype)
        kh, kw = (window_size, window_size) if isinstance(window_size, int) else window_size
        # Direct window statistics are independent of the filtered-moments implementation.
        padded = torch.nn.functional.pad(image.cpu().double(), (kw // 2, kw // 2, kh // 2, kh // 2), mode="reflect")
        windows = padded.unfold(-2, kh, 1).unfold(-2, kw, 1)
        mean = windows.mean((-2, -1))
        std = windows.std((-2, -1), correction=0)
        expected = mean * (1 + 0.2 * (std / 0.5 - 1))
        mask, actual = sauvola_threshold(image, window_size)
        self.assert_close(actual.cpu().double(), expected, atol=2e-6, rtol=2e-6)
        self.assert_close(mask.cpu(), image.cpu().double() > expected)

    def test_batch_and_channels_are_independent(self, device, dtype):
        image = torch.rand(2, 3, 7, 9, device=device, dtype=dtype)
        actual = sauvola_threshold(image, 3)
        for b in range(2):
            for c in range(3):
                expected = sauvola_threshold(image[b : b + 1, c : c + 1], 3)
                for output, reference in zip(actual, expected):
                    self.assert_close(output[b : b + 1, c : c + 1], reference)

    @pytest.mark.parametrize("value", [0.0, 0.7, 1.0])
    def test_constant_value_and_backward(self, device, dtype, value):
        image = torch.full((1, 1, 7, 9), value, device=device, dtype=dtype, requires_grad=True)
        mask, threshold = sauvola_threshold(image, 3)
        expected = (image.double() * 0.8).to(threshold.dtype)
        self.assert_close(threshold, expected, atol=0, rtol=0)
        self.assert_close(mask.cpu(), image.cpu().double() > expected)
        threshold.sum().backward()
        assert image.grad is not None
        assert torch.isfinite(image.grad).all()

    @pytest.mark.device_agnostic
    def test_constant_region_in_nonconstant_page(self):
        # A dark first pixel must not give an otherwise flat, white region positive
        # contrast through rounded box-filter weights or accumulated moment cancellation.
        image = torch.ones(1, 1, 128, 128)
        image[..., 0, 0] = 0
        _, threshold = sauvola_threshold(image, 31)
        self.assert_close(threshold[..., 16:, 16:], torch.full((1, 1, 112, 112), 0.8), atol=0, rtol=0)

    def test_uint8_scale(self, device):
        image = torch.arange(63, device=device, dtype=torch.uint8).reshape(1, 1, 7, 9) * 4
        mask, threshold = sauvola_threshold(image, 3)
        normalized = sauvola_threshold(image.float() / 255, 3)
        self.assert_close(mask, normalized[0])
        self.assert_close(threshold / 255, normalized[1], atol=2e-6, rtol=2e-6)

    @pytest.mark.parametrize("shape,window", [((0, 2, 5, 7), 3), ((2, 0, 5, 7), 3), ((2, 1, 1, 7), (1, 3))])
    def test_empty_and_thin_images(self, device, dtype, shape, window):
        image = torch.ones(shape, device=device, dtype=dtype)
        mask, threshold = sauvola_threshold(image, window)
        assert mask.shape == threshold.shape == image.shape

    @pytest.mark.parametrize("window_size", [0, 2, -3, (3, 2), (3.5, 3), (3,)])
    def test_invalid_window(self, device, dtype, window_size):
        with pytest.raises(BaseError):
            sauvola_threshold(torch.ones(1, 1, 7, 9, device=device, dtype=dtype), window_size)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"r": 0},
            {"r": -1},
            {"r": float("nan")},
            {"r": float("inf")},
            {"k": -1},
            {"k": float("nan")},
            {"k": float("inf")},
        ],
    )
    def test_invalid_parameters(self, device, dtype, kwargs):
        with pytest.raises(BaseError):
            sauvola_threshold(torch.ones(1, 1, 7, 9, device=device, dtype=dtype), 3, **kwargs)

    def test_invalid_input(self, device):
        for image in [torch.ones(5, 7, device=device), torch.ones(1, 1, 5, 7, device=device, dtype=torch.long)]:
            with pytest.raises(BaseError):
                sauvola_threshold(image, 3)
        with pytest.raises(BaseError, match="height"):
            sauvola_threshold(torch.ones(1, 1, 1, 7, device=device), 3)

    def test_gradcheck(self, device):
        if device.type == "mps":
            pytest.skip("MPS does not support float64 gradcheck.")
        image = torch.linspace(0.1, 0.9, 25, device=device, dtype=torch.float64).reshape(1, 1, 5, 5)
        self.gradcheck(lambda x: sauvola_threshold(x, 3)[1], (image,))

    @pytest.mark.device_agnostic
    def test_autocast(self):
        image = torch.linspace(0.1, 0.9, 63).reshape(1, 1, 7, 9)
        expected = sauvola_threshold(image, 3)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = sauvola_threshold(image, 3)
        self.assert_close(actual[0], expected[0])
        self.assert_close(actual[1], expected[1], atol=0, rtol=0)

    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    def test_dynamo(self, device, dtype, torch_optimizer):
        image = torch.linspace(0.1, 0.9, 63, device=device, dtype=dtype).reshape(1, 1, 7, 9)
        optimized = torch_optimizer(SauvolaThreshold(3), fullgraph=True)
        actual = optimized(image)
        expected = sauvola_threshold(image, 3)
        self.assert_close(actual[0], expected[0])
        self.assert_close(actual[1], expected[1], atol=2e-6, rtol=2e-6)

    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    def test_export(self, device, dtype):
        image = torch.linspace(0.1, 0.9, 63, device=device, dtype=dtype).reshape(1, 1, 7, 9)
        exported = torch.export.export(SauvolaThreshold(3), (image,), strict=True).module()
        for data in (image, torch.ones_like(image)):
            actual = exported(data)
            expected = sauvola_threshold(data, 3)
            self.assert_close(actual[0], expected[0])
            self.assert_close(actual[1], expected[1])
