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

from kornia.models.qwen3_vl import (
    Qwen3VLImageProcessor,
    Qwen3VLImageProcessorConfig,
    smart_resize,
)

from testing.base import BaseTester


class TestSmartResize:
    def test_already_in_band_rounds_to_factor(self):
        # 224x224 with patch_size=14, merge=2 -> factor=28; 224 is divisible by 28.
        h, w = smart_resize(224, 224, factor=28, min_pixels=56 * 56, max_pixels=28 * 28 * 1280)
        assert h == 224
        assert w == 224

    def test_shrinks_when_over_max(self):
        h, w = smart_resize(2048, 2048, factor=28, min_pixels=56 * 56, max_pixels=28 * 28 * 100)
        assert h % 28 == 0
        assert w % 28 == 0
        assert h * w <= 28 * 28 * 100
        assert abs(h - w) <= 28

    def test_expands_when_under_min(self):
        h, w = smart_resize(8, 8, factor=28, min_pixels=56 * 56, max_pixels=28 * 28 * 1280)
        assert h % 28 == 0
        assert w % 28 == 0
        assert h * w >= 56 * 56

    def test_preserves_aspect_ratio_within_factor(self):
        h, w = smart_resize(700, 350, factor=28, min_pixels=56 * 56, max_pixels=28 * 28 * 1280)
        # Original ratio is 2.0; rounding to factor=28 should keep it close.
        ratio = max(h, w) / min(h, w)
        assert 1.7 <= ratio <= 2.3

    def test_below_factor_clamped_up(self):
        # Tiny inputs should still produce dims >= factor.
        h, w = smart_resize(3, 5, factor=28, min_pixels=28 * 28, max_pixels=28 * 28 * 100)
        assert h >= 28
        assert w >= 28

    def test_aspect_ratio_guard(self):
        with pytest.raises(ValueError, match="aspect ratio"):
            smart_resize(10, 5000, factor=28, min_pixels=56 * 56, max_pixels=28 * 28 * 1280)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError, match="positive"):
            smart_resize(0, 100, factor=28, min_pixels=1, max_pixels=10)
        with pytest.raises(ValueError, match="positive"):
            smart_resize(100, 100, factor=0, min_pixels=1, max_pixels=10)
        with pytest.raises(ValueError, match="must not exceed"):
            smart_resize(100, 100, factor=28, min_pixels=10_000, max_pixels=100)


@pytest.fixture
def small_config():
    # Tiny bands so resize logic actually triggers in tests.
    return Qwen3VLImageProcessorConfig(
        patch_size=4,
        spatial_merge_size=2,
        min_pixels=8 * 8,
        max_pixels=32 * 32,
    )


@pytest.fixture
def processor(device, dtype, small_config):
    return Qwen3VLImageProcessor(small_config).to(device=device, dtype=dtype)


class TestQwen3VLImageProcessor(BaseTester):
    def test_smoke(self, processor, small_config):
        assert processor is not None
        assert processor.factor == small_config.patch_size * small_config.spatial_merge_size

    def test_passthrough_when_in_band(self, device, dtype, processor, small_config):
        # 24x24 is divisible by factor=8 and 24*24=576 is in [64, 1024].
        x = torch.zeros(1, 3, 24, 24, device=device, dtype=dtype)
        out = processor(x)
        assert out.shape == (1, 3, 24, 24)

    def test_resizes_when_outside_band(self, device, dtype, processor):
        big = torch.zeros(2, 3, 64, 64, device=device, dtype=dtype)
        out = processor(big)
        assert out.shape[0] == 2
        assert out.shape[1] == 3
        assert out.shape[-2] % processor.factor == 0
        assert out.shape[-1] % processor.factor == 0
        assert out.shape[-2] * out.shape[-1] <= 32 * 32

    def test_normalize_zero_input_yields_minus_mean_over_std(self, device, dtype, processor, small_config):
        x = torch.zeros(1, 3, 24, 24, device=device, dtype=dtype)
        out = processor(x)
        expected = -torch.tensor(small_config.image_mean, device=device, dtype=dtype) / torch.tensor(
            small_config.image_std, device=device, dtype=dtype
        )
        # All spatial positions should equal the per-channel constant.
        for c in range(3):
            assert torch.allclose(out[0, c], expected[c].expand_as(out[0, c]), rtol=1e-4, atol=1e-4)

    def test_exception(self, device, dtype, processor):
        with pytest.raises(ValueError, match="Expected 4D"):
            processor(torch.zeros(3, 24, 24, device=device, dtype=dtype))
        with pytest.raises(ValueError, match="channels"):
            processor(torch.zeros(1, 1, 24, 24, device=device, dtype=dtype))

    def test_target_size_helper(self, processor):
        h, w = processor.target_size(64, 64)
        assert h % processor.factor == 0
        assert w % processor.factor == 0

    def test_gradcheck(self, device, small_config):
        proc = Qwen3VLImageProcessor(small_config).to(device=device, dtype=torch.float64)
        x = torch.randn(1, 3, 24, 24, device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(proc, x, raise_exception=True, fast_mode=True)
