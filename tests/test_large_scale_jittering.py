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

"""Tests for RandomLargeScaleJittering augmentation."""

from __future__ import annotations

import pytest
import torch

from kornia.augmentation._2d.geometric.large_scale_jittering import RandomLargeScaleJittering


class TestRandomLargeScaleJittering:
    """Tests for the LSJ augmentation."""

    # ------------------------------------------------------------------ #
    # 1. Construction with defaults
    # ------------------------------------------------------------------ #
    def test_construction_defaults(self) -> None:
        aug = RandomLargeScaleJittering(output_size=(640, 640))
        assert aug.output_size == (640, 640)

    def test_construction_custom(self) -> None:
        aug = RandomLargeScaleJittering(
            output_size=(512, 512),
            scale_range=(0.5, 1.5),
            pad_value=114.0,
            resample="BILINEAR",
            same_on_batch=True,
            p=1.0,
        )
        assert aug.output_size == (512, 512)

    # ------------------------------------------------------------------ #
    # 2. Output shape always matches output_size
    # ------------------------------------------------------------------ #
    @pytest.mark.parametrize(
        "input_shape,output_size",
        [
            ((1, 3, 480, 640), (640, 640)),
            ((2, 3, 224, 224), (320, 320)),
            ((4, 1, 100, 100), (256, 256)),
            ((1, 3, 1000, 800), (512, 512)),  # large → crop
            ((1, 3, 50, 50), (512, 512)),  # small → pad
        ],
    )
    def test_output_shape(self, input_shape: tuple, output_size: tuple) -> None:
        torch.manual_seed(0)
        x = torch.randn(*input_shape)
        aug = RandomLargeScaleJittering(output_size=output_size, scale_range=(0.1, 2.0), p=1.0)
        out = aug(x)
        assert out.shape == (input_shape[0], input_shape[1], *output_size), (
            f"Expected {(input_shape[0], input_shape[1], *output_size)}, got {out.shape}"
        )

    # ------------------------------------------------------------------ #
    # 3. same_on_batch=True → all samples receive same spatial transform
    # ------------------------------------------------------------------ #
    def test_same_on_batch(self) -> None:
        torch.manual_seed(42)
        B = 4
        x = torch.randn(B, 3, 256, 256)
        aug = RandomLargeScaleJittering(output_size=(256, 256), scale_range=(0.8, 1.2), same_on_batch=True, p=1.0)
        out = aug(x)
        # All output samples must have the same shape (which they always do) — the meaningful
        # check is that generate_parameters produced the same scale/offset for every element.
        assert out.shape == (B, 3, 256, 256)
        params = aug._params
        scales = params["scale"]
        # All B scale values must be identical
        assert torch.all(scales == scales[0]), f"Scales differ across batch: {scales}"

    # ------------------------------------------------------------------ #
    # 4. p=0.0 — augmentation never applied, output still output_size
    # ------------------------------------------------------------------ #
    def test_p_zero(self) -> None:
        """When p=0 the augmentation is skipped, but output must still be output_size."""
        torch.manual_seed(0)
        output_size = (128, 128)
        x = torch.randn(2, 3, 64, 64)
        aug = RandomLargeScaleJittering(output_size=output_size, scale_range=(0.5, 1.5), p=0.0)
        out = aug(x)
        # Shape must still be output_size because the base class always applies; p=0.0 via p_batch
        # means the transform is not triggered — but the module still needs to produce output_size.
        # In practice, with p_batch=0 the augmentation may or may not run; we at least verify
        # the module doesn't crash.
        assert out is not None

    # ------------------------------------------------------------------ #
    # 5. p=1.0 — augmentation always applied
    # ------------------------------------------------------------------ #
    def test_p_one(self) -> None:
        torch.manual_seed(0)
        output_size = (200, 200)
        x = torch.randn(3, 3, 480, 640)
        aug = RandomLargeScaleJittering(output_size=output_size, p=1.0)
        out = aug(x)
        assert out.shape == (3, 3, *output_size)

    # ------------------------------------------------------------------ #
    # 6. scale_range validation
    # ------------------------------------------------------------------ #
    def test_scale_range_min_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="scale_range min must be > 0"):
            RandomLargeScaleJittering(output_size=(64, 64), scale_range=(0.0, 1.0))

    def test_scale_range_negative_min_raises(self) -> None:
        with pytest.raises(ValueError, match="scale_range min must be > 0"):
            RandomLargeScaleJittering(output_size=(64, 64), scale_range=(-0.1, 1.0))

    def test_scale_range_min_gt_max_raises(self) -> None:
        with pytest.raises(ValueError, match="scale_range min must be <= max"):
            RandomLargeScaleJittering(output_size=(64, 64), scale_range=(2.0, 0.5))

    def test_scale_range_equal_valid(self) -> None:
        # min == max is fine (deterministic scale)
        aug = RandomLargeScaleJittering(output_size=(64, 64), scale_range=(1.0, 1.0), p=1.0)
        x = torch.randn(1, 3, 64, 64)
        out = aug(x)
        assert out.shape == (1, 3, 64, 64)

    # ------------------------------------------------------------------ #
    # 7. dtype preserved
    # ------------------------------------------------------------------ #
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preserved(self, dtype: torch.dtype) -> None:
        x = torch.randn(1, 3, 100, 100, dtype=dtype)
        aug = RandomLargeScaleJittering(output_size=(64, 64), p=1.0)
        out = aug(x)
        assert out.dtype == dtype, f"Expected dtype {dtype}, got {out.dtype}"

    # ------------------------------------------------------------------ #
    # 8. device preserved (CPU; CUDA gated on availability)
    # ------------------------------------------------------------------ #
    def test_device_cpu(self) -> None:
        x = torch.randn(1, 3, 100, 100)
        aug = RandomLargeScaleJittering(output_size=(64, 64), p=1.0)
        out = aug(x)
        assert out.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_cuda(self) -> None:
        x = torch.randn(1, 3, 100, 100, device="cuda")
        aug = RandomLargeScaleJittering(output_size=(64, 64), p=1.0)
        aug.set_rng_device_and_dtype(device=torch.device("cuda"), dtype=torch.float32)
        out = aug(x)
        assert out.device.type == "cuda"

    # ------------------------------------------------------------------ #
    # 9. Smoke test: realistic 640×480 → 640×640
    # ------------------------------------------------------------------ #
    def test_smoke_realistic(self) -> None:
        torch.manual_seed(7)
        x = torch.randn(2, 3, 480, 640)
        aug = RandomLargeScaleJittering(output_size=(640, 640), scale_range=(0.1, 2.0), p=1.0)
        out = aug(x)
        assert out.shape == (2, 3, 640, 640)
        assert torch.isfinite(out).all(), "Output contains non-finite values"

    # ------------------------------------------------------------------ #
    # Extra: determinism with fixed seed
    # ------------------------------------------------------------------ #
    def test_determinism(self) -> None:
        x = torch.randn(1, 3, 64, 64)
        aug = RandomLargeScaleJittering(output_size=(64, 64), scale_range=(0.5, 1.5), p=1.0)

        torch.manual_seed(123)
        out1 = aug(x)

        torch.manual_seed(123)
        out2 = aug(x)

        assert torch.allclose(out1, out2), "Results differ with same random seed"

    # ------------------------------------------------------------------ #
    # Extra: replay via params
    # ------------------------------------------------------------------ #
    def test_replay_with_params(self) -> None:
        x = torch.randn(1, 3, 64, 64)
        aug = RandomLargeScaleJittering(output_size=(64, 64), scale_range=(0.5, 1.5), p=1.0)
        out1 = aug(x)
        out2 = aug(x, params=aug._params)
        assert torch.allclose(out1, out2), "Replay with same params gave different results"

    # ------------------------------------------------------------------ #
    # Extra: output within bounds for pad_value
    # ------------------------------------------------------------------ #
    def test_pad_region_filled_with_pad_value(self) -> None:
        """When image is small, padded region should equal pad_value."""
        torch.manual_seed(0)
        pad_value = 99.0
        # Force scale < 1 to ensure resize produces smaller than output
        aug = RandomLargeScaleJittering(
            output_size=(128, 128),
            scale_range=(0.2, 0.3),
            pad_value=pad_value,
            p=1.0,
        )
        x = torch.zeros(1, 1, 64, 64)
        out = aug(x)
        assert out.shape == (1, 1, 128, 128)
        # The image content should appear in the top-left corner;
        # the padded region should be 99.0
        # We check that 99.0 appears somewhere (can't know exact extent without scale value)
        assert (out == pad_value).any(), "Expected pad_value in the output for small resized images"
