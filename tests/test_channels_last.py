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
"""Tests for channels-last (NHWC) memory-format preservation in AugmentationSequential.

PR-G7: channels-last memory format awareness.

Scope: AugmentationSequential must preserve the memory format of the
primary image input — channels-last in → channels-last out, NCHW in →
NCHW out.  No silent format promotion in either direction.
"""

from __future__ import annotations

import pytest
import torch

import kornia.augmentation as K
from kornia.augmentation.container import AugmentationSequential

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline() -> AugmentationSequential:
    """Representative composition used for all format-preservation tests."""
    return AugmentationSequential(
        K.RandomHorizontalFlip(p=1.0),
        K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
        K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
        data_keys=["input"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestChannelsLastPreservation:
    """AugmentationSequential must preserve the input memory format."""

    def test_channels_last_input_produces_channels_last_output(self) -> None:
        """channels_last input → channels_last output (no silent NCHW downgrade)."""
        pipeline = _make_pipeline()
        x = torch.rand(2, 3, 32, 32).contiguous(memory_format=torch.channels_last)

        assert x.is_contiguous(memory_format=torch.channels_last), "Precondition: input must be channels-last"

        with torch.no_grad():
            y = pipeline(x)

        assert isinstance(y, torch.Tensor)
        assert y.shape == x.shape
        assert y.is_contiguous(memory_format=torch.channels_last), (
            "Output should be channels-last when input was channels-last"
        )

    def test_nchw_input_stays_nchw(self) -> None:
        """NCHW (contiguous) input → NCHW output (no spurious channels-last conversion)."""
        pipeline = _make_pipeline()
        x = torch.rand(2, 3, 32, 32)  # default contiguous NCHW

        assert x.is_contiguous(), "Precondition: input must be contiguous"
        assert not x.is_contiguous(memory_format=torch.channels_last), (
            "Precondition: input must not already be channels-last"
        )

        with torch.no_grad():
            y = pipeline(x)

        assert isinstance(y, torch.Tensor)
        assert y.shape == x.shape
        assert not y.is_contiguous(memory_format=torch.channels_last), (
            "Output should NOT be channels-last when input was NCHW"
        )

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("spatial", [16, 64])
    def test_channels_last_various_shapes(self, batch_size: int, spatial: int) -> None:
        """channels-last preservation holds across batch sizes and spatial dims."""
        pipeline = _make_pipeline()
        x = torch.rand(batch_size, 3, spatial, spatial).contiguous(memory_format=torch.channels_last)

        with torch.no_grad():
            y = pipeline(x)

        assert y.is_contiguous(memory_format=torch.channels_last), (
            f"Failed for batch_size={batch_size}, spatial={spatial}"
        )

    def test_channels_last_values_unchanged_by_format_restore(self) -> None:
        """Restoring channels-last must not alter tensor values."""
        pipeline = _make_pipeline()
        torch.manual_seed(0)
        x_nchw = torch.rand(1, 3, 16, 16)
        x_cl = x_nchw.contiguous(memory_format=torch.channels_last)

        with torch.no_grad():
            y_nchw = pipeline(x_nchw, params=pipeline.forward_parameters(x_nchw.shape))
            y_cl = pipeline(x_cl, params=pipeline._params)

        # Values must be identical regardless of memory layout
        assert torch.allclose(y_nchw, y_cl.contiguous()), "Channels-last and NCHW runs must produce the same values"
