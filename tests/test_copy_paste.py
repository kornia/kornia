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
"""Tests for RandomCopyPaste augmentation."""

from __future__ import annotations

import pytest
import torch

from kornia.augmentation import RandomCopyPaste


# -----------------------------------------------------------------------
# 1. Construction
# -----------------------------------------------------------------------
class TestConstruction:
    def test_default(self) -> None:
        aug = RandomCopyPaste()
        assert aug.p == 0.5
        assert aug.scale_range == (0.5, 1.5)

    def test_custom_params(self) -> None:
        aug = RandomCopyPaste(scale_range=(0.8, 1.2), p=0.9, same_on_batch=True)
        assert aug.scale_range == (0.8, 1.2)
        assert aug.p == 0.9
        assert aug.same_on_batch is True

    def test_repr(self) -> None:
        aug = RandomCopyPaste(p=0.7)
        assert "RandomCopyPaste" in repr(aug)


# -----------------------------------------------------------------------
# 2. Image-only forward: shape preserved
# -----------------------------------------------------------------------
class TestImageOnlyForward:
    def test_output_shape(self) -> None:
        aug = RandomCopyPaste(p=1.0)
        x = torch.randn(4, 3, 64, 64)
        out = aug(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == x.shape

    def test_single_sample(self) -> None:
        """Batch-size 1 should not crash even though j==i."""
        aug = RandomCopyPaste(p=1.0)
        x = torch.randn(1, 3, 32, 32)
        out = aug(x)
        assert out.shape == x.shape

    def test_batch_size_2(self) -> None:
        aug = RandomCopyPaste(p=1.0)
        x = torch.randn(2, 3, 32, 32)
        out = aug(x)
        assert out.shape == x.shape


# -----------------------------------------------------------------------
# 3. Image + mask forward: both shapes preserved; output mask is union
# -----------------------------------------------------------------------
class TestImageMaskForward:
    def test_output_shapes_4d_mask(self) -> None:
        aug = RandomCopyPaste(p=1.0)
        x = torch.randn(4, 3, 64, 64)
        m = (torch.rand(4, 1, 64, 64) > 0.7).float()
        result = aug(x, masks=m)
        assert isinstance(result, list)
        assert len(result) == 2
        img_out, mask_out = result
        assert img_out.shape == x.shape
        assert mask_out.shape == m.shape

    def test_output_shapes_3d_mask(self) -> None:
        """3-D mask (B, H, W) should be returned as 3-D."""
        aug = RandomCopyPaste(p=1.0)
        x = torch.randn(4, 3, 64, 64)
        m = (torch.rand(4, 64, 64) > 0.7).float()
        result = aug(x, masks=m)
        assert isinstance(result, list)
        img_out, mask_out = result
        assert img_out.shape == x.shape
        assert mask_out.shape == m.shape

    def test_output_mask_union(self) -> None:
        """Where source mask is 1, output mask should be >= input target mask."""
        torch.manual_seed(0)
        aug = RandomCopyPaste(p=1.0, scale_range=(1.0, 1.0))
        x = torch.randn(4, 3, 32, 32)
        # target mask all zeros, source masks all ones → output mask should have some ones
        m = torch.zeros(4, 1, 32, 32)
        m[0] = 1.0  # source sample 0 has full mask
        result = aug(x, masks=m)
        img_out, mask_out = result  # type: ignore[misc]
        # At least one target should have gained non-zero mask values
        assert mask_out.sum() >= m.sum()


# -----------------------------------------------------------------------
# 4. p=0.0: identity
# -----------------------------------------------------------------------
class TestProbabilityZero:
    def test_image_identity(self) -> None:
        aug = RandomCopyPaste(p=0.0)
        x = torch.randn(4, 3, 32, 32)
        out = aug(x)
        assert torch.allclose(out, x)

    def test_mask_identity(self) -> None:
        aug = RandomCopyPaste(p=0.0)
        x = torch.randn(4, 3, 32, 32)
        m = (torch.rand(4, 1, 32, 32) > 0.5).float()
        result = aug(x, masks=m)
        img_out, mask_out = result  # type: ignore[misc]
        assert torch.allclose(img_out, x)
        assert torch.allclose(mask_out, m)


# -----------------------------------------------------------------------
# 5. p=1.0: result differs from input (for sufficiently large batch)
# -----------------------------------------------------------------------
class TestProbabilityOne:
    def test_image_changes(self) -> None:
        torch.manual_seed(42)
        aug = RandomCopyPaste(p=1.0, scale_range=(1.0, 1.0))
        # Use distinct images so pasting always changes pixels
        x = torch.zeros(4, 3, 32, 32)
        x[1] = 1.0
        x[2] = 0.5
        x[3] = 0.25
        # Full-coverage mask so every paste overwrites something
        m = torch.ones(4, 1, 32, 32)
        result = aug(x, masks=m)
        img_out, _ = result  # type: ignore[misc]
        assert not torch.allclose(img_out, x), "Output should differ from input when p=1 and mask covers full image."


# -----------------------------------------------------------------------
# 6. same_on_batch behavior
# -----------------------------------------------------------------------
class TestSameOnBatch:
    def test_same_on_batch_flag(self) -> None:
        """With same_on_batch=True the augmentation still runs without error."""
        aug = RandomCopyPaste(p=1.0, same_on_batch=True)
        x = torch.randn(4, 3, 16, 16)
        out = aug(x)
        assert out.shape == x.shape


# -----------------------------------------------------------------------
# 7. scale_range validation
# -----------------------------------------------------------------------
class TestScaleRangeValidation:
    def test_invalid_scale_negative(self) -> None:
        with pytest.raises(Exception):
            RandomCopyPaste(scale_range=(-0.1, 1.0))

    def test_invalid_scale_min_gt_max(self) -> None:
        with pytest.raises(Exception):
            RandomCopyPaste(scale_range=(1.5, 0.5))

    def test_valid_unit_scale(self) -> None:
        aug = RandomCopyPaste(scale_range=(1.0, 1.0))
        assert aug.scale_range == (1.0, 1.0)


# -----------------------------------------------------------------------
# 8. dtype preserved
# -----------------------------------------------------------------------
class TestDtypePreserved:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
    def test_image_dtype(self, dtype: torch.dtype) -> None:
        aug = RandomCopyPaste(p=1.0)
        x = torch.randn(2, 3, 32, 32).to(dtype)
        out = aug(x)
        assert out.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_mask_dtype(self, dtype: torch.dtype) -> None:
        aug = RandomCopyPaste(p=1.0)
        x = torch.randn(2, 3, 32, 32).to(dtype)
        m = (torch.rand(2, 1, 32, 32) > 0.5).to(dtype)
        result = aug(x, masks=m)
        img_out, mask_out = result  # type: ignore[misc]
        assert img_out.dtype == dtype
        assert mask_out.dtype == dtype


# -----------------------------------------------------------------------
# 9. Empty mask (all zeros): should be near-identity for the image
# -----------------------------------------------------------------------
class TestEmptyMask:
    def test_zero_mask_no_change(self) -> None:
        """When all source masks are zero there is nothing to paste."""
        torch.manual_seed(7)
        aug = RandomCopyPaste(p=1.0, scale_range=(1.0, 1.0))
        x = torch.randn(4, 3, 32, 32)
        m = torch.zeros(4, 1, 32, 32)
        result = aug(x, masks=m)
        img_out, mask_out = result  # type: ignore[misc]
        # With zero mask there's nothing non-zero to paste → image unchanged
        assert torch.allclose(img_out, x), "Zero-mask paste should leave image unchanged."
        assert torch.allclose(mask_out, m), "Zero-mask paste should leave mask unchanged."
