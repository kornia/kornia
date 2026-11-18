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

from __future__ import annotations

import pytest
import torch

from kornia.core.exceptions import BaseError
from kornia.models.structures import Prompts, SegmentationResults


class TestSegmentationResults:
    def _make_results(self, B=2, C=3, H=16, W=16, threshold=0.0):
        logits = torch.randn(B, C, H, W)
        scores = torch.rand(B, C)
        return SegmentationResults(logits=logits, scores=scores, mask_threshold=threshold)

    def test_binary_masks_uses_original_res_logits_when_set(self):
        r = self._make_results()
        # Simulate having called original_res_logits()
        fake_hires = torch.randn(2, 3, 32, 32) + 10.0  # all positive -> all True
        r._original_res_logits = fake_hires
        masks = r.binary_masks
        assert masks.shape == (2, 3, 32, 32)
        assert masks.all()

    def test_original_res_logits_without_encoder_resize(self):
        r = self._make_results(B=1, C=1, H=8, W=8)
        # No encoder resize (image_size_encoder=None), just crop and resize
        result = r.original_res_logits(input_size=(8, 8), original_size=(32, 32), image_size_encoder=None)
        assert result.shape == (1, 1, 32, 32)
        assert r._original_res_logits is not None

    def test_original_res_logits_with_encoder_resize(self):
        r = self._make_results(B=1, C=1, H=8, W=8)
        # With encoder resize: first resize to (16, 16), then crop, then resize to (32, 32)
        result = r.original_res_logits(input_size=(16, 16), original_size=(32, 32), image_size_encoder=(16, 16))
        assert result.shape == (1, 1, 32, 32)

    def test_original_res_logits_crops_padding(self):
        # 10 x 10 logits whose valid 8 x 6 top-left region is the plane 10 * row + col; the bottom and right padding
        # holds 1000, which any resize of the uncropped logits to 4 x 3 reads.
        logits = torch.full((1, 1, 10, 10), 1000.0)
        rows, cols = torch.meshgrid(torch.arange(8.0), torch.arange(6.0), indexing="ij")
        logits[0, 0, :8, :6] = 10 * rows + cols
        r = SegmentationResults(logits=logits, scores=torch.ones(1, 1))

        result = r.original_res_logits(input_size=(8, 6), original_size=(4, 3), image_size_encoder=None)

        # Halving 8 x 6 bilinearly without corner alignment samples the plane at rows and cols 2 * i + 0.5, where a
        # plane interpolates exactly: 10 * (2 * i + 0.5) + (2 * j + 0.5).
        i, j = torch.meshgrid(torch.arange(4.0), torch.arange(3.0), indexing="ij")
        expected = (10 * (2 * i + 0.5) + (2 * j + 0.5))[None, None]
        torch.testing.assert_close(result, expected)

    def test_squeeze_without_original_res_logits(self):
        r = self._make_results(B=1, C=3, H=8, W=8)
        squeezed = r.squeeze(dim=0)
        assert squeezed.logits.shape == (3, 8, 8)
        assert squeezed.scores.shape == (3,)

    def test_squeeze_with_original_res_logits(self):
        r = self._make_results(B=1, C=3, H=8, W=8)
        r._original_res_logits = torch.randn(1, 3, 32, 32)
        squeezed = r.squeeze(dim=0)
        assert squeezed.logits.shape == (3, 8, 8)
        assert isinstance(squeezed._original_res_logits, torch.Tensor)
        assert squeezed._original_res_logits.shape == (3, 32, 32)

    def test_binary_masks_threshold(self):
        logits = torch.tensor([[[[0.5, -0.5], [0.3, 0.1]]]])
        scores = torch.ones(1, 1)
        r = SegmentationResults(logits=logits, scores=scores, mask_threshold=0.2)
        masks = r.binary_masks
        # 0.5 > 0.2 -> True, -0.5 > 0.2 -> False, 0.3 > 0.2 -> True, 0.1 > 0.2 -> False
        expected = torch.tensor([[[[True, False], [True, False]]]])
        assert masks.dtype == torch.bool
        assert torch.equal(masks, expected)


class TestPrompts:
    def test_no_prompts(self):
        p = Prompts()
        assert p.points is None
        assert p.boxes is None
        assert p.masks is None
        assert p.keypoints is None
        assert p.keypoints_labels is None

    def test_keypoints_from_tuple(self):
        coords = torch.rand(2, 5, 2)
        labels = torch.randint(0, 2, (2, 5)).float()
        p = Prompts(points=(coords, labels))
        assert torch.equal(p.keypoints, coords)
        assert torch.equal(p.keypoints_labels, labels)

    def test_keypoints_and_boxes_matching_batch(self):
        coords = torch.rand(3, 5, 2)
        labels = torch.rand(3, 5)
        boxes = torch.rand(3, 4)
        # Should not raise: batch sizes match
        p = Prompts(points=(coords, labels), boxes=boxes)
        assert p.keypoints.shape[0] == 3
        assert p.boxes.shape[0] == 3

    def test_keypoints_and_boxes_mismatched_batch_raises(self):
        coords = torch.rand(2, 5, 2)
        labels = torch.rand(2, 5)
        boxes = torch.rand(3, 4)  # different batch size
        with pytest.raises(BaseError, match="same batch size"):
            Prompts(points=(coords, labels), boxes=boxes)
