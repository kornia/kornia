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

"""Tests for SAM-3 Phase 2 architecture modules."""

from __future__ import annotations

import torch

from kornia.models.sam3.architecture.mask_decoder import MaskDecoder
from kornia.models.sam3.architecture.prompt_encoder import PromptEncoder


class TestPromptEncoderPoints:
    """Test PromptEncoder with point prompts."""

    def test_prompt_encoder_with_points(self) -> None:
        """Test basic prompt encoding with point prompts."""
        embed_dim = 256
        batch_size = 2
        num_points = 5

        encoder = PromptEncoder(embed_dim=embed_dim)

        # Create dummy point prompts
        coords = torch.rand(batch_size, num_points, 2)  # (B, N, 2)
        labels = torch.randint(0, 2, (batch_size, num_points))  # (B, N) with 0 or 1

        sparse_emb, dense_emb = encoder(points=(coords, labels))

        # Check output shapes
        assert sparse_emb.shape == (batch_size, num_points, embed_dim), f"Got {sparse_emb.shape}"
        assert dense_emb.shape[0] == batch_size, f"Got {dense_emb.shape}"
        assert dense_emb.shape[1] == embed_dim, f"Got {dense_emb.shape}"

    def test_prompt_encoder_without_prompts(self) -> None:
        """Test prompt encoder with no prompts."""
        embed_dim = 256
        batch_size = 1

        encoder = PromptEncoder(embed_dim=embed_dim)

        sparse_emb, dense_emb = encoder()

        # Check output shapes
        assert sparse_emb.shape[0] == batch_size, f"Got {sparse_emb.shape}"
        assert sparse_emb.shape[2] == embed_dim, f"Got {sparse_emb.shape}"
        assert dense_emb.shape[0] == batch_size, f"Got {dense_emb.shape}"
        assert dense_emb.shape[1] == embed_dim, f"Got {dense_emb.shape}"

    def test_prompt_encoder_with_boxes(self) -> None:
        """Test prompt encoder with box prompts."""
        embed_dim = 256
        batch_size = 2
        num_boxes = 3

        encoder = PromptEncoder(embed_dim=embed_dim)

        # Create dummy box prompts
        boxes = torch.rand(batch_size, num_boxes, 4)

        sparse_emb, dense_emb = encoder(boxes=boxes)

        # Check output shapes
        assert sparse_emb.shape == (batch_size, num_boxes, embed_dim), f"Got {sparse_emb.shape}"
        assert dense_emb.shape[0] == batch_size, f"Got {dense_emb.shape}"

    def test_prompt_encoder_with_masks(self) -> None:
        """Test prompt encoder with mask prompts."""
        embed_dim = 256
        batch_size = 2
        mask_in_chans = 16

        encoder = PromptEncoder(embed_dim=embed_dim, input_image_size=256, mask_in_chans=mask_in_chans)

        # Create dummy mask prompts
        masks = torch.rand(batch_size, 1, 256, 256)

        sparse_emb, dense_emb = encoder(masks=masks)

        # Check output shapes
        assert sparse_emb.shape[0] == batch_size, f"Got {sparse_emb.shape}"
        assert sparse_emb.shape[2] == embed_dim, f"Got {sparse_emb.shape}"
        assert dense_emb.shape[0] == batch_size, f"Got {dense_emb.shape}"
        assert dense_emb.ndim == 4, f"Dense embedding should be 4D, got {dense_emb.ndim}D"
        assert dense_emb.shape[2] == 64, f"Got spatial size {dense_emb.shape[2]}"  # 256 // 4

    def test_prompt_encoder_with_combined_prompts(self) -> None:
        """Test prompt encoder with combined point and box prompts."""
        embed_dim = 256
        batch_size = 2
        num_points = 3
        num_boxes = 2

        encoder = PromptEncoder(embed_dim=embed_dim)

        # Create dummy prompts
        coords = torch.rand(batch_size, num_points, 2)
        labels = torch.randint(0, 2, (batch_size, num_points))
        boxes = torch.rand(batch_size, num_boxes, 4)

        sparse_emb, dense_emb = encoder(points=(coords, labels), boxes=boxes)

        # Check output shapes
        expected_num_sparse = num_points + num_boxes
        assert sparse_emb.shape == (batch_size, expected_num_sparse, embed_dim), f"Got {sparse_emb.shape}"
        assert dense_emb.shape[0] == batch_size, f"Got {dense_emb.shape}"


class TestMaskDecoderSmoke:
    """Smoke tests for MaskDecoder."""

    def test_mask_decoder_forward(self) -> None:
        """Test basic mask decoder forward pass."""
        embed_dim = 256
        batch_size = 2
        num_patches = 1024  # 32x32
        num_prompts = 5

        decoder = MaskDecoder(embed_dim=embed_dim)

        # Create dummy embeddings
        image_embeddings = torch.randn(batch_size, num_patches, embed_dim)
        sparse_prompts = torch.randn(batch_size, num_prompts, embed_dim)
        dense_prompts = torch.randn(batch_size, embed_dim, 32, 32)

        masks, iou_pred = decoder(
            image_embeddings,
            sparse_prompts,
            dense_prompts,
            multimask_output=True,
        )

        # Check output shapes
        assert masks.ndim == 4, f"Masks should be 4D, got {masks.ndim}D"
        assert masks.shape[0] == batch_size, f"Got {masks.shape}"
        assert iou_pred.shape == (batch_size, decoder.num_multimask_outputs), f"Got {iou_pred.shape}"

    def test_mask_decoder_single_mask_output(self) -> None:
        """Test mask decoder with single mask output."""
        embed_dim = 256
        batch_size = 1
        num_patches = 1024
        num_prompts = 3

        decoder = MaskDecoder(embed_dim=embed_dim)

        # Create dummy embeddings
        image_embeddings = torch.randn(batch_size, num_patches, embed_dim)
        sparse_prompts = torch.randn(batch_size, num_prompts, embed_dim)
        dense_prompts = torch.randn(batch_size, embed_dim, 32, 32)

        masks, iou_pred = decoder(
            image_embeddings,
            sparse_prompts,
            dense_prompts,
            multimask_output=False,
        )

        # Check output shapes
        assert masks.ndim == 4, f"Masks should be 4D, got {masks.ndim}D"
        assert masks.shape[1] == 1, f"Should have single mask output, got {masks.shape[1]}"
        assert iou_pred.shape == (batch_size, 1), f"Got {iou_pred.shape}"

    def test_mask_decoder_no_sparse_prompts(self) -> None:
        """Test mask decoder with no sparse prompts."""
        embed_dim = 256
        batch_size = 1
        num_patches = 1024

        decoder = MaskDecoder(embed_dim=embed_dim)

        # Create dummy embeddings with empty sparse prompts
        image_embeddings = torch.randn(batch_size, num_patches, embed_dim)
        sparse_prompts = torch.randn(batch_size, 0, embed_dim)
        dense_prompts = torch.randn(batch_size, embed_dim, 32, 32)

        masks, iou_pred = decoder(
            image_embeddings,
            sparse_prompts,
            dense_prompts,
        )

        # Check output shapes
        assert masks.ndim == 4, f"Masks should be 4D, got {masks.ndim}D"
        assert iou_pred.shape[0] == batch_size, f"Got {iou_pred.shape}"

    def test_mask_decoder_no_dense_prompts(self) -> None:
        """Test mask decoder with no dense prompts."""
        embed_dim = 256
        batch_size = 1
        num_patches = 1024
        num_prompts = 2

        decoder = MaskDecoder(embed_dim=embed_dim)

        # Create dummy embeddings with zero dense prompts
        image_embeddings = torch.randn(batch_size, num_patches, embed_dim)
        sparse_prompts = torch.randn(batch_size, num_prompts, embed_dim)
        dense_prompts = torch.zeros(batch_size, embed_dim, 32, 32)

        masks, iou_pred = decoder(
            image_embeddings,
            sparse_prompts,
            dense_prompts,
        )

        # Check output shapes
        assert masks.ndim == 4, f"Masks should be 4D, got {masks.ndim}D"
        assert iou_pred.shape == (batch_size, decoder.num_multimask_outputs), f"Got {iou_pred.shape}"

    def test_mask_decoder_batch_processing(self) -> None:
        """Test mask decoder with different batch sizes."""
        embed_dim = 256
        num_patches = 1024

        decoder = MaskDecoder(embed_dim=embed_dim)

        for batch_size in [1, 2, 4]:
            image_embeddings = torch.randn(batch_size, num_patches, embed_dim)
            sparse_prompts = torch.randn(batch_size, 3, embed_dim)
            dense_prompts = torch.randn(batch_size, embed_dim, 32, 32)

            masks, iou_pred = decoder(
                image_embeddings,
                sparse_prompts,
                dense_prompts,
            )

            assert masks.shape[0] == batch_size, f"Batch size mismatch: {masks.shape[0]} vs {batch_size}"
            assert iou_pred.shape[0] == batch_size, f"Batch size mismatch in IoU: {iou_pred.shape[0]} vs {batch_size}"


if __name__ == "__main__":
    # Run basic tests
    test_prompt = TestPromptEncoderPoints()
    test_prompt.test_prompt_encoder_with_points()
    test_prompt.test_prompt_encoder_without_prompts()
    test_prompt.test_prompt_encoder_with_boxes()
    test_prompt.test_prompt_encoder_with_masks()
    test_prompt.test_prompt_encoder_with_combined_prompts()

    test_decoder = TestMaskDecoderSmoke()
    test_decoder.test_mask_decoder_forward()
    test_decoder.test_mask_decoder_single_mask_output()
    test_decoder.test_mask_decoder_no_sparse_prompts()
    test_decoder.test_mask_decoder_no_dense_prompts()
    test_decoder.test_mask_decoder_batch_processing()

    print("All tests passed!")
