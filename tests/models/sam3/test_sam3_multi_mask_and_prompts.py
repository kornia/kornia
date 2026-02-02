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

"""Tests for SAM-3 Phase 3 (Complete Prompt & Mask Semantics) features."""

from __future__ import annotations

import torch

from kornia.models.sam3 import Sam3
from kornia.models.sam3.architecture.mask_decoder import MaskDecoder
from kornia.models.sam3.architecture.prompt_encoder import PromptEncoder


class TestPromptEncoderPhase3:
    """Test PromptEncoder Phase 3 features: box and mask encoding."""

    def test_box_prompt_encoding(self) -> None:
        """Test box prompt encoding produces corner embeddings."""
        embed_dim = 256
        batch_size = 2
        num_boxes = 3

        encoder = PromptEncoder(embed_dim=embed_dim)

        # Create dummy box prompts [x_min, y_min, x_max, y_max]
        boxes = torch.rand(batch_size, num_boxes, 4)

        sparse_emb, dense_emb = encoder(boxes=boxes)

        # Phase 3: boxes produce 2 embeddings per box (top-left + bottom-right corners)
        expected_sparse_size = num_boxes * 2
        assert sparse_emb.shape == (batch_size, expected_sparse_size, embed_dim), f"Got {sparse_emb.shape}"
        assert sparse_emb.dtype == boxes.dtype
        assert not torch.allclose(sparse_emb, torch.zeros_like(sparse_emb)), "Box embeddings should be non-zero"

    def test_mask_prompt_encoding(self) -> None:
        """Test mask prompt encoding produces semantic embeddings."""
        embed_dim = 256
        batch_size = 2
        mask_in_chans = 16
        img_size = 256

        encoder = PromptEncoder(
            embed_dim=embed_dim,
            input_image_size=img_size,
            mask_in_chans=mask_in_chans,
        )

        # Create dummy mask prompts
        masks = torch.rand(batch_size, 1, img_size, img_size)

        sparse_emb, dense_emb = encoder(masks=masks)

        # Phase 3: mask encoding produces 1 sparse embedding per batch
        assert sparse_emb.shape == (batch_size, 1, embed_dim), f"Got {sparse_emb.shape}"
        assert sparse_emb.dtype == masks.dtype
        assert not torch.allclose(sparse_emb, torch.zeros_like(sparse_emb)), "Mask embeddings should be non-zero"

    def test_combined_box_and_mask_prompts(self) -> None:
        """Test combining box and mask prompts in single forward pass."""
        embed_dim = 256
        batch_size = 2
        num_boxes = 2
        mask_in_chans = 16
        img_size = 256

        encoder = PromptEncoder(
            embed_dim=embed_dim,
            input_image_size=img_size,
            mask_in_chans=mask_in_chans,
        )

        # Create dummy prompts
        boxes = torch.rand(batch_size, num_boxes, 4)
        masks = torch.rand(batch_size, 1, img_size, img_size)

        sparse_emb, dense_emb = encoder(boxes=boxes, masks=masks)

        # Phase 3: combined box and mask prompts
        # boxes: num_boxes * 2 embeddings
        # mask: 1 embedding
        expected_sparse_size = num_boxes * 2 + 1
        assert sparse_emb.shape == (batch_size, expected_sparse_size, embed_dim), f"Got {sparse_emb.shape}"
        assert dense_emb.shape[0] == batch_size

    def test_all_prompt_types_combined(self) -> None:
        """Test combining points, boxes, and masks in single forward pass."""
        embed_dim = 256
        batch_size = 2
        num_points = 2
        num_boxes = 2
        mask_in_chans = 16
        img_size = 256

        encoder = PromptEncoder(
            embed_dim=embed_dim,
            input_image_size=img_size,
            mask_in_chans=mask_in_chans,
        )

        # Create all prompt types
        coords = torch.rand(batch_size, num_points, 2)
        labels = torch.randint(0, 2, (batch_size, num_points))
        boxes = torch.rand(batch_size, num_boxes, 4)
        masks = torch.rand(batch_size, 1, img_size, img_size)

        sparse_emb, dense_emb = encoder(points=(coords, labels), boxes=boxes, masks=masks)

        # Phase 3: all prompt types combined
        # points: num_points
        # boxes: num_boxes * 2
        # mask: 1
        expected_sparse_size = num_points + num_boxes * 2 + 1
        assert sparse_emb.shape == (batch_size, expected_sparse_size, embed_dim), f"Got {sparse_emb.shape}"


class TestMaskDecoderPhase3:
    """Test MaskDecoder Phase 3 features: multi-mask generation."""

    def test_multimask_output_true(self) -> None:
        """Test mask decoder generates multiple masks when multimask_output=True."""
        embed_dim = 256
        num_multimask_outputs = 3
        batch_size = 2
        num_patches = 1024
        num_prompts = 2

        decoder = MaskDecoder(embed_dim=embed_dim, num_multimask_outputs=num_multimask_outputs)

        image_embeddings = torch.randn(batch_size, num_patches, embed_dim)
        sparse_prompts = torch.randn(batch_size, num_prompts, embed_dim)
        dense_prompts = torch.randn(batch_size, embed_dim, 32, 32)

        masks, iou_pred = decoder(
            image_embeddings,
            sparse_prompts,
            dense_prompts,
            multimask_output=True,
        )

        # Phase 3: generate num_multimask_outputs masks
        assert masks.shape[1] == num_multimask_outputs, f"Expected {num_multimask_outputs} masks, got {masks.shape[1]}"
        assert masks.ndim == 4
        assert iou_pred.shape == (batch_size, num_multimask_outputs)

    def test_multimask_output_false(self) -> None:
        """Test mask decoder generates single mask when multimask_output=False."""
        embed_dim = 256
        num_multimask_outputs = 3
        batch_size = 2
        num_patches = 1024
        num_prompts = 2

        decoder = MaskDecoder(embed_dim=embed_dim, num_multimask_outputs=num_multimask_outputs)

        image_embeddings = torch.randn(batch_size, num_patches, embed_dim)
        sparse_prompts = torch.randn(batch_size, num_prompts, embed_dim)
        dense_prompts = torch.randn(batch_size, embed_dim, 32, 32)

        masks, iou_pred = decoder(
            image_embeddings,
            sparse_prompts,
            dense_prompts,
            multimask_output=False,
        )

        # Phase 3: generate single mask when multimask_output=False
        assert masks.shape[1] == 1, f"Expected 1 mask, got {masks.shape[1]}"
        assert masks.ndim == 4
        assert iou_pred.shape == (batch_size, num_multimask_outputs)

    def test_different_num_multimask_outputs(self) -> None:
        """Test mask decoder with different num_multimask_outputs values."""
        embed_dim = 256
        batch_size = 1
        num_patches = 1024

        for num_outputs in [1, 2, 3, 5]:
            decoder = MaskDecoder(embed_dim=embed_dim, num_multimask_outputs=num_outputs)

            image_embeddings = torch.randn(batch_size, num_patches, embed_dim)
            sparse_prompts = torch.randn(batch_size, 2, embed_dim)
            dense_prompts = torch.randn(batch_size, embed_dim, 32, 32)

            masks, iou_pred = decoder(
                image_embeddings,
                sparse_prompts,
                dense_prompts,
                multimask_output=True,
            )

            assert masks.shape[1] == num_outputs
            assert iou_pred.shape[1] == num_outputs

    def test_mask_tokens_are_learned(self) -> None:
        """Test that mask tokens are learnable parameters."""
        embed_dim = 256
        num_multimask_outputs = 3

        decoder = MaskDecoder(embed_dim=embed_dim, num_multimask_outputs=num_multimask_outputs)

        # Check mask tokens exist and are parameters
        assert len(decoder.mask_tokens) == num_multimask_outputs
        for mask_token in decoder.mask_tokens:
            assert isinstance(mask_token, torch.nn.Parameter)
            assert mask_token.shape == (1, 1, embed_dim)

    def test_hypernetwork_mlps_affect_output(self) -> None:
        """Test that hypernetwork MLPs modulate mask generation."""
        embed_dim = 256
        num_multimask_outputs = 3
        batch_size = 1
        num_patches = 1024

        decoder = MaskDecoder(embed_dim=embed_dim, num_multimask_outputs=num_multimask_outputs)

        image_embeddings = torch.randn(batch_size, num_patches, embed_dim)
        sparse_prompts = torch.randn(batch_size, 2, embed_dim)
        dense_prompts = torch.randn(batch_size, embed_dim, 32, 32)

        # Generate masks twice with same input
        masks1, _ = decoder(
            image_embeddings,
            sparse_prompts,
            dense_prompts,
            multimask_output=True,
        )

        masks2, _ = decoder(
            image_embeddings,
            sparse_prompts,
            dense_prompts,
            multimask_output=True,
        )

        # Masks should be identical (deterministic with fixed model and input)
        assert torch.allclose(masks1, masks2, atol=1e-5)

        # Different masks should exist for different mask indices
        # (mask generation varies based on mask_tokens and hypernetwork MLPs)
        assert not torch.allclose(masks1[:, 0], masks1[:, 1], atol=1e-3)


class TestSam3ModelSmoke:
    """Smoke tests for the complete Sam3 model."""

    def test_sam3_forward_with_points(self) -> None:
        """Test Sam3 forward pass with point prompts."""
        img_size = 256
        embed_dim = 128
        batch_size = 1

        model = Sam3(img_size=img_size, patch_size=16, embed_dim=embed_dim)

        images = torch.randn(batch_size, 3, img_size, img_size)
        coords = torch.rand(batch_size, 2, 2)
        labels = torch.tensor([[0, 1]])

        masks, iou_pred = model(images, points=(coords, labels))

        assert masks.ndim == 4
        assert masks.shape[0] == batch_size
        assert iou_pred.shape[0] == batch_size

    def test_sam3_forward_with_boxes(self) -> None:
        """Test Sam3 forward pass with box prompts."""
        img_size = 256
        embed_dim = 128
        batch_size = 1

        model = Sam3(img_size=img_size, patch_size=16, embed_dim=embed_dim)

        images = torch.randn(batch_size, 3, img_size, img_size)
        boxes = torch.tensor([[[0.1, 0.1, 0.5, 0.5]]])

        masks, iou_pred = model(images, boxes=boxes)

        assert masks.ndim == 4
        assert masks.shape[0] == batch_size

    def test_sam3_forward_with_masks(self) -> None:
        """Test Sam3 forward pass with mask prompts."""
        img_size = 256
        embed_dim = 128
        batch_size = 1

        model = Sam3(img_size=img_size, patch_size=16, embed_dim=embed_dim)

        images = torch.randn(batch_size, 3, img_size, img_size)
        mask_prompts = torch.rand(batch_size, 1, img_size, img_size) > 0.5

        masks, iou_pred = model(images, masks=mask_prompts.float())

        assert masks.ndim == 4
        assert masks.shape[0] == batch_size

    def test_sam3_multimask_control(self) -> None:
        """Test Sam3 respects multimask_output parameter."""
        img_size = 256
        embed_dim = 128
        num_multimask_outputs = 3
        batch_size = 1

        model = Sam3(
            img_size=img_size,
            patch_size=16,
            embed_dim=embed_dim,
            num_multimask_outputs=num_multimask_outputs,
        )

        images = torch.randn(batch_size, 3, img_size, img_size)
        coords = torch.rand(batch_size, 1, 2)
        labels = torch.tensor([[1]])

        # Multiple masks
        masks_multi, _ = model(images, points=(coords, labels), multimask_output=True)
        assert masks_multi.shape[1] == num_multimask_outputs

        # Single mask
        masks_single, _ = model(images, points=(coords, labels), multimask_output=False)
        assert masks_single.shape[1] == 1

    def test_sam3_combined_prompts(self) -> None:
        """Test Sam3 with combined prompt types."""
        img_size = 256
        embed_dim = 128
        batch_size = 1

        model = Sam3(img_size=img_size, patch_size=16, embed_dim=embed_dim)

        images = torch.randn(batch_size, 3, img_size, img_size)
        coords = torch.rand(batch_size, 1, 2)
        labels = torch.tensor([[1]])
        boxes = torch.tensor([[[0.2, 0.2, 0.8, 0.8]]])

        masks, iou_pred = model(images, points=(coords, labels), boxes=boxes)

        assert masks.ndim == 4
        assert masks.shape[0] == batch_size

    def test_sam3_no_prompts(self) -> None:
        """Test Sam3 with no prompts."""
        img_size = 256
        embed_dim = 128
        batch_size = 1

        model = Sam3(img_size=img_size, patch_size=16, embed_dim=embed_dim)

        images = torch.randn(batch_size, 3, img_size, img_size)

        masks, iou_pred = model(images)

        assert masks.ndim == 4
        assert masks.shape[0] == batch_size

    def test_sam3_batch_processing(self) -> None:
        """Test Sam3 with different batch sizes."""
        img_size = 256
        embed_dim = 128

        model = Sam3(img_size=img_size, patch_size=16, embed_dim=embed_dim)

        for batch_size in [1, 2, 4]:
            images = torch.randn(batch_size, 3, img_size, img_size)
            masks, iou_pred = model(images)

            assert masks.shape[0] == batch_size
            assert iou_pred.shape[0] == batch_size

    def test_sam3_gradient_flow(self) -> None:
        """Test that gradients flow through the entire model."""
        img_size = 256
        embed_dim = 128
        batch_size = 1

        model = Sam3(img_size=img_size, patch_size=16, embed_dim=embed_dim)

        images = torch.randn(batch_size, 3, img_size, img_size, requires_grad=True)
        coords = torch.rand(batch_size, 1, 2)
        labels = torch.tensor([[1]])

        masks, iou_pred = model(images, points=(coords, labels))

        # Compute simple loss and backprop
        loss = masks.sum() + iou_pred.sum()
        loss.backward()

        # Check that gradients exist for model and input
        assert images.grad is not None
        assert any(p.grad is not None for p in model.parameters())


__all__ = [
    "TestPromptEncoderPhase3",
    "TestMaskDecoderPhase3",
    "TestSam3ModelSmoke",
]
