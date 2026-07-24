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

"""Behavioral validation tests for SAM-3 MaskDecoder and PromptEncoder.

These tests go beyond shape checks to verify that components behave correctly:
- Dense prompts influence mask outputs
- Multi-mask outputs are diverse across indices
- Both sparse and dense paths independently affect the output
- Gradients flow through the dense prompt modulation path
- mask_tokens is stored as nn.Embedding with correct shape and learned weights
"""

from __future__ import annotations

import torch

from kornia.models.sam3.architecture.mask_decoder import MaskDecoder
from kornia.models.sam3.architecture.prompt_encoder import PromptEncoder


class TestMaskDecoderBehavior:
    """Behavioral tests for MaskDecoder verifying functional correctness."""

    def _make_decoder(self, embed_dim: int = 64, num_multimask_outputs: int = 3) -> MaskDecoder:
        """Return a small MaskDecoder for fast tests.

        Uses embed_dim=64 so num_patches=64 forms an 8x8 grid.
        """
        return MaskDecoder(embed_dim=embed_dim, num_multimask_outputs=num_multimask_outputs)

    # ------------------------------------------------------------------
    # Dense prompt influence
    # ------------------------------------------------------------------

    def test_dense_prompt_influences_output(self) -> None:
        """Non-zero dense prompts must produce different masks than zero dense prompts.

        Validates that dense_to_feature_modulation is exercised and genuinely
        changes mask predictions compared to a zero dense input.
        """
        torch.manual_seed(0)
        embed_dim = 64
        batch_size = 1
        num_patches = 64  # 8x8 grid

        decoder = self._make_decoder(embed_dim=embed_dim)
        decoder.eval()

        image_embeddings = torch.randn(batch_size, num_patches, embed_dim)
        sparse_prompts = torch.randn(batch_size, 2, embed_dim)
        dense_zeros = torch.zeros(batch_size, embed_dim, 8, 8)
        dense_nonzero = torch.randn(batch_size, embed_dim, 8, 8)

        with torch.no_grad():
            masks_no_dense, _ = decoder(image_embeddings, sparse_prompts, dense_zeros)
            masks_with_dense, _ = decoder(image_embeddings, sparse_prompts, dense_nonzero)

        diff = (masks_no_dense - masks_with_dense).abs().mean().item()
        assert diff > 1e-4, f"Dense prompt should influence mask output, but mean absolute difference={diff:.6f}"

    # ------------------------------------------------------------------
    # Multi-mask diversity
    # ------------------------------------------------------------------

    def test_multimask_output_diversity(self) -> None:
        """Every pair of mask indices must produce meaningfully different masks.

        Ensures that distinct mask tokens and hypernetwork MLPs lead to diverse
        outputs rather than collapsing to identical predictions.
        """
        torch.manual_seed(42)
        embed_dim = 64
        num_multimask_outputs = 3
        batch_size = 1
        num_patches = 64

        decoder = self._make_decoder(embed_dim=embed_dim, num_multimask_outputs=num_multimask_outputs)
        decoder.eval()

        image_embeddings = torch.randn(batch_size, num_patches, embed_dim)
        sparse_prompts = torch.randn(batch_size, 2, embed_dim)
        dense_prompts = torch.randn(batch_size, embed_dim, 8, 8)

        with torch.no_grad():
            masks, _ = decoder(image_embeddings, sparse_prompts, dense_prompts, multimask_output=True)

        assert masks.shape == (batch_size, num_multimask_outputs, 32, 32), f"Unexpected shape: {masks.shape}"

        for i in range(num_multimask_outputs):
            for j in range(i + 1, num_multimask_outputs):
                diff = (masks[:, i] - masks[:, j]).abs().mean().item()
                assert diff > 1e-4, (
                    f"Masks at indices {i} and {j} are too similar (mean abs diff={diff:.6f}). "
                    "Distinct mask tokens and hypernetwork MLPs should produce diverse outputs."
                )

    # ------------------------------------------------------------------
    # Sparse vs dense independence
    # ------------------------------------------------------------------

    def test_sparse_and_dense_both_influence_output(self) -> None:
        """Sparse and dense embeddings must each independently affect mask output.

        Runs three forward passes:
        1. Both sparse and dense nonzero (baseline).
        2. Only dense nonzero (sparse replaced with empty tensor).
        3. Only sparse nonzero (dense replaced with zeros).

        The baseline must differ from both ablations, confirming each path
        contributes a unique signal.
        """
        torch.manual_seed(7)
        embed_dim = 64
        batch_size = 1
        num_patches = 64

        decoder = self._make_decoder(embed_dim=embed_dim)
        decoder.eval()

        image_embeddings = torch.randn(batch_size, num_patches, embed_dim)
        sparse_nonzero = torch.randn(batch_size, 2, embed_dim)
        dense_nonzero = torch.randn(batch_size, embed_dim, 8, 8)
        sparse_empty = torch.zeros(batch_size, 0, embed_dim)  # no sparse tokens
        dense_zero = torch.zeros(batch_size, embed_dim, 8, 8)

        with torch.no_grad():
            masks_both, _ = decoder(image_embeddings, sparse_nonzero, dense_nonzero)
            masks_only_dense, _ = decoder(image_embeddings, sparse_empty, dense_nonzero)
            masks_only_sparse, _ = decoder(image_embeddings, sparse_nonzero, dense_zero)

        diff_sparse = (masks_both - masks_only_dense).abs().mean().item()
        diff_dense = (masks_both - masks_only_sparse).abs().mean().item()

        assert diff_sparse > 1e-4, (
            f"Sparse prompts should influence output, but diff with/without sparse={diff_sparse:.6f}"
        )
        assert diff_dense > 1e-4, f"Dense prompts should influence output, but diff with/without dense={diff_dense:.6f}"

    # ------------------------------------------------------------------
    # Gradient flow through dense prompt path
    # ------------------------------------------------------------------

    def test_gradient_flow_through_dense_prompt_path(self) -> None:
        """Gradients must propagate through dense_to_feature_modulation.

        Checks that:
        - dense_prompts tensor itself receives a nonzero gradient.
        - dense_to_feature_modulation Conv2d weight receives a nonzero gradient.
        This confirms the dense prompt path is wired into the computation graph.
        """
        torch.manual_seed(0)
        embed_dim = 64
        batch_size = 1
        num_patches = 64

        decoder = self._make_decoder(embed_dim=embed_dim)
        decoder.train()

        image_embeddings = torch.randn(batch_size, num_patches, embed_dim)
        sparse_prompts = torch.randn(batch_size, 2, embed_dim)
        dense_prompts = torch.randn(batch_size, embed_dim, 8, 8, requires_grad=True)

        masks, iou_pred = decoder(image_embeddings, sparse_prompts, dense_prompts)
        loss = masks.sum() + iou_pred.sum()
        loss.backward()

        assert dense_prompts.grad is not None, "dense_prompts should receive gradients"
        assert dense_prompts.grad.abs().sum().item() > 0, "dense_prompts gradients should be nonzero"

        conv = decoder.dense_to_feature_modulation
        assert conv.weight.grad is not None, "dense_to_feature_modulation.weight should receive gradients"
        assert conv.weight.grad.abs().sum().item() > 0, "dense_to_feature_modulation.weight gradients should be nonzero"

    # ------------------------------------------------------------------
    # Mask token embedding type and structure
    # ------------------------------------------------------------------

    def test_mask_tokens_are_embedding(self) -> None:
        """mask_tokens must be nn.Embedding with shape (num_multimask_outputs, embed_dim)."""
        embed_dim = 64
        num_multimask_outputs = 3

        decoder = self._make_decoder(embed_dim=embed_dim, num_multimask_outputs=num_multimask_outputs)

        assert isinstance(decoder.mask_tokens, torch.nn.Embedding), (
            f"mask_tokens should be nn.Embedding, got {type(decoder.mask_tokens)}"
        )
        assert decoder.mask_tokens.weight.shape == (num_multimask_outputs, embed_dim), (
            f"Expected shape ({num_multimask_outputs}, {embed_dim}), got {decoder.mask_tokens.weight.shape}"
        )
        assert decoder.mask_tokens.weight.requires_grad, "mask_tokens.weight should be a learnable parameter"

    def test_mask_token_embedding_gradient_flow(self) -> None:
        """Gradients must flow through each row of mask_tokens.weight."""
        torch.manual_seed(0)
        embed_dim = 64
        num_multimask_outputs = 3
        batch_size = 1
        num_patches = 64

        decoder = self._make_decoder(embed_dim=embed_dim, num_multimask_outputs=num_multimask_outputs)
        decoder.train()

        image_embeddings = torch.randn(batch_size, num_patches, embed_dim)
        sparse_prompts = torch.randn(batch_size, 2, embed_dim)
        dense_prompts = torch.randn(batch_size, embed_dim, 8, 8)

        masks, iou_pred = decoder(image_embeddings, sparse_prompts, dense_prompts, multimask_output=True)
        loss = masks.sum() + iou_pred.sum()
        loss.backward()

        assert decoder.mask_tokens.weight.grad is not None, "mask_tokens.weight should receive gradients"
        assert decoder.mask_tokens.weight.grad.abs().sum().item() > 0, "mask_tokens.weight gradients should be nonzero"

    # ------------------------------------------------------------------
    # Output shape consistency
    # ------------------------------------------------------------------

    def test_multimask_vs_single_mask_output_shapes(self) -> None:
        """multimask_output flag must correctly control output tensor dimensions."""
        torch.manual_seed(0)
        embed_dim = 64
        num_multimask_outputs = 3
        batch_size = 2
        num_patches = 64

        decoder = self._make_decoder(embed_dim=embed_dim, num_multimask_outputs=num_multimask_outputs)
        decoder.eval()

        image_embeddings = torch.randn(batch_size, num_patches, embed_dim)
        sparse_prompts = torch.randn(batch_size, 2, embed_dim)
        dense_prompts = torch.randn(batch_size, embed_dim, 8, 8)

        with torch.no_grad():
            masks_multi, iou_multi = decoder(image_embeddings, sparse_prompts, dense_prompts, multimask_output=True)
            masks_single, iou_single = decoder(image_embeddings, sparse_prompts, dense_prompts, multimask_output=False)

        assert masks_multi.shape == (batch_size, num_multimask_outputs, 32, 32), (
            f"Multi-mask shape mismatch: {masks_multi.shape}"
        )
        assert iou_multi.shape == (batch_size, num_multimask_outputs), (
            f"Multi-mask IoU shape mismatch: {iou_multi.shape}"
        )
        assert masks_single.shape == (batch_size, 1, 32, 32), f"Single mask shape mismatch: {masks_single.shape}"
        assert iou_single.shape == (batch_size, 1), f"Single mask IoU shape mismatch: {iou_single.shape}"


class TestPromptEncoderBehavior:
    """Behavioral tests for PromptEncoder."""

    def test_mask_prompt_affects_dense_embedding(self) -> None:
        """A mask prompt must produce a different dense embedding than no mask.

        Confirms the mask_downscaling → mask_proj → interpolate path is
        functional and distinct from the no_mask_embed broadcast path.
        """
        torch.manual_seed(0)
        embed_dim = 64
        img_size = 64
        batch_size = 1

        encoder = PromptEncoder(embed_dim=embed_dim, input_image_size=img_size)
        encoder.eval()

        mask = torch.rand(batch_size, 1, img_size, img_size)

        with torch.no_grad():
            _, dense_with_mask = encoder(masks=mask)
            _, dense_no_mask = encoder()

        diff = (dense_with_mask - dense_no_mask).abs().mean().item()
        assert diff > 1e-4, f"Mask prompt should change dense embedding, but mean abs diff={diff:.6f}"

    def test_box_and_mask_prompt_sparse_shape(self) -> None:
        """Combined box + mask prompt must produce the expected sparse token count."""
        torch.manual_seed(0)
        embed_dim = 64
        img_size = 64
        batch_size = 2
        num_boxes = 2

        encoder = PromptEncoder(embed_dim=embed_dim, input_image_size=img_size)
        encoder.eval()

        boxes = torch.rand(batch_size, num_boxes, 4)
        mask = torch.rand(batch_size, 1, img_size, img_size)

        with torch.no_grad():
            sparse_emb, dense_emb = encoder(boxes=boxes, masks=mask)

        # boxes → num_boxes * 2 corner tokens; mask → 1 semantic token
        expected_sparse = num_boxes * 2 + 1
        assert sparse_emb.shape == (batch_size, expected_sparse, embed_dim), (
            f"Expected ({batch_size}, {expected_sparse}, {embed_dim}), got {sparse_emb.shape}"
        )
        assert dense_emb.shape[0] == batch_size


__all__ = ["TestMaskDecoderBehavior", "TestPromptEncoderBehavior"]
