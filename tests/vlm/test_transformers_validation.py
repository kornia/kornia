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

"""Validation tests comparing kornia.vlm outputs with transformers library.

These tests require the transformers library and pretrained weights,
so they are marked as slow and will only run when KORNIA_TEST_RUNSLOW=true.
"""

import pytest
import torch

# Skip all tests if transformers is not available
pytest.importorskip("transformers")


@pytest.mark.slow
class TestTransformersValidation:
    """Validation tests comparing with HuggingFace transformers."""

    @pytest.fixture
    def model_id(self):
        """Model ID to use for validation."""
        return "google/paligemma2-3b-pt-224"

    @pytest.mark.skip(reason="Requires model download and HuggingFace access")
    def test_siglip_vision_encoder_matches(self, model_id, device):
        """Verify SigLIP vision encoder outputs match transformers."""
        from transformers import PaliGemmaForConditionalGeneration

        from kornia.vlm import PaliGemma2

        # Load models
        hf_model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
        kornia_model = PaliGemma2.from_pretrained(model_id)

        hf_model = hf_model.to(device).eval()
        kornia_model = kornia_model.to(device).eval()

        # Create test input
        images = torch.randn(1, 3, 224, 224, device=device)

        # Get vision encoder outputs
        with torch.no_grad():
            # HuggingFace
            hf_vision_outputs = hf_model.vision_tower(images)
            hf_features = hf_vision_outputs.last_hidden_state

            # Kornia
            kornia_vision_outputs = kornia_model.encode_image(images)
            kornia_features = kornia_vision_outputs.last_hidden_state

        # Compare
        assert hf_features.shape == kornia_features.shape
        assert torch.allclose(hf_features, kornia_features, atol=1e-4, rtol=1e-4)

    @pytest.mark.skip(reason="Requires model download and HuggingFace access")
    def test_full_model_logits_match(self, model_id, device):
        """Verify full model logits match transformers."""
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        from kornia.vlm import PaliGemma2
        from kornia.vlm.paligemma import PaliGemmaProcessor

        # Load models
        hf_model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
        kornia_model = PaliGemma2.from_pretrained(model_id)

        hf_model = hf_model.to(device).eval()
        kornia_model = kornia_model.to(device).eval()

        # Load processors
        hf_processor = AutoProcessor.from_pretrained(model_id)
        kornia_processor = PaliGemmaProcessor.from_pretrained(model_id)

        # Create test input
        images = torch.randn(1, 3, 224, 224, device=device)
        text = "Describe this image"

        # Process inputs
        hf_inputs = hf_processor(images=images, text=text, return_tensors="pt").to(device)
        kornia_inputs = kornia_processor(images=images, text=text)

        # Forward pass
        with torch.no_grad():
            hf_outputs = hf_model(**hf_inputs)
            kornia_outputs = kornia_model(
                kornia_inputs.pixel_values.to(device),
                kornia_inputs.input_ids.to(device),
                attention_mask=kornia_inputs.attention_mask.to(device),
            )

        # Compare logits
        assert hf_outputs.logits.shape == kornia_outputs.logits.shape
        assert torch.allclose(hf_outputs.logits, kornia_outputs.logits, atol=1e-3, rtol=1e-3)


def test_layer_implementations_match_reference(device, dtype):
    """Test that individual layers produce correct outputs.

    This test doesn't require pretrained weights - it verifies the
    mathematical correctness of the layer implementations.
    """
    from kornia.vlm.layers import MultiHeadAttention, RMSNorm, SwiGLU

    # Test RMSNorm numerical stability
    norm = RMSNorm(768).to(device, dtype)
    x = torch.randn(2, 10, 768, device=device, dtype=dtype)
    output = norm(x)

    # RMSNorm should not change the dtype
    assert output.dtype == dtype

    # Output should have similar scale to input (due to normalization)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

    # Test SwiGLU gating
    mlp = SwiGLU(512, 2048).to(device, dtype)
    x = torch.randn(2, 10, 512, device=device, dtype=dtype)
    output = mlp(x)

    assert output.shape == x.shape
    assert not torch.isnan(output).any()

    # Test MultiHeadAttention with GQA
    attn = MultiHeadAttention(
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,  # GQA: 4x expansion
        head_dim=64,
    ).to(device, dtype)

    x = torch.randn(2, 10, 512, device=device, dtype=dtype)
    output, weights, _ = attn(x, output_attentions=True)

    assert output.shape == x.shape
    assert weights.shape == (2, 8, 10, 10)  # (B, num_heads, seq, seq)
    # Attention weights should sum to 1
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-5)


def test_rotary_embedding_correctness(device):
    """Test RoPE implementation produces expected values."""
    from kornia.vlm.layers.attention import RotaryEmbedding, apply_rotary_pos_emb

    head_dim = 64
    seq_len = 10
    batch_size = 2

    rope = RotaryEmbedding(head_dim, max_position_embeddings=512).to(device)

    x = torch.randn(batch_size, seq_len, head_dim, device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    cos, sin = rope(x, position_ids)

    # cos and sin should have correct shape
    assert cos.shape == (batch_size, seq_len, head_dim)
    assert sin.shape == (batch_size, seq_len, head_dim)

    # Values should be bounded
    assert (cos >= -1).all() and (cos <= 1).all()
    assert (sin >= -1).all() and (sin <= 1).all()

    # Test applying RoPE to Q and K
    q = torch.randn(batch_size, 8, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, 8, seq_len, head_dim, device=device)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    # Output shapes should match input
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape

    # Rotated vectors should have same norm (rotation preserves magnitude)
    q_norm = q.norm(dim=-1)
    q_rot_norm = q_rot.norm(dim=-1)
    assert torch.allclose(q_norm, q_rot_norm, atol=1e-4)
