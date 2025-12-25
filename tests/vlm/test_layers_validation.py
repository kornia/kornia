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

"""Validation tests comparing kornia.vlm layers with transformers implementations.

These tests verify that our from-scratch implementations produce identical
outputs to the official transformers library implementations.
"""

import pytest
import torch

# Skip tests if transformers is not available
transformers = pytest.importorskip("transformers")


class TestGemmaRMSNormValidation:
    """Validate GemmaRMSNorm against transformers GemmaRMSNorm."""

    def test_gemma_rmsnorm_matches_transformers(self, device, dtype):
        """Test that our GemmaRMSNorm matches transformers GemmaRMSNorm."""
        from transformers.models.gemma.modeling_gemma import GemmaRMSNorm as HFGemmaRMSNorm

        from kornia.vlm.layers import GemmaRMSNorm

        hidden_size = 768
        eps = 1e-6

        # Create both implementations with eval mode
        kornia_norm = GemmaRMSNorm(hidden_size, eps=eps).to(device, dtype).eval()
        hf_norm = HFGemmaRMSNorm(hidden_size, eps=eps).to(device, dtype).eval()

        # Copy weights from HF to kornia
        with torch.no_grad():
            kornia_norm.weight.copy_(hf_norm.weight)

        # Test input
        x = torch.randn(2, 10, hidden_size, device=device, dtype=dtype)

        # Forward pass
        with torch.no_grad():
            kornia_output = kornia_norm(x)
            hf_output = hf_norm(x)

        # Compare
        assert kornia_output.shape == hf_output.shape
        assert torch.allclose(kornia_output, hf_output, atol=1e-5, rtol=1e-5), (
            f"Max diff: {(kornia_output - hf_output).abs().max().item()}"
        )

    def test_gemma_rmsnorm_gradient_matches(self, device):
        """Test that gradients match between implementations."""
        from transformers.models.gemma.modeling_gemma import GemmaRMSNorm as HFGemmaRMSNorm

        from kornia.vlm.layers import GemmaRMSNorm

        hidden_size = 256
        dtype = torch.float32  # Use float32 for gradient comparison

        kornia_norm = GemmaRMSNorm(hidden_size).to(device, dtype)
        hf_norm = HFGemmaRMSNorm(hidden_size).to(device, dtype)

        # Copy weights from HF to kornia
        with torch.no_grad():
            kornia_norm.weight.copy_(hf_norm.weight)

        # Test input (needs grad) - use same random seed
        torch.manual_seed(42)
        x_kornia = torch.randn(2, 10, hidden_size, device=device, dtype=dtype, requires_grad=True)
        torch.manual_seed(42)
        x_hf = torch.randn(2, 10, hidden_size, device=device, dtype=dtype, requires_grad=True)

        # Forward + backward
        kornia_output = kornia_norm(x_kornia)
        hf_output = hf_norm(x_hf)

        kornia_output.sum().backward()
        hf_output.sum().backward()

        # Compare gradients
        assert torch.allclose(x_kornia.grad, x_hf.grad, atol=1e-5, rtol=1e-5), (
            f"Gradient max diff: {(x_kornia.grad - x_hf.grad).abs().max().item()}"
        )


class TestRMSNormValidation:
    """Validate RMSNorm (LLaMA-style) implementation."""

    def test_rmsnorm_basic_properties(self, device, dtype):
        """Test basic RMSNorm properties."""
        from kornia.vlm.layers import RMSNorm

        hidden_size = 768
        norm = RMSNorm(hidden_size).to(device, dtype).eval()

        # Weight should be initialized to ones
        assert torch.allclose(norm.weight, torch.ones(hidden_size, device=device, dtype=dtype))

        # Test forward pass
        x = torch.randn(2, 10, hidden_size, device=device, dtype=dtype)
        with torch.no_grad():
            output = norm(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestSwiGLUValidation:
    """Validate SwiGLU against transformers GemmaMLP with silu activation."""

    def test_swiglu_matches_transformers(self, device, dtype):
        """Test that our SwiGLU matches transformers GemmaMLP with silu activation.

        Note: HF Gemma defaults to gelu_pytorch_tanh, so we explicitly set silu
        to match our SwiGLU implementation.
        """
        from transformers import GemmaConfig as HFGemmaConfig
        from transformers.models.gemma.modeling_gemma import GemmaMLP

        from kornia.vlm.layers import SwiGLU

        hidden_size = 512
        intermediate_size = 2048

        # Create HF config with silu activation to match our SwiGLU
        hf_config = HFGemmaConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act="silu",  # Match our SwiGLU implementation
        )
        hf_mlp = GemmaMLP(hf_config).to(device, dtype).eval()

        # Create our implementation
        kornia_mlp = SwiGLU(hidden_size, intermediate_size, bias=False).to(device, dtype).eval()

        # Copy weights from HF to kornia
        with torch.no_grad():
            kornia_mlp.gate_proj.weight.copy_(hf_mlp.gate_proj.weight)
            kornia_mlp.up_proj.weight.copy_(hf_mlp.up_proj.weight)
            kornia_mlp.down_proj.weight.copy_(hf_mlp.down_proj.weight)

        # Test input
        x = torch.randn(2, 10, hidden_size, device=device, dtype=dtype)

        # Forward pass
        with torch.no_grad():
            kornia_output = kornia_mlp(x)
            hf_output = hf_mlp(x)

        # Compare
        assert kornia_output.shape == hf_output.shape
        assert torch.allclose(kornia_output, hf_output, atol=1e-4, rtol=1e-4), (
            f"Max diff: {(kornia_output - hf_output).abs().max().item()}"
        )


class TestGeLUMLPValidation:
    """Validate GeLUMLP against transformers SigLIP MLP."""

    def test_gelu_mlp_matches_transformers(self, device, dtype):
        """Test that our GeLUMLP matches transformers SiglipMLP."""
        from transformers import SiglipVisionConfig
        from transformers.models.siglip.modeling_siglip import SiglipMLP

        from kornia.vlm.layers import GeLUMLP

        hidden_size = 384
        intermediate_size = 1536

        # Create HF config and model
        hf_config = SiglipVisionConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        hf_mlp = SiglipMLP(hf_config).to(device, dtype).eval()

        # Create our implementation
        kornia_mlp = GeLUMLP(hidden_size, intermediate_size, bias=True).to(device, dtype).eval()

        # Copy weights from HF to kornia
        with torch.no_grad():
            kornia_mlp.fc1.weight.copy_(hf_mlp.fc1.weight)
            kornia_mlp.fc1.bias.copy_(hf_mlp.fc1.bias)
            kornia_mlp.fc2.weight.copy_(hf_mlp.fc2.weight)
            kornia_mlp.fc2.bias.copy_(hf_mlp.fc2.bias)

        # Test input
        x = torch.randn(2, 10, hidden_size, device=device, dtype=dtype)

        # Forward pass
        with torch.no_grad():
            kornia_output = kornia_mlp(x)
            hf_output = hf_mlp(x)

        # Compare
        assert kornia_output.shape == hf_output.shape
        assert torch.allclose(kornia_output, hf_output, atol=1e-4, rtol=1e-4), (
            f"Max diff: {(kornia_output - hf_output).abs().max().item()}"
        )


class TestLayerNormValidation:
    """Validate LayerNorm against PyTorch and transformers."""

    def test_layernorm_matches_pytorch(self, device, dtype):
        """Test that our LayerNorm matches PyTorch nn.LayerNorm."""
        from kornia.vlm.layers import LayerNorm

        hidden_size = 768
        eps = 1e-6

        # Create both implementations
        kornia_norm = LayerNorm(hidden_size, eps=eps, bias=True).to(device, dtype).eval()
        torch_norm = torch.nn.LayerNorm(hidden_size, eps=eps).to(device, dtype).eval()

        # Copy weights from torch to kornia
        with torch.no_grad():
            kornia_norm.weight.copy_(torch_norm.weight)
            kornia_norm.bias.copy_(torch_norm.bias)

        # Test input
        x = torch.randn(2, 10, hidden_size, device=device, dtype=dtype)

        # Forward pass
        with torch.no_grad():
            kornia_output = kornia_norm(x)
            torch_output = torch_norm(x)

        # Compare
        assert kornia_output.shape == torch_output.shape
        assert torch.allclose(kornia_output, torch_output, atol=1e-5, rtol=1e-5), (
            f"Max diff: {(kornia_output - torch_output).abs().max().item()}"
        )


class TestPatchEmbeddingValidation:
    """Validate PatchEmbedding against transformers SiglipVisionEmbeddings."""

    def test_patch_embedding_matches_transformers(self, device, dtype):
        """Test that our PatchEmbedding matches transformers."""
        from transformers import SiglipVisionConfig
        from transformers.models.siglip.modeling_siglip import SiglipVisionEmbeddings

        from kornia.vlm.layers import PatchEmbedding

        image_size = 224
        patch_size = 14
        hidden_size = 1152
        num_channels = 3

        # Create HF config and embeddings
        hf_config = SiglipVisionConfig(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_channels=num_channels,
        )
        hf_embed = SiglipVisionEmbeddings(hf_config).to(device, dtype).eval()

        # Create our implementation
        kornia_embed = (
            PatchEmbedding(
                image_size=image_size,
                patch_size=patch_size,
                num_channels=num_channels,
                embed_dim=hidden_size,
            )
            .to(device, dtype)
            .eval()
        )

        # Copy weights (patch projection)
        with torch.no_grad():
            kornia_embed.proj.weight.copy_(hf_embed.patch_embedding.weight)
            kornia_embed.proj.bias.copy_(hf_embed.patch_embedding.bias)

        # Test input
        images = torch.randn(2, num_channels, image_size, image_size, device=device, dtype=dtype)

        # Forward pass - just patch embedding without position
        with torch.no_grad():
            kornia_output = kornia_embed(images)
            hf_output = hf_embed.patch_embedding(images).flatten(2).transpose(1, 2)

        # Compare
        assert kornia_output.shape == hf_output.shape, f"Shape mismatch: {kornia_output.shape} vs {hf_output.shape}"
        assert torch.allclose(kornia_output, hf_output, atol=1e-4, rtol=1e-4), (
            f"Max diff: {(kornia_output - hf_output).abs().max().item()}"
        )


class TestRotaryEmbeddingValidation:
    """Validate RotaryEmbedding computation."""

    def test_rotary_embedding_values(self, device):
        """Test that RoPE produces expected values."""
        from kornia.vlm.layers.attention import RotaryEmbedding

        head_dim = 64
        max_position = 2048
        base = 10000.0

        rope = RotaryEmbedding(head_dim, max_position, base).to(device)

        # Check that cos/sin are cached
        seq_len = 10
        assert rope.cos_cached.shape[0] >= seq_len
        assert rope.sin_cached.shape[0] >= seq_len

        # cos and sin should be in valid range [-1, 1]
        assert rope.cos_cached.abs().max() <= 1.0 + 1e-6
        assert rope.sin_cached.abs().max() <= 1.0 + 1e-6

        # Test with larger sequence (should re-cache)
        seq_len = 3000
        x = torch.randn(1, seq_len, head_dim, device=device)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        cos, sin = rope(x, position_ids)
        assert cos.shape[1] >= seq_len


class TestMultiHeadAttentionValidation:
    """Validate MultiHeadAttention components."""

    def test_attention_output_shape(self, device, dtype):
        """Test that attention produces correct output shapes."""
        from kornia.vlm.layers import MultiHeadAttention

        hidden_size = 512
        num_heads = 8
        num_kv_heads = 2
        head_dim = 64
        batch_size = 2
        seq_len = 10

        attn = (
            MultiHeadAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                head_dim=head_dim,
                use_rotary=True,
            )
            .to(device, dtype)
            .eval()
        )

        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

        with torch.no_grad():
            output, weights, cache = attn(x, output_attentions=True, use_cache=True)

        assert output.shape == (batch_size, seq_len, hidden_size)
        assert weights.shape == (batch_size, num_heads, seq_len, seq_len)
        assert cache is not None
        assert len(cache) == 2  # key and value

    def test_grouped_query_attention_expansion(self, device, dtype):
        """Test that GQA correctly expands key-value heads."""
        from kornia.vlm.layers import MultiHeadAttention

        hidden_size = 512
        num_heads = 8
        num_kv_heads = 2  # 4x expansion
        head_dim = 64

        attn = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            use_rotary=False,
        ).to(device, dtype)

        # Check that internal K/V projections have correct size
        assert attn.k_proj.out_features == num_kv_heads * head_dim
        assert attn.v_proj.out_features == num_kv_heads * head_dim
        assert attn.q_proj.out_features == num_heads * head_dim

    def test_attention_causal_mask(self, device, dtype):
        """Test that causal masking works correctly."""
        from kornia.vlm.layers import MultiHeadAttention

        hidden_size = 256
        num_heads = 4
        head_dim = 64
        batch_size = 1
        seq_len = 5

        attn = (
            MultiHeadAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                num_key_value_heads=num_heads,
                head_dim=head_dim,
                use_rotary=False,
            )
            .to(device, dtype)
            .eval()
        )

        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

        # Create causal mask
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
            diagonal=1,
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output, weights, _ = attn(x, attention_mask=causal_mask, output_attentions=True)

        # Check that attention weights are causal (upper triangle should be ~0)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert weights[0, :, i, j].abs().max() < 1e-6, f"Non-causal attention at position ({i}, {j})"


class TestAttentionWeightsNormalization:
    """Test that attention weights are properly normalized."""

    def test_attention_weights_sum_to_one(self, device, dtype):
        """Attention weights should sum to 1 along the key dimension."""
        from kornia.vlm.layers import MultiHeadAttention

        hidden_size = 256
        num_heads = 4
        head_dim = 64
        batch_size = 2
        seq_len = 10

        attn = (
            MultiHeadAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                num_key_value_heads=num_heads,
                head_dim=head_dim,
                use_rotary=False,
            )
            .to(device, dtype)
            .eval()
        )

        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

        with torch.no_grad():
            _, weights, _ = attn(x, output_attentions=True)

        # Check normalization
        weight_sums = weights.sum(dim=-1)
        expected = torch.ones_like(weight_sums)
        assert torch.allclose(weight_sums, expected, atol=1e-5), (
            f"Attention weights don't sum to 1: max deviation = {(weight_sums - 1).abs().max().item()}"
        )
