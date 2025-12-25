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

"""Tests for PaliGemma model implementation."""

import pytest
import torch

from kornia.vlm import (
    GemmaConfig,
    GemmaDecoder,
    GemmaLM,
    PaliGemma2,
    PaliGemma2Config,
    SigLIPVisionConfig,
    SiglipVisionEncoder,
)
from kornia.vlm.layers import (
    GeLUMLP,
    LayerNorm,
    MultiHeadAttention,
    PatchEmbedding,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
)


class TestLayers:
    """Test transformer building blocks."""

    def test_rms_norm(self, device, dtype):
        """Test RMSNorm layer."""
        hidden_size = 768
        batch_size = 2
        seq_len = 10

        norm = RMSNorm(hidden_size).to(device, dtype)
        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

        output = norm(x)

        assert output.shape == x.shape
        assert output.dtype == dtype

    def test_layer_norm(self, device, dtype):
        """Test LayerNorm layer."""
        hidden_size = 768
        batch_size = 2
        seq_len = 10

        norm = LayerNorm(hidden_size).to(device, dtype)
        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

        output = norm(x)

        assert output.shape == x.shape
        assert output.dtype == dtype

    def test_gelu_mlp(self, device, dtype):
        """Test GeLU MLP layer."""
        hidden_size = 768
        intermediate_size = 3072
        batch_size = 2
        seq_len = 10

        mlp = GeLUMLP(hidden_size, intermediate_size).to(device, dtype)
        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

        output = mlp(x)

        assert output.shape == x.shape
        assert output.dtype == dtype

    def test_swiglu(self, device, dtype):
        """Test SwiGLU MLP layer."""
        hidden_size = 2048
        intermediate_size = 16384
        batch_size = 2
        seq_len = 10

        mlp = SwiGLU(hidden_size, intermediate_size).to(device, dtype)
        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

        output = mlp(x)

        assert output.shape == x.shape
        assert output.dtype == dtype

    def test_patch_embedding(self, device, dtype):
        """Test PatchEmbedding layer."""
        image_size = 224
        patch_size = 14
        embed_dim = 1152
        batch_size = 2

        embed = PatchEmbedding(image_size, patch_size, 3, embed_dim).to(device, dtype)
        images = torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=dtype)

        output = embed(images)

        num_patches = (image_size // patch_size) ** 2
        assert output.shape == (batch_size, num_patches, embed_dim)
        assert output.dtype == dtype

    def test_rotary_embedding(self, device, dtype):
        """Test RotaryEmbedding."""
        head_dim = 256
        max_seq_len = 512
        batch_size = 2
        seq_len = 10

        rope = RotaryEmbedding(head_dim, max_seq_len).to(device)
        x = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=dtype)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        cos, sin = rope(x, position_ids)

        assert cos.shape == (batch_size, seq_len, head_dim)
        assert sin.shape == (batch_size, seq_len, head_dim)

    def test_multi_head_attention(self, device, dtype):
        """Test MultiHeadAttention layer."""
        hidden_size = 2048
        num_attention_heads = 8
        num_key_value_heads = 1
        head_dim = 256
        batch_size = 2
        seq_len = 10

        attn = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        ).to(device, dtype)

        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

        output, weights, cache = attn(x, output_attentions=True)

        assert output.shape == x.shape
        assert weights is not None
        assert weights.shape == (batch_size, num_attention_heads, seq_len, seq_len)


class TestSiglipVisionEncoder:
    """Test Siglip vision encoder."""

    @pytest.fixture
    def config(self):
        """Create a small config for testing."""
        return SigLIPVisionConfig(
            image_size=112,  # Smaller for faster tests
            patch_size=14,
            hidden_size=384,
            intermediate_size=1536,
            num_hidden_layers=4,  # Fewer layers
            num_attention_heads=6,
        )

    def test_forward(self, config, device, dtype):
        """Test Siglip forward pass."""
        model = SiglipVisionEncoder(config).to(device, dtype)
        batch_size = 2

        images = torch.randn(batch_size, 3, config.image_size, config.image_size, device=device, dtype=dtype)

        output = model(images)

        num_patches = (config.image_size // config.patch_size) ** 2
        assert output.features.shape == (batch_size, num_patches, config.hidden_size)
        assert output.layer_features is None
        assert output.attention_weights is None

    def test_return_intermediates(self, config, device, dtype):
        """Test Siglip with return_intermediates=True."""
        model = SiglipVisionEncoder(config).to(device, dtype)
        batch_size = 2

        images = torch.randn(batch_size, 3, config.image_size, config.image_size, device=device, dtype=dtype)

        output = model(images, return_intermediates=True)

        assert output.layer_features is not None
        # layer_features includes embedding + each layer output
        assert len(output.layer_features) == config.num_hidden_layers + 1

    def test_return_attention_weights(self, config, device, dtype):
        """Test Siglip with return_attention_weights=True."""
        model = SiglipVisionEncoder(config).to(device, dtype)
        batch_size = 2

        images = torch.randn(batch_size, 3, config.image_size, config.image_size, device=device, dtype=dtype)

        output = model(images, return_attention_weights=True)

        assert output.attention_weights is not None
        assert len(output.attention_weights) == config.num_hidden_layers


class TestGemmaDecoder:
    """Test Gemma language decoder."""

    @pytest.fixture
    def config(self):
        """Create a small config for testing."""
        return GemmaConfig(
            vocab_size=1000,
            hidden_size=512,
            intermediate_size=2048,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=128,
            max_position_embeddings=256,
        )

    def test_forward(self, config, device, dtype):
        """Test Gemma forward pass."""
        model = GemmaDecoder(config).to(device, dtype)
        batch_size = 2
        seq_len = 10

        token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

        output = model(token_ids=token_ids)

        features = output[0]
        assert features.shape == (batch_size, seq_len, config.hidden_size)

    def test_causal_lm(self, config, device, dtype):
        """Test GemmaLM."""
        model = GemmaLM(config).to(device, dtype)
        batch_size = 2
        seq_len = 10

        token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

        loss, logits, _, _, _ = model(input_ids=token_ids, labels=token_ids)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert loss is not None


class TestPaliGemma2:
    """Test PaliGemma2 VLM."""

    @pytest.fixture
    def config(self):
        """Create a small config for testing."""
        vision_config = SigLIPVisionConfig(
            image_size=112,
            patch_size=14,
            hidden_size=384,
            intermediate_size=1536,
            num_hidden_layers=4,
            num_attention_heads=6,
        )
        text_config = GemmaConfig(
            vocab_size=1000,
            hidden_size=512,
            intermediate_size=2048,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=128,
            max_position_embeddings=256,
        )
        return PaliGemma2Config(
            vision_config=vision_config,
            text_config=text_config,
            image_token_index=999,
        )

    def test_forward(self, config, device, dtype):
        """Test PaliGemma2 forward pass."""
        model = PaliGemma2(config).to(device, dtype)
        batch_size = 2
        seq_len = 10 + config.vision_config.num_patches  # Text + image tokens

        images = torch.randn(
            batch_size, 3, config.vision_config.image_size, config.vision_config.image_size, device=device, dtype=dtype
        )

        # Create token_ids with image tokens
        num_patches = config.vision_config.num_patches
        image_tokens = torch.full((batch_size, num_patches), config.image_token_index, device=device)
        text_tokens = torch.randint(0, 100, (batch_size, 10), device=device)
        token_ids = torch.cat([image_tokens, text_tokens], dim=1)

        output = model(images, token_ids)

        assert output.logits.shape == (batch_size, seq_len, config.text_config.vocab_size)

    def test_extract_vision_features(self, config, device, dtype):
        """Test vision feature extraction."""
        model = PaliGemma2(config).to(device, dtype)
        batch_size = 2

        images = torch.randn(
            batch_size, 3, config.vision_config.image_size, config.vision_config.image_size, device=device, dtype=dtype
        )

        output = model.extract_vision_features(images)

        num_patches = config.vision_config.num_patches
        assert output.features.shape == (batch_size, num_patches, config.vision_config.hidden_size)

    def test_extract_vision_features_with_intermediates(self, config, device, dtype):
        """Test vision feature extraction with intermediate states."""
        model = PaliGemma2(config).to(device, dtype)
        batch_size = 2

        images = torch.randn(
            batch_size, 3, config.vision_config.image_size, config.vision_config.image_size, device=device, dtype=dtype
        )

        output = model.extract_vision_features(images, return_intermediates=True, return_attention_weights=True)

        assert output.layer_features is not None
        assert output.attention_weights is not None
        assert len(output.layer_features) == config.vision_config.num_hidden_layers + 1

    def test_return_intermediates(self, config, device, dtype):
        """Test PaliGemma2 with return_intermediates=True."""
        model = PaliGemma2(config).to(device, dtype)
        batch_size = 2

        images = torch.randn(
            batch_size, 3, config.vision_config.image_size, config.vision_config.image_size, device=device, dtype=dtype
        )

        num_patches = config.vision_config.num_patches
        image_tokens = torch.full((batch_size, num_patches), config.image_token_index, device=device)
        text_tokens = torch.randint(0, 100, (batch_size, 10), device=device)
        token_ids = torch.cat([image_tokens, text_tokens], dim=1)

        output = model(images, token_ids, return_intermediates=True)

        assert output.vision_features is not None
        assert output.text_features is not None
        assert output.projected is not None

    def test_from_config(self, config):
        """Test model creation from config."""
        model = PaliGemma2.from_config(config)
        assert isinstance(model, PaliGemma2)

    def test_vision_tower_access(self, config, device, dtype):
        """Test direct access to vision tower."""
        model = PaliGemma2(config).to(device, dtype)

        # Should be able to access vision tower directly
        assert hasattr(model, "vision_tower")
        assert isinstance(model.vision_tower, SiglipVisionEncoder)

        # Should be able to use it independently
        batch_size = 2
        images = torch.randn(
            batch_size, 3, config.vision_config.image_size, config.vision_config.image_size, device=device, dtype=dtype
        )

        output = model.vision_tower(images)
        assert output.features.shape[0] == batch_size
