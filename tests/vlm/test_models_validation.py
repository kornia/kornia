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

"""Validation tests for kornia.vlm models.

These tests verify model outputs, shapes, and behavior without requiring
pretrained weights. More comprehensive validation against transformers
can be done with separate integration tests.
"""

import pytest
import torch


class TestSiglipVisionEncoderShape:
    """Test Siglip vision encoder output shapes."""

    @pytest.fixture
    def config(self):
        """Create a small Siglip config for fast testing."""
        from kornia.vlm.paligemma.config import SigLIPVisionConfig

        return SigLIPVisionConfig(
            image_size=112,
            patch_size=14,
            hidden_size=384,
            intermediate_size=1536,
            num_hidden_layers=4,
            num_attention_heads=6,
            layer_norm_eps=1e-6,
        )

    def test_output_shape(self, config, device, dtype):
        """Test that Siglip produces correct output shapes."""
        from kornia.vlm.paligemma.siglip import SiglipVisionEncoder

        model = SiglipVisionEncoder(config).to(device, dtype).eval()

        batch_size = 2
        images = torch.randn(batch_size, 3, config.image_size, config.image_size, device=device, dtype=dtype)

        with torch.no_grad():
            output = model(images)

        num_patches = (config.image_size // config.patch_size) ** 2
        assert output.features.shape == (batch_size, num_patches, config.hidden_size)

    def test_layer_features(self, config, device, dtype):
        """Test that layer features are returned correctly."""
        from kornia.vlm.paligemma.siglip import SiglipVisionEncoder

        model = SiglipVisionEncoder(config).to(device, dtype).eval()

        batch_size = 2
        images = torch.randn(batch_size, 3, config.image_size, config.image_size, device=device, dtype=dtype)

        with torch.no_grad():
            output = model(images, return_intermediates=True)

        # Should have features for embedding + each layer
        expected_n_layers = config.num_hidden_layers + 1
        assert len(output.layer_features) == expected_n_layers

        num_patches = (config.image_size // config.patch_size) ** 2
        for i, features in enumerate(output.layer_features):
            assert features.shape == (batch_size, num_patches, config.hidden_size), (
                f"Layer {i} features have wrong shape"
            )

    def test_attention_weights(self, config, device, dtype):
        """Test that attention weights are returned correctly."""
        from kornia.vlm.paligemma.siglip import SiglipVisionEncoder

        model = SiglipVisionEncoder(config).to(device, dtype).eval()

        batch_size = 2
        images = torch.randn(batch_size, 3, config.image_size, config.image_size, device=device, dtype=dtype)

        with torch.no_grad():
            output = model(images, return_attention_weights=True)

        assert len(output.attention_weights) == config.num_hidden_layers

        num_patches = (config.image_size // config.patch_size) ** 2
        for i, attn in enumerate(output.attention_weights):
            assert attn.shape == (
                batch_size,
                config.num_attention_heads,
                num_patches,
                num_patches,
            ), f"Attention {i} has wrong shape"


class TestGemmaDecoderShape:
    """Test Gemma decoder output shapes."""

    @pytest.fixture
    def config(self):
        """Create a small Gemma config for fast testing."""
        from kornia.vlm.paligemma.config import GemmaConfig

        return GemmaConfig(
            vocab_size=1000,
            hidden_size=512,
            intermediate_size=2048,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=64,
            max_position_embeddings=256,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
        )

    def test_output_shape(self, config, device, dtype):
        """Test that Gemma produces correct output shapes."""
        from kornia.vlm.paligemma.gemma import GemmaLM

        model = GemmaLM(config).to(device, dtype).eval()

        batch_size = 2
        seq_len = 10
        token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

        with torch.no_grad():
            loss, logits, layer_features, attentions, cache = model(input_ids=token_ids)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert loss is None  # No labels provided

    def test_layer_features(self, config, device, dtype):
        """Test that layer features are returned correctly."""
        from kornia.vlm.paligemma.gemma import GemmaLM

        model = GemmaLM(config).to(device, dtype).eval()

        batch_size = 2
        seq_len = 10
        token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

        with torch.no_grad():
            loss, logits, layer_features, attentions, cache = model(input_ids=token_ids, output_hidden_states=True)

        # Should have features for embedding + each layer
        expected_n_layers = config.num_hidden_layers + 1
        assert len(layer_features) == expected_n_layers

        for i, features in enumerate(layer_features):
            assert features.shape == (batch_size, seq_len, config.hidden_size), f"Layer {i} features have wrong shape"

    def test_attention_weights(self, config, device, dtype):
        """Test that attention weights are returned correctly."""
        from kornia.vlm.paligemma.gemma import GemmaLM

        model = GemmaLM(config).to(device, dtype).eval()

        batch_size = 2
        seq_len = 10
        token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

        with torch.no_grad():
            loss, logits, layer_features, attentions, cache = model(input_ids=token_ids, output_attentions=True)

        assert len(attentions) == config.num_hidden_layers

        for i, attn in enumerate(attentions):
            assert attn.shape == (batch_size, config.num_attention_heads, seq_len, seq_len), (
                f"Attention {i} has wrong shape"
            )

    def test_kv_cache(self, config, device, dtype):
        """Test that KV cache works correctly."""
        from kornia.vlm.paligemma.gemma import GemmaLM

        model = GemmaLM(config).to(device, dtype).eval()

        batch_size = 1
        seq_len = 5
        token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

        with torch.no_grad():
            _, logits, _, _, cache = model(input_ids=token_ids, use_cache=True)

        assert cache is not None
        assert len(cache) == config.num_hidden_layers

        # Each cache entry should have (key, value)
        for layer_cache in cache:
            assert len(layer_cache) == 2
            key, value = layer_cache
            assert key.shape == (batch_size, config.num_key_value_heads, seq_len, config.head_dim)
            assert value.shape == (batch_size, config.num_key_value_heads, seq_len, config.head_dim)


class TestPaliGemma2Shape:
    """Test PaliGemma2 model output shapes."""

    @pytest.fixture
    def config(self):
        """Create small PaliGemma config for testing."""
        from kornia.vlm.paligemma.config import GemmaConfig, PaliGemma2Config, SigLIPVisionConfig

        vision_config = SigLIPVisionConfig(
            image_size=112,
            patch_size=14,
            hidden_size=384,
            intermediate_size=1536,
            num_hidden_layers=2,
            num_attention_heads=6,
        )
        text_config = GemmaConfig(
            vocab_size=1000,
            hidden_size=512,
            intermediate_size=2048,
            num_hidden_layers=2,
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

    def test_output_shape(self, config, device, dtype):
        """Test that PaliGemma produces correct output shapes."""
        from kornia.vlm import PaliGemma2

        model = PaliGemma2(config).to(device, dtype).eval()

        batch_size = 2
        num_patches = config.vision_config.num_patches
        text_len = 10

        images = torch.randn(
            batch_size,
            3,
            config.vision_config.image_size,
            config.vision_config.image_size,
            device=device,
            dtype=dtype,
        )

        image_tokens = torch.full((batch_size, num_patches), config.image_token_index, device=device)
        text_tokens = torch.randint(0, 100, (batch_size, text_len), device=device)
        token_ids = torch.cat([image_tokens, text_tokens], dim=1)

        with torch.no_grad():
            output = model(images, token_ids)

        expected_seq_len = num_patches + text_len
        expected_vocab_size = config.text_config.vocab_size

        assert output.logits.shape == (batch_size, expected_seq_len, expected_vocab_size)

    def test_vision_features_shape(self, config, device, dtype):
        """Test that vision features have correct shape."""
        from kornia.vlm import PaliGemma2

        model = PaliGemma2(config).to(device, dtype).eval()

        batch_size = 2
        images = torch.randn(
            batch_size,
            3,
            config.vision_config.image_size,
            config.vision_config.image_size,
            device=device,
            dtype=dtype,
        )

        with torch.no_grad():
            vision_output = model.extract_vision_features(images)

        expected_n_patches = config.vision_config.num_patches
        expected_dim = config.vision_config.hidden_size

        assert vision_output.features.shape == (
            batch_size,
            expected_n_patches,
            expected_dim,
        )

    def test_projected_features(self, config, device, dtype):
        """Test that projected features have correct dimensions."""
        from kornia.vlm import PaliGemma2

        model = PaliGemma2(config).to(device, dtype).eval()

        batch_size = 2
        num_patches = config.vision_config.num_patches
        text_len = 10

        images = torch.randn(
            batch_size,
            3,
            config.vision_config.image_size,
            config.vision_config.image_size,
            device=device,
            dtype=dtype,
        )

        image_tokens = torch.full((batch_size, num_patches), config.image_token_index, device=device)
        text_tokens = torch.randint(0, 100, (batch_size, text_len), device=device)
        token_ids = torch.cat([image_tokens, text_tokens], dim=1)

        with torch.no_grad():
            output = model(images, token_ids, return_intermediates=True)

        # Projected features should match text hidden size
        assert output.projected.shape == (
            batch_size,
            num_patches,
            config.text_config.hidden_size,
        )
