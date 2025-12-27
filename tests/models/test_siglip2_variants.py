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

"""Tests for all SigLip2 model variants."""

import pytest
import torch

from kornia.models.siglip2 import SigLip2Builder, SigLip2Config
from kornia.testing import BaseTester

# Model variants with expected configurations
# (model_name, image_size, hidden_size, num_layers, num_heads, vocab_size, projection_dim)
MODEL_VARIANTS = [
    ("google/siglip-base-patch16-224", 224, 768, 12, 12, 32000, 768),
    ("google/siglip2-base-patch16-224", 224, 768, 12, 12, 256000, 768),
    ("google/siglip2-base-patch16-256", 256, 768, 12, 12, 256000, 768),
    ("google/siglip2-base-patch16-384", 384, 768, 12, 12, 256000, 768),
    ("google/siglip2-base-patch16-512", 512, 768, 12, 12, 256000, 768),
    ("google/siglip2-large-patch16-256", 256, 1024, 24, 16, 256000, 1024),
    ("google/siglip2-large-patch16-384", 384, 1024, 24, 16, 256000, 1024),
    ("google/siglip2-large-patch16-512", 512, 1024, 24, 16, 256000, 1024),
]


@pytest.mark.slow
class TestSigLip2Variants(BaseTester):
    """Test suite for all SigLip2 model variants."""

    @pytest.mark.parametrize("model_name,img_size,hidden,layers,heads,vocab,proj", MODEL_VARIANTS)
    def test_config_from_name(self, model_name, img_size, hidden, layers, heads, vocab, proj):
        """Test that config.from_name() creates correct configuration."""
        config = SigLip2Config.from_name(model_name)

        assert config.vision_config.image_size == img_size
        assert config.vision_config.hidden_size == hidden
        assert config.vision_config.num_hidden_layers == layers
        assert config.vision_config.num_attention_heads == heads
        assert config.text_config.vocab_size == vocab
        assert config.projection_dim == proj

    @pytest.mark.parametrize("model_name,img_size,hidden,layers,heads,vocab,proj", MODEL_VARIANTS)
    def test_model_creation(self, model_name, img_size, hidden, layers, heads, vocab, proj, device, dtype):
        """Test that models can be created for all variants."""
        model = SigLip2Builder.from_name(model_name).to(device, dtype)
        assert model is not None

        # Verify structure
        assert model.vision_model.config.image_size == img_size
        assert model.vision_model.config.hidden_size == hidden
        assert len(model.vision_model.encoder.layers) == layers
        assert model.text_model.config.vocab_size == vocab

    @pytest.mark.parametrize("model_name,img_size,hidden,layers,heads,vocab,proj", MODEL_VARIANTS)
    def test_forward_pass(self, model_name, img_size, hidden, layers, heads, vocab, proj, device, dtype):
        """Test forward pass for all variants."""
        model = SigLip2Builder.from_name(model_name).to(device, dtype).eval()

        batch_size = 1
        pixel_values = torch.randn(batch_size, 3, img_size, img_size, device=device, dtype=dtype)
        # Use smaller range for testing to avoid memory issues with large vocab
        input_ids = torch.randint(0, min(1000, vocab), (batch_size, 10), device=device)
        attention_mask = torch.ones(batch_size, 10, device=device)

        with torch.no_grad():
            output = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)

        assert output["image_embeds"].shape == (batch_size, proj)
        assert output["text_embeds"].shape == (batch_size, proj)
        assert output["logits_per_image"].shape == (batch_size, batch_size)

    @pytest.mark.parametrize("model_name,img_size,hidden,layers,heads,vocab,proj", MODEL_VARIANTS)
    def test_feature_extraction(self, model_name, img_size, hidden, layers, heads, vocab, proj, device, dtype):
        """Test feature extraction methods for all variants."""
        model = SigLip2Builder.from_name(model_name).to(device, dtype).eval()

        batch_size = 1
        pixel_values = torch.randn(batch_size, 3, img_size, img_size, device=device, dtype=dtype)
        input_ids = torch.randint(0, min(1000, vocab), (batch_size, 10), device=device)
        attention_mask = torch.ones(batch_size, 10, device=device)

        with torch.no_grad():
            img_feat = model.get_image_features(pixel_values)
            txt_feat = model.get_text_features(input_ids, attention_mask=attention_mask)

        assert img_feat.shape == (batch_size, proj)
        assert txt_feat.shape == (batch_size, proj)

        # Check normalization
        img_norm = img_feat.norm(dim=-1)
        txt_norm = txt_feat.norm(dim=-1)
        self.assert_close(img_norm, torch.ones_like(img_norm), rtol=1e-5, atol=1e-5)
        self.assert_close(txt_norm, torch.ones_like(txt_norm), rtol=1e-5, atol=1e-5)
