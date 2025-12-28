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

"""Tests for SigLip2 model."""

import pytest
import torch

from kornia.models.siglip2 import SigLip2Builder, SigLip2Config, SigLip2Model


class TestSigLip2Model:
    """Test suite for SigLip2 model."""

    def test_smoke(self, device, dtype):
        """Test basic model instantiation."""
        config = SigLip2Config()
        model = SigLip2Model(config).to(device, dtype)
        assert model is not None

    def test_forward_image_only(self, device, dtype):
        """Test forward pass with images only."""
        config = SigLip2Config()
        model = SigLip2Model(config).to(device, dtype).eval()

        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)

        with torch.no_grad():
            output = model(pixel_values=pixel_values)

        assert output.image_embeds is not None
        assert output.image_embeds.shape == (batch_size, config.projection_dim)
        assert output.text_embeds is None

    def test_forward_text_only(self, device, dtype):
        """Test forward pass with text only."""
        config = SigLip2Config()
        model = SigLip2Model(config).to(device, dtype).eval()

        batch_size = 2
        seq_len = 10
        # Use smaller range for testing to avoid memory issues with large vocab
        input_ids = torch.randint(0, min(1000, config.text_config.vocab_size), (batch_size, seq_len), device=device)

        with torch.no_grad():
            output = model(input_ids=input_ids)

        assert output.text_embeds is not None
        assert output.text_embeds.shape == (batch_size, config.projection_dim)
        assert output.image_embeds is None

    def test_forward_joint(self, device, dtype):
        """Test forward pass with both image and text."""
        config = SigLip2Config()
        model = SigLip2Model(config).to(device, dtype).eval()

        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)
        seq_len = 10
        # Use smaller range for testing to avoid memory issues with large vocab
        input_ids = torch.randint(0, min(1000, config.text_config.vocab_size), (batch_size, seq_len), device=device)

        with torch.no_grad():
            output = model(pixel_values=pixel_values, input_ids=input_ids)

        assert output.image_embeds is not None
        assert output.text_embeds is not None
        assert output.logits_per_image is not None
        assert output.logits_per_text is not None
        assert output.image_embeds.shape == (batch_size, config.projection_dim)
        assert output.text_embeds.shape == (batch_size, config.projection_dim)
        assert output.logits_per_image.shape == (batch_size, batch_size)
        assert output.logits_per_text.shape == (batch_size, batch_size)

    def test_get_image_features(self, device, dtype):
        """Test get_image_features method."""
        config = SigLip2Config()
        model = SigLip2Model(config).to(device, dtype).eval()

        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)

        with torch.no_grad():
            features = model.get_image_features(pixel_values)

        assert features.shape == (batch_size, config.projection_dim)
        # Check normalization
        norms = features.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-5)

    def test_get_text_features(self, device, dtype):
        """Test get_text_features method."""
        config = SigLip2Config()
        model = SigLip2Model(config).to(device, dtype).eval()

        batch_size = 2
        seq_len = 10
        # Use smaller range for testing to avoid memory issues with large vocab
        input_ids = torch.randint(0, min(1000, config.text_config.vocab_size), (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        with torch.no_grad():
            features = model.get_text_features(input_ids, attention_mask=attention_mask)

        assert features.shape == (batch_size, config.projection_dim)
        # Check normalization
        norms = features.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-5)

    def test_attention_mask_handling(self, device, dtype):
        """Test attention mask handling in text encoder."""
        config = SigLip2Config()
        model = SigLip2Model(config).to(device, dtype).eval()

        batch_size = 2
        seq_len = 10
        # Use smaller range for testing to avoid memory issues with large vocab
        input_ids = torch.randint(0, min(1000, config.text_config.vocab_size), (batch_size, seq_len), device=device)

        # Create attention mask with different lengths
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        attention_mask[0, 5:] = 0  # First sequence has 5 tokens
        attention_mask[1, 8:] = 0  # Second sequence has 8 tokens

        with torch.no_grad():
            features = model.get_text_features(input_ids, attention_mask=attention_mask)

        assert features.shape == (batch_size, config.projection_dim)

    def test_different_batch_sizes(self, device, dtype):
        """Test model with different batch sizes."""
        config = SigLip2Config()
        model = SigLip2Model(config).to(device, dtype).eval()

        for batch_size in [1, 2, 4]:
            pixel_values = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)
            # Use smaller range for testing to avoid memory issues with large vocab
            input_ids = torch.randint(0, min(1000, config.text_config.vocab_size), (batch_size, 10), device=device)

            with torch.no_grad():
                output = model(pixel_values=pixel_values, input_ids=input_ids)

            assert output.image_embeds is not None
            assert output.text_embeds is not None
            assert output.image_embeds.shape[0] == batch_size
            assert output.text_embeds.shape[0] == batch_size

    @pytest.mark.slow
    def test_torch_fullgraph(self, device, dtype):
        """Test that torch.compile with fullgraph=True works (non-breaking graph)."""
        model_name = "google/siglip2-base-patch16-224"

        # Load model
        model = SigLip2Builder.from_pretrained_hf(model_name).to(device, dtype).eval()

        # Try to compile with fullgraph=True
        # This will raise an error if the graph is not fully traceable
        try:
            compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

            # Create dummy inputs
            batch_size = 1
            pixel_values = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)
            input_ids = torch.randint(0, 1000, (batch_size, 10), device=device)
            attention_mask = torch.ones(batch_size, 10, device=device)

            # Test forward pass
            with torch.no_grad():
                outputs = compiled_model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)

            # If we get here, fullgraph compilation succeeded
            assert outputs.image_embeds is not None
            assert outputs.text_embeds is not None
            assert outputs.logits_per_image is not None

        except Exception as e:
            raise e
