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
from kornia.testing import BaseTester

try:
    from transformers import Siglip2Model as HFSigLip2Model
    from transformers import Siglip2Processor

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers library not available")
class TestSigLip2Model(BaseTester):
    """Test suite for SigLip2 model."""

    @pytest.mark.slow
    def test_smoke(self, device, dtype):
        """Test basic model instantiation."""
        config = SigLip2Config()
        model = SigLip2Model(config).to(device, dtype)
        assert model is not None

    @pytest.mark.slow
    def test_forward_image_only(self, device, dtype):
        """Test forward pass with images only."""
        config = SigLip2Config()
        model = SigLip2Model(config).to(device, dtype).eval()

        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)

        with torch.no_grad():
            output = model(pixel_values=pixel_values)

        assert "image_embeds" in output
        assert output["image_embeds"].shape == (batch_size, config.projection_dim)
        assert output["text_embeds"] is None

    @pytest.mark.slow
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

        assert "text_embeds" in output
        assert output["text_embeds"].shape == (batch_size, config.projection_dim)
        assert output["image_embeds"] is None

    @pytest.mark.slow
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

        assert "image_embeds" in output
        assert "text_embeds" in output
        assert "logits_per_image" in output
        assert "logits_per_text" in output
        assert output["image_embeds"].shape == (batch_size, config.projection_dim)
        assert output["text_embeds"].shape == (batch_size, config.projection_dim)
        assert output["logits_per_image"].shape == (batch_size, batch_size)
        assert output["logits_per_text"].shape == (batch_size, batch_size)

    @pytest.mark.slow
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
        self.assert_close(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-5)

    @pytest.mark.slow
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
        self.assert_close(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-5)

    @pytest.mark.slow
    @pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers library not available")
    def test_output_matching_with_hf(self, device, dtype):
        """Test that outputs match HuggingFace transformers implementation."""
        model_name = "google/siglip2-base-patch16-224"

        # Load HF model
        hf_model = HFSigLip2Model.from_pretrained(model_name).to(device).eval()
        processor = Siglip2Processor.from_pretrained(model_name)

        # Load Kornia model
        kornia_model = SigLip2Builder.from_pretrained(model_name).to(device).eval()

        # Prepare test inputs
        texts = ["a photo of a cat", "a photo of a dog"]
        images = [torch.randn(3, 224, 224, device=device, dtype=dtype) for _ in range(2)]

        # Process inputs
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(device)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Forward pass with HF model
        with torch.no_grad():
            hf_outputs = hf_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Forward pass with Kornia model
        with torch.no_grad():
            kornia_outputs = kornia_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Compare outputs
        # Note: We use relaxed tolerances as there may be minor numerical differences
        # due to implementation details (e.g., attention mask handling, pooling)

        # Compare image embeddings
        if hf_outputs.image_embeds is not None and kornia_outputs["image_embeds"] is not None:
            self.assert_close(
                kornia_outputs["image_embeds"],
                hf_outputs.image_embeds,
                rtol=1e-3,
                atol=1e-3,
            )

        # Compare text embeddings
        if hf_outputs.text_embeds is not None and kornia_outputs["text_embeds"] is not None:
            self.assert_close(
                kornia_outputs["text_embeds"],
                hf_outputs.text_embeds,
                rtol=1e-3,
                atol=1e-3,
            )

        # Compare logits if available
        if hasattr(hf_outputs, "logits_per_image") and "logits_per_image" in kornia_outputs:
            self.assert_close(
                kornia_outputs["logits_per_image"],
                hf_outputs.logits_per_image,
                rtol=1e-2,
                atol=1e-2,
            )

    @pytest.mark.slow
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

    @pytest.mark.slow
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

            assert output["image_embeds"].shape[0] == batch_size
            assert output["text_embeds"].shape[0] == batch_size

    @pytest.mark.slow
    def test_module(self, device, dtype):
        """Test that model is a proper PyTorch module."""
        config = SigLip2Config()
        model = SigLip2Model(config).to(device, dtype)

        # Test that it's a Module
        assert isinstance(model, torch.nn.Module)

        # Test that it has required components
        assert hasattr(model, "vision_model")
        assert hasattr(model, "text_model")
        assert hasattr(model, "vision_projection")
        assert hasattr(model, "text_projection")
        assert hasattr(model, "logit_scale")
