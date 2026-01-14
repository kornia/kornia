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

from kornia.models.siglip2 import SigLip2Config, SigLip2Model, SigLip2Result
from kornia.models.siglip2.attention import SigLip2Attention
from kornia.models.siglip2.config import SigLip2TextConfig, SigLip2VisionConfig
from kornia.models.siglip2.preprocessor import SigLip2ImagePreprocessor
from kornia.models.siglip2.text_encoder import SigLip2TextEmbeddings, SigLip2TextEncoder, SigLip2TextModel
from kornia.models.siglip2.vision_encoder import SigLip2VisionEmbeddings, SigLip2VisionEncoder, SigLip2VisionModel

from testing.base import BaseTester


@pytest.fixture
def config():
    """Fixture for SigLip2Config."""
    return SigLip2Config()


@pytest.fixture
def model(device, dtype, config):
    """Fixture for SigLip2Model."""
    return SigLip2Model(config).to(device, dtype).eval()


def _create_input_ids(batch_size, seq_len, config, device):
    """Create input_ids with smaller range to avoid memory issues with large vocab."""
    return torch.randint(0, min(100, config.text_config.vocab_size), (batch_size, seq_len), device=device)


class TestSigLip2Model(BaseTester):
    """Test suite for SigLip2 model."""

    def test_siglip2_logit_scale_no_nan(self):
        import torch

        from kornia.models.siglip2 import SigLip2Config, SigLip2Model

        config = SigLip2Config()
        model = SigLip2Model(config)

        # simulate extreme value
        model.logit_scale.data.fill_(100.0)

        pixel_values = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (2, 10))

        output = model(pixel_values=pixel_values, input_ids=input_ids)

        assert torch.isfinite(output.logits_per_image).all()
        assert torch.isfinite(output.logits_per_text).all()

    def test_smoke(self, device, dtype, config):
        """Test basic model instantiation."""
        model = SigLip2Model(config).to(device, dtype)
        assert model is not None

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_cardinality(self, device, dtype, model, config, batch_size):
        """Test output shapes with different inputs and batch sizes."""
        pixel_values = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)
        seq_len = 10
        input_ids = _create_input_ids(batch_size, seq_len, config, device)

        with torch.no_grad():
            # Image only
            output: SigLip2Result = model(pixel_values=pixel_values)
            assert output.image_embeds is not None
            assert output.image_embeds.shape == (batch_size, config.projection_dim)
            assert output.text_embeds is None

            # Text only
            output: SigLip2Result = model(input_ids=input_ids)
            assert output.text_embeds is not None
            assert output.text_embeds.shape == (batch_size, config.projection_dim)
            assert output.image_embeds is None

            # Joint
            output: SigLip2Result = model(pixel_values=pixel_values, input_ids=input_ids)
            assert output.image_embeds.shape == (batch_size, config.projection_dim)
            assert output.text_embeds.shape == (batch_size, config.projection_dim)
            assert output.logits_per_image.shape == (batch_size, batch_size)
            assert output.logits_per_text.shape == (batch_size, batch_size)

    def test_exception(self, device, dtype, model, config):
        """Test exception handling."""
        # Test invalid pixel_values shape (wrong number of dimensions)
        with pytest.raises((RuntimeError, ValueError, IndexError)):
            invalid_pixel_values = torch.randn(3, 224, 224, device=device, dtype=dtype)  # Missing batch dimension
            model.get_image_features(invalid_pixel_values)

        # Test invalid attention mask shape
        with pytest.raises((RuntimeError, ValueError, IndexError)):
            input_ids = _create_input_ids(2, 10, config, device)
            invalid_attention_mask = torch.ones(2, 5, device=device)  # Wrong sequence length
            model.get_text_features(input_ids, attention_mask=invalid_attention_mask)

        # Test input_ids with wrong number of dimensions
        with pytest.raises((RuntimeError, ValueError, IndexError)):
            invalid_input_ids = torch.randint(0, 100, (10,), device=device)  # Missing batch dimension
            model.get_text_features(invalid_input_ids)

    def test_get_image_features(self, device, dtype, model, config):
        """Test get_image_features method."""

        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)

        with torch.no_grad():
            features = model.get_image_features(pixel_values)

        assert features.shape == (batch_size, config.projection_dim)
        # Check normalization
        norms = features.norm(dim=-1)
        self.assert_close(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-5)

    def test_get_text_features(self, device, dtype, model, config):
        """Test get_text_features method."""

        batch_size = 2
        seq_len = 10
        input_ids = _create_input_ids(batch_size, seq_len, config, device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        with torch.no_grad():
            features = model.get_text_features(input_ids, attention_mask=attention_mask)

        assert features.shape == (batch_size, config.projection_dim)
        # Check normalization
        norms = features.norm(dim=-1)
        self.assert_close(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-5)

    def test_attention_mask_handling(self, device, dtype, model, config):
        """Test attention mask handling in text encoder."""

        batch_size = 2
        seq_len = 10
        input_ids = _create_input_ids(batch_size, seq_len, config, device)

        # Create attention mask with different lengths
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        attention_mask[0, 5:] = 0  # First sequence has 5 tokens
        attention_mask[1, 8:] = 0  # Second sequence has 8 tokens

        with torch.no_grad():
            features = model.get_text_features(input_ids, attention_mask=attention_mask)

        assert features.shape == (batch_size, config.projection_dim)

    def test_return_loss(self, device, dtype, model, config):
        """Test forward pass with return_loss=True and verify logit_scale clamping."""
        import math

        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)
        seq_len = 10
        input_ids = _create_input_ids(batch_size, seq_len, config, device)

        with torch.no_grad():
            output = model(pixel_values=pixel_values, input_ids=input_ids, return_loss=True)

        assert output.loss is not None
        assert output.loss.item() >= 0.0  # Loss should be non-negative

        # Test logit_scale clamping with extreme values
        with torch.no_grad():
            # Test max clamping
            model.logit_scale.data.fill_(100.0)
            output_max = model(pixel_values=pixel_values, input_ids=input_ids)
            assert torch.isfinite(output_max.logits_per_image).all(), "Max clamp: logits contain non-finite values"
            assert math.isclose(output_max.logit_scale.item(), config.logit_scale_max, rel_tol=1e-5, abs_tol=1e-5), (
                f"Max clamp failed: {output_max.logit_scale.item()} != {config.logit_scale_max}"
            )

            # Test min clamping
            model.logit_scale.data.fill_(-10.0)
            output_min = model(pixel_values=pixel_values, input_ids=input_ids)
            assert output_min.logit_scale.item() >= 1.0, f"Min clamp failed: {output_min.logit_scale.item()} < 1.0"

    def test_gradcheck(self, device, dtype, config):
        """Test gradient computation correctness."""
        # Convert model to float64 for gradcheck
        model = SigLip2Model(config).to(device, torch.float64).train()
        batch_size = 1
        pixel_values = torch.randn(batch_size, 3, 224, 224, device=device, dtype=torch.float64, requires_grad=True)
        seq_len = 5
        input_ids = _create_input_ids(batch_size, seq_len, config, device).to(torch.int64)

        # Only check gradients for pixel_values (input_ids are indices, not differentiable)
        def func(pixel_vals):
            # Use input_ids as closure variable, not as gradcheck input
            return model.get_image_features(pixel_vals) + model.get_text_features(input_ids)

        self.gradcheck(func, pixel_values, raise_exception=True, fast_mode=True)

    def test_dynamo(self, device, dtype, torch_optimizer, model, config):
        """Test torch.compile compatibility."""
        batch_size = 1
        pixel_values = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)
        seq_len = 10
        input_ids = _create_input_ids(batch_size, seq_len, config, device)

        model_optimized = torch_optimizer(model)

        with torch.no_grad():
            expected = model(pixel_values=pixel_values, input_ids=input_ids)
            actual = model_optimized(pixel_values=pixel_values, input_ids=input_ids)

        self.assert_close(actual.image_embeds, expected.image_embeds)
        self.assert_close(actual.text_embeds, expected.text_embeds)


class TestSigLip2Components(BaseTester):
    """Test suite for SigLip2 individual components."""

    def test_vision_embeddings(self, device, dtype):
        """Test SigLip2VisionEmbeddings."""
        config = SigLip2VisionConfig(image_size=224, patch_size=16, hidden_size=768)
        embeddings = SigLip2VisionEmbeddings(config).to(device, dtype)

        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)

        with torch.no_grad():
            output = embeddings(pixel_values)

        num_patches = (224 // 16) ** 2
        assert output.shape == (batch_size, num_patches, config.hidden_size)

    def test_vision_encoder(self, device, dtype):
        """Test SigLip2VisionEncoder."""
        config = SigLip2VisionConfig(
            image_size=224, patch_size=16, hidden_size=768, num_hidden_layers=2, num_attention_heads=12
        )
        encoder = SigLip2VisionEncoder(config).to(device, dtype)
        embeddings = SigLip2VisionEmbeddings(config).to(device, dtype)

        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)

        with torch.no_grad():
            # Encoder expects embeddings, not raw pixel values
            hidden_states = embeddings(pixel_values)
            output = encoder(hidden_states)

        num_patches = (224 // 16) ** 2
        assert output[0].shape == (batch_size, num_patches, config.hidden_size)

    def test_vision_model(self, device, dtype):
        """Test SigLip2VisionModel."""
        config = SigLip2VisionConfig(
            image_size=224, patch_size=16, hidden_size=768, num_hidden_layers=2, num_attention_heads=12
        )
        model = SigLip2VisionModel(config).to(device, dtype)

        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)

        with torch.no_grad():
            pooled_output, last_hidden_state = model(pixel_values)

        assert pooled_output.shape == (batch_size, config.hidden_size)
        num_patches = (224 // 16) ** 2
        assert last_hidden_state.shape == (batch_size, num_patches, config.hidden_size)

    def test_text_embeddings(self, device, dtype):
        """Test SigLip2TextEmbeddings."""
        config = SigLip2TextConfig(vocab_size=1000, hidden_size=768, max_position_embeddings=512)
        embeddings = SigLip2TextEmbeddings(config).to(device, dtype)

        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

        with torch.no_grad():
            output = embeddings(input_ids)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_text_encoder(self, device, dtype):
        """Test SigLip2TextEncoder."""
        config = SigLip2TextConfig(
            vocab_size=1000,
            hidden_size=768,
            num_hidden_layers=2,
            num_attention_heads=12,
            max_position_embeddings=512,
        )
        encoder = SigLip2TextEncoder(config).to(device, dtype)
        embeddings = SigLip2TextEmbeddings(config).to(device, dtype)

        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        with torch.no_grad():
            # Encoder expects hidden_states (embeddings), not input_ids
            hidden_states = embeddings(input_ids)
            output = encoder(hidden_states, attention_mask=attention_mask)

        assert output[0].shape == (batch_size, seq_len, config.hidden_size)

    def test_text_model(self, device, dtype):
        """Test SigLip2TextModel."""
        config = SigLip2TextConfig(
            vocab_size=1000,
            hidden_size=768,
            num_hidden_layers=2,
            num_attention_heads=12,
            max_position_embeddings=512,
        )
        model = SigLip2TextModel(config).to(device, dtype)

        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        with torch.no_grad():
            pooled_output, last_hidden_state = model(input_ids=input_ids, attention_mask=attention_mask)

        assert pooled_output.shape == (batch_size, config.hidden_size)
        assert last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)

    def test_attention(self, device, dtype):
        """Test SigLip2Attention."""
        hidden_size = 768
        num_heads = 12
        attention = SigLip2Attention(hidden_size=hidden_size, num_heads=num_heads).to(device, dtype)

        batch_size = 2
        seq_len = 10
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        with torch.no_grad():
            output = attention(hidden_states, attention_mask=attention_mask)

        # Attention returns a single tensor, not a tuple
        assert output.shape == (batch_size, seq_len, hidden_size)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("input_size", [(256, 256), (300, 400), (512, 512)])
    @pytest.mark.parametrize("image_size", [(224, 224), (256, 256), (384, 384)])
    def test_image_preprocessor(self, device, dtype, batch_size, input_size, image_size):
        """Test SigLip2ImagePreprocessor with different configurations."""
        preprocessor = SigLip2ImagePreprocessor(image_size=image_size).to(device, dtype)

        # Test with batch of images (4D tensor)
        images = torch.randint(0, 255, (batch_size, 3, *input_size), device=device, dtype=dtype)
        with torch.no_grad():
            output = preprocessor(images)
        assert output.shape == (batch_size, 3, image_size[0], image_size[1])

    def test_image_preprocessor_single_image(self, device, dtype):
        """Test SigLip2ImagePreprocessor with single image (3D tensor)."""
        image_size = (224, 224)
        preprocessor = SigLip2ImagePreprocessor(image_size=image_size).to(device, dtype)

        # Test with single image (3D tensor) - preprocessor adds batch dimension
        image = torch.randint(0, 255, (3, 256, 256), device=device, dtype=dtype)
        with torch.no_grad():
            output = preprocessor(image)
        assert output.shape == (1, 3, image_size[0], image_size[1])
