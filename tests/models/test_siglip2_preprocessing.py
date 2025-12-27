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

"""Test SigLip2 preprocessing matches HuggingFace."""

import numpy as np
import pytest
import torch
from PIL import Image

from kornia.models.siglip2 import SigLip2Builder

try:
    from transformers import AutoModel, GemmaTokenizer, SiglipImageProcessor

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not available")
def test_preprocessing_matches_hf():
    """Test that preprocessing produces same results as HF."""
    model_name = "google/siglip2-base-patch16-224"
    device = "cpu"

    # Set seeds for reproducibility
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    # Create test image
    img_array = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
    img_pil = Image.fromarray(img_array)

    # Get HF image processor
    img_proc = SiglipImageProcessor.from_pretrained(model_name)

    # Process with HF processor
    hf_processed = img_proc(img_pil, return_tensors="pt")
    hf_pixel_values = hf_processed["pixel_values"]

    # Load models
    torch.manual_seed(42)
    hf_model = AutoModel.from_pretrained(model_name).to(device).eval()

    torch.manual_seed(42)
    kornia_model = SigLip2Builder.from_pretrained(model_name).to(device).eval()

    # Test with same preprocessed image
    tokenizer = GemmaTokenizer.from_pretrained(model_name)
    texts = ["a photo of a cat"]
    encoded = tokenizer(texts, padding=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    with torch.no_grad():
        hf_outputs = hf_model(pixel_values=hf_pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        kornia_outputs = kornia_model(pixel_values=hf_pixel_values, input_ids=input_ids, attention_mask=attention_mask)

        # Check image embeddings
        img_diff = (hf_outputs.image_embeds - kornia_outputs["image_embeds"]).abs()
        img_cosine = torch.nn.functional.cosine_similarity(
            hf_outputs.image_embeds, kornia_outputs["image_embeds"], dim=-1
        ).item()

        assert img_diff.max() < 1e-3, f"Image embeddings differ: max_diff={img_diff.max():.6e}"
        assert abs(img_cosine - 1.0) < 1e-3, f"Image embeddings cosine similarity: {img_cosine:.9f}"

        # Check text embeddings
        txt_diff = (hf_outputs.text_embeds - kornia_outputs["text_embeds"]).abs()
        txt_cosine = torch.nn.functional.cosine_similarity(
            hf_outputs.text_embeds, kornia_outputs["text_embeds"], dim=-1
        ).item()

        assert txt_diff.max() < 1e-3, f"Text embeddings differ: max_diff={txt_diff.max():.6e}"
        assert abs(txt_cosine - 1.0) < 1e-3, f"Text embeddings cosine similarity: {txt_cosine:.9f}"

        # Check logits
        logits_diff = (hf_outputs.logits_per_image - kornia_outputs["logits_per_image"]).abs()
        assert logits_diff.max() < 1e-2, f"Logits differ: max_diff={logits_diff.max():.6e}"


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not available")
def test_preprocessing_parameters():
    """Test that we document the correct preprocessing parameters."""
    model_name = "google/siglip2-base-patch16-224"

    # Get HF image processor
    img_proc = SiglipImageProcessor.from_pretrained(model_name)

    # Verify preprocessing parameters
    assert img_proc.image_mean == [0.5, 0.5, 0.5], "Image mean should be [0.5, 0.5, 0.5]"
    assert img_proc.image_std == [0.5, 0.5, 0.5], "Image std should be [0.5, 0.5, 0.5]"
    assert img_proc.size == {"height": 224, "width": 224}, "Image size should be 224x224"
    assert img_proc.do_resize is True, "Should resize images"
    assert img_proc.do_rescale is True, "Should rescale images"
    assert img_proc.do_normalize is True, "Should normalize images"
    assert abs(img_proc.rescale_factor - 1.0 / 255.0) < 1e-6, "Rescale factor should be 1/255"


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not available")
def test_kornia_preprocessor():
    """Test that Kornia preprocessor works correctly."""
    import torch

    from kornia.models.siglip2 import SigLip2ImagePreprocessor

    # Create preprocessor
    preprocessor = SigLip2ImagePreprocessor(image_size=(224, 224))

    # Test with single image [0, 255] range
    image = torch.randint(0, 255, (3, 300, 400), dtype=torch.float32)
    processed = preprocessor(image)

    assert processed.shape == (1, 3, 224, 224)
    assert processed.min() >= -1.5  # Normalized range
    assert processed.max() <= 1.5

    # Test with batch
    batch = torch.randint(0, 255, (2, 3, 300, 400), dtype=torch.float32)
    processed_batch = preprocessor(batch)

    assert processed_batch.shape == (2, 3, 224, 224)

    # Test with [0, 1] range (auto-converted)
    image_01 = torch.rand(3, 300, 400)
    processed_01 = preprocessor(image_01)

    assert processed_01.shape == (1, 3, 224, 224)
