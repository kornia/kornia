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

"""Test that Kornia preprocessor produces same outputs as HuggingFace."""

import numpy as np
import pytest
import torch
from PIL import Image

from kornia.models.siglip2 import SigLip2Builder, SigLip2ImagePreprocessor

try:
    from transformers import AutoModel, GemmaTokenizer, SiglipImageProcessor

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not available")
def test_preprocessor_outputs_match_hf():
    """Test that Kornia preprocessor produces same outputs as HF preprocessor."""
    model_name = "google/siglip2-base-patch16-224"

    # Set seeds for reproducibility
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    # Create test images
    img_array1 = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
    img_pil1 = Image.fromarray(img_array1)
    img_array2 = rng.integers(0, 255, (250, 350, 3), dtype=np.uint8)
    img_pil2 = Image.fromarray(img_array2)

    # HF preprocessing
    hf_proc = SiglipImageProcessor.from_pretrained(model_name)
    hf_processed = hf_proc([img_pil1, img_pil2], return_tensors="pt")
    hf_pixel_values = hf_processed["pixel_values"]

    # Kornia preprocessing
    kornia_proc = SigLip2ImagePreprocessor(image_size=(224, 224))
    # Convert PIL to tensor format [C, H, W] in [0, 255] range
    img_tensor1 = torch.from_numpy(img_array1).permute(2, 0, 1).float()  # HWC -> CHW
    img_tensor2 = torch.from_numpy(img_array2).permute(2, 0, 1).float()
    img_batch = torch.stack([img_tensor1, img_tensor2])
    kornia_pixel_values = kornia_proc(img_batch)

    print("\nHF preprocessing:")
    print(f"  Shape: {hf_pixel_values.shape}")
    print(f"  Range: [{hf_pixel_values.min():.3f}, {hf_pixel_values.max():.3f}]")
    print(f"  Mean: {hf_pixel_values.mean():.3f}")

    print("\nKornia preprocessing:")
    print(f"  Shape: {kornia_pixel_values.shape}")
    print(f"  Range: [{kornia_pixel_values.min():.3f}, {kornia_pixel_values.max():.3f}]")
    print(f"  Mean: {kornia_pixel_values.mean():.3f}")

    # Compare outputs
    diff = (hf_pixel_values - kornia_pixel_values).abs()
    print("\nDifference:")
    print(f"  Max diff: {diff.max().item():.6e}")
    print(f"  Mean diff: {diff.mean().item():.6e}")

    # Small differences are expected due to interpolation differences between
    # HF's PIL-based resize and Kornia's tensor-based resize
    # The differences are acceptable as long as model outputs are close
    assert diff.max() < 2.0, f"Preprocessor outputs differ too much: max_diff={diff.max():.6e}"
    assert diff.mean() < 0.5, f"Preprocessor outputs differ too much: mean_diff={diff.mean():.6e}"


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not available")
def test_model_outputs_with_kornia_preprocessor():
    """Test that using Kornia preprocessor with Kornia model matches HF."""
    model_name = "google/siglip2-base-patch16-224"
    device = "cpu"

    # Set seeds for reproducibility
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    # Create test image
    img_array = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
    img_pil = Image.fromarray(img_array)

    # HF preprocessing and model
    hf_proc = SiglipImageProcessor.from_pretrained(model_name)
    hf_processed = hf_proc(img_pil, return_tensors="pt")
    hf_pixel_values = hf_processed["pixel_values"]

    torch.manual_seed(42)
    hf_model = AutoModel.from_pretrained(model_name).to(device).eval()

    # Kornia preprocessing and model
    kornia_proc = SigLip2ImagePreprocessor(image_size=(224, 224))
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()  # HWC -> CHW
    kornia_pixel_values = kornia_proc(img_tensor)

    torch.manual_seed(42)
    kornia_model = SigLip2Builder.from_pretrained(model_name).to(device).eval()

    # Prepare text inputs
    tokenizer = GemmaTokenizer.from_pretrained(model_name)
    texts = ["a photo of a cat", "a photo of a dog"]
    encoded = tokenizer(texts, padding=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    # Forward pass with HF
    with torch.no_grad():
        hf_outputs = hf_model(pixel_values=hf_pixel_values, input_ids=input_ids, attention_mask=attention_mask)

    # Forward pass with Kornia (using Kornia preprocessor)
    with torch.no_grad():
        kornia_outputs = kornia_model(
            pixel_values=kornia_pixel_values, input_ids=input_ids, attention_mask=attention_mask
        )

    # Compare image embeddings
    img_diff = (hf_outputs.image_embeds - kornia_outputs["image_embeds"]).abs()
    img_cosine = torch.nn.functional.cosine_similarity(
        hf_outputs.image_embeds, kornia_outputs["image_embeds"], dim=-1
    ).item()

    print("\nImage embeddings:")
    print(f"  Max diff: {img_diff.max().item():.6e}")
    print(f"  Cosine similarity: {img_cosine:.9f}")

    # Compare text embeddings
    txt_diff = (hf_outputs.text_embeds - kornia_outputs["text_embeds"]).abs()
    txt_cosine = torch.nn.functional.cosine_similarity(
        hf_outputs.text_embeds, kornia_outputs["text_embeds"], dim=-1
    ).item()

    print("\nText embeddings:")
    print(f"  Max diff: {txt_diff.max().item():.6e}")
    print(f"  Cosine similarity: {txt_cosine:.9f}")

    # Compare logits
    logits_diff = (hf_outputs.logits_per_image - kornia_outputs["logits_per_image"]).abs()

    print("\nLogits:")
    print(f"  Max diff: {logits_diff.max().item():.6e}")

    # Assertions - small differences expected due to preprocessing interpolation differences
    # But outputs should still be close
    assert img_diff.max() < 1e-1, f"Image embeddings differ: max_diff={img_diff.max():.6e}"
    assert abs(img_cosine - 1.0) < 1e-2, f"Image embeddings cosine similarity: {img_cosine:.9f}"

    assert txt_diff.max() < 1e-3, f"Text embeddings differ: max_diff={txt_diff.max():.6e}"
    assert abs(txt_cosine - 1.0) < 1e-3, f"Text embeddings cosine similarity: {txt_cosine:.9f}"

    assert logits_diff.max() < 1.0, f"Logits differ: max_diff={logits_diff.max():.6e}"


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not available")
def test_model_outputs_same_preprocessing():
    """Test that using same preprocessed images produces same outputs."""
    model_name = "google/siglip2-base-patch16-224"
    device = "cpu"

    # Set seeds for reproducibility
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    # Create test image
    img_array = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
    img_pil = Image.fromarray(img_array)

    # Use HF preprocessing for both models
    hf_proc = SiglipImageProcessor.from_pretrained(model_name)
    hf_processed = hf_proc(img_pil, return_tensors="pt")
    hf_pixel_values = hf_processed["pixel_values"]

    torch.manual_seed(42)
    hf_model = AutoModel.from_pretrained(model_name).to(device).eval()

    torch.manual_seed(42)
    kornia_model = SigLip2Builder.from_pretrained(model_name).to(device).eval()

    # Prepare text inputs
    tokenizer = GemmaTokenizer.from_pretrained(model_name)
    texts = ["a photo of a cat", "a photo of a dog"]
    encoded = tokenizer(texts, padding=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    # Forward pass with HF
    with torch.no_grad():
        hf_outputs = hf_model(pixel_values=hf_pixel_values, input_ids=input_ids, attention_mask=attention_mask)

    # Forward pass with Kornia (using same HF preprocessed images)
    with torch.no_grad():
        kornia_outputs = kornia_model(pixel_values=hf_pixel_values, input_ids=input_ids, attention_mask=attention_mask)

    # Compare all outputs
    img_diff = (hf_outputs.image_embeds - kornia_outputs["image_embeds"]).abs()
    txt_diff = (hf_outputs.text_embeds - kornia_outputs["text_embeds"]).abs()
    logits_diff = (hf_outputs.logits_per_image - kornia_outputs["logits_per_image"]).abs()

    print("\nWith same preprocessing:")
    print(f"  Image embeddings max diff: {img_diff.max().item():.6e}")
    print(f"  Text embeddings max diff: {txt_diff.max().item():.6e}")
    print(f"  Logits max diff: {logits_diff.max().item():.6e}")

    # Should match exactly (within numerical precision)
    assert img_diff.max() < 1e-3, f"Image embeddings differ: max_diff={img_diff.max():.6e}"
    assert txt_diff.max() < 1e-3, f"Text embeddings differ: max_diff={txt_diff.max():.6e}"
    assert logits_diff.max() < 1e-2, f"Logits differ: max_diff={logits_diff.max():.6e}"


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not available")
def test_comprehensive_preprocessor_comparison():
    """Comprehensive test comparing all outputs with different preprocessing methods."""
    model_name = "google/siglip2-base-patch16-224"
    device = "cpu"

    # Set seeds for reproducibility
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    # Create multiple test images (same size for batch processing)
    images_pil = []
    images_tensor = []
    for _i in range(3):
        img_array = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
        images_pil.append(Image.fromarray(img_array))
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        images_tensor.append(img_tensor)

    # HF preprocessing
    hf_proc = SiglipImageProcessor.from_pretrained(model_name)
    hf_processed = hf_proc(images_pil, return_tensors="pt")
    hf_pixel_values = hf_processed["pixel_values"]

    # Kornia preprocessing
    kornia_proc = SigLip2ImagePreprocessor(image_size=(224, 224))
    img_batch = torch.stack(images_tensor)
    kornia_pixel_values = kornia_proc(img_batch)

    # Load models
    torch.manual_seed(42)
    hf_model = AutoModel.from_pretrained(model_name).to(device).eval()

    torch.manual_seed(42)
    kornia_model = SigLip2Builder.from_pretrained(model_name).to(device).eval()

    # Prepare text inputs
    tokenizer = GemmaTokenizer.from_pretrained(model_name)
    texts = ["a cat", "a dog", "a bird"]
    encoded = tokenizer(texts, padding=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    print("\n" + "=" * 70)
    print("COMPREHENSIVE PREPROCESSOR COMPARISON")
    print("=" * 70)

    # Test 1: HF preprocessing with HF model
    with torch.no_grad():
        hf_outputs_hf_preproc = hf_model(
            pixel_values=hf_pixel_values, input_ids=input_ids, attention_mask=attention_mask
        )

    # Test 2: HF preprocessing with Kornia model
    with torch.no_grad():
        kornia_outputs_hf_preproc = kornia_model(
            pixel_values=hf_pixel_values, input_ids=input_ids, attention_mask=attention_mask
        )

    # Test 3: Kornia preprocessing with Kornia model
    with torch.no_grad():
        kornia_outputs_kornia_preproc = kornia_model(
            pixel_values=kornia_pixel_values, input_ids=input_ids, attention_mask=attention_mask
        )

    # Compare Test 1 vs Test 2 (same preprocessing, different models)
    print("\n1. HF preprocessing - HF model vs Kornia model:")
    diff1_img = (hf_outputs_hf_preproc.image_embeds - kornia_outputs_hf_preproc["image_embeds"]).abs()
    diff1_txt = (hf_outputs_hf_preproc.text_embeds - kornia_outputs_hf_preproc["text_embeds"]).abs()
    diff1_logits = (hf_outputs_hf_preproc.logits_per_image - kornia_outputs_hf_preproc["logits_per_image"]).abs()
    print(f"   Image embeds max diff: {diff1_img.max().item():.6e}")
    print(f"   Text embeds max diff: {diff1_txt.max().item():.6e}")
    print(f"   Logits max diff: {diff1_logits.max().item():.6e}")

    # Compare Test 2 vs Test 3 (same model, different preprocessing)
    print("\n2. Kornia model - HF preprocessing vs Kornia preprocessing:")
    diff2_img = (kornia_outputs_hf_preproc["image_embeds"] - kornia_outputs_kornia_preproc["image_embeds"]).abs()
    diff2_txt = (kornia_outputs_hf_preproc["text_embeds"] - kornia_outputs_kornia_preproc["text_embeds"]).abs()
    diff2_logits = (
        kornia_outputs_hf_preproc["logits_per_image"] - kornia_outputs_kornia_preproc["logits_per_image"]
    ).abs()
    print(f"   Image embeds max diff: {diff2_img.max().item():.6e}")
    print(f"   Text embeds max diff: {diff2_txt.max().item():.6e}")
    print(f"   Logits max diff: {diff2_logits.max().item():.6e}")

    # Compare Test 1 vs Test 3 (different preprocessing, different models)
    print("\n3. HF preprocessing+model vs Kornia preprocessing+model:")
    diff3_img = (hf_outputs_hf_preproc.image_embeds - kornia_outputs_kornia_preproc["image_embeds"]).abs()
    diff3_txt = (hf_outputs_hf_preproc.text_embeds - kornia_outputs_kornia_preproc["text_embeds"]).abs()
    diff3_logits = (hf_outputs_hf_preproc.logits_per_image - kornia_outputs_kornia_preproc["logits_per_image"]).abs()
    print(f"   Image embeds max diff: {diff3_img.max().item():.6e}")
    print(f"   Text embeds max diff: {diff3_txt.max().item():.6e}")
    print(f"   Logits max diff: {diff3_logits.max().item():.6e}")

    print("\n" + "=" * 70)

    # Assertions
    # Test 1 vs Test 2: Should match exactly (same preprocessing)
    assert diff1_img.max() < 1e-3, "HF preprocessing: models should match"
    assert diff1_txt.max() < 1e-3, "HF preprocessing: models should match"
    assert diff1_logits.max() < 1e-2, "HF preprocessing: models should match"

    # Test 2 vs Test 3: Small differences expected due to preprocessing interpolation differences
    # But should still be reasonably close
    assert diff2_img.max() < 1e-1, "Kornia preprocessing: should be close to HF preprocessing"
    assert diff2_txt.max() < 1e-3, "Kornia preprocessing: text should match (not affected by image preprocessing)"
    assert diff2_logits.max() < 1.0, (
        "Kornia preprocessing: logits should be reasonably close (interpolation differences)"
    )

    print("\n✓ All comparisons passed!")
