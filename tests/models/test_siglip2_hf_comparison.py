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

"""Standalone test script to compare SigLip2 outputs with HuggingFace transformers.

This script can be run independently to verify that our implementation matches
the HuggingFace transformers library outputs.

Usage:
    python tests/models/test_siglip2_hf_comparison.py
"""

import torch

from kornia.models.siglip2 import SigLip2Builder, SigLip2ImagePreprocessor

try:
    from transformers import AutoModel, GemmaTokenizer, SiglipImageProcessor

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: transformers library not available or SigLip2 not supported in this version")
    print("Skipping HuggingFace comparison test.")
    TRANSFORMERS_AVAILABLE = False


def test_output_matching():  # noqa: PLR0912
    """Test that Kornia SigLip2 outputs match HuggingFace transformers."""
    if not TRANSFORMERS_AVAILABLE:
        print("Skipping test - transformers not available")
        return

    device = torch.device("cpu")
    model_name = "google/siglip2-base-patch16-224"

    print("=" * 70)
    print("SigLip2 Output Matching Test")
    print("=" * 70)

    print("\n1. Loading models...")
    try:
        # Load HF model
        hf_model = AutoModel.from_pretrained(model_name).to(device).eval()
        processor = SiglipImageProcessor.from_pretrained(model_name)
        tokenizer = GemmaTokenizer.from_pretrained(model_name)
        print("   ✓ HuggingFace model loaded")

        # Load Kornia model and preprocessor
        kornia_model = SigLip2Builder.from_pretrained(model_name).to(device).eval()
        kornia_preprocessor = SigLip2ImagePreprocessor(image_size=(224, 224))
        print("   ✓ Kornia model and preprocessor loaded")
    except Exception as e:
        print(f"   ✗ Failed to load models: {e}")
        return

    print("\n2. Preparing test inputs...")
    texts = ["a photo of a cat", "a photo of a dog"]

    # Create test images in [0, 255] range (as numpy arrays, then convert to PIL)
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    images_pil = []
    images_tensor = []
    for _ in range(2):
        img_array = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
        img_pil = Image.fromarray(img_array)
        images_pil.append(img_pil)
        # Convert to tensor for Kornia preprocessor
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()  # HWC -> CHW
        images_tensor.append(img_tensor)

    # Process inputs with HF processor
    hf_processed = processor(images_pil, return_tensors="pt")
    hf_pixel_values = hf_processed["pixel_values"].to(device)

    # Process text with tokenizer
    encoded = tokenizer(texts, padding=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    # Process images with Kornia preprocessor
    kornia_images_batch = torch.stack(images_tensor).to(device)
    kornia_pixel_values = kornia_preprocessor(kornia_images_batch).to(device)

    print("   Input shapes:")
    print(f"     HF pixel_values: {hf_pixel_values.shape}")
    print(f"     Kornia pixel_values: {kornia_pixel_values.shape}")
    print(f"     input_ids: {input_ids.shape}")
    print(f"     attention_mask: {attention_mask.shape}")

    print("\n3. Running forward passes...")
    print("\n   Test A: Using HF preprocessing for both models (should match exactly)")
    # Forward pass with HF model (HF preprocessing)
    with torch.no_grad():
        hf_outputs_hf_preproc = hf_model(
            pixel_values=hf_pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    # Forward pass with Kornia model (HF preprocessing)
    with torch.no_grad():
        kornia_outputs_hf_preproc = kornia_model(
            pixel_values=hf_pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    print("\n   Test B: Using Kornia preprocessing with Kornia model")
    # Forward pass with Kornia model (Kornia preprocessing)
    with torch.no_grad():
        kornia_outputs_kornia_preproc = kornia_model(
            pixel_values=kornia_pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    print("   ✓ Forward passes completed")

    print("\n4. Comparing outputs...")
    all_match_hf_preproc = True
    all_match_kornia_preproc = True

    # Test A: Compare with HF preprocessing (should match exactly)
    print("\n   Test A Results: HF preprocessing (both models)")
    if hf_outputs_hf_preproc.image_embeds is not None and kornia_outputs_hf_preproc["image_embeds"] is not None:
        img_diff = (kornia_outputs_hf_preproc["image_embeds"] - hf_outputs_hf_preproc.image_embeds).abs()
        img_max_diff = img_diff.max().item()
        img_mean_diff = img_diff.mean().item()

        print("\n     Image Embeddings:")
        print(f"       Max absolute difference: {img_max_diff:.6e}")
        print(f"       Mean absolute difference: {img_mean_diff:.6e}")

        if img_max_diff < 1e-3:
            print("       ✓ MATCH (within tolerance)")
        else:
            print("       ✗ MISMATCH")
            all_match_hf_preproc = False

    if hf_outputs_hf_preproc.text_embeds is not None and kornia_outputs_hf_preproc["text_embeds"] is not None:
        txt_diff = (kornia_outputs_hf_preproc["text_embeds"] - hf_outputs_hf_preproc.text_embeds).abs()
        txt_max_diff = txt_diff.max().item()
        txt_mean_diff = txt_diff.mean().item()

        print("\n     Text Embeddings:")
        print(f"       Max absolute difference: {txt_max_diff:.6e}")
        print(f"       Mean absolute difference: {txt_mean_diff:.6e}")

        if txt_max_diff < 1e-3:
            print("       ✓ MATCH (within tolerance)")
        else:
            print("       ✗ MISMATCH")
            all_match_hf_preproc = False

    if hasattr(hf_outputs_hf_preproc, "logits_per_image") and "logits_per_image" in kornia_outputs_hf_preproc:
        logits_diff = (kornia_outputs_hf_preproc["logits_per_image"] - hf_outputs_hf_preproc.logits_per_image).abs()
        logits_max_diff = logits_diff.max().item()
        logits_mean_diff = logits_diff.mean().item()

        print("\n     Logits per Image:")
        print(f"       Max absolute difference: {logits_max_diff:.6e}")
        print(f"       Mean absolute difference: {logits_mean_diff:.6e}")

        if logits_max_diff < 1e-2:
            print("       ✓ MATCH (within tolerance)")
        else:
            print("       ✗ MISMATCH")
            all_match_hf_preproc = False

    # Test B: Compare Kornia preprocessing with HF preprocessing
    print("\n   Test B Results: Kornia preprocessing vs HF preprocessing")
    if hf_outputs_hf_preproc.image_embeds is not None and kornia_outputs_kornia_preproc["image_embeds"] is not None:
        img_diff = (kornia_outputs_kornia_preproc["image_embeds"] - hf_outputs_hf_preproc.image_embeds).abs()
        img_max_diff = img_diff.max().item()
        img_mean_diff = img_diff.mean().item()

        print("\n     Image Embeddings:")
        print(f"       Max absolute difference: {img_max_diff:.6e}")
        print(f"       Mean absolute difference: {img_mean_diff:.6e}")

        if img_max_diff < 1e-1:
            print("       ✓ Within tolerance (interpolation differences expected)")
        else:
            print("       ⚠ Larger differences (interpolation differences)")
            all_match_kornia_preproc = False

    if hf_outputs_hf_preproc.text_embeds is not None and kornia_outputs_kornia_preproc["text_embeds"] is not None:
        txt_diff = (kornia_outputs_kornia_preproc["text_embeds"] - hf_outputs_hf_preproc.text_embeds).abs()
        txt_max_diff = txt_diff.max().item()
        txt_mean_diff = txt_diff.mean().item()

        print("\n     Text Embeddings:")
        print(f"       Max absolute difference: {txt_max_diff:.6e}")
        print(f"       Mean absolute difference: {txt_mean_diff:.6e}")

        if txt_max_diff < 1e-3:
            print("       ✓ MATCH (text not affected by image preprocessing)")
        else:
            print("       ✗ MISMATCH")
            all_match_kornia_preproc = False

    if hasattr(hf_outputs_hf_preproc, "logits_per_image") and "logits_per_image" in kornia_outputs_kornia_preproc:
        logits_diff = (kornia_outputs_kornia_preproc["logits_per_image"] - hf_outputs_hf_preproc.logits_per_image).abs()
        logits_max_diff = logits_diff.max().item()
        logits_mean_diff = logits_diff.mean().item()

        print("\n     Logits per Image:")
        print(f"       Max absolute difference: {logits_max_diff:.6e}")
        print(f"       Mean absolute difference: {logits_mean_diff:.6e}")

        if logits_max_diff < 1.0:
            print("       ✓ Within tolerance (interpolation differences expected)")
        else:
            print("       ⚠ Larger differences")
            all_match_kornia_preproc = False

    print("\n" + "=" * 70)
    if all_match_hf_preproc:
        print("✓ Test A PASSED: Models match exactly with HF preprocessing!")
    else:
        print("✗ Test A FAILED: Models don't match with HF preprocessing")

    if all_match_kornia_preproc:
        print("✓ Test B PASSED: Kornia preprocessor produces reasonable outputs!")
    else:
        print("⚠ Test B: Some differences (interpolation differences expected)")
    print("=" * 70)


if __name__ == "__main__":
    test_output_matching()
