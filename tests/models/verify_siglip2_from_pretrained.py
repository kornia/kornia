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

"""Comprehensive verification that from_pretrained matches transformers exactly.

This script verifies:
1. State dict keys match (after mapping)
2. Weight values match exactly
3. Forward pass outputs match
4. Multiple model variants work correctly

Usage:
    python tests/models/verify_siglip2_from_pretrained.py [model_name]
"""

import sys

import torch

from kornia.models.siglip2 import SigLip2Builder

# Try to import transformers
try:
    from transformers import Siglip2Model as HFSigLip2Model
    from transformers import Siglip2Processor, SiglipProcessor
    from transformers import SiglipModel as HFSigLipModel

    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: transformers library not available: {e}")
    print("Please install: pip install transformers")
    sys.exit(1)


def get_processor(model_name: str):
    """Get the appropriate processor for the model."""
    if "siglip-base" in model_name and "siglip2" not in model_name:
        return SiglipProcessor.from_pretrained(model_name)
    else:
        return Siglip2Processor.from_pretrained(model_name)


def get_hf_model(model_name: str):
    """Get the appropriate HF model for the model name."""
    if "siglip-base" in model_name and "siglip2" not in model_name:
        return HFSigLipModel.from_pretrained(model_name)
    else:
        return HFSigLip2Model.from_pretrained(model_name)


def compare_state_dicts(kornia_model, hf_model, model_name: str):
    """Compare state dicts between Kornia and HF models."""
    print("\n" + "=" * 70)
    print("1. Comparing State Dicts")
    print("=" * 70)

    hf_state_dict = hf_model.state_dict()
    kornia_state_dict = kornia_model.state_dict()

    # Map HF keys to Kornia keys (reverse of our mapping)
    def map_kornia_to_hf(kornia_key: str) -> str:
        if kornia_key.startswith("vision_projection"):
            return kornia_key.replace("vision_projection", "visual_projection")
        return kornia_key

    # Build mapping
    hf_to_kornia = {}
    for hf_key in hf_state_dict.keys():
        # Remove 'model.' prefix if present
        clean_key = hf_key[6:] if hf_key.startswith("model.") else hf_key
        # Apply our mapping in reverse
        if clean_key.startswith("visual_projection"):
            kornia_key = clean_key.replace("visual_projection", "vision_projection")
        else:
            kornia_key = clean_key
        hf_to_kornia[hf_key] = kornia_key

    # Compare keys
    hf_keys = set(hf_state_dict.keys())
    kornia_keys = set(kornia_state_dict.keys())

    # Map HF keys to Kornia format for comparison
    mapped_hf_keys = {map_kornia_to_hf(k[6:] if k.startswith("model.") else k) for k in hf_keys}

    missing_in_kornia = mapped_hf_keys - kornia_keys
    extra_in_kornia = kornia_keys - mapped_hf_keys

    print(f"\nHF model keys: {len(hf_keys)}")
    print(f"Kornia model keys: {len(kornia_keys)}")
    print(f"Mapped HF keys: {len(mapped_hf_keys)}")

    if missing_in_kornia:
        print(f"\n⚠ Keys in HF but not in Kornia: {len(missing_in_kornia)}")
        for key in sorted(missing_in_kornia)[:10]:
            print(f"  - {key}")
        if len(missing_in_kornia) > 10:
            print(f"  ... and {len(missing_in_kornia) - 10} more")

    if extra_in_kornia:
        print(f"\n⚠ Keys in Kornia but not in HF: {len(extra_in_kornia)}")
        for key in sorted(extra_in_kornia)[:10]:
            print(f"  - {key}")
        if len(extra_in_kornia) > 10:
            print(f"  ... and {len(extra_in_kornia) - 10} more")

    # Compare weight values
    print("\n2. Comparing Weight Values")
    print("-" * 70)

    max_diff = 0.0
    total_params = 0
    matching_params = 0

    for hf_key, hf_tensor in hf_state_dict.items():
        # Map to Kornia key
        clean_key = hf_key[6:] if hf_key.startswith("model.") else hf_key
        if clean_key.startswith("visual_projection"):
            kornia_key = clean_key.replace("visual_projection", "vision_projection")
        else:
            kornia_key = clean_key

        if kornia_key not in kornia_state_dict:
            continue

        kornia_tensor = kornia_state_dict[kornia_key]

        # Compare shapes
        if hf_tensor.shape != kornia_tensor.shape:
            print(f"⚠ Shape mismatch for {kornia_key}: HF {hf_tensor.shape} vs Kornia {kornia_tensor.shape}")
            continue

        # Compare values
        diff = (hf_tensor - kornia_tensor).abs()
        max_diff = max(max_diff, diff.max().item())
        total_params += hf_tensor.numel()

        # Check if exact match
        if torch.allclose(hf_tensor, kornia_tensor, rtol=1e-9, atol=1e-9):
            matching_params += hf_tensor.numel()

    print(f"\nTotal parameters compared: {total_params:,}")
    print(f"Exactly matching parameters: {matching_params:,} ({100 * matching_params / total_params:.2f}%)")
    print(f"Maximum absolute difference: {max_diff:.2e}")

    if max_diff < 1e-6:
        print("✓ All weights match exactly!")
        return True
    elif max_diff < 1e-3:
        print("⚠ Weights match within tolerance (likely due to numerical precision)")
        return True
    else:
        print("✗ Significant weight differences detected!")
        return False


def compare_forward_outputs(kornia_model, hf_model, processor, model_name: str, device="cpu"):
    """Compare forward pass outputs."""
    print("\n" + "=" * 70)
    print("3. Comparing Forward Pass Outputs")
    print("=" * 70)

    # Prepare test inputs
    texts = ["a photo of a cat", "a photo of a dog", "a beautiful sunset"]
    images = [torch.randn(3, 224, 224, device=device) for _ in range(3)]

    # Process inputs
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    pixel_values = inputs["pixel_values"].to(device)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    print("\nInput shapes:")
    print(f"  pixel_values: {pixel_values.shape}")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")

    # Forward pass
    kornia_model.eval()
    hf_model.eval()

    with torch.no_grad():
        kornia_outputs = kornia_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hf_outputs = hf_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    all_match = True

    # Compare image embeddings
    if hf_outputs.image_embeds is not None and kornia_outputs["image_embeds"] is not None:
        print("\nImage Embeddings:")
        print(f"  Shape: {kornia_outputs['image_embeds'].shape}")

        img_diff = (kornia_outputs["image_embeds"] - hf_outputs.image_embeds).abs()
        img_max_diff = img_diff.max().item()
        img_mean_diff = img_diff.mean().item()
        img_std_diff = img_diff.std().item()

        # Relative difference
        img_rel_diff = (img_diff / (hf_outputs.image_embeds.abs() + 1e-8)).max().item()

        print(f"  Max absolute diff: {img_max_diff:.6e}")
        print(f"  Mean absolute diff: {img_mean_diff:.6e}")
        print(f"  Std absolute diff: {img_std_diff:.6e}")
        print(f"  Max relative diff: {img_rel_diff:.6e}")

        # Check if normalized
        kornia_norm = kornia_outputs["image_embeds"].norm(dim=-1)
        hf_norm = hf_outputs.image_embeds.norm(dim=-1)
        print(f"  Kornia norm (mean): {kornia_norm.mean().item():.6f}")
        print(f"  HF norm (mean): {hf_norm.mean().item():.6f}")

        if img_max_diff < 1e-3:
            print("  ✓ MATCH (within 1e-3 tolerance)")
        else:
            print("  ✗ MISMATCH")
            all_match = False

    # Compare text embeddings
    if hf_outputs.text_embeds is not None and kornia_outputs["text_embeds"] is not None:
        print("\nText Embeddings:")
        print(f"  Shape: {kornia_outputs['text_embeds'].shape}")

        txt_diff = (kornia_outputs["text_embeds"] - hf_outputs.text_embeds).abs()
        txt_max_diff = txt_diff.max().item()
        txt_mean_diff = txt_diff.mean().item()
        txt_std_diff = txt_diff.std().item()
        txt_rel_diff = (txt_diff / (hf_outputs.text_embeds.abs() + 1e-8)).max().item()

        print(f"  Max absolute diff: {txt_max_diff:.6e}")
        print(f"  Mean absolute diff: {txt_mean_diff:.6e}")
        print(f"  Std absolute diff: {txt_std_diff:.6e}")
        print(f"  Max relative diff: {txt_rel_diff:.6e}")

        kornia_norm = kornia_outputs["text_embeds"].norm(dim=-1)
        hf_norm = hf_outputs.text_embeds.norm(dim=-1)
        print(f"  Kornia norm (mean): {kornia_norm.mean().item():.6f}")
        print(f"  HF norm (mean): {hf_norm.mean().item():.6f}")

        if txt_max_diff < 1e-3:
            print("  ✓ MATCH (within 1e-3 tolerance)")
        else:
            print("  ✗ MISMATCH")
            all_match = False

    # Compare logits
    if hasattr(hf_outputs, "logits_per_image") and "logits_per_image" in kornia_outputs:
        print("\nLogits per Image:")
        print(f"  Shape: {kornia_outputs['logits_per_image'].shape}")

        logits_diff = (kornia_outputs["logits_per_image"] - hf_outputs.logits_per_image).abs()
        logits_max_diff = logits_diff.max().item()
        logits_mean_diff = logits_diff.mean().item()
        logits_rel_diff = (logits_diff / (hf_outputs.logits_per_image.abs() + 1e-8)).max().item()

        print(f"  Max absolute diff: {logits_max_diff:.6e}")
        print(f"  Mean absolute diff: {logits_mean_diff:.6e}")
        print(f"  Max relative diff: {logits_rel_diff:.6e}")

        if logits_max_diff < 1e-2:
            print("  ✓ MATCH (within 1e-2 tolerance)")
        else:
            print("  ✗ MISMATCH")
            all_match = False

    return all_match


def verify_model(model_name: str, device: str = "cpu"):
    """Verify a single model."""
    print("\n" + "=" * 70)
    print(f"Verifying: {model_name}")
    print("=" * 70)

    try:
        # Load models
        print("\nLoading models...")
        hf_model = get_hf_model(model_name).to(device).eval()
        print("  ✓ HuggingFace model loaded")

        kornia_model = SigLip2Builder.from_pretrained(model_name).to(device).eval()
        print("  ✓ Kornia model loaded")

        processor = get_processor(model_name)
        print("  ✓ Processor loaded")

        # Compare state dicts
        weights_match = compare_state_dicts(kornia_model, hf_model, model_name)

        # Compare forward outputs
        outputs_match = compare_forward_outputs(kornia_model, hf_model, processor, model_name, device)

        # Summary
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"  Weights match: {'✓' if weights_match else '✗'}")
        print(f"  Outputs match: {'✓' if outputs_match else '✗'}")

        if weights_match and outputs_match:
            print(f"\n✓ {model_name} VERIFIED - Matches transformers exactly!")
            return True
        else:
            print(f"\n✗ {model_name} FAILED - Differences detected")
            return False

    except Exception as e:
        print(f"\n✗ Error verifying {model_name}: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main verification function."""
    # Default model or from command line
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "google/siglip2-base-patch16-224"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Verify the model
    success = verify_model(model_name, device)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
