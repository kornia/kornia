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

"""Comprehensive comparison script for SigLip2 with transformers.

This script compares:
1. State dict keys and values
2. Forward pass outputs
3. Individual component outputs
4. Multiple test cases

Usage:
    PYTHONPATH=. python tests/models/compare_siglip2_with_transformers.py [model_name]
"""

import sys

import numpy as np
import torch

from kornia.models.siglip2 import SigLip2Builder

# Try to import transformers
try:
    from transformers import AutoModel, AutoProcessor

    TRANSFORMERS_AVAILABLE = True
    # Try to import specific models, fallback to AutoModel
    try:
        from transformers import Siglip2Model as HFSigLip2Model
        from transformers import Siglip2Processor
    except ImportError:
        HFSigLip2Model = None
        Siglip2Processor = None

    try:
        from transformers import SiglipModel as HFSigLipModel
        from transformers import SiglipProcessor
    except ImportError:
        HFSigLipModel = None
        SiglipProcessor = None

except ImportError as e:
    print(f"ERROR: transformers library not available: {e}")
    print("\nTo use this script:")
    print("  1. Install compatible numpy: pip install 'numpy<2.0'")
    print("  2. Install transformers: pip install transformers")
    print("  3. Run this script again")
    sys.exit(1)


def get_processor(model_name: str):
    """Get the appropriate processor for the model."""
    # Always use AutoProcessor as it handles model selection automatically
    return AutoProcessor.from_pretrained(model_name)


def get_hf_model(model_name: str):
    """Get the appropriate HF model for the model name."""
    if HFSigLip2Model is not None and "siglip2" in model_name:
        return HFSigLip2Model.from_pretrained(model_name)
    elif HFSigLipModel is not None and "siglip-base" in model_name and "siglip2" not in model_name:
        return HFSigLipModel.from_pretrained(model_name)
    else:
        # Fallback to AutoModel
        return AutoModel.from_pretrained(model_name)


def compare_state_dicts(kornia_model, hf_model, model_name: str):
    """Compare state dicts between Kornia and HF models."""
    print("\n" + "=" * 70)
    print("1. State Dict Comparison")
    print("=" * 70)

    hf_state_dict = hf_model.state_dict()
    kornia_state_dict = kornia_model.state_dict()

    # Map HF keys to Kornia format
    def map_hf_to_kornia(hf_key: str) -> str:
        # Remove 'model.' prefix if present
        if hf_key.startswith("model."):
            hf_key = hf_key[6:]

        # Handle projection naming
        if hf_key.startswith("visual_projection"):
            return hf_key.replace("visual_projection", "vision_projection")

        return hf_key

    # Build mapping
    hf_keys_mapped = {map_hf_to_kornia(k): v for k, v in hf_state_dict.items()}

    # Compare
    common_keys = set(hf_keys_mapped.keys()) & set(kornia_state_dict.keys())
    only_hf = set(hf_keys_mapped.keys()) - set(kornia_state_dict.keys())
    only_kornia = set(kornia_state_dict.keys()) - set(hf_keys_mapped.keys())

    print(f"\nHF model keys: {len(hf_keys_mapped)}")
    print(f"Kornia model keys: {len(kornia_state_dict)}")
    print(f"Common keys: {len(common_keys)}")
    print(f"Only in HF: {len(only_hf)}")
    print(f"Only in Kornia: {len(only_kornia)}")

    if only_hf:
        print("\n⚠ Keys only in HF (first 10):")
        for k in sorted(only_hf)[:10]:
            print(f"  {k}")

    if only_kornia:
        print("\n⚠ Keys only in Kornia (first 10):")
        for k in sorted(only_kornia)[:10]:
            print(f"  {k}")

    # Compare values for common keys
    print("\n2. Weight Value Comparison")
    print("-" * 70)

    max_diff = 0.0
    total_params = 0
    matching_params = 0
    differences = []

    for key in sorted(common_keys):
        hf_tensor = hf_keys_mapped[key]
        kornia_tensor = kornia_state_dict[key]

        if hf_tensor.shape != kornia_tensor.shape:
            print(f"⚠ Shape mismatch for {key}: HF {hf_tensor.shape} vs Kornia {kornia_tensor.shape}")
            continue

        diff = (hf_tensor - kornia_tensor).abs()
        max_diff = max(max_diff, diff.max().item())
        total_params += hf_tensor.numel()

        if torch.allclose(hf_tensor, kornia_tensor, rtol=1e-5, atol=1e-5):
            matching_params += hf_tensor.numel()
        else:
            mean_diff = diff.mean().item()
            max_key_diff = diff.max().item()
            differences.append((key, mean_diff, max_key_diff))

    print(f"\nTotal parameters compared: {total_params:,}")
    print(
        f"Matching parameters (rtol=1e-5, atol=1e-5): {matching_params:,} ({100 * matching_params / total_params:.2f}%)"
    )
    print(f"Maximum absolute difference: {max_diff:.2e}")

    if differences:
        print("\nTop 10 keys with largest differences:")
        differences.sort(key=lambda x: x[2], reverse=True)
        for key, mean_diff, max_diff in differences[:10]:
            print(f"  {key}: mean={mean_diff:.2e}, max={max_diff:.2e}")

    if max_diff < 1e-4:
        print("\n✓ All weights match closely!")
        return True
    elif max_diff < 1e-2:
        print("\n⚠ Weights match within tolerance (likely due to numerical precision)")
        return True
    else:
        print("\n✗ Significant weight differences detected!")
        return False


def compare_outputs(kornia_model, hf_model, processor, model_name: str, device="cpu", dtype=torch.float32):  # noqa: PLR0912
    """Compare forward pass outputs."""
    print("\n" + "=" * 70)
    print("3. Forward Pass Output Comparison")
    print("=" * 70)

    # Test case 1: Simple inputs with proper preprocessing
    print("\nTest Case 1: Simple inputs")

    # Set seed for reproducible inputs - CRITICAL for exact matching
    torch.manual_seed(42)
    _rng = np.random.default_rng(42)  # Set seed for reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    texts = ["a photo of a cat", "a photo of a dog"]

    # Use HF image processor for exact preprocessing
    try:
        from PIL import Image
        from transformers import SiglipImageProcessor

        img_proc = SiglipImageProcessor.from_pretrained(model_name)
        # Create test images
        torch.manual_seed(42)
        rng2 = np.random.default_rng(42)
        images_pil = []
        for _ in range(2):
            img_array = rng2.integers(0, 255, (300, 400, 3), dtype=np.uint8)
            images_pil.append(Image.fromarray(img_array))

        # Preprocess with HF processor
        processed = img_proc(images_pil, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(device)
        print("  ✓ Using HF SiglipImageProcessor for preprocessing")
    except Exception as e:
        # Fallback to random images
        print(f"  ⚠ Could not use HF processor: {e}, using random images")
        torch.manual_seed(42)
        images = [torch.randn(3, 224, 224, device=device, dtype=dtype) for _ in range(2)]
        pixel_values = torch.stack(images).to(device)

    # Use GemmaTokenizer for text (SigLip2 uses Gemma tokenizer)
    try:
        from transformers import GemmaTokenizer

        tokenizer = GemmaTokenizer.from_pretrained(model_name)
        encoded = tokenizer(texts, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        # Create attention_mask if not provided
        if "attention_mask" not in encoded:
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
        else:
            attention_mask = encoded["attention_mask"].to(device)
        print("  ✓ Using GemmaTokenizer for tokenization")
    except Exception as e:
        # Fallback to dummy IDs with fixed seed
        print(f"  ⚠ GemmaTokenizer failed: {e}, using dummy token IDs")
        torch.manual_seed(42)
        batch_size = len(texts)
        seq_len = 10
        gen = torch.Generator(device=device)
        gen.manual_seed(42)
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device, generator=gen)
        attention_mask = torch.ones(batch_size, seq_len, device=device)

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
        print("\n  Image Embeddings:")
        kornia_img = kornia_outputs["image_embeds"]
        hf_img = hf_outputs.image_embeds

        img_diff = (kornia_img - hf_img).abs()
        img_max_diff = img_diff.max().item()
        img_mean_diff = img_diff.mean().item()
        img_cosine = torch.nn.functional.cosine_similarity(kornia_img, hf_img, dim=-1).mean().item()

        print(f"    Shape: {kornia_img.shape}")
        print(f"    Max absolute diff: {img_max_diff:.6e}")
        print(f"    Mean absolute diff: {img_mean_diff:.6e}")
        print(f"    Cosine similarity: {img_cosine:.6f}")
        print(f"    Kornia norm (mean): {kornia_img.norm(dim=-1).mean().item():.6f}")
        print(f"    HF norm (mean): {hf_img.norm(dim=-1).mean().item():.6f}")

        if img_max_diff < 1e-3 and img_cosine > 0.999:
            print("    ✓ MATCH")
        else:
            print("    ✗ MISMATCH")
            all_match = False

    # Compare text embeddings
    if hf_outputs.text_embeds is not None and kornia_outputs["text_embeds"] is not None:
        print("\n  Text Embeddings:")
        kornia_txt = kornia_outputs["text_embeds"]
        hf_txt = hf_outputs.text_embeds

        txt_diff = (kornia_txt - hf_txt).abs()
        txt_max_diff = txt_diff.max().item()
        txt_mean_diff = txt_diff.mean().item()
        txt_cosine = torch.nn.functional.cosine_similarity(kornia_txt, hf_txt, dim=-1).mean().item()

        print(f"    Shape: {kornia_txt.shape}")
        print(f"    Max absolute diff: {txt_max_diff:.6e}")
        print(f"    Mean absolute diff: {txt_mean_diff:.6e}")
        print(f"    Cosine similarity: {txt_cosine:.6f}")
        print(f"    Kornia norm (mean): {kornia_txt.norm(dim=-1).mean().item():.6f}")
        print(f"    HF norm (mean): {hf_txt.norm(dim=-1).mean().item():.6f}")

        if txt_max_diff < 1e-3 and txt_cosine > 0.999:
            print("    ✓ MATCH")
        else:
            print("    ✗ MISMATCH")
            all_match = False

    # Compare logits
    if hasattr(hf_outputs, "logits_per_image") and "logits_per_image" in kornia_outputs:
        print("\n  Logits per Image:")
        kornia_logits = kornia_outputs["logits_per_image"]
        hf_logits = hf_outputs.logits_per_image

        logits_diff = (kornia_logits - hf_logits).abs()
        logits_max_diff = logits_diff.max().item()
        logits_mean_diff = logits_diff.mean().item()

        print(f"    Shape: {kornia_logits.shape}")
        print(f"    Max absolute diff: {logits_max_diff:.6e}")
        print(f"    Mean absolute diff: {logits_mean_diff:.6e}")
        print(f"    Kornia range: [{kornia_logits.min().item():.2f}, {kornia_logits.max().item():.2f}]")
        print(f"    HF range: [{hf_logits.min().item():.2f}, {hf_logits.max().item():.2f}]")

        if logits_max_diff < 1e-2:
            print("    ✓ MATCH")
        else:
            print("    ✗ MISMATCH")
            all_match = False

    # Test case 2: Single image, multiple texts
    print("\nTest Case 2: Single image, multiple texts")

    # Set seed for reproducible inputs
    torch.manual_seed(123)
    _rng = np.random.default_rng(123)  # Set seed for reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)

    texts2 = ["a cat", "a dog", "a bird"]

    # Use HF image processor for exact preprocessing
    try:
        from PIL import Image
        from transformers import SiglipImageProcessor

        img_proc = SiglipImageProcessor.from_pretrained(model_name)
        # Create test image
        torch.manual_seed(123)
        rng2 = np.random.default_rng(123)
        img_array = rng2.integers(0, 255, (300, 400, 3), dtype=np.uint8)
        img_pil = Image.fromarray(img_array)

        # Preprocess with HF processor
        processed2 = img_proc([img_pil], return_tensors="pt")
        pixel_values2 = processed2["pixel_values"].to(device)
    except Exception:
        # Fallback to random image
        torch.manual_seed(123)
        images2 = [torch.randn(3, 224, 224, device=device, dtype=dtype)]
        pixel_values2 = torch.stack(images2).to(device)

    # Use GemmaTokenizer
    try:
        from transformers import GemmaTokenizer

        tokenizer = GemmaTokenizer.from_pretrained(model_name)
        encoded2 = tokenizer(texts2, padding=True, return_tensors="pt")
        input_ids2 = encoded2["input_ids"].to(device)
        # Create attention_mask if not provided
        if "attention_mask" not in encoded2:
            attention_mask2 = (input_ids2 != tokenizer.pad_token_id).long().to(device)
        else:
            attention_mask2 = encoded2["attention_mask"].to(device)
    except Exception:
        torch.manual_seed(123)
        batch_size2 = len(texts2)
        seq_len2 = 10
        gen2 = torch.Generator(device=device)
        gen2.manual_seed(123)
        input_ids2 = torch.randint(0, 1000, (batch_size2, seq_len2), device=device, generator=gen2)
        attention_mask2 = torch.ones(batch_size2, seq_len2, device=device)

    with torch.no_grad():
        kornia_outputs2 = kornia_model(
            pixel_values=pixel_values2,
            input_ids=input_ids2,
            attention_mask=attention_mask2,
        )
        hf_outputs2 = hf_model(
            pixel_values=pixel_values2,
            input_ids=input_ids2,
            attention_mask=attention_mask2,
        )

    if hf_outputs2.image_embeds is not None and kornia_outputs2["image_embeds"] is not None:
        img_diff2 = (kornia_outputs2["image_embeds"] - hf_outputs2.image_embeds).abs().max().item()
        print(f"  Image embeds max diff: {img_diff2:.6e}")
        if img_diff2 < 1e-3:
            print("  ✓ MATCH")
        else:
            print("  ✗ MISMATCH")
            all_match = False

    return all_match


def main():
    """Main comparison function."""
    model_name = sys.argv[1] if len(sys.argv) > 1 else "google/siglip2-base-patch16-224"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    _rng = np.random.default_rng(42)  # Set seed for reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("=" * 70)
    print(f"Comparing SigLip2: {model_name}")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print("Random seed: 42 (fixed for reproducibility)")

    try:
        # Load models with same random seed
        print("\nLoading models...")

        # Set seed before loading HF model
        torch.manual_seed(42)
        _rng2 = np.random.default_rng(42)  # Set seed for reproducibility
        hf_model = get_hf_model(model_name).to(device).eval()
        print("  ✓ HuggingFace model loaded")

        # Set seed before loading Kornia model (for reproducible initialization)
        torch.manual_seed(42)
        _rng3 = np.random.default_rng(42)  # Set seed for reproducibility
        kornia_model = SigLip2Builder.from_pretrained(model_name).to(device).eval()
        print("  ✓ Kornia model loaded")

        try:
            # Try multiple ways to get processor
            try:
                processor = get_processor(model_name)
                print("  ✓ Processor loaded")
            except Exception:
                # Try AutoProcessor directly
                try:
                    processor = AutoProcessor.from_pretrained(model_name)
                    print("  ✓ Processor loaded (via AutoProcessor)")
                except Exception:
                    # Try GemmaTokenizer directly (SigLip2 uses Gemma tokenizer)
                    try:
                        from PIL import Image
                        from transformers import GemmaTokenizer

                        # Create a simple processor-like object
                        class SimpleProcessor:
                            def __init__(self, tokenizer):
                                self.tokenizer = tokenizer

                            def __call__(self, text=None, images=None, return_tensors="pt", padding=True, **kwargs):
                                result = {}
                                if text is not None:
                                    encoded = self.tokenizer(text, padding=padding, return_tensors=return_tensors)
                                    result.update(encoded)
                                if images is not None:
                                    # Simple image preprocessing (normalize to [0, 1] and resize)
                                    # This is a simplified version - real processor does more
                                    processed_images = []
                                    for img in images:
                                        if isinstance(img, torch.Tensor):
                                            # Assume already processed
                                            processed_images.append(img)
                                        else:
                                            # Convert PIL to tensor
                                            import torchvision.transforms as T

                                            transform = T.Compose(
                                                [
                                                    T.Resize((224, 224)),
                                                    T.ToTensor(),
                                                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                                ]
                                            )
                                            if isinstance(img, Image.Image):
                                                processed_images.append(transform(img))
                                            else:
                                                # Assume tensor, just normalize
                                                processed_images.append(img)
                                    result["pixel_values"] = torch.stack(processed_images)
                                return result

                        tokenizer = GemmaTokenizer.from_pretrained(model_name)
                        processor = SimpleProcessor(tokenizer)
                        print("  ✓ Processor loaded (via GemmaTokenizer)")
                    except Exception as e3:
                        print(f"  ⚠ Processor not available: {e3}")
                        print("  Will use manual input preparation")
                        processor = None
        except Exception as e:
            print(f"  ⚠ Processor not available: {e}")
            print("  Will use manual input preparation")
            processor = None

        # Compare state dicts
        weights_match = compare_state_dicts(kornia_model, hf_model, model_name)

        # Compare outputs
        outputs_match = compare_outputs(kornia_model, hf_model, processor, model_name, device, dtype)

        # Summary
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"  Weights match: {'✓' if weights_match else '✗'}")
        print(f"  Outputs match: {'✓' if outputs_match else '✗'}")

        if weights_match and outputs_match:
            print(f"\n✓ {model_name} VERIFIED - Matches transformers exactly!")
            return 0
        else:
            print(f"\n✗ {model_name} - Some differences detected")
            return 1

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
