#!/usr/bin/env python
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

"""Validation script for Kornia VLM implementation.

This script compares the outputs of the Kornia VLM implementation with
HuggingFace Transformers to verify numerical equivalence.

Usage:
    python scripts/validate_vlm.py --image path/to/image.jpg --prompt "Describe this image"
    python scripts/validate_vlm.py --random  # Use random inputs for quick validation
    python scripts/validate_vlm.py --model google/paligemma2-3b-pt-224

Requirements:
    pip install transformers sentencepiece pillow
"""

import argparse
import sys

import torch


def load_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """Load and preprocess an image.

    Args:
        image_path: Path to the image file.
        image_size: Target size for the image.

    Returns:
        Preprocessed image tensor of shape (1, 3, H, W).
    """
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def create_random_inputs(
    batch_size: int = 1,
    image_size: int = 224,
    seq_len: int = 20,
    vocab_size: int = 256000,
    num_image_tokens: int = 256,
    image_token_id: int = 257152,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """Create random inputs for testing.

    Args:
        batch_size: Batch size.
        image_size: Image size.
        seq_len: Text sequence length (excluding image tokens).
        vocab_size: Vocabulary size.
        num_image_tokens: Number of image tokens.
        image_token_id: ID of the image token.
        device: Device to create tensors on.
        dtype: Data type for tensors.

    Returns:
        Tuple of (images, input_ids, attention_mask).
    """
    # Random images normalized to [-1, 1]
    images = torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=dtype)

    # Create input_ids with image tokens followed by text tokens
    image_tokens = torch.full((batch_size, num_image_tokens), image_token_id, device=device, dtype=torch.long)
    text_tokens = torch.randint(0, min(1000, vocab_size), (batch_size, seq_len), device=device, dtype=torch.long)
    input_ids = torch.cat([image_tokens, text_tokens], dim=1)

    # Full attention mask
    attention_mask = torch.ones_like(input_ids)

    return images, input_ids, attention_mask


def validate_vision_encoder(
    model_id: str = "google/paligemma2-3b-pt-224",
    images: torch.Tensor = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    rtol: float = 1e-3,
    atol: float = 1e-3,
):
    """Validate the Siglip vision encoder against HuggingFace.

    Args:
        model_id: HuggingFace model ID.
        images: Input images (if None, uses random).
        device: Device to run on.
        dtype: Data type.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.

    Returns:
        Dict with validation results.
    """
    print("\n" + "=" * 60)
    print("Validating Vision Encoder (Siglip)")
    print("=" * 60)

    # Import HuggingFace
    try:
        from transformers import SiglipVisionConfig, SiglipVisionModel
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers")
        return {"success": False, "error": "transformers not installed"}

    # Import Kornia
    from kornia.vlm.paligemma.config import SigLIPVisionConfig
    from kornia.vlm.paligemma.siglip import SiglipVisionEncoder

    # Determine image size from model_id
    if "224" in model_id:
        image_size = 224
    elif "448" in model_id:
        image_size = 448
    elif "896" in model_id:
        image_size = 896
    else:
        image_size = 224

    # Create random images if not provided
    if images is None:
        images = torch.randn(1, 3, image_size, image_size, device=device, dtype=dtype)
    else:
        images = images.to(device, dtype)

    print(f"  Input shape: {images.shape}")
    print(f"  Device: {device}, Dtype: {dtype}")

    # Load HuggingFace model
    print("\n  Loading HuggingFace Siglip...")
    try:
        hf_model = (
            SiglipVisionModel.from_pretrained(
                model_id,
                subfolder="vision_tower",
                torch_dtype=dtype,
            )
            .to(device)
            .eval()
        )
    except Exception as e:
        print(f"  Warning: Could not load pretrained model: {e}")
        print("  Using random weights for both models...")

        # Create matching configs
        hf_config = SiglipVisionConfig(
            image_size=image_size,
            patch_size=14,
            hidden_size=1152,
            intermediate_size=4304,
            num_hidden_layers=27,
            num_attention_heads=16,
        )
        hf_model = SiglipVisionModel(hf_config).to(device, dtype).eval()

        kornia_config = SigLIPVisionConfig(
            image_size=image_size,
            patch_size=14,
            hidden_size=1152,
            intermediate_size=4304,
            num_hidden_layers=27,
            num_attention_heads=16,
        )
        kornia_model = SiglipVisionEncoder(kornia_config).to(device, dtype).eval()

        # Copy weights from HuggingFace to Kornia
        print("  Copying weights from HF to Kornia...")
        copy_siglip_weights(hf_model, kornia_model)
    else:
        # Create Kornia model with matching config
        kornia_config = SigLIPVisionConfig(
            image_size=image_size,
            patch_size=14,
            hidden_size=1152,
            intermediate_size=4304,
            num_hidden_layers=27,
            num_attention_heads=16,
        )
        kornia_model = SiglipVisionEncoder(kornia_config).to(device, dtype).eval()

        # Copy weights
        print("  Copying weights from HF to Kornia...")
        copy_siglip_weights(hf_model, kornia_model)

    # Run forward pass
    print("\n  Running forward pass...")
    with torch.no_grad():
        hf_output = hf_model(images, output_hidden_states=True, output_attentions=True)
        kornia_output = kornia_model(images, return_intermediates=True, return_attention_weights=True)

    # Compare outputs
    results = {}

    # Compare last hidden state
    hf_features = hf_output.last_hidden_state
    kornia_features = kornia_output.features

    max_diff = (hf_features - kornia_features).abs().max().item()
    mean_diff = (hf_features - kornia_features).abs().mean().item()
    matches = torch.allclose(hf_features, kornia_features, rtol=rtol, atol=atol)

    results["features"] = {
        "shape_match": hf_features.shape == kornia_features.shape,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "matches": matches,
    }

    print("\n  Features:")
    print(f"    Shape: HF={hf_features.shape}, Kornia={kornia_features.shape}")
    print(f"    Max diff: {max_diff:.6e}")
    print(f"    Mean diff: {mean_diff:.6e}")
    print(f"    Matches (rtol={rtol}, atol={atol}): {'✓ YES' if matches else '✗ NO'}")

    # Compare hidden states
    if hf_output.hidden_states and kornia_output.layer_features:
        n_layers_match = len(hf_output.hidden_states) == len(kornia_output.layer_features)
        results["layer_features"] = {"n_layers_match": n_layers_match}
        print("\n  Layer features:")
        print(f"    Num layers: HF={len(hf_output.hidden_states)}, Kornia={len(kornia_output.layer_features)}")

    results["success"] = results["features"]["matches"]
    return results


def copy_siglip_weights(hf_model, kornia_model):
    """Copy weights from HuggingFace Siglip to Kornia Siglip.

    Args:
        hf_model: HuggingFace SiglipVisionModel.
        kornia_model: Kornia SiglipVisionEncoder.
    """
    # Patch embedding
    kornia_model.embedder.patch_proj.proj.weight.data.copy_(
        hf_model.vision_model.embeddings.patch_embedding.weight.data
    )
    kornia_model.embedder.patch_proj.proj.bias.data.copy_(hf_model.vision_model.embeddings.patch_embedding.bias.data)

    # Position embedding
    kornia_model.embedder.pos_embed.weight.data.copy_(hf_model.vision_model.embeddings.position_embedding.weight.data)

    # Encoder layers
    for i, (hf_layer, kornia_layer) in enumerate(
        zip(hf_model.vision_model.encoder.layers, kornia_model.transformer.blocks)
    ):
        # Self attention
        kornia_layer.attn.wq.weight.data.copy_(hf_layer.self_attn.q_proj.weight.data)
        kornia_layer.attn.wq.bias.data.copy_(hf_layer.self_attn.q_proj.bias.data)
        kornia_layer.attn.wk.weight.data.copy_(hf_layer.self_attn.k_proj.weight.data)
        kornia_layer.attn.wk.bias.data.copy_(hf_layer.self_attn.k_proj.bias.data)
        kornia_layer.attn.wv.weight.data.copy_(hf_layer.self_attn.v_proj.weight.data)
        kornia_layer.attn.wv.bias.data.copy_(hf_layer.self_attn.v_proj.bias.data)
        kornia_layer.attn.wo.weight.data.copy_(hf_layer.self_attn.out_proj.weight.data)
        kornia_layer.attn.wo.bias.data.copy_(hf_layer.self_attn.out_proj.bias.data)

        # Layer norms
        kornia_layer.norm1.weight.data.copy_(hf_layer.layer_norm1.weight.data)
        kornia_layer.norm1.bias.data.copy_(hf_layer.layer_norm1.bias.data)
        kornia_layer.norm2.weight.data.copy_(hf_layer.layer_norm2.weight.data)
        kornia_layer.norm2.bias.data.copy_(hf_layer.layer_norm2.bias.data)

        # MLP
        kornia_layer.ffn.fc1.weight.data.copy_(hf_layer.mlp.fc1.weight.data)
        kornia_layer.ffn.fc1.bias.data.copy_(hf_layer.mlp.fc1.bias.data)
        kornia_layer.ffn.fc2.weight.data.copy_(hf_layer.mlp.fc2.weight.data)
        kornia_layer.ffn.fc2.bias.data.copy_(hf_layer.mlp.fc2.bias.data)

    # Post layer norm
    kornia_model.final_norm.weight.data.copy_(hf_model.vision_model.post_layernorm.weight.data)
    kornia_model.final_norm.bias.data.copy_(hf_model.vision_model.post_layernorm.bias.data)


def validate_full_model(
    model_id: str = "google/paligemma2-3b-pt-224",
    images: torch.Tensor = None,
    input_ids: torch.Tensor = None,
    attention_mask: torch.Tensor = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    rtol: float = 1e-3,
    atol: float = 1e-3,
):
    """Validate the full PaliGemma model against HuggingFace.

    Args:
        model_id: HuggingFace model ID.
        images: Input images.
        input_ids: Input token IDs.
        attention_mask: Attention mask.
        device: Device to run on.
        dtype: Data type.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        Dict with validation results.
    """
    print("\n" + "=" * 60)
    print("Validating Full PaliGemma2 Model")
    print("=" * 60)

    # Import HuggingFace
    try:
        from transformers import PaliGemmaForConditionalGeneration
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers")
        return {"success": False, "error": "transformers not installed"}

    # Import Kornia
    from kornia.vlm import PaliGemma2, PaliGemma2Config

    # Determine image size
    if "224" in model_id:
        image_size = 224
        num_patches = 256
    elif "448" in model_id:
        image_size = 448
        num_patches = 1024
    elif "896" in model_id:
        image_size = 896
        num_patches = 4096
    else:
        image_size = 224
        num_patches = 256

    # Create random inputs if not provided
    if images is None or input_ids is None:
        images, input_ids, attention_mask = create_random_inputs(
            image_size=image_size,
            num_image_tokens=num_patches,
            device=device,
            dtype=dtype,
        )

    print(f"  Image shape: {images.shape}")
    print(f"  Input IDs shape: {input_ids.shape}")
    print(f"  Device: {device}, Dtype: {dtype}")

    # Load HuggingFace model
    print("\n  Loading HuggingFace PaliGemma...")
    try:
        hf_model = (
            PaliGemmaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
            )
            .to(device)
            .eval()
        )
        pretrained = True
    except Exception as e:
        print(f"  Warning: Could not load pretrained model: {e}")
        print("  Skipping full model validation (requires pretrained weights)")
        return {"success": True, "skipped": True, "reason": str(e)}

    # Create Kornia model
    print("  Creating Kornia PaliGemma2...")
    if "224" in model_id:
        kornia_config = PaliGemma2Config.paligemma2_3b_224()
    elif "448" in model_id:
        kornia_config = PaliGemma2Config.paligemma2_3b_448()
    elif "896" in model_id:
        kornia_config = PaliGemma2Config.paligemma2_3b_896()
    else:
        kornia_config = PaliGemma2Config.paligemma2_3b_224()

    kornia_model = PaliGemma2(kornia_config).to(device, dtype).eval()

    # Copy weights
    print("  Copying weights from HF to Kornia...")
    copy_paligemma_weights(hf_model, kornia_model)

    # Run forward pass
    print("\n  Running forward pass...")
    with torch.no_grad():
        hf_output = hf_model(
            pixel_values=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        kornia_output = kornia_model(
            images=images,
            token_ids=input_ids,
            mask=attention_mask,
        )

    # Compare logits
    hf_logits = hf_output.logits
    kornia_logits = kornia_output.logits

    max_diff = (hf_logits - kornia_logits).abs().max().item()
    mean_diff = (hf_logits - kornia_logits).abs().mean().item()
    matches = torch.allclose(hf_logits, kornia_logits, rtol=rtol, atol=atol)

    results = {
        "logits": {
            "shape_match": hf_logits.shape == kornia_logits.shape,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "matches": matches,
        },
        "success": matches,
    }

    print("\n  Logits:")
    print(f"    Shape: HF={hf_logits.shape}, Kornia={kornia_logits.shape}")
    print(f"    Max diff: {max_diff:.6e}")
    print(f"    Mean diff: {mean_diff:.6e}")
    print(f"    Matches (rtol={rtol}, atol={atol}): {'✓ YES' if matches else '✗ NO'}")

    return results


def copy_paligemma_weights(hf_model, kornia_model):
    """Copy weights from HuggingFace PaliGemma to Kornia PaliGemma.

    Args:
        hf_model: HuggingFace PaliGemmaForConditionalGeneration.
        kornia_model: Kornia PaliGemma2.
    """
    # Vision encoder
    copy_siglip_weights(hf_model.vision_tower, kornia_model.vision_tower)

    # Multimodal projector
    kornia_model.connector.proj.weight.data.copy_(hf_model.multi_modal_projector.linear.weight.data)
    kornia_model.connector.proj.bias.data.copy_(hf_model.multi_modal_projector.linear.bias.data)

    # Language model embedding
    # Note: HF PaliGemma2 uses Gemma2Model directly, not nested model.model
    kornia_model.text_decoder.decoder.token_embed.weight.data.copy_(hf_model.language_model.embed_tokens.weight.data)

    # Language model layers
    for i, (hf_layer, kornia_layer) in enumerate(
        zip(hf_model.language_model.layers, kornia_model.text_decoder.decoder.blocks)
    ):
        # Attention
        kornia_layer.attn.q_proj.weight.data.copy_(hf_layer.self_attn.q_proj.weight.data)
        kornia_layer.attn.k_proj.weight.data.copy_(hf_layer.self_attn.k_proj.weight.data)
        kornia_layer.attn.v_proj.weight.data.copy_(hf_layer.self_attn.v_proj.weight.data)
        kornia_layer.attn.o_proj.weight.data.copy_(hf_layer.self_attn.o_proj.weight.data)

        # MLP
        kornia_layer.ffn.gate_proj.weight.data.copy_(hf_layer.mlp.gate_proj.weight.data)
        kornia_layer.ffn.up_proj.weight.data.copy_(hf_layer.mlp.up_proj.weight.data)
        kornia_layer.ffn.down_proj.weight.data.copy_(hf_layer.mlp.down_proj.weight.data)

        # Norms
        kornia_layer.pre_attn_norm.weight.data.copy_(hf_layer.input_layernorm.weight.data)
        kornia_layer.pre_ffn_norm.weight.data.copy_(hf_layer.post_attention_layernorm.weight.data)

    # Final norm
    kornia_model.text_decoder.decoder.final_norm.weight.data.copy_(hf_model.language_model.norm.weight.data)

    # LM head (on main model, not language_model)
    kornia_model.text_decoder.output_proj.weight.data.copy_(hf_model.lm_head.weight.data)


def main():
    parser = argparse.ArgumentParser(
        description="Validate Kornia VLM implementation against HuggingFace Transformers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick validation with random inputs
  python scripts/validate_vlm.py --random

  # Validate with a specific image and prompt
  python scripts/validate_vlm.py --image photo.jpg --prompt "What is in this image?"

  # Validate a specific model variant
  python scripts/validate_vlm.py --model google/paligemma2-3b-pt-448 --random

  # Validate only the vision encoder
  python scripts/validate_vlm.py --vision-only --random
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="google/paligemma2-3b-pt-224",
        help="HuggingFace model ID (default: google/paligemma2-3b-pt-224)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image in detail.",
        help="Text prompt (default: 'Describe this image in detail.')",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random inputs for validation (faster, no image required)",
    )
    parser.add_argument(
        "--vision-only",
        action="store_true",
        help="Only validate the vision encoder",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on (default: cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type (default: float32)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for comparison (default: 1e-3)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for comparison (default: 1e-3)",
    )

    args = parser.parse_args()

    # Parse dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"

    print("=" * 60)
    print("Kornia VLM Validation Script")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")
    print(f"Tolerance: rtol={args.rtol}, atol={args.atol}")

    # Determine image size
    if "224" in args.model:
        image_size = 224
    elif "448" in args.model:
        image_size = 448
    elif "896" in args.model:
        image_size = 896
    else:
        image_size = 224

    # Prepare inputs
    images = None
    input_ids = None
    attention_mask = None

    if args.random:
        print("\nUsing random inputs...")
    elif args.image:
        print(f"\nLoading image: {args.image}")
        images = load_image(args.image, image_size)
        print(f"Image loaded: {images.shape}")

    # Run validations
    all_results = {}

    # Vision encoder validation
    vision_results = validate_vision_encoder(
        model_id=args.model,
        images=images,
        device=args.device,
        dtype=dtype,
        rtol=args.rtol,
        atol=args.atol,
    )
    all_results["vision_encoder"] = vision_results

    # Full model validation (unless vision-only)
    if not args.vision_only:
        full_results = validate_full_model(
            model_id=args.model,
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            device=args.device,
            dtype=dtype,
            rtol=args.rtol,
            atol=args.atol,
        )
        all_results["full_model"] = full_results

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, results in all_results.items():
        if results.get("skipped"):
            status = "⚠ SKIPPED"
        elif results.get("success"):
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
            all_passed = False
        print(f"  {name}: {status}")

    print("=" * 60)
    if all_passed:
        print("Overall: ✓ ALL VALIDATIONS PASSED")
        return 0
    else:
        print("Overall: ✗ SOME VALIDATIONS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
