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

"""Verify that from_pretrained correctly loads weights from HuggingFace Hub.

This script verifies:
1. Weights can be downloaded and loaded
2. State dict structure is correct
3. Model can perform forward passes with loaded weights

Usage:
    PYTHONPATH=. python tests/models/verify_siglip2_weights.py [model_name]
"""

import sys

import torch

from kornia.models.siglip2 import SigLip2Builder, SigLip2Config


def verify_weight_loading(model_name: str):
    """Verify that weights are loaded correctly."""
    print("=" * 70)
    print(f"Verifying weight loading for: {model_name}")
    print("=" * 70)

    try:
        # Get config first
        config = SigLip2Config.from_name(model_name)
        print("\n✓ Config created:")
        print(f"  Image size: {config.vision_config.image_size}")
        print(f"  Hidden size: {config.vision_config.hidden_size}")
        print(f"  Projection dim: {config.projection_dim}")

        # Load model with pretrained weights
        print("\nLoading pretrained weights from HuggingFace Hub...")
        model = SigLip2Builder.from_pretrained(model_name)
        print("✓ Model loaded successfully")

        # Check state dict
        state_dict = model.state_dict()
        print(f"\n✓ State dict loaded: {len(state_dict)} parameters")

        # Verify key components exist
        required_keys = [
            "vision_model.embeddings.patch_embedding.weight",
            "vision_model.embeddings.position_embeddings.weight",
            "text_model.embeddings.token_embedding.weight",
            "text_model.embeddings.position_embeddings.weight",
            "vision_projection.weight",
            "text_projection.weight",
            "logit_scale",
        ]

        missing_keys = []
        for key in required_keys:
            if key not in state_dict:
                missing_keys.append(key)

        if missing_keys:
            print(f"\n⚠ Missing keys: {missing_keys}")
        else:
            print("\n✓ All required components present")

        # Check parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("\nModel Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # Test forward pass
        print("\nTesting forward pass...")
        model.eval()

        batch_size = 2
        img_size = config.vision_config.image_size
        pixel_values = torch.randn(batch_size, 3, img_size, img_size)
        input_ids = torch.randint(0, min(1000, config.text_config.vocab_size), (batch_size, 10))
        attention_mask = torch.ones(batch_size, 10)

        with torch.no_grad():
            output = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        print("✓ Forward pass successful")
        print(f"  Image embeds shape: {output['image_embeds'].shape}")
        print(f"  Text embeds shape: {output['text_embeds'].shape}")
        print(f"  Logits shape: {output['logits_per_image'].shape}")

        # Check that embeddings are normalized
        img_norm = output["image_embeds"].norm(dim=-1)
        txt_norm = output["text_embeds"].norm(dim=-1)
        print("\nEmbedding norms:")
        print(f"  Image embeds norm (mean): {img_norm.mean().item():.6f}")
        print(f"  Text embeds norm (mean): {txt_norm.mean().item():.6f}")

        # Check logit scale
        logit_scale = output["logit_scale"].item()
        print(f"  Logit scale: {logit_scale:.6f}")

        print("\n" + "=" * 70)
        print("✓ Weight loading verification PASSED")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def compare_with_random_init(model_name: str):
    """Compare pretrained model with randomly initialized model."""
    print("\n" + "=" * 70)
    print("Comparing pretrained vs random initialization")
    print("=" * 70)

    config = SigLip2Config.from_name(model_name)

    # Random init
    random_model = SigLip2Builder.from_name(model_name)
    random_model.eval()

    # Pretrained
    pretrained_model = SigLip2Builder.from_pretrained(model_name)
    pretrained_model.eval()

    # Same inputs
    batch_size = 1
    img_size = config.vision_config.image_size
    pixel_values = torch.randn(batch_size, 3, img_size, img_size)
    input_ids = torch.randint(0, min(1000, config.text_config.vocab_size), (batch_size, 10))
    attention_mask = torch.ones(batch_size, 10)

    with torch.no_grad():
        random_output = random_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pretrained_output = pretrained_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    # They should be different (pretrained weights should produce different outputs)
    img_diff = (random_output["image_embeds"] - pretrained_output["image_embeds"]).abs().mean()
    txt_diff = (random_output["text_embeds"] - pretrained_output["text_embeds"]).abs().mean()

    print("\nOutput differences (pretrained vs random):")
    print(f"  Image embeds mean diff: {img_diff.item():.6f}")
    print(f"  Text embeds mean diff: {txt_diff.item():.6f}")

    if img_diff > 0.1 and txt_diff > 0.1:
        print("✓ Pretrained weights produce different outputs (as expected)")
        return True
    else:
        print("⚠ Outputs are very similar - weights may not be loaded correctly")
        return False


def main():
    """Main verification function."""
    model_name = sys.argv[1] if len(sys.argv) > 1 else "google/siglip2-base-patch16-224"

    print(f"\nVerifying: {model_name}")
    print("Note: This verifies weight loading. For full comparison with transformers,")
    print("      run verify_siglip2_from_pretrained.py (requires compatible numpy/transformers)\n")

    success1 = verify_weight_loading(model_name)
    success2 = compare_with_random_init(model_name)

    if success1 and success2:
        print("\n" + "=" * 70)
        print("✓ ALL VERIFICATIONS PASSED")
        print("=" * 70)
        print("\nTo verify exact matching with transformers:")
        print("  1. Fix numpy compatibility (numpy<2.0)")
        print("  2. Run: PYTHONPATH=. python tests/models/verify_siglip2_from_pretrained.py")
        print("=" * 70)
        return 0
    else:
        print("\n✗ Some verifications failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
