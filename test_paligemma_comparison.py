#!/usr/bin/env python3
"""Test script to compare Kornia PaliGemma with transformers implementation."""

import sys
import torch
import numpy as np

# Try to import transformers
try:
    from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("WARNING: transformers not available. Install with: pip install transformers")
    TRANSFORMERS_AVAILABLE = False
    sys.exit(1)

# Import Kornia
from kornia.vlm import PaliGemma2, PaliGemma2Config
from kornia.vlm.paligemma import PaliGemmaProcessor

def copy_weights_hf_to_kornia(hf_model, kornia_model):
    """Copy weights from HuggingFace model to Kornia model."""
    print("Copying weights from HuggingFace to Kornia...")
    
    # Vision encoder
    print("  Copying vision encoder...")
    for i, (hf_layer, kornia_layer) in enumerate(
        zip(hf_model.vision_tower.vision_model.encoder.layers, kornia_model.vision_tower.transformer.blocks)
    ):
        # Self attention
        kornia_layer.attn.wq.weight.data.copy_(hf_layer.self_attn.q_proj.weight.data)
        if hasattr(kornia_layer.attn.wq, 'bias') and hf_layer.self_attn.q_proj.bias is not None:
            kornia_layer.attn.wq.bias.data.copy_(hf_layer.self_attn.q_proj.bias.data)
        kornia_layer.attn.wk.weight.data.copy_(hf_layer.self_attn.k_proj.weight.data)
        if hasattr(kornia_layer.attn.wk, 'bias') and hf_layer.self_attn.k_proj.bias is not None:
            kornia_layer.attn.wk.bias.data.copy_(hf_layer.self_attn.k_proj.bias.data)
        kornia_layer.attn.wv.weight.data.copy_(hf_layer.self_attn.v_proj.weight.data)
        if hasattr(kornia_layer.attn.wv, 'bias') and hf_layer.self_attn.v_proj.bias is not None:
            kornia_layer.attn.wv.bias.data.copy_(hf_layer.self_attn.v_proj.bias.data)
        kornia_layer.attn.wo.weight.data.copy_(hf_layer.self_attn.out_proj.weight.data)
        if hasattr(kornia_layer.attn.wo, 'bias') and hf_layer.self_attn.out_proj.bias is not None:
            kornia_layer.attn.wo.bias.data.copy_(hf_layer.self_attn.out_proj.bias.data)
        
        # Layer norms
        kornia_layer.norm1.weight.data.copy_(hf_layer.layer_norm1.weight.data)
        if hasattr(kornia_layer.norm1, 'bias') and hf_layer.layer_norm1.bias is not None:
            kornia_layer.norm1.bias.data.copy_(hf_layer.layer_norm1.bias.data)
        kornia_layer.norm2.weight.data.copy_(hf_layer.layer_norm2.weight.data)
        if hasattr(kornia_layer.norm2, 'bias') and hf_layer.layer_norm2.bias is not None:
            kornia_layer.norm2.bias.data.copy_(hf_layer.layer_norm2.bias.data)
        
        # MLP
        kornia_layer.ffn.fc1.weight.data.copy_(hf_layer.mlp.fc1.weight.data)
        if hasattr(kornia_layer.ffn.fc1, 'bias') and hf_layer.mlp.fc1.bias is not None:
            kornia_layer.ffn.fc1.bias.data.copy_(hf_layer.mlp.fc1.bias.data)
        kornia_layer.ffn.fc2.weight.data.copy_(hf_layer.mlp.fc2.weight.data)
        if hasattr(kornia_layer.ffn.fc2, 'bias') and hf_layer.mlp.fc2.bias is not None:
            kornia_layer.ffn.fc2.bias.data.copy_(hf_layer.mlp.fc2.bias.data)
    
    # Patch embedding
    kornia_model.vision_tower.embedder.patch_proj.proj.weight.data.copy_(
        hf_model.vision_tower.vision_model.embeddings.patch_embedding.weight.data
    )
    if hasattr(kornia_model.vision_tower.embedder.patch_proj.proj, 'bias'):
        kornia_model.vision_tower.embedder.patch_proj.proj.bias.data.copy_(
            hf_model.vision_tower.vision_model.embeddings.patch_embedding.bias.data
        )
    
    # Position embedding
    kornia_model.vision_tower.embedder.pos_embed.weight.data.copy_(
        hf_model.vision_tower.vision_model.embeddings.position_embedding.weight.data
    )
    
    # Final norm
    kornia_model.vision_tower.final_norm.weight.data.copy_(
        hf_model.vision_tower.vision_model.post_layernorm.weight.data
    )
    if hasattr(kornia_model.vision_tower.final_norm, 'bias') and hf_model.vision_tower.vision_model.post_layernorm.bias is not None:
        kornia_model.vision_tower.final_norm.bias.data.copy_(
            hf_model.vision_tower.vision_model.post_layernorm.bias.data
        )
    
    # Connector
    print("  Copying connector...")
    kornia_model.connector.proj.weight.data.copy_(hf_model.multi_modal_projector.linear.weight.data)
    kornia_model.connector.proj.bias.data.copy_(hf_model.multi_modal_projector.linear.bias.data)
    
    # Language model
    print("  Copying language model...")
    kornia_model.text_decoder.decoder.token_embed.weight.data.copy_(
        hf_model.language_model.embed_tokens.weight.data
    )
    
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
    kornia_model.text_decoder.decoder.final_norm.weight.data.copy_(
        hf_model.language_model.norm.weight.data
    )
    
    # Output projection
    kornia_model.text_decoder.output_proj.weight.data.copy_(hf_model.lm_head.weight.data)
    
    print("  Weight copying complete!")

def compare_outputs(name, hf_out, kornia_out, rtol=1e-3, atol=1e-3):
    """Compare two outputs and print results."""
    if isinstance(hf_out, torch.Tensor) and isinstance(kornia_out, torch.Tensor):
        if hf_out.shape != kornia_out.shape:
            print(f"  {name}: SHAPE MISMATCH - HF: {hf_out.shape}, Kornia: {kornia_out.shape}")
            return False
        
        max_diff = (hf_out - kornia_out).abs().max().item()
        mean_diff = (hf_out - kornia_out).abs().mean().item()
        matches = torch.allclose(hf_out, kornia_out, rtol=rtol, atol=atol)
        
        print(f"  {name}:")
        print(f"    Shape: {hf_out.shape}")
        print(f"    Max diff: {max_diff:.6e}")
        print(f"    Mean diff: {mean_diff:.6e}")
        print(f"    Matches: {'✓ YES' if matches else '✗ NO'}")
        
        if not matches:
            # Print some sample values
            print(f"    HF sample: {hf_out.flatten()[:5].tolist()}")
            print(f"    Kornia sample: {kornia_out.flatten()[:5].tolist()}")
        
        return matches
    else:
        print(f"  {name}: Type mismatch - HF: {type(hf_out)}, Kornia: {type(kornia_out)}")
        return False

def main():
    model_id = "google/paligemma2-3b-pt-224"
    device = "cpu"
    dtype = torch.float32
    
    print("=" * 80)
    print("PaliGemma Output Comparison Test")
    print("=" * 80)
    print(f"Model: {model_id}")
    print(f"Device: {device}, Dtype: {dtype}")
    
    # Load HuggingFace model
    print("\nLoading HuggingFace model...")
    hf_model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device).eval()
    
    # Create Kornia model
    print("Creating Kornia model...")
    kornia_config = PaliGemma2Config.paligemma2_3b_224()
    kornia_model = PaliGemma2(kornia_config).to(device).to(dtype).eval()
    
    # Copy weights
    copy_weights_hf_to_kornia(hf_model, kornia_model)
    
    # Create test inputs
    print("\nCreating test inputs...")
    batch_size = 1
    image_size = 224
    images = torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=dtype)
    
    # Create token IDs with image tokens
    num_image_tokens = 256  # 224x224 / 14x14 = 256 patches
    image_token_id = 257152
    text_tokens = torch.randint(0, 1000, (batch_size, 10), device=device, dtype=torch.long)
    image_tokens = torch.full((batch_size, num_image_tokens), image_token_id, device=device, dtype=torch.long)
    input_ids = torch.cat([image_tokens, text_tokens], dim=1)
    attention_mask = torch.ones_like(input_ids)
    
    print(f"  Images: {images.shape}")
    print(f"  Input IDs: {input_ids.shape}")
    
    # Test 1: Vision encoder only
    print("\n" + "=" * 80)
    print("Test 1: Vision Encoder Output")
    print("=" * 80)
    
    with torch.no_grad():
        hf_vision_out = hf_model.vision_tower(images, output_hidden_states=True)
        kornia_vision_out = kornia_model.vision_tower(images, return_intermediates=True)
    
    vision_match = compare_outputs(
        "Vision features",
        hf_vision_out.last_hidden_state,
        kornia_vision_out.features,
        rtol=1e-4,
        atol=1e-4,
    )
    
    # Test 2: Full model forward
    print("\n" + "=" * 80)
    print("Test 2: Full Model Forward Pass")
    print("=" * 80)
    
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
    
    logits_match = compare_outputs(
        "Logits",
        hf_output.logits,
        kornia_output.logits,
        rtol=1e-3,
        atol=1e-3,
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Vision encoder match: {'✓ PASS' if vision_match else '✗ FAIL'}")
    print(f"Full model logits match: {'✓ PASS' if logits_match else '✗ FAIL'}")
    
    if vision_match and logits_match:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
