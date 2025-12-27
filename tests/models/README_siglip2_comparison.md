# SigLip2 Comparison with Transformers

This directory contains scripts to compare the Kornia SigLip2 implementation with HuggingFace transformers.

## Prerequisites

1. **Compatible NumPy version**: The transformers library requires `numpy<2.0`
   ```bash
   pip install 'numpy<2.0'
   ```

2. **Transformers library**:
   ```bash
   pip install transformers
   ```

3. **HuggingFace Hub** (for downloading models):
   ```bash
   pip install huggingface_hub
   ```

## Comparison Scripts

### `compare_siglip2_with_transformers.py`

Comprehensive comparison script that:
- Compares state dict keys and weight values
- Compares forward pass outputs (embeddings, logits)
- Tests multiple input scenarios
- Provides detailed statistics

**Usage:**
```bash
PYTHONPATH=. python tests/models/compare_siglip2_with_transformers.py [model_name]
```

**Example:**
```bash
PYTHONPATH=. python tests/models/compare_siglip2_with_transformers.py google/siglip2-base-patch16-224
```

### `verify_siglip2_weights.py`

Verifies that weights can be loaded from HuggingFace Hub (doesn't require transformers):
```bash
PYTHONPATH=. python tests/models/verify_siglip2_weights.py [model_name]
```

## Expected Results

### Weight Loading

The checkpoint from HuggingFace Hub contains:
- ✅ Vision encoder weights
- ✅ Text encoder weights
- ✅ Position embeddings
- ✅ Attention weights (q/k/v combined into qkv_proj)
- ✅ MLP weights
- ✅ Layer norms (layer_norm1, layer_norm2)
- ✅ Logit scale
- ❌ Projection layers (visual_projection, text_projection) - these are randomly initialized
- ❌ Some embeddings layer_norms - may be initialized differently

### Output Comparison

When comparing outputs with transformers:
- **Image embeddings**: Should match closely (within 1e-3 tolerance)
- **Text embeddings**: Should match closely (within 1e-3 tolerance)
- **Logits**: May differ if projection layers aren't loaded from checkpoint

## Troubleshooting

### NumPy Compatibility Error

If you see:
```
AttributeError: _ARRAY_API not found
ImportError: numpy.core.multiarray failed to import
```

**Solution**: Downgrade numpy:
```bash
pip install 'numpy<2.0'
```

### Missing Weights Warning

If you see warnings about missing weights:
- `vision_projection`, `text_projection`: These are not in the checkpoint and are randomly initialized
- `embeddings.layer_norm`: May not be in checkpoint depending on model version
- `encoder.layer_norm`: Should be mapped from `final_layer_norm` in checkpoint

### Output Mismatches

If outputs don't match:
1. Check that all encoder weights loaded correctly
2. Verify position embedding sizes match
3. Check that attention weights were properly fused (q/k/v -> qkv_proj)
4. Note that projection layers may be randomly initialized if not in checkpoint

## Notes

- The checkpoint structure may vary between model versions
- Some components (like projection layers) may be in a wrapper model, not the base checkpoint
- The comparison script handles these differences and reports them
