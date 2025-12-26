# PaliGemma Output Matching Fixes

This document summarizes the fixes made to ensure Kornia's PaliGemma implementation matches transformers output.

## Key Fixes

### 1. Position IDs Calculation (`model.py`)

**Issue**: Position IDs were calculated using `mask.long().cumsum(-1) - 1`, which doesn't match transformers' approach.

**Fix**: Changed to sequential position IDs starting from 0:
```python
positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
positions = positions.unsqueeze(0).expand(B, -1)
```

Position IDs are now sequential indices for all tokens. Padding is handled by the attention mask, not position IDs.

### 2. Causal Mask Construction (`gemma.py`)

**Issue**: The causal mask construction had issues with:
- Not properly allowing self-attention (diagonal)
- Incorrect handling of cached positions

**Fix**: Rewrote `_make_causal_mask` to:
- Properly create lower triangular mask (including diagonal)
- Handle cached positions correctly (all new positions can attend to all cached positions)
- Use efficient tensor operations instead of loops where possible

```python
# All new positions can attend to all cached positions
if cache_len > 0:
    causal[:, :cache_len] = 0.0

# Create lower triangular mask for new positions
for i in range(L):
    causal[i, cache_len : cache_len + i + 1] = 0.0
```

### 3. Removed Debug Logging (`gemma.py`)

**Issue**: Debug logging code was left in the implementation.

**Fix**: Removed all `_debug_log` calls and the `_debug_log` method to clean up the codebase.

### 4. API Compatibility (`model.py`)

**Enhancement**: Added transformers-compatible argument names while keeping our cleaner API:
- `pixel_values` as alias for `images`
- `input_ids` as alias for `token_ids`
- `attention_mask` as alias for `mask`

The forward method now accepts both naming conventions, making it easier to compare with transformers while maintaining our cleaner API.

## API Design Philosophy

Our API is **similar to transformers but not a copy-paste**:

- **Cleaner naming**: `images` instead of `pixel_values`, `token_ids` instead of `input_ids`
- **Better for research**: Easy access to intermediate representations via `return_intermediates` and `return_attention_weights`
- **Compatibility**: Supports transformers argument names for easy comparison
- **Simpler structure**: More maintainable codebase with clear separation of concerns

## Testing

To verify the fixes work correctly, use the validation script:

```bash
python scripts/validate_vlm.py --random
```

Or use the test script:

```bash
python test_paligemma_comparison.py
```

## Next Steps

1. Run validation tests with pretrained weights
2. Verify numerical equivalence with transformers
3. Test generation functionality
4. Ensure all edge cases are handled correctly
