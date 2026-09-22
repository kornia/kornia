Implemented the benchmark-driven `auto` policy in `82ae603e`:

```python
if tensor.device.type == "cuda":
    return "unfold"
return "shift"
```

This keeps the rule device-only and deliberately avoids kernel-size, batch-size, dtype, and compile-mode branches. `convolution` remains available explicitly but is no longer selected by `auto`; CUDA users with unusually memory-constrained windows can still select `engine="shift"` explicitly.

Also updated:

- the dilation and erosion resolver tests, including explicit `shift` pass-through;
- all seven public morphology engine docstrings;
- the existing breaking changelog fragment.

Verification:

- CPU morphology, float32/float64/float16/bfloat16: 981 passed;
- CUDA morphology, float32: 252 passed;
- morphology doctests: 7 passed;
- `pre-commit run --all-files`: passed;
- `ty check kornia`: exited successfully (warning-level existing diagnostics only).

Written by Codex (GPT-5) on behalf of @ducha-aiki
