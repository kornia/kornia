Fix the inverted `SigLip2` attention-mask polarity, so every call that passes an `attention_mask` changes (#4043).
All three mask branches of `SigLip2Attention` (2-D `(B, N)`, 3-D `(B, N, N)` and 4-D `(B, 1, N, N)`) negated the
mask before handing it to `torch.nn.functional.scaled_dot_product_attention`, whose boolean form reads `True` as
*attend*, not as *mask out*. Every position the caller asked to attend to was masked out and every padded position
attended to, so an all-ones mask left every query row fully masked. The symptom is version-dependent because a
fully masked row returns `0` from SDPA on PyTorch 2.5 and later and `nan` on PyTorch 2.4 and earlier:
`SigLip2.get_text_features(..., attention_mask=...)` has been returning a zeroed attention output on current
PyTorch and `nan` text features on the declared `torch>=2.0` floor. 2-D padding masks are now broadcast over the
key axis only, matching `SiglipTextTransformer.create_bidirectional_mask` in Hugging Face `transformers`, so a
partially padded sequence no longer produces a fully masked row. A row whose mask is entirely zero — an empty
caption in a batch — is still fully masked and still follows the SDPA behavior above.
