`extract_patches_from_pyramid` now samples ordinary-sized inputs once from a packed pyramid atlas instead of
sampling every LAF at every pyramid level. A statically selected levelwise fallback keeps atlases larger than
128 MiB from becoming a memory hazard. LAFs whose nominal level is coarser than the last usable pyramid level now
sample that actual coarsest level; they previously returned an all-zero patch. For `float16` and `bfloat16` inputs
the whole pyramid -- padding, `pyrdown`, coordinate arithmetic and sampling -- runs in `float32` and the patches
are cast back, on every device: this avoids the normalized-coordinate precision loss caused by the wider atlas,
keeps the extraction off `replication_pad2d`/`grid_sampler_2d` kernels that old torch builds lack for half dtypes
on the CPU (torch <= 2.5.1 would otherwise crash), and avoids recasting the full atlas on every sampling chunk.
`extract_patches_simple` computes its sampling grid in `float32` for reduced-precision inputs on every device for
the same reason -- its output matches the pyramid extractor's level-0 output again. Half-precision patches
therefore change on CUDA and MPS as well as the CPU, and floating-point results are not byte-identical to the
former levelwise implementation.
The universal `float32` sampling is a real cost where the native half kernels were fine: on CUDA, `float16`
`extract_patches_simple` is roughly 2x slower at high N, and peak CUDA memory rises at moderate N (the atlas plus
the one-time upcast copy) even though the high-N peaks drop.
Both extractors now reject an image/LAF batch-size mismatch at the boundary with a clear error: former releases
raised an internal `grid_sample` error for most batch mixes, except a smaller LAF batch, where both extractors
silently returned patches for only the first `min(B_img, B_laf)` images. A LAF on another device is moved once to
the image, while a different LAF dtype is promoted with the image dtype for grid arithmetic rather than rejected or
rounded down; this preserves mixed-precision autocast pipelines and subpixel LAF coordinates. Patches still return
with the image's dtype and device. Both extractors chunk their sampling along N, in the spirit of `batched_forward`, so
the folded `(B, N*PS, PS, 2)` grid and channel-scaled sampled chunk each stay within a fixed 64 MiB budget (remap
intermediates can still make the total peak a multiple of that). Batched high-N extraction now peaks below even the
pre-batching per-image loop (468 MiB
against 495 MiB on `main` for B=8,
N=4000, PS=32 on a 512x512 float32 image, where the unchunked atlas peaked at 1.24 GiB), eager throughput stays
within ~5% of the unchunked atlas, and float32/float64 patches are bitwise identical to it; a `torch.compile`d
call whose grid exceeds the budget splits into several `grid_sample` calls (~1.5x the unbounded atlas latency at
B=8, N=2000 on CPU inductor, still ~10x faster than the per-level loop it replaces). Both extractors return an
empty `(B, N, CH, PS, PS)` tensor for `B == 0` or `N == 0` instead of raising from inside `affine_grid` --
`extract_patches_from_pyramid` with `B == 0` returned the empty result before the batched rewrite below
regressed it; every other empty combination raised on all prior releases. `extract_patches_from_pyramid` now
also accepts a LAF on a different device than the image, as `extract_patches_simple` always has, and returns
patches on the image's device; both extractors move the LAF to the image's device up front, so a cross-device
call now samples on the image's device and matches the same-device result (`extract_patches_simple` previously
built the full sampling grid on the LAF's device).
