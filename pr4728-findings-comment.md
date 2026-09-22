I ran the requested morphology engine sweeps at `eb40a588` on:

- NVIDIA RTX 4090, CUDA 13.0, cuDNN 9.24, PyTorch 2.14.0
- Intel i7-14700K, 14 physical cores, AVX2

The `shift` engine here is the shift-and-running-max/min implementation. All `shift-unfold` differences were exactly zero. CUDA `convolution-unfold` differences were also zero, including with `cudnn.allow_tf32=True`.

### CUDA

For the 63 eager forward configurations (3 dtypes x 3 batches x 7 kernels), after replacing one contaminated long-sweep row with its fresh-process rerun:

- `unfold` won 48/63.
- `shift` won 15/63, mainly batch-32 float16/bfloat16.
- `convolution` won 0/63, although the current `auto` policy selects it.

Representative eager timings in ms:

| dtype / shape | k | unfold | convolution | shift | winner |
|---|---:|---:|---:|---:|---|
| float32, B1 | 11 | 0.258 | 0.420 | 2.430 | unfold |
| float16, B32 | 7 | 2.216 | 3.575 | 1.512 | shift |
| bfloat16, B32 | 21 | 14.776 | 95.751 | 10.750 | shift |
| float32, B32 | 21 | 27.180 | 92.901 | 46.091 | unfold |

The last two rows were checked in fresh processes. The original all-in-one compile sweep accumulated enough state to produce a non-reproducible float32 B32 k21 result (747 ms unfold / 1397 ms convolution); it should not be used for policy.

Compiled CUDA has a clean kernel-size split across every dtype and batch measured:

- k=3..11: `shift` won all 45 rows.
- k=15..21: `unfold` won all 18 rows.
- `shift` cold compile time grew from about 0.6 s to 39 s across the full sweep.

For forward + backward, float32 B8, eager `unfold` won both sampled cases:

| k | unfold | convolution | shift |
|---:|---:|---:|---:|
| 3 | 0.641 | 0.725 | 1.730 |
| 7 | 2.476 | 3.294 | 8.864 |

### x86 CPU

Eager `shift` won all 14 float32 rows, by 3.1-12.7x versus unfold, and all 12 float16/bfloat16 rows, by 3.4-16.3x. Examples:

| dtype / shape | k | unfold | convolution | shift |
|---|---:|---:|---:|---:|
| float32, B1 | 3 | 1.746 | 1.793 | 0.189 |
| float32, B8 | 21 | 230.347 | 947.654 | 46.602 |
| float16, B8 | 15 | 154.056 | skipped | 16.008 |
| bfloat16, B8 | 15 | 145.752 | skipped | 29.863 |

Compiled CPU `shift` won 10/14 rows. `unfold` won only B1 at k=7, 11, 15, and 21; those do not justify a more complicated default, especially given shift's bounded intermediate memory.

### Proposed `auto` policy

I would keep this deliberately coarse and device-only:

```python
if device.type == "cuda":
    return "unfold"
return "shift"
```

In other words:

- CPU: `shift`. The x86 wins are large and universal; on the existing M1 data, the large-k unfold advantage is small enough that it is not worth a kernel-size branch, while shift uses much less memory.
- CUDA: `unfold`. It wins most eager rows, both backward samples, and compiled large kernels. Batch-32 half precision sometimes favors shift, but that is too shape-specific to encode and the gains are not compelling enough to complicate `auto`.
- MPS and other accelerators: `shift`. On the existing MPS data it is competitive in eager, substantially better when compiled, exact, and low-memory. The modest eager convolution wins around k=5-7 are not worth a special case.
- `convolution`: keep it available as an explicit engine, but do not select it from `auto`; it never won on this CUDA or x86 machine and has the largest intermediate-memory behavior.

I would not branch on `torch.compiler.is_compiling()`, batch size, dtype, or kernel size. I also would not add a hard-coded CUDA memory threshold yet: the unfolded workspace estimate is useful, but a safe cutoff depends on GPU capacity and allocator state. For memory-constrained CUDA workloads, `engine="shift"` is the simple explicit escape hatch. If automatic memory fallback is desired later, it should be based on dedicated peak-memory measurements rather than these timing results.

Focused `auto`/`shift` dilation and erosion tests also passed: 198 passed, 19 deselected.

Written by Codex (GPT-5) on behalf of @ducha-aiki
