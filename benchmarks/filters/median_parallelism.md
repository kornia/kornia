# Median blur parallelism notes

Checked 2026-09-18 against PyTorch [`fc553088`](https://github.com/pytorch/pytorch/tree/fc5530887482e83d171231ad1f7c1dd6498da99e).
These are implementation observations, except where explicitly identified as
upstream measurements. They are not a substitute for the benchmark suite.

Kornia's regular path first creates a `B x C x K x H x W` window tensor, then
calls `median(dim=2)`. A 3x3 or 5x5 filter therefore has `K=9` or `K=25` and
has `B*C*H*W` independent reductions. Median blur should parallelize across
those output pixels, channels, and batch elements. Assigning CPU threads *within*
one 9/25-value window is too fine grained. GPU threads can cooperate within a
window, but that mapping still needs to minimize work and memory traffic per pixel.

## Current PyTorch implementations

| Backend | Source observation for `median(dim)` | Consequence for small Kornia windows |
| --- | --- | --- |
| CPU | [`median_with_indices_impl`](https://github.com/pytorch/pytorch/blob/fc5530887482e83d171231ad1f7c1dd6498da99e/aten/src/ATen/native/Sorting.cpp#L641-L740) uses a `TensorIterator` and `iter.for_each`, so independent output slices are CPU-thread parallel. If the reduced dimension is not stride-one, it makes a contiguous permuted copy. Per slice, it allocates an index vector and calls `std::nth_element`. | The standard path has enough outer parallelism, but its general selection and copy are expensive for `K=9/25`. The CPU no-grad fixed selection network avoids both materialized windows and general selection. Its separate elementwise passes are still less efficient than a fused native CPU kernel could be. |
| CUDA | [`gatherMedian`](https://github.com/pytorch/pytorch/blob/fc5530887482e83d171231ad1f7c1dd6498da99e/aten/src/ATen/native/cuda/Sorting.cu#L106-L263) assigns one CUDA block to a reduction slice. Its block size rounds `K` up to a warp, so 3x3 and 5x5 use 32 threads. The kernel scans for NaNs, uses radix selection, then scans again to find the index. | There are many independent blocks, so output-level parallelism is present. The generic selection and index work are disproportionate for a 9/25-value window. A Python selection network would introduce many eager kernel launches and is not a suitable CUDA replacement. |
| MPS | [`median_with_indices_impl_mps`](https://github.com/pytorch/pytorch/blob/fc5530887482e83d171231ad1f7c1dd6498da99e/aten/src/ATen/native/mps/operations/Sort.mm#L512-L628) handles floating `median(dim)` by sorting each row and gathering the selected value/index; integral types use rank-select TopK. | Rows are independent, so it can occupy the device for image-sized inputs, but sorting is excessive for 9/25 values. An eager chain of min/max operations would also be launch-bound. A device-fused small-window kernel is the credible further optimization. |

## Semantics that limit the fast path

`median(dim)` returns both a value and an index. The index determines the
backward path. CPU's current comparator orders equal values by their original
indices; CUDA documents the selected tie index as nondeterministic
([source](https://github.com/pytorch/pytorch/blob/fc5530887482e83d171231ad1f7c1dd6498da99e/aten/src/ATen/native/cuda/Sorting.cpp#L80-L90),
[PyTorch issue #161841](https://github.com/pytorch/pytorch/issues/161841)).
A min/max selection network returns the same median value but need not preserve
that index or its gradient. It must therefore remain restricted to inference:
the existing Kornia CPU fast path excludes autograd and forward-mode AD. It
also has to preserve PyTorch's lower-middle choice and NaN/Inf propagation.

## Upstream status

- CPU: [issue #51450](https://github.com/pytorch/pytorch/issues/51450),
  `torch.median(dim)` slower than sort, remains open. Its report is historical
  performance data, not a measurement of the current Kornia branch.
- CUDA: I found no relevant open PyTorch CUDA *performance* pull request for
  `median` in the GitHub search checked on 2026-09-18. CUDA has a specialized
  generic median kernel; that is evidence of implementation, not a claim that
  it is optimal for median blur.
- MPS: [PR #187060](https://github.com/pytorch/pytorch/pull/187060) was merged
  on 2026-06-12. It moved median/nanmedian to Metal and reports large-input
  speedups, including `median(dim)` cases. Its results do not cover 3x3 or
  5x5 image windows, where the source still shows sort-plus-gather for floats.

The practical direction is to retain the current CPU 3x3/5x5 inference fast
path, and only pursue CUDA/MPS work as a fused device kernel with focused
device benchmarks and explicit value/NaN/Inf/gradient tests. `torch.compile`
is deliberately outside this eager-path assessment.

## Measured effect of moving axes

There are two different axes here. `C` is the image channel axis, while `K`
contains the 9 or 25 neighborhood samples being reduced. Moving `K` with
`movedim` changes the view, but does not make its values adjacent in memory.
Adding `contiguous()` does that at the cost of copying the expanded window
tensor. The CPU median implementation already performs this rearrangement.

An eager probe on Apple M1, PyTorch 2.14.0, float32, four CPU threads, input
`1 x 3 x 256 x 256`, and a 5x5 window measured the following milliseconds.
Numbers are medians of three repeated measurements, each using
`common.time_us(min_run_time=0.35)`; MPS was synchronized inside timing.
Each variant includes its extraction or layout-conversion cost, and outputs
were checked against the original convolution/median path.

| Variant | CPU ms | MPS ms |
| --- | ---: | ---: |
| Convolution, then original `median(dim=2)` | 25.61 | 48.28 |
| Same extraction, move `K` to last, then median | 26.45 | 46.53 |
| Same extraction, move `K` to last and make contiguous | 25.62 | 46.54 |
| Current public `median_blur`, contiguous BCHW input | 16.94 | 46.91 |
| Current public API, already channels-last image storage | 12.84 | 52.32 |
| Current public API, including conversion to channels-last | 13.09 | 51.90 |

The first three rows are diagnostic variants of the generic fallback, not
alternative public implementations. They do not outperform the current CPU
3x3/5x5 inference network. Across additional 3x3/7x7 and batched probes, moving
`K` was not a consistent improvement; explicit copying also regressed the
batched MPS case.

Physical channels-last **image** storage can help the existing CPU selection
network, without changing the BCHW shape:

```python
image = image.contiguous(memory_format=torch.channels_last)
result = kornia.filters.median_blur(image, 5)
```

The broader CPU probe included batches 1/8, channels 1/3/16/32, image sizes
64/128/256, and one/four threads. It found gains for larger multichannel images,
but conversion regressed RGB 64x64 at four threads (about 0.65x throughput for
5x5), and the MPS example above regressed too. The implementation therefore
does not force a memory format. Benchmark the caller's actual shapes and thread
count before keeping an image pipeline channels-last. No CUDA device was
available for this layout study.
