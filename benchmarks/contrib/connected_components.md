# Exact block-based connected components

`kornia.contrib.connected_components_union_find` is an opt-in pure PyTorch
alternative to fixed-count max-pool label propagation. It addresses the
pure-PyTorch direction discussed in [#3657](https://github.com/kornia/kornia/issues/3657).
The block contraction is based on Allegretti, Bolelli and Grana,
[Optimized Block-Based Algorithms to Label Connected Components on GPUs](https://doi.org/10.1109/TPDS.2019.2934683),
TPDS 31(2), 423–438. This is a tensor adaptation, not a reproduction of the
paper's CUDA BUF/BKE kernels or their published performance.

## Why this algorithm

Max pooling moves a label by at most one edge per iteration. The required budget
depends on the foreground graph, not just the image dimensions. A 33×65 snake
in this benchmark needs 1,088 iterations; the default 100 produces several
labels for a single component. Choosing a large constant wastes work on other
masks and still cannot guarantee correctness for arbitrary inputs.

The new API runs to convergence and returns `int64`, avoiding float label
collisions for half-precision masks. The old API, iteration behavior and output
dtype are unchanged. Component IDs differ between the APIs; compare partitions.

## Correctness invariants

1. All foreground pixels in a 2×2 block are 8-connected, including diagonals.
   Contracting them to one graph vertex preserves the component partition.
2. The four predecessor directions cover every possible undirected edge between
   blocks exactly once: left, above, above-left and above-right. Horizontal and
   vertical edges require foreground on both touching sides; diagonal edges
   require both touching corner pixels. A nonempty neighboring block alone is
   insufficient. Odd dimensions are padded with background, and batches have
   disjoint index ranges and no cross-image edges.
3. Parents initially equal their own indices. A reduction hooks only roots,
   always toward a smaller index. This cannot create a cycle or split a tree.
   If several edges compete for a root, the smallest proposal wins and the
   remaining edges are reconsidered in the next round.
4. Pointer jumping compresses every path before the next root-hooking round.
   Each nonterminal round strictly decreases at least one nonnegative parent.
   At termination every original edge joins equal roots, so precisely the
   connected components share a root. Their root is their minimum block index.

The implementation keeps fixed-size edge arrays, representing missing edges as
self-loops. This avoids dynamic `nonzero` output allocation during edge
construction, but uses more memory than sparse edge storage. Working memory is
O(BHW). The scalar convergence checks synchronize CPU and accelerator: this API
does not support CUDA graph capture or fullgraph compilation. It is discrete
and has no gradients. No new runtime dependency or compiled extension is needed.

## Validation

The dependency-free pytest suite uses an independent stack-based flood fill and
exhausts every 3×3, 2×4 and 4×2 binary mask. It also checks random masks, snakes,
diagonal paths, distinct IDs across batch axes, odd sizes, noncontiguous input,
empty shapes, singleton blocks, boolean/integer inputs, exact foreground
selection and half-precision ID collisions. CPU/CUDA and all four floating
dtypes are exercised; CUDA half tests run in isolated subprocesses.

An additional SciPy check exhausts all 65,536 4×4 binary masks (103,696 foreground
components). Reproduce from the repository root with SciPy installed:

```python
import numpy as np
import torch
from scipy.ndimage import label
from kornia.contrib import connected_components_union_find

values = torch.arange(65536)
mask = ((values[:, None] >> torch.arange(16)) & 1).reshape(-1, 1, 4, 4)
actual = connected_components_union_find(mask).numpy()[:, 0]
structure = np.zeros((3, 3, 3), dtype=int)
structure[1] = 1  # No connections across the batch axis.
expected, count = label(mask[:, 0].numpy(), structure=structure)
pairs = np.unique(np.stack((actual.ravel(), expected.ravel()), axis=1), axis=0)
assert len(pairs) == len(np.unique(actual)) == count + 1
assert np.array_equal(actual != 0, expected != 0)
```

## Benchmark method

Run `python -m benchmarks.contrib.connected_components` from each checkout root.
The harness prints the interpreter and imported Kornia paths so an editable
install cannot silently invalidate the A/B comparison. On the base checkout,
copy the unchanged benchmark script into the same untracked relative path;
it detects the missing new API and benchmarks pooling only.

Both APIs receive the same resident float32 tensors. Seeds are pinned. Timing
uses `benchmarks/common.py` (`blocked_autorange`, warmup, synchronized device
execution, median and IQR). Correctness is checked against SciPy outside timing
by establishing a bijection between label IDs. Transfers and reference labeling
are excluded. The optional `--calibrate` search finds the minimum correct pooling
budget outside timing. This favors pooling with an oracle unavailable in normal
use, rather than inflating speedups with an arbitrary overlarge budget.

The `pool100` rows show the existing default, including its incorrect partitions.
Only correct `pool_calibrated` rows are used for equivalent-result comparisons.
CUDA peak allocation includes the output and temporary tensors above the
resident input, excluding allocator cache. CPU performance is measured with one
thread. These synthetic cases characterize algorithmic regimes, not downstream
segmentation accuracy or end-to-end application performance.

Commands (append `--json PATH` to save metadata and raw rows):

```text
python -m benchmarks.contrib.connected_components --device cpu --sizes 256 --calibrate --min-run-time 2
python -m benchmarks.contrib.connected_components --device cuda --sizes 256 1024 --calibrate
python -m benchmarks.contrib.connected_components --device cuda --sizes 256 --batch 4 --calibrate
```

## Results

Measured 2026-09-19 on Windows 11, AMD EPYC 9654 (one PyTorch CPU thread),
NVIDIA RTX 5090, Python 3.13.5, PyTorch 2.10.0+cu130, NumPy 2.1.3 and
SciPy 1.15.3. Base: `e1b79c2`; implementation and benchmark: `8041e8e`.
CPU autorange used at least 2 seconds; CUDA at least 1 second.
Raw A/B results: [connected_components_results](connected_components_results/).

All union-find rows match SciPy. The default 100-step pool fails on both dense
sizes, the snake and the 1024 random mask. Ratios below compare the two correct
methods within the implementation run. Values are median / IQR milliseconds;
a speedup below 1 means union-find loses.


### CUDA, batch 1

| Mask | Minimum pool steps | Pool ms / IQR | Union-find ms / IQR | Speedup |
| --- | ---: | ---: | ---: | ---: |
| dense 256x256 | 255 | 9.352 / 0.077 | 2.101 / 0.023 | 4.45x |
| random 256x256 | 95 | 3.555 / 0.079 | 3.573 / 0.073 | 1.00x |
| isolated 256x256 | 1 | 0.154 / 0.001 | 1.089 / 0.009 | 0.14x |
| snake 33x65 | 1088 | 38.439 / 1.356 | 2.744 / 0.018 | 14.01x |
| dense 1024x1024 | 1023 | 35.136 / 0.602 | 2.913 / 0.019 | 12.06x |
| random 1024x1024 | 197 | 6.918 / 0.065 | 4.480 / 0.078 | 1.54x |
| isolated 1024x1024 | 1 | 0.152 / 0.002 | 1.123 / 0.008 | 0.14x |

### CPU, batch 1

| Mask | Minimum pool steps | Pool ms / IQR | Union-find ms / IQR | Speedup |
| --- | ---: | ---: | ---: | ---: |
| dense 256x256 | 255 | 515.523 / 9.786 | 2.518 / 0.041 | 204.76x |
| random 256x256 | 95 | 340.587 / 2.054 | 5.308 / 0.042 | 64.16x |
| isolated 256x256 | 1 | 2.300 / 0.014 | 1.621 / 0.019 | 1.42x |
| snake 33x65 | 1088 | 90.508 / 2.612 | 0.518 / 0.005 | 174.64x |

### CUDA, batch 4

| Mask | Minimum pool steps | Pool ms / IQR | Union-find ms / IQR | Speedup |
| --- | ---: | ---: | ---: | ---: |
| dense 256x256 | 255 | 8.993 / 0.198 | 2.122 / 0.005 | 4.24x |
| random 256x256 | 152 | 5.437 / 0.090 | 3.736 / 0.085 | 1.46x |
| isolated 256x256 | 1 | 0.154 / 0.002 | 1.123 / 0.013 | 0.14x |
| snake 33x65 | 1088 | 38.797 / 0.930 | 2.735 / 0.023 | 14.19x |

### Interpretation and limits

The 1024x1024 dense case is about 12x faster and the small snake about 14x
faster on this GPU than an oracle-tuned correct pooling run. The random 256 mask
is effectively a tie; isolated pixels are about 7x slower than a known one-step
pooling budget. This is why the new API is opt-in rather than a default replacement.
CPU results reflect PyTorch's pooling/reduction implementations on this machine;
they are not a comparison against optimized native CPU CCL implementations.

Peak extra allocated CUDA memory at 1024x1024 is 17.0 MiB for pooling and
57.0-59.0 MiB for union-find (output included), roughly 3.5x higher. Block contraction
reduces the graph size but the tensor edge arrays, root proposals and integer
output still add memory. This matters for high-resolution batches.

Unchanged pooling code varies between separate base and branch measurements:
the 1024 dense calibrated pool is 48.491 ms on base and 35.136 ms in the branch
run. The raw base results disclose this run-to-run variation; headline ratios use
the paired branch measurements. Re-run on the intended hardware before using
these numbers for a deployment decision. MPS/TPU, minimum-supported PyTorch,
natural-image datasets and end-to-end segmentation pipelines have not been
benchmarked here.
