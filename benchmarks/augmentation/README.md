# Augmentation benchmarks

Reproducible throughput benchmarks for kornia's augmentation stack, measured against every
common image-augmentation library. The goal is an **honest, durable baseline** — one that makes
each performance regime measurable and gives future work a concrete target to improve against.

## Scripts

| Script | What it measures |
| --- | --- |
| [`flagship.py`](flagship.py) | **The flagship suite** — augmentations benchmarked through each library's random-transform class API (parameter sampling included) vs torchvision v2, albumentations, and OpenCV/PIL where comparable; shared `common.py` methodology, `--json` export. Supersedes `all_libraries.py`. |
| [`vs_torchvision.py`](vs_torchvision.py) | Per-op kornia (eager + `torch.compile`) vs torchvision v2, with a `best/tv` ratio and win/lose verdict per op. |
| [`cross_library.py`](cross_library.py) | A focused per-op comparison of kornia vs torchvision v2 vs albumentations. |
| [`pipeline.py`](pipeline.py) | End-to-end **multi-op pipeline** throughput (the shape a training loop runs); supports `--compile` and `--half` (fp16/AMP). |

Run:

```bash
python benchmarks/augmentation/flagship.py --batches 1,8,32 --size 256 --device cpu --compile
python benchmarks/augmentation/flagship.py --batches 1,8,32 --size 256 --device cuda --compile --json aug_cuda.json
python benchmarks/augmentation/pipeline.py --batch 32 --size 224 --device cuda --compile
```

Each script prints the git commit, platform, and (on CUDA) the device name, per the methodology
contract in [`benchmarks/README.md`](../README.md). Optional libraries that are not installed are reported as a
skip line rather than failing the run.

PR [#4659](https://github.com/kornia/kornia/pull/4659) fixes the crop-parameter recompilation
and shared augmentation compile-cache defects tracked in
[#4658](https://github.com/kornia/kornia/issues/4658). The Intel CPU/CUDA snapshots below were
re-measured after merging that fix. The pre-fix M1 Pro augmentation snapshots have moved to
[`superseded/`](../results/superseded/README.md) until that hardware can be re-measured; their
compiled crop rows measured compilation storms rather than steady-state throughput.

## Intel i7-14700K and RTX 4090 snapshot after #4659 (2026-09-18)

Kornia 0.9.0rc1 at `5f96e3ca` (CPU) and `5f96e3ca-dirty` (CUDA), including merged PR #4659
(`1eb06974`), WSL2, Python 3.11.14, PyTorch 2.14.0+cu130, four PyTorch threads,
float32 RGB 256×256, batches 1/8/32. The CUDA dirty marker reflects only pending benchmark
results, documentation and export-test skips; the runtime and benchmark harness are unchanged
from `5f96e3ca`. Both runs use the public random-transform classes,
include parameter sampling and allocation, verify the checkout import, and warm the CPU workers
before the sweep. CPU and CUDA ran sequentially. Timing uses the shared `blocked_autorange`
helper with a one-second minimum and CUDA synchronization. All suite baselines were installed:
torchvision 0.29.0+cu130, albumentations 2.0.8, OpenCV 4.11.0 and Pillow 12.3.0.

Full rows, timing IQRs and metadata:
[`CPU JSON`](../results/0.9.0rc1/augmentation--i7-14700k-rtx-4090--cpu.json) and
[`CUDA JSON`](../results/0.9.0rc1/augmentation--i7-14700k-rtx-4090--cuda.json).
These are single-run snapshots, not a main-versus-PR comparison. The table below is batch 32,
in input images/s; the JSON also contains every cross-library baseline.

| Operation | CPU eager | CPU compiled | CUDA eager | CUDA compiled |
| --- | ---: | ---: | ---: | ---: |
| RandomHorizontalFlip | 36,897 | 27,724 | 469,309 | 439,993 |
| RandomAffine | 6,207 | 2,819 | 7,754 | 49,545 |
| RandomPerspective | 2,423 | 2,424 | 16,484 | 51,682 |
| RandomResizedCrop | 14,966 | 11,848 | 17,390 | 74,338 |
| ColorJiggle | 314 | 1,337 | 12,703 | 40,025 |
| RandomGaussianBlur | 1,544 | 1,439 | 47,015 | 224,664 |
| RandomBrightness | 16,143 | 30,413 | 149,602 | 188,938 |
| RandomGrayscale | 21,298 | 32,669 | 177,511 | 464,424 |

Compilation helps CUDA Gaussian blur by 4.78× and ColorJiggle by 3.15× in this batch-32 slice.
It also loses: CPU Gaussian blur runs at 0.93× eager throughput, CPU horizontal flip at
0.75×, and CPU affine at 0.45×. See the JSON for all baseline columns and the regime descriptions below
before comparing CPU uint8 loops with batched float tensors.

**RandomResizedCrop after the fix**, input images/s:

| Batch | CPU eager | CPU compiled | CUDA eager | CUDA compiled |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2,693 | 5,340 | 1,001 | 2,261 |
| 8 | 10,831 | 15,212 | 5,950 | 16,379 |
| 32 | 14,966 | 11,848 | 17,390 | 74,338 |

All 24 compiled Kornia rows per device completed without compile failures or recompilation-limit
warnings. The old Kornia 0.9.0rc1 snapshots at `d579a572` (CPU) and `43cf6b46` (CUDA)
measured roughly 1–2 images/s at batches 8/32 while repeatedly compiling crop parameters. Those
snapshots have been replaced. The new CPU batch-32 compiled crop is still slower than eager;
fixing recompilation does not make compilation the fastest choice for every workload. Other
between-run differences should not be attributed to #4659: these are separate snapshots, not
an interleaved A/B experiment.

Reproduce with the project interpreter:

```bash
python benchmarks/augmentation/flagship.py --batches 1,8,32 --size 256 --threads 4 --device cpu --compile --json aug_cpu.json
python benchmarks/augmentation/flagship.py --batches 1,8,32 --size 256 --threads 4 --device cuda --compile --json aug_cuda.json
```

## The regimes — why the numbers are not apples-to-apples

The libraries do not solve the same problem, and reading a single column as "the winner" is
misleading. What each one operates on:

| Library | Data | Batch | Device | Differentiable |
| --- | --- | --- | --- | --- |
| **kornia** | `float` `BCHW` tensor | ✅ batched | CPU **or GPU** | ✅ yes |
| **torchvision v2** | `float` `BCHW` tensor | ✅ batched | CPU or GPU | ❌ no |
| **albumentations** | `uint8` `HWC` numpy | ❌ per-image loop | CPU only | ❌ no |
| **OpenCV** (`cv2`) | `uint8` `HWC` numpy | ❌ per-image loop | CPU only | ❌ no |
| **PIL** (Pillow) | `uint8` `HWC` image | ❌ per-image loop | CPU only | ❌ no |
| **kornia-rs** | `uint8` `HWC` numpy | ❌ per-image | CPU only (native Rust) | ❌ no |

Two distinct races follow:

- **CPU / `uint8` / single-image** — albumentations, OpenCV, PIL and kornia-rs live here. This is
  the classic data-loader regime, and the SIMD/native backends win it. kornia's `float` path is
  *not* built for this and should not be expected to lead.
- **GPU / `float` / batched / differentiable** — kornia's home turf. torchvision v2 competes on
  raw speed but is not differentiable; nothing else runs here at all. This is the regime kornia
  is designed to lead, and where `torch.compile` fusion pays off.

## Historical sample results (all_libraries.py, superseded 2026-08-08)

The tables below were measured with the superseded per-op script `all_libraries.py` (deterministic
parameters, albumentations transforms constructed inside the timed loop, PIL/kornia-rs columns);
it predates the shared `common.py` methodology and JSON export. Fresh flagship numbers live in the
root [`benchmarks/README.md`](../README.md); the script itself remains in git history.

Directional numbers only — reproduce on your own hardware for anything you cite. Measured on an
NVIDIA Jetson Orin (aarch64), torch 2.10, **CPU**, batch 8, 128×128, throughput in **img/s**
(higher is better):

| op | kornia (eager) | kornia (compiled) | torchvision v2 | albumentations | OpenCV | PIL | kornia-rs |
| --- | --: | --: | --: | --: | --: | --: | --: |
| HorizontalFlip | 12269 | 30262 | 37147 | 4498 | **156846** | 18174 | 65304 |
| VerticalFlip | 14088 | 27319 | 46584 | 4497 | **145922** | 20634 | 72195 |
| Resize (½) | 2079 | 3961 | 5892 | 2840 | 45657 | 3812 | **75643** |
| GaussianBlur | 978 | 194 | 912 | 1875 | 15266 | – | **62848** |
| Brightness | 8275 | 25746 | 26425 | 2657 | 22872 | – | **173588** |
| Grayscale | 7655 | 32267 | 31502 | 3497 | **67327** | 16789 | 15492 |

What this particular slice shows (CPU):

- **kornia-rs is the fastest CPU/`uint8` backend on most ops** — it beats OpenCV on Resize,
  GaussianBlur and Brightness, and only trails it on the two ops that are a single memory pass
  (flips) or a fused color reduction (grayscale). This is the evidence behind the planned
  **opt-in `kornia-rs` augmentation backend** (roadmap): kornia already ships the fastest CPU
  kernels via its Rust sibling — the work is API integration, not raw speed.
- **`torch.compile` is kornia's CPU lever for pointwise ops** — Brightness 8.3k → 25.7k (3.1×),
  Grayscale 7.7k → 32.3k (4.2×), closing or beating the torchvision gap. It does **not** help
  conv-bound ops on CPU (GaussianBlur regresses — the compile/launch overhead exceeds the tiny
  kernel).
GPU (NVIDIA Jetson Orin, torch 2.10, `--device cuda`, batch 32, 256×256, img/s). The `uint8`
single-image backends are CPU-only and repeated for reference:

| op | kornia (eager) | kornia (compiled) | torchvision v2 | OpenCV | kornia-rs |
| --- | --: | --: | --: | --: | --: |
| HorizontalFlip | 11458 | ✗ | **52120** | 22134 | 32284 |
| VerticalFlip | 10599 | ✗ | **50601** | 22683 | 29592 |
| Resize (½) | ✗ nan | ✗ | 26149 | 21145 | **40773** |
| GaussianBlur | 474 | 1042 | **6904** | 8342 | 19336 |
| Brightness | 6538 | 12284 | **29785** | 4044 | 9621 |
| Grayscale | 19062 | 24777 | **42555** | 18211 | 4698 |

What the GPU rows show — **read the Jetson caveats, do not treat them as a datacenter result**:

- `✗` in the compiled column is a **Jetson-wheel limitation**, not a kornia bug: `torch.compile`
  on the Orin's torch 2.10 CUDA build errors (`CUDA driver error: invalid argument`) for several
  ops, so the compiled row is unavailable. `Resize`'s eager `nan` is the same story — the Orin
  wheel's `cusolver` is broken, so the `get_perspective_transform` linear solve fails on-device.
  On a normal (datacenter) CUDA wheel both work; the Orin simply *understates* kornia here.
- Where compile **is** available on-device (Brightness, GaussianBlur, Grayscale) it gives the
  expected 2–4× over eager, and torchvision v2 leads the batched float throughput — consistent
  with the wrapper-overhead finding (≈78% of a cheap-op GPU forward is base-class orchestration,
  not the kernel; see the improvement list below).
- **This box's Orin is the standard GPU-bench target for this project** — run all GPU numbers
  here with `--device cuda`, and report the two caveats above rather than omitting the rows.

## Known improvement opportunities (the "improve later" list)

Concrete, measured levers, roughly in leverage order. Each is a place this baseline can move.

1. **Wrapper overhead on cheap ops (GPU).** Profiling shows ~78% of a GPU `RandomHorizontalFlip`
   forward is per-call orchestration in the augmentation base (transform-matrix construction, the
   `where`-blend, dtype/shape bookkeeping) — not the kernel. The lever is a leaner, fully
   `torch.compile`-able base `forward` plus CUDA-graph capture, not faster kernels. Partially
   addressed (the `p == 1` transform-matrix fast path); CUDA-graph capture is the next step.
2. **A `kornia-rs` CPU/`uint8` backend.** The table above shows kornia-rs already leads the CPU
   race. An opt-in `backend="rust"` selector that routes non-differentiable CPU `uint8`
   augmentation through `kornia_rs.imgproc` would make kornia the fastest option in the data-loader
   regime too — an API-integration problem, not a kernel one.
3. **Triton kernels for the kernel-bound minority.** For ops `inductor` cannot fuse — the fused
   warp sampler (`warp_affine` / `grid_sample` / `remap`, which every geometric aug routes
   through), `median_blur`, `equalize`/histogram, bilateral/guided filters, morphology. Pointwise
   ops stay on `inductor`; a Triton rewrite there only re-derives what the compiler emits. (Note:
   kornia's dependency policy is PyTorch-only, so a core Triton kernel is a deliberate
   optional-dependency decision, not a drop-in.)
4. **A `uint8` fast path in the float pipeline** to cut the `float`↔`uint8` conversions that make
   the batched path lose to the native `uint8` backends on cheap ops.
5. **Compiled end-to-end pipelines.** `pipeline.py` already shows a compiled kornia pipeline
   beating torchvision v2 and reaching albumentations parity on CPU; the remaining win is keeping
   the whole pipeline (including parameter generation) fullgraph so it captures into a CUDA graph.

When you improve one of these, re-run the relevant script, drop the new numbers in the PR
description, and — if the change is durable — update the sample table above with fresh hardware.
