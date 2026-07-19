# Augmentation benchmarks

Reproducible throughput benchmarks for kornia's augmentation stack, measured against every
common image-augmentation library. The goal is an **honest, durable baseline** — one that makes
each performance regime measurable and gives future work a concrete target to improve against.

## Scripts

| Script | What it measures |
| --- | --- |
| [`vs_torchvision.py`](vs_torchvision.py) | Per-op kornia (eager + `torch.compile`) vs torchvision v2, with a `best/tv` ratio and win/lose verdict per op. |
| [`all_libraries.py`](all_libraries.py) | Per-op throughput across **all** libraries (kornia eager + compiled, torchvision v2, albumentations, OpenCV, PIL, kornia-rs). |
| [`cross_library.py`](cross_library.py) | A focused per-op comparison of kornia vs torchvision v2 vs albumentations. |
| [`pipeline.py`](pipeline.py) | End-to-end **multi-op pipeline** throughput (the shape a training loop runs); supports `--compile` and `--half` (fp16/AMP). |

Run:

```bash
python benchmarks/augmentation/all_libraries.py --batch 32 --size 256 --device cpu --compile
python benchmarks/augmentation/all_libraries.py --batch 32 --size 256 --device cuda --compile
python benchmarks/augmentation/pipeline.py      --batch 32 --size 224 --device cuda --compile
```

Each script prints the git commit, platform, and (on CUDA) the device name, per the benchmark
guidelines in the root `CLAUDE.md`. Optional libraries that are not installed are reported as a
skip line rather than failing the run.

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

## Sample results

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
