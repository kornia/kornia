# Handoff: kornia.augmentations 2.0 RFC + funnel work

**Reader: Jian (and Jian's agent).** This is your entry point. Read this file first.

**PR:** https://github.com/kornia/kornia/pull/3710 (DRAFT — do NOT merge until §11 RFC vote)
**Branch:** `feat/augmentations-2-0-rfc`
**Status:** RFC + funnel foundation work landed; awaiting hardware-validated benchmarks + maintainer review

## Read these files in order

1. **`HANDOFF.md`** (this file) — orientation + next-step tasks
2. **`RFC_2_0.md`** — the architectural proposal you're being asked to vote on
3. **`docs/migration_kornia_2.md`** — what downstream users will see when 2.0 ships
4. **`benchmarks/profile/bottlenecks_categorized.md`** — empirical bottleneck data per op (37 transforms × 5 categories)
5. **`benchmarks/comparative/leaderboard_v6.md`** — final comparative bench numbers (kornia patched vs torchvision.v2 vs Albumentations)

The funnel code (RFC 1.0 work that justifies 2.0) is in `kornia/augmentation/*` and `kornia/augmentations/*` — you don't need to read it line-by-line; the RFC §3 summarizes what changed and why.

## What's done (so you don't redo it)

✅ Both falsifiable headline outcomes from RFC 1.0:
- torchgeo issue #3108 closeable: `torch.export.export(K.deterministic.Resize(560), x)` works (9/9 export tests + 3/3 ONNX tests pass)
- rf-detr Phase 2 unblocked: `KA.BboxParams(format, min_area, min_visibility, label_fields)` shipped

✅ Funnel optimizations (1.x scope, all goldens green):
- 5 eager-mode patches: Normalize buffers, HFlip cache, hflip/vflip no-contiguous, Affine closed-form (v1+v2), ColorJiggle HSV-fusion
- 11 RandomGenerator subclasses migrated to GPU-side RNG (no host syncs)
- 5 new transforms (Mosaic 2×2 mask-aware, MixUp box-aware, CutMix box-aware + min_area, LSJ, Copy-Paste)
- BboxParams + filter_bboxes utility
- Path A lightweight forward bypass + aggressive forward override on 12 transforms
- CutMix `torch.where` rewrite (4× speedup)

✅ Benchmark + profile suite:
- 6 versions of comparative bench against torchvision.v2 + Albumentations
- Per-op CUDA-event timing for 37 transforms
- torch.profiler bottleneck categorization for all 37 transforms (dispatch / allocation / kernel / sync / fusion-eligible)

✅ Documentation:
- `RFC.md` — funnel RFC (1.0)
- `RFC_2_0.md` — architectural redesign proposal (the document for vote)
- `docs/migration_kornia_2.md` — downstream user migration guide
- `benchmarks/profile/bottlenecks_categorized.md` — categorized profile data

## What needs YOUR hardware (not available on the dev Jetson)

### Hardware gap

- **CUPTI privileges** — kernel-level CUDA self-times require root. Jetson dev rig didn't have this; bottleneck categorization fell back to self-CPU + sync-blocked proxy. With CUPTI on your machine you get kernel timings directly.
- **Triton** — not available on JetPack aarch64. All `torch.compile(mode='reduce-overhead'/'inductor')` rows in the bench are SKIPPED. Need x86 + CUDA + Triton to validate.
- **CUDA Graphs** — capture currently fails for both kornia and torchvision because of in-forward `torch.tensor()` calls. Both libraries need the same fix. We need to verify the fix works on real CUDA hardware, not just on a CPU-only pixi env.
- **Data-center GPU (A100/H100)** — Orin's 1792-core iGPU has a different CPU/GPU ratio than data-center GPUs. The bench numbers we measured are conservative for what tv looks like on Orin (CPU is abundant); on A100 the kornia framework overhead becomes proportionally MORE visible (kernel time is faster, dispatch overhead is the same). Need A100 numbers to validate the "1.3-2.5× of tv after 2.0 redesign" projection.

### Next-step tasks (ordered by priority)

#### Task 1: Reproduce baseline bench on data-center GPU (1 day)

**Goal:** confirm or refute the per-op gap measurements on A100/4090.

```bash
# 1. Clone the branch (you have push access)
git clone -b feat/augmentations-2-0-rfc https://github.com/kornia/kornia.git
cd kornia

# 2. Set up Python 3.11 with CUDA torch + Triton
pixi install
# Or use whatever env has CUDA torch + Triton

# 3. Run the per-op comparative bench (the v6 patches are on disk; just import kornia from this branch)
pip install -e .
python benchmarks/comparative/run_v6.py
# Outputs: benchmarks/comparative/leaderboard_v6.md (overwrites with your numbers)

# 4. Compare to Orin baseline at benchmarks/comparative/leaderboard_v6.md (committed)
```

**Acceptance:** the per-op kornia-vs-tv ratios on A100 should be SAME OR LARGER than on Orin (kernel time shrinks, framework overhead stays). If kornia is suddenly competitive on A100 by accident, that's also informative.

#### Task 2: CUDA Graph capture experiment (2-3 days)

**Goal:** validate that the kornia 2.0 design lifts param sampling out of forward and that the resulting pipeline graph-captures cleanly.

The `benchmarks/comparative/run_v3.py` Scope 3 already attempts CUDA Graph capture and reports `FAILED — operation not permitted when stream is capturing` for both kornia and torchvision. Both libraries need the same fix: pre-allocate `torch.tensor(...)` outside forward.

**Specific experiments:**

```python
# Experiment 1: prove the kornia fix
# Take RandomHorizontalFlip with our v6 patches
# Capture into torch.cuda.CUDAGraph at fixed shape
# Should succeed because v6 patches moved torch.tensor() out of compute_transformation

# Experiment 2: prove the torchvision symmetric fix doesn't exist
# torchvision.transforms.v2.RandomHorizontalFlip — try the same capture
# Will still fail at _geometry.py:affine_image's torch.tensor(matrix, ...) call
# This is the architectural moat: kornia 2.0 fixes their own, tv won't follow

# Experiment 3: pipeline-level graph capture
# Build a 4-op DETR pipeline (HFlip + Affine + ColorJiggle + Normalize)
# Graph-capture it; replay 1000×; measure replay time
# Expected: 5-8× speedup over eager
```

**Acceptance:** at least one transform graph-captures successfully; at least one pipeline graph-captures with measured speedup ≥ 3× over eager.

Reference: `benchmarks/cuda_graph/per_transform.py` is the harness skeleton for this.

#### Task 3: Triton fused-kernel proof-of-concept (1 week)

**Goal:** validate that fused composite kernels can beat torchvision on ColorJitter / Mosaic / MixUp / Affine.

The categorized profile (`bottlenecks_categorized.md`) identifies 8 fusion-eligible ops. **Highest ROI: ColorJitter** because it's at 2.3× of tv (closest to flip).

```python
# Triton kernel sketch
@triton.jit
def fused_color_jitter_kernel(
    img_ptr, out_ptr,
    brightness, contrast, saturation, hue,
    H, W, BLOCK: tl.constexpr,
):
    # Single-pass: read pixel → brightness → contrast → RGB↔HSV → saturation → hue → write
    # vs current kornia: 4-5 separate kernels with HSV roundtrip
    ...
```

Compare to current kornia 52ms / tv 23ms / projected fused ~5ms. **If you hit 5-8ms it beats tv by 3-5×.** That's the headline kornia 2.0 number.

**Acceptance:** fused ColorJitter kernel produces output within ε of unfused (numerical equivalence test); benchmarks within 2× of projection.

#### Task 4: Phase 1 of RFC 2.0 implementation (3 weeks, after RFC vote)

**Goal:** deliver the foundation per RFC §7 Phase 1.

Per the RFC: new base class + 4 mixins under `kornia/augmentations2/`, tagged tensor types, type-dispatch kernel registry, lazy metadata, functional API mirror, 5 exemplar transforms migrated.

The complete code skeleton for `kornia/augmentations2/base.py` is in RFC_2_0.md §4.1 (~50 lines). The exemplar transform `RandomHorizontalFlip` is in §4.7 (~75 lines). Start there.

**Acceptance:** RFC §7 Phase 1 acceptance criteria met (5 transforms benchmark within projection on A100).

#### Task 5: Update perf projections in RFC §6 with real A100 numbers

**Goal:** the projections in RFC §6 are extrapolations from Orin measurements. Replace with real A100 numbers post Task 1.

Edit `RFC_2_0.md` §6.1 table — add columns for A100 measured numbers. The relative ratios (k vs tv) should hold; absolute numbers will scale.

## Open questions for human review (not for the agent)

These are for Jian as a maintainer, not for the agent to decide:

1. **RFC §11 vote** — approve / approve-with-changes / reject
2. **RFC §9.2 Q1** — `data_keys=` removal target: kornia 1.0 (post-funnel) or 1.1?
3. **RFC §9.2 Q2** — Pydantic v2 dep floor: accept it as the first non-PyTorch dep, or vendor a minimal validator?
4. **RFC §9.2 Q3** — channels-last default in 1.0?
5. **RFC §9.2 Q5** — release as 2.0.0 (major) or 1.x.0 (minor)?
6. **RFC §9.2 Q6** — MedianBlur kernel rewrite as separate PR or rolled into 2.0?

## Files Jian's agent should touch

For Tasks 1-3 (validation/profiling on stronger hardware), the agent edits:

```
benchmarks/comparative/run_v6.py        # baseline reproducible bench
benchmarks/comparative/run_v7_a100.py   # NEW — your A100 bench
benchmarks/cuda_graph/per_transform.py  # Graph capture harness
benchmarks/profile/profile_a100.py      # NEW — A100 profile, with CUPTI privileges
RFC_2_0.md                               # update §6 perf projections with A100 numbers
HANDOFF.md                               # log progress
```

For Task 4 (Phase 1 implementation, after RFC approval), the agent creates:

```
kornia/augmentations2/                   # NEW package — RFC §10 layout
  __init__.py
  base.py                                # ~50 lines per RFC §4.1
  functional/
    __init__.py
    geometric.py                         # functional API mirror
  _2d/
    geometric/
      horizontal_flip.py                 # exemplar per RFC §4.7
    intensity/
      normalize.py

kornia/tensors/                          # NEW — tagged tensor types
  __init__.py

tests/augmentations2/
  test_base.py
  test_horizontal_flip.py
  test_perf_baseline.py                  # CG-7 acceptance gate
```

## Reproducible commands (the dev rig had these working)

### Run the comparative bench
```bash
PYTHONNOUSERSITE=1 cd /tmp && \
  /path/to/python_with_cuda_torch/python3 \
  /path/to/kornia/benchmarks/comparative/run_v6.py
# Outputs: results_v6.json + leaderboard_v6.md in benchmarks/comparative/
```

### Run the per-op profile
```bash
PYTHONNOUSERSITE=1 cd /tmp && \
  /path/to/python_with_cuda_torch/python3 \
  /path/to/kornia/benchmarks/profile/profile_all.py
# Outputs: bottlenecks_all.json + bottlenecks_categorized.md
# With CUPTI privileges (root or capabilities), CUDA self-times will be populated
```

### Run the goldens (must stay green)
```bash
pixi run uv run python -m pytest tests/augmentation/goldens/test_goldens.py --noconftest -v
# Expected: 194 passed
```

### Run the BC contract tests
```bash
pixi run uv run python -m pytest tests/test_namespace_shim.py tests/test_normalize_with_grad.py tests/test_rfdetr_signature_freeze.py tests/test_pydantic_params.py tests/test_bbox_params.py tests/test_export_torch.py -v --noconftest
# Expected: 89 passed
```

## What this PR is NOT

- ❌ Not a merge candidate. RFC review only. Do NOT merge before §11 vote.
- ❌ Not a finished implementation. Phase 1 work follows after RFC approval.
- ❌ Not breaking changes for existing kornia.augmentation users. 1.x namespace fully preserved.
- ❌ Not yet validated on x86 + CUDA + Triton. Projections are from Jetson Orin; A100/4090 validation is Task 1.

## How to ping for help

The dev session left `notepad` artifacts under `.omc/notepads/` (gitignored). If you need session context beyond what's documented, those are there but not in version control.

Edgar (`@edgarriba` on GitHub) is the original author and can answer questions about the funnel work and architectural decisions. The RFC text is the source of truth for the 2.0 design — when in doubt, cite it section by section.

## Bottom line

You're inheriting a draft PR with:
- A complete RFC document for 2.0
- Funnel foundation work that empirically justifies the design
- Comprehensive bench + profile data (with hardware caveats)
- A migration guide ready for downstream users

Your job:
- Validate on real hardware (A100/4090/H100 + Triton)
- Get the RFC voted on by maintainers
- Implement Phase 1 (~3 weeks per the RFC plan)
- Update RFC §6 with measured numbers as they come in

Everything you need is in this branch. Start with this file, then RFC_2_0.md, then your hardware validation tasks. Goodspeed.
