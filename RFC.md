# RFC: kornia.augmentations funnel

## 1. Goals

- **G1 — Export-clean determinism.** `torch.export.export(K.deterministic.Resize(560), x)` succeeds on CPU and CUDA, and ONNX opset-18 export of all deterministic preprocessing transforms passes round-trip numerical equality. This is the falsifiable closing condition for torchgeo issue #3108.
- **G2 — Bit-identical legacy surface for rf-detr.** The seven transforms hard-coded in rf-detr's `_REGISTRY` (`RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation`, `RandomAffine`, `ColorJiggle`, `RandomGaussianBlur`, `RandomGaussianNoise`) keep their constructor signatures and class names byte-identical, and we ship `KA.BboxParams(min_area, min_visibility, label_fields)` so rf-detr PR #874 Phase 2 can delete its `collate_boxes` / `unpack_boxes` glue.
- **G3 — Five new detection-grade transforms.** Mosaic 2×2 (box+mask aware), MixUp box-aware, CutMix box-aware, Large-Scale Jittering, Copy-Paste — all box-and-mask aware, all CUDA-Graph-capturable.
- **G4 — Three GPU correctness/perf upgrades.** GPU-side RNG (PR-G1), pipeline-wide `torch.compile` (PR-G4), channels-last awareness (PR-G7).
- **G5 — Per-transform CUDA Graph capture is the structural gate.** A binary pass/fail CI contract (`torch.cuda.graph` capture at fixed shape) for every transform — this is the only mechanically falsifiable proof that no transform issues a host sync.
- **G6 — Agent-installable skill.** A `kornia-augmentations` skill with USER mode (compose pipelines) and DEVELOPER mode (author new transforms).

## 2. Non-goals

- We are **not** preserving TorchScript scripting compatibility for augmentation transforms. The no-autograd contract makes scripting redundant for training pipelines, and Pydantic v2 in `__init__` is incompatible with `torch.jit.script`. `torch.export` + ONNX is the supported export path.
- We are **not** introducing a new public namespace beyond `kornia.augmentations` (plural). No `kornia.augv2`, no `kornia.transforms`.
- We are **not** changing the `kornia/augmentation/_2d/base.py` class hierarchy depth (`AugmentationBase2D` → `RigidAffineAugmentationBase2D` → concrete) in this funnel — that refactor is deferred.
- We are **not** removing `data_keys=` in this funnel; removal is scheduled for kornia 1.0.

## 3. Design decisions

### 3.1 Single namespace + transparent shim

`kornia/augmentation/` (singular) remains the source-of-truth on disk; `kornia/augmentations/__init__.py` (plural) re-exports via `from kornia.augmentation import *` plus a lazy module-level `__getattr__` for sub-modules (`auto`, `container`, `_2d`, `_3d`). Every existing import in the wild — including rf-detr's `import kornia.augmentation as K` and torchgeo's `from kornia.augmentation import ...` — keeps working unchanged. The plural form becomes the documented canonical spelling because every other library in the ecosystem (torchvision, albumentations, timm) uses the plural; the singular shim is permanent, not deprecated.

### 3.2 No-autograd contract + NormalizeWithGrad

`@torch.no_grad()` is applied at the unified Transform base (`_AugmentationBase` in `kornia/augmentation/base.py`, the parent of `AugmentationBase2D` per `_2d/base.py:23`). This is the single largest perf win in the funnel and the precondition for clean `torch.export` graphs (G1). The one carved-out exception is `NormalizeWithGrad`, which lives in `kornia.augmentations.deterministic` as a **distinct class**, not a `requires_grad=True` flag on `Normalize`. Rationale: a flag would create silent module-path-based behavior (the same `Normalize` symbol behaving differently depending on construction args), which is exactly the footgun that motivated the carve-out in the first place. A separate class is greppable, type-checkable, and impossible to import by accident.

### 3.3 Pydantic v2 params + decorator registry + catalog

Each transform's `__init__` validates its parameters via a Pydantic v2 model (`RandomAffineParams`, `ColorJiggleParams`, …). This replaces the ad-hoc `_range_bound` / `_tuple_range_reader` helpers, gives us free JSON-schema generation for the agent skill (G6), and produces clear error messages — which agent-authored transforms desperately need. We accept the cost: **TorchScript scripting of augmentation transforms is deprecated** (non-goal §2). Catalog finalization uses a hybrid lazy `@functools.cache`d access (so `kornia.augmentations.RandomAffine` doesn't pay catalog cost on import) plus eager finalization at module-end via `__all__` walk — this avoids the import-order hazard where a transform registered after first catalog access would be silently missing.

### 3.4 Per-transform CUDA Graph capture contract

Every concrete transform must pass `torch.cuda.graph` capture at a fixed shape, asserted in CI. This is non-negotiable and binary: it is the only **mechanical** proof that a transform contains zero host syncs (no `.item()`, no `.cpu()`, no Python-level shape branches on tensor data). PR-00a lands the harness with the current leaderboard ("X/82 OK, Y failed"). PR-G1 (GPU-side RNG migration) is the single biggest ratchet on the leaderboard — most current failures trace to CPU-side `torch.rand` calls in `generate_parameters`. Target: PR-G1 closes with **82/82 OK**. After PR-00a, any new transform PR that breaks the gate is rejected at CI.

### 3.5 Test preservation gate

All existing kornia tests pass at every commit in the funnel. Numerical goldens (sampled via fixed-seed runs of the current `main`) are checked at every commit. The **single explicit exception** is PR-NG (No-Grad), which removes ~12 autograd/gradcheck test files (`test_*_gradcheck.py` under `tests/augmentation/`). The RFC-level justification: these tests assert a property (gradient flow through randomized augmentation) that the no-autograd contract explicitly removes. Keeping them as `xfail` is misleading; deleting them with this RFC as the citation is honest. `NormalizeWithGrad` retains its own `gradcheck` test.

### 3.6 `data_keys=` deprecation timeline

`data_keys=` continues to work indefinitely through the 9-week funnel. The replacement (typed multi-input call: `aug(image=..., boxes=..., masks=...)`) lands in PR-DK1 as additive. Removal of `data_keys=` is scheduled for kornia 1.0, **not** in this funnel. Rationale: `AugmentationSequential` is the most-cited container in downstream code (rf-detr, torchgeo, lightly), and `data_keys=` is in every example in `container/augment.py:107`. A breaking change here would torpedo G2.

### 3.7 Pre-existing per-op `torch.compile` rationalization

`gaussian_blur.py`, `color_jitter.py`, and `gaussian_illumination.py` already wrap inner ops in `torch.compile`. PR-G4 introduces pipeline-wide `torch.compile` on `AugmentationSequential.forward`. Per-op compile becomes **opt-in fallback only** (gated by an env var or sequential constructor flag), because nesting `torch.compile` regions inside an outer compiled region forces graph breaks and is strictly worse than a single outer compile. We do not delete the per-op decorators in this funnel — they become dead-but-harmless under the default path.

### 3.8 Agent skill — dual audience

The `kornia-augmentations` skill ships two modes selected at install time. **USER mode**: pipeline composition, data-key wiring, common recipe templates (detection / segmentation / classification). **DEVELOPER mode**: transform authoring template, Pydantic param model boilerplate, CUDA-Graph capture self-test, golden-value harness. The Pydantic schemas from §3.3 are what makes DEVELOPER mode tractable for an agent — the parameter contract is machine-readable.

## 4. Backward compatibility contract (what cannot break)

1. **Class names and constructor positional/keyword signatures** for the rf-detr seven (G2). Verified by a frozen-signature test in PR-00b.
2. **Import paths.** `from kornia.augmentation import RandomAffine` works; `from kornia.augmentations import RandomAffine` works; `import kornia.augmentation as K; K.RandomAffine` works. See current `kornia/augmentation/__init__.py:88-104` for the full surface to preserve.
3. **`AugmentationSequential(*args, data_keys=[...])`** call form (`container/augment.py:51`).
4. **Numerical outputs at fixed seed** for every transform present on `main` today. Goldens captured in PR-00c.
5. **`transform_matrix` property** on rigid transforms (`_2d/base.py:87`).
6. **`auto` and `container` submodules** as importable attributes of `kornia.augmentation` (current `__init__.py:24`).

## 5. PR sequencing — 18 PRs, 5 phases, 9 weeks

**Phase 0 — Foundations (week 1, 3 PRs)**
- PR-00a: CUDA Graph capture harness + leaderboard CI gate.
- PR-00b: Frozen-signature test for the rf-detr seven.
- PR-00c: Numerical golden harness (fixed-seed snapshots).

**Phase 1 — Contract landing (weeks 2-3, 4 PRs)**
- PR-NS: Namespace shim — `kornia/augmentations/__init__.py` re-export.
- PR-NG: No-autograd contract at base + delete the ~12 gradcheck files (RFC-cited).
- PR-NWG: `NormalizeWithGrad` as distinct class in `kornia.augmentations.deterministic`.
- PR-PV: Pydantic v2 param models for the rf-detr seven (sets the pattern).

**Phase 2 — GPU correctness (weeks 4-5, 3 PRs)**
- PR-G1: GPU-side RNG migration. Ratchets CUDA Graph leaderboard to 82/82.
- PR-G4: Pipeline-wide `torch.compile` on `AugmentationSequential`; per-op compile demoted to opt-in.
- PR-G7: channels-last awareness (preserve memory format end-to-end).

**Phase 3 — Detection transforms (weeks 6-7, 5 PRs)**
- PR-T1: Mosaic 2×2 (box + mask aware).
- PR-T2: MixUp box-aware (upgrade `RandomMixUpV2`).
- PR-T3: CutMix box-aware (upgrade `RandomCutMixV2`).
- PR-T4: Large-Scale Jittering.
- PR-T5: Copy-Paste augmentation.

**Phase 4 — Export + skill + bbox params (weeks 8-9, 3 PRs)**
- PR-EX: `torch.export` + ONNX opset-18 path for `kornia.augmentations.deterministic` (closes torchgeo #3108).
- PR-BB: `KA.BboxParams(min_area, min_visibility, label_fields)` (unblocks rf-detr Phase 2).
- PR-SK: `kornia-augmentations` skill (USER + DEVELOPER modes).

## 6. CI gates

- **CG-1** CUDA Graph capture: every transform, every PR, fixed shape. Binary pass/fail. Blocking.
- **CG-2** Frozen-signature test for the rf-detr seven. Blocking.
- **CG-3** Numerical goldens. Blocking; regenerate only via explicit `--update-goldens` flag with reviewer sign-off.
- **CG-4** `torch.export` round-trip on `kornia.augmentations.deterministic.*`. Blocking from PR-EX onward.
- **CG-5** ONNX opset-18 export + onnxruntime numerical equality. Blocking from PR-EX onward.
- **CG-6** No new `.item()` / `.cpu()` / `.tolist()` in `kornia/augmentation/**` (lint rule). Blocking.
- **CG-7** `pixi run lint`, `pixi run typecheck`, `pixi run doctest`. Blocking (existing).

## 7. Open questions for maintainers

1. **`data_keys=` removal target.** Confirm kornia 1.0 (post-funnel) is the right horizon, or push to 1.1?
2. **Pydantic v2 dependency floor.** Kornia currently has zero non-PyTorch deps (CLAUDE.md). Pydantic v2 would be the first. Acceptable, or vendor a minimal validator?
3. **Channels-last default.** PR-G7 makes channels-last *aware*; should `AugmentationSequential` default to channels-last on CUDA in 1.0?
4. **Per-op `torch.compile` removal.** Once PR-G4 lands, do we delete the per-op decorators in `gaussian_blur.py` / `color_jitter.py` / `gaussian_illumination.py` in a follow-up, or leave them as opt-in fallback indefinitely?
5. **3D transform parity.** This funnel is 2D-focused. Do `_3d/*` transforms need CUDA Graph capture in scope, or deferred?
6. **`RandomMosaic` already exists** (`__init__.py:62`) — is PR-T1 an upgrade in place, or a new `Mosaic2x2` class? Recommendation: upgrade in place, keep class name, add `box+mask` arg paths; verify against CG-2 frozen-signature gate.
