# Kornia Roadmap

This document describes where Kornia is headed. It exists to help contributors find
high-impact work, to show users the direction of the project, and to make our
priorities legible to the wider community.

It is a living document. Dates are intentions, not commitments, and priorities shift
as the ecosystem does. If you want to work on something here, open or comment on the
linked issue first so we can coordinate (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## Guiding themes

Three engineering north stars shape most of the roadmap below. The important insight
is that the first two are largely **the same body of work**:

1. **Compile-first (`torch.compile` / dynamo).** The numeric core (filters, color,
   geometry, enhance, losses) is already largely compile-clean. The remaining work is
   concentrated in the stochastic augmentation pipeline and the dynamic-shape feature
   detectors, which branch on tensor values and break the graph.

2. **Export-first (ONNX).** ONNX export fails on the *same* patterns that break
   `torch.compile`: data-dependent control flow, `.item()` calls, and dynamic shapes.
   [#3722](https://github.com/kornia/kornia/pull/3722) demonstrated the fix — replacing
   the stochastic apply path with `torch.where` blends and gating linear-algebra
   fallbacks on `is_tracing()` — which advanced both goals in a single change. The
   remaining spine work pays out to compile and export together.

3. **Performance.** A static-shape, data-independent augmentation graph — the same one
   the compile/export work produces — is what fuses under `inductor` and what a native
   `kornia-rs` kernel can accelerate. Our differentiator is GPU-batched, on-device,
   differentiable augmentation; our goal is to make that regime dramatically faster and
   to measure it honestly against alternatives.

A fourth theme, **breadth**, runs in parallel and is largely community-driven: closing
classical-CV gaps and hardening the model zoo.

## Short term — next release (v0.8.4)

- **Stabilize CI and the augmentation core.** Recent fixes repaired regressions from the
  ONNX-exportability refactor and healed the scheduled test matrix. Ship these in a
  release so users get the fixes.
- **Progressively enable more `ruff` rule sets** to raise code health
  ([#2445](https://github.com/kornia/kornia/issues/2445)).
- **Docs modernization** — evaluate the migration to MkDocs
  ([#3454](https://github.com/kornia/kornia/issues/3454)).
- **Harden the recently-landed model zoo.** Several VLM/model integrations shipped without
  full test coverage; add tests before building further on them
  ([#3554](https://github.com/kornia/kornia/issues/3554),
  [#3555](https://github.com/kornia/kornia/issues/3555),
  [#3556](https://github.com/kornia/kornia/issues/3556),
  [#3481](https://github.com/kornia/kornia/issues/3481)).

## Medium term — ~6 months

- **The compile / export spine.** Extend the `torch.where` static-shape approach to the
  shape-changing augmentations (crop, resize, mosaic) — currently the shared blocker for
  both `torch.compile` and ONNX. Sweep the `torch.compiler.is_compiling()` guard across
  value-dependent checks in the numeric core. Solve multi-output ONNX export generically
  so tuple-returning modules (e.g. `Canny`, YUV conversions) are no longer blocked, and
  converge on a single modern ONNX opset.
- **Turn on dynamo tests in CI** for the compile-clean core so it stays a defended floor,
  and publish an ONNX export conformance matrix (exportable / numerically-verified /
  blocked-with-reason) for the public API.
- **Augmentation performance.** Introduce a `uint8` fast path and begin moving the hottest
  ops (warps, blurs, color conversions) toward native `kornia-rs` kernels. Land a
  cross-library benchmark harness (CPU and GPU-batched) so performance claims are
  reproducible.
- **VLM / VLA focus.** Complete and test the native-PyTorch model integrations that are the
  project's current priority: e.g. SmolVLM2
  ([#3455](https://github.com/kornia/kornia/issues/3455),
  [#3770](https://github.com/kornia/kornia/issues/3770)),
  PaliGemma ([#3471](https://github.com/kornia/kornia/issues/3471)),
  Qwen2.5-VL ([#3410](https://github.com/kornia/kornia/issues/3410)),
  and SAM-3 ([#3406](https://github.com/kornia/kornia/issues/3406)).

## Long term — vision

- **A fully compilable, fully exportable library.** Every public operator runs under
  `torch.compile` without graph breaks and exports to ONNX with verified numerical
  equivalence — including the stochastic augmentation and dynamic-shape feature paths.
- **The fastest differentiable, GPU-batched augmentation stack**, with a published,
  reproducible benchmark that states precisely the regime in which we lead.
- **End-to-end spatial-AI and VLM/VLA workflows** — pairing Kornia's deep geometry and
  feature-matching foundations with modern learned models, usable from classical
  calibration through to vision-language reasoning.
- **Breadth toward feature-completeness** for a differentiable CV library — closing the
  notable gaps below.

## Areas seeking contributors

We especially welcome help in these areas. Comment on the linked issue (or open one) to
get started. Issues labelled [`help wanted`](https://github.com/kornia/kornia/labels/help%20wanted)
and [`good first issue`](https://github.com/kornia/kornia/labels/good%20first%20issue)
are good entry points.

**Modern deep-learning CV (largest gaps):**

- **Learned optical flow** (RAFT / SEA-RAFT). A significant gap — the endpoint-error
  metric (AEPE) already exists, but there is no flow model.
- **Pose estimation** models (human / hand keypoints).
- **Object-detection breadth** beyond RT-DETR and YuNet (e.g. YOLO, open-vocabulary
  detection).
- **Feature-matching extensions** — Efficient LoFTR
  ([#3282](https://github.com/kornia/kornia/issues/3282)),
  SANDesc ([#3752](https://github.com/kornia/kornia/issues/3752)).

**Classical computer vision:**

- **Camera intrinsic calibration** (Zhang's method, checkerboard / ChArUco detection).
  Kornia can undistort and solve PnP but cannot yet calibrate a camera from scratch.
- **Fiducial markers** (ArUco / ChArUco).
- **Classical tracking** — Lucas-Kanade, KCF
  ([#1381](https://github.com/kornia/kornia/issues/1381)).
- Dense stereo matching, Hough transforms, contour/shape analysis, template matching.

**Augmentation parity** (vs. common augmentation libraries): dropout-family transforms
(CoarseDropout, GridDropout), grid/optical distortion, weather effects (fog, sun-flare),
noise variants (ISO noise), and additional compression transforms.

**Code health & infrastructure:**

- Expanding `ruff` rule coverage ([#2445](https://github.com/kornia/kornia/issues/2445)).
- Migrating test classes to `BaseTester`
  ([#2752](https://github.com/kornia/kornia/issues/2752)).
- `torch.compile` and ONNX-export coverage for existing operators.

## How to contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and guidelines. In short: link your PR to
an approved issue, include local test output, and cite a reference (paper, OpenCV,
PyTorch, etc.) for any new algorithm. For larger roadmap items, start a discussion on the
issue first so we can align on design before you write code.
