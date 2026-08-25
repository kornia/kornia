# Kornia Roadmap

This document describes where Kornia is headed. It exists to help contributors find
high-impact work, to show users the direction of the project, and to make our
priorities legible to the wider community.

It is a living document. Dates are intentions, not commitments, and priorities shift
as the ecosystem does. If you want to work on something here, the technical guidelines
and social contract are in [CONTRIBUTING.md](CONTRIBUTING.md). For large features or
behavior changes, an early conversation may save work and surface compatibility concerns.

**Where Kornia is going.** Kornia is becoming the **reference implementation and
executable specification** for differentiable computer vision and geometry in the
PyTorch ecosystem. At ~3M downloads/month the highest-leverage work is not growing
the API — it is making the differentiated core (warping and sampling, homographies,
cameras, epipolar geometry, rotations and Lie groups, geometry-consistent
augmentation) unusually trustworthy: conventions explicit and pinned by tests,
behavior validated by conformance vectors, guarantees enforced by CI. Progress is
measured in confidence, not API count — a strong release can contain zero new public
APIs and still deliver better `torch.compile` coverage, fixed convention ambiguity,
improved numerical stability, and stronger dtype/device behavior.

## Guiding themes

Four engineering north stars shape the roadmap below.

1. **The reference core: conventions, conformance, tiers.** Geometric code has a
   uniquely dangerous failure mode: small convention mistakes (`align_corners`,
   pixel-center vs corner, (x,y) vs (row,col), normalized vs pixel coordinates,
   rotation direction, homography normalization) produce code that *runs correctly
   while being mathematically wrong* — and this is exactly the error class LLMs
   reproduce. The program, in order, per API surface:

   1. **Document** — every core op gets a Convention block in its docs plus
      `test_convention_*` pinning tests, describing actual behavior including known
      warts. The first batches have landed
      ([#3924](https://github.com/kornia/kornia/pull/3924),
      [#3926](https://github.com/kornia/kornia/pull/3926)); the canonical
      [Conventions & Pitfalls](https://kornia.readthedocs.io/en/latest/get-started/conventions.html)
      page is live. Behavior bugs found by the audit get dedicated issues.
   2. **Repair & deprecate** — one coordinated window fixes accumulated semantic and
      default-value bugs, so downstream users face a single loud migration instead of
      a drizzle of breaking changes. The window runs *under* the
      [stability policy](https://kornia.readthedocs.io/en/latest/get-started/stability.html),
      not around it: every semantic or default change gets the policy's deprecation
      treatment (at least one minor release of warnings, landing at a 0.x minor
      boundary) — "coordinated" means the changes are batched into one migration,
      not that the deprecation rules are waived. Only clearly-broken-output bugs
      (NaN, crashes) use the policy's correctness escape hatch and land early as
      ordinary bugfixes.
   3. **Freeze** — after the window, a **Tier A ("Kornia Core")** set is declared:
      symbols with the strongest guarantees (autograd, gradcheck, compile, dtype/
      device correctness, explicit conventions, semantic stability). Tier A starts
      brutally small and a guarantee is only claimed where CI enforces it.
      **These tiers are proposed, not yet in force**: until the tier policy is
      published after the window, the current
      [stability policy](https://kornia.readthedocs.io/en/latest/get-started/stability.html)
      (stable core / best-effort / experimental) remains the authoritative contract.

   Alongside this grows a **conformance corpus**: framework-neutral golden vectors
   (inputs, expected outputs, tolerances, convention metadata) that external
   implementations — JAX ports, Rust/Triton kernels, vendored copies, and
   LLM-generated code — can validate against. Long-term, Kornia defining the
   conformance tests for differentiable geometric vision may matter more than owning
   every implementation.

2. **Compile-first (`torch.compile` / dynamo).** The numeric core (filters, color,
   geometry, enhance, losses) is already largely compile-clean. The remaining work is
   concentrated in the stochastic augmentation pipeline and the dynamic-shape feature
   detectors, which branch on tensor values and break the graph — the augmentation
   classes fail structurally in data-dependent parameter generation
   ([#3913](https://github.com/kornia/kornia/issues/3913)). Compile support is
   claimed **per evidenced surface**: today that means the functional core, not the
   augmentation classes as a class-level guarantee — many individual augmentations
   already run fullgraph in the PR-time dynamo job (see the compile/export spine
   below), but the data-dependent parameter generators keep the class-level claim
   off the table — and docs state the scope explicitly.

3. **Export-first (ONNX).** ONNX export fails on the *same* patterns that break
   `torch.compile`: data-dependent control flow, `.item()` calls, and dynamic shapes.
   [#3722](https://github.com/kornia/kornia/pull/3722) demonstrated the fix — replacing
   the stochastic apply path with `torch.where` blends and gating linear-algebra
   fallbacks on `is_tracing()` — which advanced both goals in a single change. The
   remaining spine work pays out to compile and export together.

4. **Performance, honestly measured.** The benchmark harness is now first-class:
   `benchmarks/` measures kornia (eager + compiled) against OpenCV, torchvision v2,
   albumentations, and PIL with median+IQR timings, hardware metadata, and an
   auto-published results page — and it publishes where Kornia *loses* (CPU,
   batch=1) alongside where it wins (large-batch GPU, differentiability, compiled
   paths), so the page is citable by skeptics. Measurement drives the fix order:
   the systemic finding is that the augmentation pipeline is **launch-bound, not
   kernel-bound** (per-call wrapper orchestration can make throughput *fall* as
   batch grows), so orchestration overhead is fixed first; then the measured kernel
   outliers (`median_blur`, `canny`, `rotate`). For the non-differentiable
   CPU/`uint8` regime an opt-in `kornia-rs` backend is the path to a step-change,
   and hand-written Triton kernels are reserved for the small set of ops `inductor`
   structurally cannot fuse.

A fifth theme, **breadth**, is community-driven. One boundary is stated openly:
**model-zoo expansion is frozen.** No new model
or VLM/VLA integrations while maintainer bandwidth concentrates on the core; shipped
wrappers (LoFTR, LightGlue, DISK, DeDoDe, SAM, XFeat, ALIKED, Kimi-VL, …) remain
available and maintained under a *usable-or-deleted* rule. The single exception
rule, applied everywhere in this document: **a new integration requires a named
maintainer sponsor who accepts ongoing ownership — a contributor's implementation
alone, however good, does not reopen the surface.** The freeze lifts when a
maintainer with model-work capacity joins the project. Contributions already in
flight before the freeze (an open issue plus a PR under review) are grandfathered
and finish under the rules they started under.

## Short term — next release

- **Stabilize CI and the augmentation core.** Recent fixes repaired regressions from the
  ONNX-exportability refactor and healed the scheduled test matrix. Ship these in a
  release so users get the fixes.
- **Convention-block batches.** Continue the per-op audit:
  `geometry.conversions` → bounding boxes
  ([#3934](https://github.com/kornia/kornia/issues/3934)) → camera → augmentation
  classes. Each batch delivers Convention blocks, pinning tests, and dedicated
  issues for any behavior bug found.
- **Broken-output bugfixes** from the audit, landing ahead of the repair window:
  pyramid bounds check ([#3927](https://github.com/kornia/kornia/issues/3927)),
  1-pixel crop NaN ([#3929](https://github.com/kornia/kornia/issues/3929)).
- **CI truthfulness.** Every advertised CI matrix axis must actually reach the code
  under test; first repair is the unused pytorch-version axis
  ([#3930](https://github.com/kornia/kornia/issues/3930)).
- **Contribution documentation:** keep the social contract, technical guidance, and
  project automation consistent and easy to understand.
- **Progressively enable more `ruff` rule sets** to raise code health
  ([#2445](https://github.com/kornia/kornia/issues/2445)).
- **Docs modernization** — evaluate the migration to MkDocs
  ([#3454](https://github.com/kornia/kornia/issues/3454)).
- **Model-zoo maintenance (best-effort tier; future Tier C).** Shipped integrations
  must be usable or deleted — kept loading, tested, and honest (recent example:
  repointing Kimi-VL to working weights). The 2025-era coverage-gap issues are
  resolved; newly found maintenance work gets a current tracking issue with
  acceptance criteria — no new integrations.
- **`help wanted` label audit.** Older issue templates auto-applied the label;
  re-triage open issues so the label points to current, well-described work where
  outside help would be useful.

## Medium term — ~6 months

- **The Repair & Deprecation window → Tier A v1.** Execute the one coordinated
  release motion for the semantic and default-value repairs accumulated by the
  audit (e.g. `warp_image_tps` `align_corners` default
  [#3928](https://github.com/kornia/kornia/issues/3928), `pyrdown` output rounding,
  inconsistent 3D-crop defaults, bbox width arithmetic
  [#3934](https://github.com/kornia/kornia/issues/3934)) — each change with an
  old-vs-new conformance vector pair, the stability policy's deprecation treatment
  (at least one minor release of warnings before landing at a 0.x minor boundary),
  and one collective migration note. Afterwards, publish the support-tier policy and
  declare Tier A v1: small, explicit, CI-enforced. Until that publication the
  current stability policy remains the authoritative contract.
- **The conformance corpus.** Define the framework-neutral format (per-dtype/device
  tolerances, convention metadata), migrate the existing convention-pinning tests
  into it, add semantic invariant tests
  (`warp(inv(H), warp(H, img)) ≈ img`, `project(unproject(d, K), K) ≈ uv`,
  `R @ R.T ≈ I`), and publish a docs page: *validate your (hand-written or
  LLM-generated) geometry code against Kornia*.
- **The agent knowledge layer.** `AGENTS.md`, `llms.txt`, and `llms-full.txt` are live and
  benchmark-fed; after the final convention batch, ship a small set of validated
  skills (image warping & homographies, batched differentiable
  augmentation with transform tracking, camera/epipolar conventions, feature
  matching) — all derived from the canonical docs, never duplicating them.
- **The compile / export spine.** *In progress — much of the augmentation surface now
  compiles fullgraph.* The stochastic-apply path, the `_extract_device_dtype`/version-check
  helpers, the transform-matrix blend, and the shape-changing crops (`Resize`, `CenterCrop`)
  have all landed as genuine single-path fixes; common intensity ops (color jitter, solarize,
  brightness/contrast, erasing, flips) are fullgraph on the CI torch. Remaining: the
  data-dependent tail — random-coordinate `RandomCrop`/`crop_by_indices`, histogram-based
  `equalize`, and random-permutation dispatch — which need redesign, not guards
  ([#3913](https://github.com/kornia/kornia/issues/3913)). On the export
  side: solve multi-output ONNX export generically so tuple-returning modules (e.g. `Canny`,
  YUV conversions) are no longer blocked, and converge on a single modern ONNX opset.
- **Dynamo tests in CI — landed for the compile-clean core.** A PR-time `dynamo` job now runs
  the compile-clean augmentation/enhance/filter/color/loss/morphology tests under `inductor` on
  the CI torch version, so a fullgraph regression is caught at PR time. Still to do: publish an
  ONNX export conformance matrix (exportable / numerically-verified / blocked-with-reason) for
  the public API, and widen the dynamo job beyond the core once model/feature paths are vetted.
- **Augmentation performance — orchestration first.** GPU profiling shows the deficit
  vs torchvision on cheap augmentations is *not* the kernels: for
  `RandomHorizontalFlip` the raw flip is ~22% of the module forward and the other
  ~78% is per-call orchestration in the augmentation base (transform-matrix build,
  the `where`-blend, dtype/shape bookkeeping). The highest-leverage work is a leaner,
  fully-compilable base `forward` (a fast path that skips `compute_transformation`
  for non-affine ops) plus CUDA-graph capture — one fix that lifts every
  augmentation. Compiled kornia pipelines already beat torchvision v2 on GPU while
  staying end-to-end differentiable (see the published benchmark page); killing
  wrapper overhead extends that lead to the eager path everyone runs by default.
- **A `kornia-rs` augmentation backend (opt-in).** Add a backend selector (e.g.
  `backend="rust"`) that routes non-differentiable, CPU, `uint8` augmentation through
  `kornia-rs`, while the PyTorch path stays the default for GPU-batched,
  differentiable, and float workloads. Upstream kernels are already fast (recent
  `kornia-rs` releases beat OpenCV raw on warps); the current bottleneck is the
  tensor↔image interop cost, and the dispatch only ships for ops where it wins
  ≥5× including conversion. Sequencing is conformance-first: `kornia-rs` outputs are
  validated against Kornia's golden vectors before any dispatch, making it the
  conformance corpus's first external consumer.
- **Triton for the kernel-bound minority.** Hand-written kernels only where
  `inductor` structurally cannot help and the benchmark data shows a real gap:
  `median_blur` (selection networks), `canny` (data-dependent NMS + hysteresis),
  possibly large-kernel morphology. Capped ambition (2–3 kernels), each with a
  registered custom op, backward, and conformance vectors; opt-in experimental until
  GPU CI can verify them.
- **Batched RANSAC.** Restructure the RANSAC hot loop to not need a compiler:
  vectorize over hypotheses (sample K minimal sets, solve batched, score batched)
  with chunked execution and confidence-based stopping. GPU-batched robust geometry
  is core mission; the variants (eager, lazy-scripted, compiled-chunked, batched)
  are benchmark-arbitrated.

## Long term — vision

- **The reference implementation for differentiable CV.** External implementations —
  JAX ports, native kernels, vendored code, and code written by LLMs — validate
  against Kornia's conformance data, and humans and coding agents consistently
  produce *correct* geometry code from Kornia's documentation. Success is fewer
  convention bugs in the wild, not more modules in the package.
- **A fully compilable, fully exportable core.** Every applicable Tier A tensor
  operator runs under `torch.compile` without unnecessary graph breaks and exports
  to ONNX where ONNX can represent its semantics, with verified numerical
  equivalence — including the stochastic augmentation and dynamic-shape feature
  paths that break today. Best-effort and experimental surfaces are covered as
  evidence allows, never by blanket claim.
- **The fastest differentiable, GPU-batched augmentation stack**, with a published,
  reproducible benchmark that states precisely the regime in which we lead.

## Areas seeking contributors

We especially welcome help in these areas. The
[`help wanted`](https://github.com/kornia/kornia/labels/help%20wanted) label is one
place to look, although some older issues still need to be checked for current scope
and context. Search for related issues and pull requests before investing substantial
work, and see [CONTRIBUTING.md](CONTRIBUTING.md) for the social contract and technical
guidelines.

**Classical vision in the core domain (highest priority):** these strengthen exactly
the geometry/camera surface the project is built around.

- **Camera intrinsic calibration** (Zhang's method, checkerboard / ChArUco
  detection). Kornia can undistort and solve PnP but cannot yet calibrate a camera
  from scratch — the largest gap in the camera core.
- **Fiducial markers** (ArUco / ChArUco) — detection feeding directly into the
  calibration and pose pipelines.
- **Classical tracking** — Lucas-Kanade, KCF
  ([#1381](https://github.com/kornia/kornia/issues/1381)).
- Dense stereo matching, Hough transforms and Hough voting, contour/shape analysis,
  template matching.

**Reference-core contributions:**

- **Benchmark results from your hardware** — run the
  suite with `--contribute` and send the JSON; CUDA numbers from diverse GPUs are
  especially wanted.
- **Convention pinning tests and conformance vectors** for core geometry operations,
  and **corrective error messages** (upgrading bare shape asserts into errors that
  state what was wrong, what was expected, which convention applies, and the likely
  fix). Because these changes can define or alter public behavior, sharing the
  intended contract early is useful.

**Augmentation parity** — low-priority and community-driven: Kornia does
not compete on generic augmentation breadth (the differentiated value is
differentiability, GPU batching, and transform tracking). Ideas in this area include
dropout-family transforms (CoarseDropout, GridDropout), grid/optical distortion,
weather effects (fog, sun-flare),
noise variants (ISO noise), and additional compression transforms.

**Learned models — frozen (see the guiding themes):**

- **Grandfathered model work.** Efficient LoFTR
  ([#3282](https://github.com/kornia/kornia/issues/3282), PR
  [#3621](https://github.com/kornia/kornia/pull/3621) under review) and SANDesc
  ([#3752](https://github.com/kornia/kornia/issues/3752), pre-freeze maintainer
  approval on the issue) were approved and started before the model-zoo freeze and
  may continue under their existing scopes. They are exceptions honoring prior
  commitments, not invitations for new model integrations.
- **Acknowledged gaps — not currently soliciting contributions.** Learned optical
  flow (RAFT / SEA-RAFT), pose estimation, and object-detection breadth beyond
  RT-DETR and YuNet are real gaps, listed for honesty. New model work remains
  frozen until a maintainer accepts ongoing ownership of the integration.

**Code health & infrastructure:**

- Expanding `ruff` rule coverage ([#2445](https://github.com/kornia/kornia/issues/2445)).
- Migrating test classes to `BaseTester`
  ([#2752](https://github.com/kornia/kornia/issues/2752)).
- `torch.compile` and ONNX-export coverage for existing operators.

## How to contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for the social contract, development setup,
and technical guidelines. Tell us what you checked and how AI helped, if you used it.
For a new algorithm, a reference such as a paper, OpenCV, or PyTorch makes the idea
and implementation easier to understand.
