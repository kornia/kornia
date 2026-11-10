# Kornia Roadmap

Kornia is becoming the **reference implementation and executable specification** for
differentiable computer vision and geometry in PyTorch. Concentrate on warping and
sampling, homographies, cameras, epipolar geometry, rotations/Lie groups, and
geometry-consistent augmentation. Progress means trustworthy results, explicit
conventions, dependable composition, and honest performance; API and PR counts are
not goals.

This document owns strategy and correctness policy. [CONTRIBUTING.md](CONTRIBUTING.md)
is the contribution entry point; [AGENTS.md](AGENTS.md) guides agent execution, and
[TESTING.md](TESTING.md) describes validation. Dates are intentions, not commitments.

## Release status — 2026-10-23

[Kornia 0.9.0](https://github.com/kornia/kornia/releases/tag/v0.9.0) shipped and its
published artifacts were verified. The repair/deprecation window is open under the
existing [stability policy](https://kornia.readthedocs.io/en/latest/get-started/stability.html).
Use [#4620](https://github.com/kornia/kornia/issues/4620) for the current release sequence,
evidence, and follow-ups. Patch releases prioritize confirmed regressions and compatible fixes.

## Guiding themes

1. **Reference core.** Pixel-centre, coordinate-frame, and sampling convention mistakes
   let code run while producing mathematically wrong results; agents can reproduce those
   mistakes. Document actual behavior and known defects, repair the contracts,
   then freeze a small dependable surface. Coordinate convention changes and migration
   examples under the stability policy; the repair window does not waive it. Convention
   tests and independent expected results are the foundation of a future portable corpus
   that JAX, Rust, Triton, and other implementations can validate against.
2. **Compile-first.** Design tensor paths that work with `torch.compile` without avoidable
   graph breaks. Claim support per evidenced surface, not for every augmentation or model.
   The [compile promise](TESTING.md#torchcompile-conformance-5819) defines what CI enforces.
3. **Export-first.** Treat ONNX graph construction and numerical runtime behavior as part
   of the implementation. Prefer designs that improve compile and export together;
   [ONNX conformance](TESTING.md#onnx-conformance-5819) defines the enforced scope.
4. **Performance, honestly measured.** Use [the benchmark methodology](benchmarks/README.md)
   to compare realistic workloads, publish losses as well as wins, and choose work from
   measured bottlenecks. Reduce augmentation orchestration overhead before adding kernels;
   specialized kernels need evidence that the compiler cannot do the job.

**Model-zoo expansion is frozen** while maintainers concentrate on the core. Shipped
wrappers remain maintained under a usable-or-deleted rule. A new integration requires
an explicitly named maintainer sponsor who accepts ongoing ownership; a contributor's
implementation alone does not reopen the surface. The freeze lifts when a maintainer
with model-work capacity joins. Contributions already in flight before the freeze
(an open issue plus a PR under review) are grandfathered under their original rules;
the named pre-freeze commitments are listed below.

**Tier A/B/C is proposed, not in force.** The current stable-core / best-effort /
experimental stability policy remains authoritative. A future Tier A starts small and
claims autograd, dtype/device, compile, and semantic guarantees only where CI enforces them.

## Correctness and backend support

**Maintain correct, useful behavior within Kornia's agreed scope.** Distinguish
mathematical defects, numerical instability, and upstream workarounds.

1. **Check the contract.** Establish supported inputs, output dtype, conventions, and
   accuracy requirements. Separate wrong mathematics or avoidable instability from
   rounding, dtype limits, and undefined behavior. Compare with plain PyTorch where
   applicable: a sum exceeding float16's range does not justify widening its output.
   This does not excuse unstable computation of a representable result.
2. **Do not expand support through bug fixes.** Do not add integer-input support or
   special paths solely for synthetic extremes: half-precision overflow far beyond
   image/coordinate ranges, `1e20` float32 image values or pixel coordinates, or a matrix
   scaled by `1e15`. These are examples, not universal magnitude cutoffs; `1e5` already
   exceeds float16's finite range. Supporting each feature does not imply supporting every
   combination; float64 ONNX export is outside Kornia’s supported scope. Stop implementation
   and explain the supported input form. Expanding scope requires an explicit maintainer decision backed by a concrete application.
   A synthetic failure is insufficient; documented exceptions remain supported.
3. **Check the existing solution first.** If a supported dtype conversion, parameter
   such as `num_iters`, or workflow adequately solves the reported need, do not add
   another implementation without demonstrated benefit. Value and maintenance cost
   can justify declining a proposal, not merely postponing it.
4. **Check upstream before adding a workaround.** Expected autocast precision changes
   and SVD backward limitations at repeated singular values are not by themselves PyTorch
   defects. Kornia owns choosing a formulation and working precision that meet its contract.
   Isolate suspected PyTorch failures
   on latest stable and an available next-version build (pre-release or nightly) on the
   affected backend; record unavailable checks. If fixed upstream, default to documenting
   affected versions and alternatives without a new workaround or silently weakening
   existing promises. Record accepted workarounds' upstream issue, verified scope,
   measured cost, and retirement/reassessment condition beside the code. This also
   applies to backend performance workarounds.
5. **Prove the correction and report consequences.** Derive expected results independently
   and pin conventions with `test_convention_*` tests. An unchanged or worse downstream
   benchmark does not disprove a correction. Report results with uncertainty where
   applicable, investigate regressions, and leave acceptance to a maintainer; release
   notes state accepted tradeoffs. Prefer a shared stable formulation; extra device,
   version, size, or grad-mode branches need a measured reason stated beside the branch.
6. **Choose the outcome explicitly.** Repair the implementation, document a limitation, propose
   a scope change, or make no change. Measure performance on affected and common
   unaffected paths. Use justified dtype tolerances, not blanket bitwise equality.
   Exposure sets priority, not whether a proven defect exists. Unknown user incidence stays unknown.

At degenerate or nonsmooth points, establish whether the value or derivative exists
and is unique. A finite fallback alone does not prove correctness; document the choice.
For ambiguous boundary behavior, consult the
[stability policy](https://kornia.readthedocs.io/en/latest/get-started/stability.html):
a proven contract violation uses the correctness-fix route; changing a working
convention uses deprecation. Missing documentation or variation across PyTorch versions
alone does not authorize a silent change; unresolved cases need a maintainer decision.

## Short term — maintenance and correctness candidates

Confirmed downstream regressions take precedence. These are investigation priorities,
not release commitments. Issue reservations and ownership remain in force; the linked
issues own live status and acceptance decisions.

- **Coordinates and sampling:** feature pixel centres
  ([#6536](https://github.com/kornia/kornia/issues/6536),
  [#6537](https://github.com/kornia/kornia/issues/6537)), measured on matching/pose tasks;
  depth-warp identity [#6317](https://github.com/kornia/kornia/issues/6317), coordinated
  with the existing [#6371](https://github.com/kornia/kornia/pull/6371) author.
- **Mathematical geometry:** thin-plate splines
  [#6404](https://github.com/kornia/kornia/issues/6404) and fundamental-matrix RANSAC
  degeneracy [#6574](https://github.com/kornia/kornia/issues/6574), including downstream
  effects and release treatment.
- **Supported execution paths:** failures established by current
  [precision evidence](https://kornia.readthedocs.io/en/latest/get-started/precision.html).
  CUDA-half support is not a blanket patch-release guarantee.
- **Usable guidance and maintenance:** runnable workflows, version-correct documentation,
  coherent contributor/agent instructions, and maintained shipped models. Keep `help wanted`
  labels tied to current, well-described work. Release checks remain in #4620.

## Medium term — ~6 months

- **Coordinated repairs and Tier A v1:** settle box, camera, sampling, and augmentation
  conventions with executable old/new examples and a collective migration note. Concrete
  cases include bbox width arithmetic ([#3934](https://github.com/kornia/kornia/issues/3934))
  and the split `align_corners` defaults in augmentation ([#4412](https://github.com/kornia/kornia/issues/4412)),
  both reserved for maintainer implementation. Follow
  the stability policy's warning period and minor boundary for deliberate changes;
  use its correctness exception only for proven defects. Publish tiers after the repairs.
- **Portable conformance corpus:** define inputs, independent expected results,
  tolerances, and convention metadata; migrate convention tests and add composition
  invariants such as projection/unprojection and rotation orthogonality. Publish how
  external implementations can validate against Kornia.
- **Knowledge for users and agents:** keep package help, canonical docs, `llms.txt`, and
  `llms-full.txt` aligned. Derive validated workflow skills from those sources rather
  than duplicate conventions; assess them through completed user tasks.
- **Compile/export design:** finish data-dependent augmentation paths,
  tracked in [#5137](https://github.com/kornia/kornia/issues/5137): partial application in
  `RandomCrop` (`p < 1`), `ColorJiggle`'s random order, and `AugmentationSequential`'s
  `random_apply` dispatch. Recorded fullgraph runs of `RandomAffine`, `RandomPerspective`,
  and `RandomResizedCrop` include CUDA successes; these are evidence for the tested cases,
  not a blanket CUDA promise. `crop_by_indices` needs an explicit `size` to avoid value-dependent output shapes.
  Dynamic-shape feature detectors remain separate design work.
  Redesign the shared implementation instead of accumulating guards. Handle multi-output
  export generically, converge on a modern ONNX opset, and extend CI promises only with evidence.
- **Augmentation throughput:** reduce wrapper/transform bookkeeping and kernel-launch
  overhead, with CUDA-graph capture where useful. The [recorded GPU profile](benchmarks/augmentation/README.md#known-improvement-opportunities-the-improve-later-list)
  attributes about 78% of `RandomHorizontalFlip` forward to orchestration and 22% to the raw
  flip; a transform-matrix fast path has partially addressed it. Remeasure after changes.
  Keep eager and compiled comparisons
  in the benchmark evidence rather than promising uniform speedups.
- **Optional specialized execution:** evaluate a `kornia-rs` backend for non-differentiable
  CPU/`uint8` workloads only after conformance validation and at least a 5x improvement
  including conversion. This is an explicit backend proposal, not permission to add integer
  paths to ordinary operators. Limit hand-written Triton work to 2–3 measured kernel
  bottlenecks, such as `median_blur` and `canny`, with registered ops, backward, and conformance
  evidence; experimental until validated. The [benchmark evidence](benchmarks/README.md#the-honest-reading-across-apple-silicon-l4-rtx-pro-6000-rtx-4090wsl2)
  requires remeasuring `rotate` with matched interpolation before treating it as an outlier.
- **Batched RANSAC:** vectorize hypothesis sampling, solving, and scoring with chunked
  execution and confidence-based stopping. Choose eager/scripted/compiled variants by
  correctness and benchmarks, without requiring a compiler for the basic batched path.

## Long term — vision

- A reference implementation whose conformance data makes correct geometric vision
  easier to implement across frameworks, languages, and human/agent workflows.
  Success is fewer convention bugs in users' code, not more modules in the package.
- A compilable and exportable core with per-symbol, CI-enforced guarantees, scoped to
  semantics ONNX can represent; best-effort and experimental coverage grows with evidence.
- A leading differentiable, GPU-batched augmentation stack, with reproducible evidence
  stating exactly where it leads and where it does not.

## Areas seeking contributors

Check [help wanted](https://github.com/kornia/kornia/labels/help%20wanted), existing issues,
and [the contribution rules](CONTRIBUTING.md) before implementation. A roadmap area is
not approval for a new API; establish the concrete need and check reservations first.

- **Documented CUDA and half-precision failures:** fix existing CUDA reports, open
  half-precision issues, and entries in the known-failure manifests for
  [`float16`](testing/half_precision_xfails/cpu_float16.txt) and
  [`bfloat16`](testing/half_precision_xfails/cpu_bfloat16.txt). Take CUDA work only when a CUDA device is available
  for reproduction and validation. Apply the [correctness policy](#correctness-and-backend-support)
  and [precision testing guidance](TESTING.md#7-half-precision-dtypes-float16--bfloat16);
  remove the corresponding xfail or manifest entry after verifying the fix.
- **Reference evidence:** convention tests, independent expected vectors, corrective
  error messages, and hardware benchmark contributions, especially CUDA. Use the
  benchmark suite's `--contribute` workflow.
- **Code health:** compile/export coverage for existing operators.
- **Augmentation breadth:** low-priority, community-driven gaps such as dropout,
  distortions, weather/noise, and compression effects; the differentiated core remains
  differentiability, batching, and annotation consistency.

**Grandfathered model work:** Efficient LoFTR
([#3282](https://github.com/kornia/kornia/issues/3282),
[#3621](https://github.com/kornia/kornia/pull/3621)) and SANDesc
([#3752](https://github.com/kornia/kornia/issues/3752)) retain their pre-freeze scopes.
Learned optical flow, broader pose/object detection, and VLM/VLA integrations are
acknowledged gaps, not open invitations; the sponsor rule in the guiding themes applies.

## How to contribute

Start with [CONTRIBUTING.md](CONTRIBUTING.md). Keep implementation, testing, release,
and attribution procedures there and in the linked technical guides rather than
adding another copy to this roadmap.
