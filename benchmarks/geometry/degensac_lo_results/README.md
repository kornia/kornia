# DEGENSAC local-optimization routing experiment

Measured on 2026-10-05 against Kornia `0.9.0rc1`, base `c7b3737e60737d6b64789d00a506a7d127d60b63`.

## Decision

Skipping ordinary fundamental-matrix LO for every H-degenerate pooled candidate is unsuitable for
Kornia's current estimator. On five calibrated Reichstag pairs, pose mAA falls from **0.881 to 0.652**
at a 4,096-sample budget, while latency changes from **33.94 to 33.74 ms**. The loss occurs on all
five pairs. At 256 samples the corresponding accuracy falls from 0.687 to 0.309.

A narrower experimental change was also evaluated: recovered models have already received
truncated-loss LM and skip its repeated final-pool pass. Raw candidates still receive their first
LO, including ones classified H-degenerate. Final Cauchy refinement is retained. This preserves
Reichstag mAA exactly in this experiment and saves about 1% there in the repeated timing run;
it does not explain away DEGENSAC's substantial overhead over plain F RANSAC.

## Native flag trace

Inspected pydegensac revision `24ad783720b41647f2ba998b165097b76d3d47a1`.
The Python `enable_degeneracy_check` argument forwards to the native DEGENSAC gate;
it does **not** globally toggle local optimization. The binding separately passes `do_lo=1`
and `inlLimit=0`. Error-function selection, symmetric checks, LAF checks, and confidence are
also configured independently.

The enabled branch nevertheless changes **which refinement runs**. In `exp_ranF.c`, a raw
record setter whose sample is H-degenerate enters homography evaluation, `innerH` refinement,
and `rFtH` plane-and-parallax F recovery. The alternative branch prepares ordinary F LO.
The recovery counter `degen_cnt` also suppresses the final fallback ordinary LO. Thus the flag
is a DEGENSAC gate with refinement-routing consequences, rather than a check layered over
an otherwise identical LO schedule.

Source anchors at that revision:

- [Python argument and forwarding](https://github.com/ducha-aiki/pydegensac/blob/24ad783720b41647f2ba998b165097b76d3d47a1/src/pydegensac/utils.py#L121)
- [Independent LO parameters and degeneracy gate](https://github.com/ducha-aiki/pydegensac/blob/24ad783720b41647f2ba998b165097b76d3d47a1/src/pydegensac/bindings.cpp#L370)
- [Detected-sample recovery branch](https://github.com/ducha-aiki/pydegensac/blob/24ad783720b41647f2ba998b165097b76d3d47a1/src/pydegensac/degensac/exp_ranF.c#L462)
- [Alternative ordinary-LO preparation](https://github.com/ducha-aiki/pydegensac/blob/24ad783720b41647f2ba998b165097b76d3d47a1/src/pydegensac/degensac/exp_ranF.c#L515)
- [Fallback-LO suppression](https://github.com/ducha-aiki/pydegensac/blob/24ad783720b41647f2ba998b165097b76d3d47a1/src/pydegensac/degensac/exp_ranF.c#L617)

## Policies

| Policy | Ordinary truncated-loss LO after sampling | Final Cauchy refinement |
| --- | --- | --- |
| Plain F | All eight pooled minimal models | Selected model |
| Current DEGENSAC (base) | All pooled raw and recovered models | Selected model |
| Planar skip (rejected) | Only raw models whose samples are not H-degenerate | Selected model |
| Recovered only (archived trial) | Raw models; recovered models skip their repeated pass | Selected model |

pydegensac routes *raw record setters* during sampling: a flagged sample goes through homography
refinement and F recovery, while the other branch schedules ordinary F LO. A degeneracy-recovery
counter also suppresses its final fallback LO. Kornia instead selects and refines an eight-model
pool after sampling. The rejected experiment adapts the candidate-level skip rule to that pool;
it does not reproduce pydegensac's exact event-driven LO timing or candidate set.

Chum's five-of-seven homography test can also flag samples in scenes without a dominant plane.
The real-data results show that Kornia still needs ordinary F refinement on many such raw models.
Avoiding that work is therefore not a valid unconditional optimization of the current pipeline.

## Latency

Mean of per-pair/per-seed **repeated-call medians**, milliseconds, at 4,096 samples and confidence
0.999. Estimator timing seeds are 0, 1, 2. The first run uses at least 0.3 s per timing cell.
Construction, data loading, and accuracy evaluation are outside the timed call.

| Input | Plain F | Current DEGENSAC | Planar skip | Recovered only |
| --- | ---: | ---: | ---: | ---: |
| 60% inliers, no plane | 23.66 | 27.10 | 29.40 | 27.48 |
| 60% inliers, 95% of inliers planar | 13.09 | 27.45 | 24.88 | 26.37 |
| 30% inliers, no plane | 48.12 | 53.03 | 53.58 | 53.66 |
| 30% inliers, 95% of inliers planar | 49.99 | 90.78 | 89.84 | 92.42 |
| Reichstag (5 pairs) | 17.08 | 33.94 | 33.74 | 33.79 |
| LoFTR indoor (1 pair) | 38.02 | 53.08 | 53.45 | 52.09 |

A second sequential base/retained run uses the same seeds with at least 0.5 s per timing cell.
The synthetic differences are mixed, and the small first-run gains do not all repeat.

| Input | Current DEGENSAC, repeat | Recovered only, repeat | Time saved |
| --- | ---: | ---: | ---: |
| 60% inliers, no plane | 26.24 | 26.68 | -1.7% |
| 60% inliers, 95% of inliers planar | 25.19 | 25.42 | -0.9% |
| 30% inliers, no plane | 50.95 | 50.80 | +0.3% |
| 30% inliers, 95% of inliers planar | 87.79 | 88.31 | -0.6% |
| Reichstag (5 pairs) | 31.05 | 30.72 | +1.1% |
| LoFTR indoor (1 pair) | 49.17 | 48.59 | +1.2% |

Plain F timings also move between rounds (Reichstag 17.08 to 16.05 ms; LoFTR 38.02 to 35.98 ms).
Treat differences of a few percent cautiously. There is no evidence here of the desired 2–4x
reduction in DEGENSAC cost, or of DEGENSAC becoming faster than plain F.

## Geometric accuracy

Twenty estimator seeds (0–19) for every fixed pair and budget. Synthetic data use scene seed 0,
1,000 correspondences, 0.5 px noise in both images, and 60% or 30% geometric inliers. The planar
scenes put 95% of those inliers on a plane. A synthetic success requires the returned F's median
Sampson distance on the **clean off-plane** points to be below 2 px; failures count as misses.

At 4,096 samples:

| Input and metric | Plain F | Current DEGENSAC | Planar skip | Recovered only |
| --- | ---: | ---: | ---: | ---: |
| 60% inliers, no plane, off-plane successes | 20/20 | 20/20 | 20/20 | 20/20 |
| 60% inliers, 95% of inliers planar, off-plane successes | 2/20 | 20/20 | 20/20 | 20/20 |
| 30% inliers, no plane, off-plane successes | 15/20 | 15/20 | 15/20 | 15/20 |
| 30% inliers, 95% of inliers planar, off-plane successes | 0/20 | 18/20 | 20/20 | 18/20 |
| Reichstag, pose mAA at 1–10 degrees | 0.886 | 0.881 | 0.652 | 0.881 |
| LoFTR, seeds with ≥2 GT errors above 10 px (lower is better) | 10/20 | 7/20 | 7/20 | 7/20 |

The planar skip improves the low-inlier synthetic planar case from 18/20 to 20/20 but severely
hurts calibrated real pairs. That is why synthetic success alone cannot justify this policy.
The retained change leaves Reichstag mAA unchanged at both budgets. On LoFTR at 256 samples,
the number of seeds with at least two gross GT errors is 14/20 plain, 16/20 current DEGENSAC,
and 15/20 for either trial. At 4,096 samples all DEGENSAC policies have 7/20 such seeds.
Both trials reproduce all 400 plain-mode quality/support records exactly against the base.

## Cost interpretation

A diagnostic CPU profile of the rejected planar-skip policy on the high-inlier planar scene
put approximately 19 of 26 ms inside `_degensac_batch`: homography refinement, plane-and-parallax
search, and recovery refinement. This is a single profiled call, not a reported latency measurement.
It identifies why removing the extra pool LO cannot remove most of the overhead. Ordinary LM is
already batched across the pool, so dropping one recovered model has little effect on total time.

## Reproduction and provenance

Host: Apple arm64, macOS 26.5.1, Python 3.11.14, PyTorch 2.14.0, four PyTorch threads, CPU float32
inputs and float64 Sampson accuracy evaluation. The OpenCV pose evaluator is shared with
`ransac_cpu.py` and uses OpenCV 5.0.0. Timing includes only public `RANSAC.forward` calls.
Each run prints and verifies `kornia.__file__` and prints `sys.executable`; both revisions use the
same explicitly named project interpreter. The JSON records git state and source digests.

Data: five public `golden_f_reichstag_*.npz` regression pairs from the local pydegensac repository,
and Kornia's cached `loftr_indoor_and_fundamental_data.safetensors` reference pair (639 matches,
10 GT correspondences). Thresholds are 0.75 px for Reichstag and 1 px for synthetic/LoFTR inputs.
The golden/input aggregate hash in the first two files covers both calibrated and LoFTR files;
the final harness also emits each real input's individual hash.

Copy the same harness into the base checkout before running it there. Set the interpreter variable
to the project environment, even while running from the base checkout:

```bash
KORNIA_PYTHON=/path/to/kornia/.venv/bin/python
"$KORNIA_PYTHON" benchmarks/geometry/degensac_lo.py \
  --golden-dir /data/pydegensac/tests/data \
  --loftr /data/loftr_indoor_and_fundamental_data.safetensors \
  --threshold-px 1.0 --json /tmp/degensac-lo.json
```

The repeat command adds `--budgets 4096 --seeds 0,1,2 --min-run-time 0.5`.
No PhotoTourism benchmark export was cached; this is five pairs from one outdoor scene and one
indoor pair, not a multi-scene accuracy study. No CUDA timings were run. MPS focused tests were
skipped by the repository's process-abort guard for virtualized Apple GPUs (#4204).

Raw results: [base](base.json), [rejected planar skip](planar_skip_rejected.json),
[retained recovered-only policy](recovered_only.json), [base repeat](base_repeat.json),
and [retained repeat](recovered_only_repeat.json). Full runs contain 800 rows each (including
both plain and DEGENSAC), and repeats contain 60 rows each. Quality uses all 20 seeds; timed
seeds are 0, 1, 2. Styling/metadata fixes to the harness between rounds do not change the estimator
configuration, inputs, quality calculation, or timed call; hashes record the actual harness used.

The [rejected patch](planar_skip_rejected.patch) and [retained patch](recovered_only.patch) apply
independently to the base revision. The rejected patch is an experiment artifact, not active code.
This report archives both implementation variants as patches. The report PR is intended to be
closed without merging; it does not apply either estimator change. Keep it as a reference before
repeating the same LO-routing investigation.

Validation of the retained source: full CPU RANSAC/DEGENSAC/refinement files in float32/float64
(**546 passed, 26 skipped, 4 non-strict XPASSes**), full pre-commit, focused benchmark-file hooks,
type checking, and comparison-artifact schema validation. Review confirmed the recovered/ordinary
mixed-pool routing and zero-LO coverage. The full planar skip also passed synthetic tests, which
is another reason the independent real-pair benchmark matters.
