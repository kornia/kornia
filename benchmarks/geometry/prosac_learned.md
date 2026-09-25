# PROSAC on RoMa and ALIKED + LightGlue

This follows [the SIFT/XFeat experiment](prosac_stopping.md), using stronger
matchers from `imc2021-simple`. Estimator settings were not tuned on these pairs.

## Data and the RoMa cache correction

ALIKED uses the existing `raw_matches_aliked_n2048.h5` caches, produced with
LightGlue (not the separate SMNN caches). Scores are `1 - match confidence`,
so lower is better. We selected ten pairs per scene with seed 19, in scene order
sacre_coeur, reichstag, st_peters_square, british_museum,
florence_cathedral_side, lincoln_memorial_statue, london_bridge. Both features
use exactly the same 70 image pairs.

The existing RoMa 10k caches have no `roma_score_version=2` stamp. Their scores
predate the companion project's fix: RoMa's sampler clamps returned confidence,
and looking up confidence from the A-side coordinates is wrong for the B-grid
half of a symmetric warp. These caches are unsuitable for judging PROSAC.
They were left untouched.

For these 70 pairs only, we reran outdoor RoMa with the current
`examples/roma_eval.py::sample_with_scores` from companion commit `1aafdcf`.
It records the sampled flat indices and reads their **unclamped** confidence,
including the correct symmetric half. Its sampling steps were checked against
the installed RoMa sampler. RoMa source: `77f8d688`; 560 coarse / 864 upsampled
resolution, symmetric matching, 2,048 balanced sampled correspondences,
`use_custom_corr=False`, extraction seed `19 + sorted pair index`.
The output has almost one distinct confidence value per correspondence.
Preparation provenance and source hashes are recorded in `prosac_learned.json`.

## Measurement

Base is merged #4902 (`dfac11ab`); changed is prefix stopping (`cbf98d55`).
Both use MSAC, fundamental eight-point estimation, a fixed 1-pixel threshold,
maximum 4,096 samples, confidence 0.999, five full-inlier LO iterations, and
seeds 0/1/2. CPU batches are 32 and 256; CUDA uses 256. The smaller CPU batch
allows confident estimates to stop sooner instead of spending at least 256
samples. All inputs are identically sorted for both samplers.

Pose mAA is mean accuracy at 1–10 degrees, equally weighted by scene and seed.
All 70 pairs per feature are scored. Repeated-call timing uses two pairs per
scene/feature/seed, warmup, synchronization, and `common.time_us` with minimum
0.05 seconds. Reported milliseconds are the mean of these per-call medians.
Feature extraction and image loading are excluded. Base and changed timings
run sequentially; imported Kornia paths are checked for each checkout.
Hardware is i7-14700K / RTX 4090 under WSL2, one CPU thread, Python 3.11.14,
PyTorch 2.14.0+cu130, Kornia 0.9.0rc1.

## Results

### CPU

| Feature | Batch | Sampler | mAA | Time (ms) | Mean sample sets |
|---|---:|---|---:|---:|---:|
| ALIKED + LightGlue | 32 | Uniform | 0.596 | 2.10 | 46 |
| ALIKED + LightGlue | 32 | PROSAC, full budget | 0.610 | 100.15 | 4096 |
| ALIKED + LightGlue | 32 | PROSAC, prefix stopping | 0.420 | 2.35 | 32 |
| ALIKED + LightGlue | 256 | Uniform | 0.590 | 5.53 | 262 |
| ALIKED + LightGlue | 256 | PROSAC, full budget | 0.624 | 69.65 | 4096 |
| ALIKED + LightGlue | 256 | PROSAC, prefix stopping | 0.542 | 5.73 | 256 |
| RoMa | 32 | Uniform | 0.770 | 2.71 | 47 |
| RoMa | 32 | PROSAC, full budget | 0.773 | 142.86 | 4096 |
| RoMa | 32 | PROSAC, prefix stopping | 0.726 | 3.14 | 32 |
| RoMa | 256 | Uniform | 0.773 | 16.34 | 267 |
| RoMa | 256 | PROSAC, full budget | 0.791 | 209.77 | 4096 |
| RoMa | 256 | PROSAC, prefix stopping | 0.774 | 16.92 | 256 |

### CUDA

| Feature | Batch | Sampler | mAA | Time (ms) | Mean sample sets |
|---|---:|---|---:|---:|---:|
| ALIKED + LightGlue | 256 | Uniform | 0.578 | 10.55 | 263 |
| ALIKED + LightGlue | 256 | PROSAC, full budget | 0.598 | 53.42 | 4096 |
| ALIKED + LightGlue | 256 | PROSAC, prefix stopping | 0.525 | 10.63 | 256 |
| RoMa | 256 | Uniform | 0.779 | 8.54 | 268 |
| RoMa | 256 | PROSAC, full budget | 0.803 | 54.56 | 4096 |
| RoMa | 256 | PROSAC, prefix stopping | 0.760 | 11.35 | 256 |

## Interpretation: do not adopt the current stopping patch

The learned-matcher experiment rejects the earlier patch as a generally useful
speed/accuracy improvement. At CPU batch 32, ALIKED + LightGlue falls from 0.596
mAA with uniform sampling to 0.420 with prefix stopping, while taking slightly
longer. RoMa also loses accuracy (0.770 to 0.726). At CPU batch 256, RoMa is comparable
to uniform, but ALIKED still loses accuracy. On CUDA, both lose accuracy. Full-budget PROSAC is more accurate
in these cells, at a large runtime cost.

The strong matchers make uniform RANSAC cheap already: at CPU batch 32 it uses
about 46–47 samples on average. Prefix stopping uses exactly 32 in every tested
case. For ALIKED, 205/210 returned models label all ten top-ranked matches as
inliers; 87 of those models nevertheless have at least 10 degrees of pose error.
The mean global inlier fraction is 0.841, so this is not simply a failure to find
a large consensus. RoMa has 207/210 all-inlier top tens, 23 with at least 10 degrees
of pose error, and mean global support 0.918.

The stopping rule can certify a perfect prefix of only ten matches after one
required draw. That confidence concerns finding an all-inlier sample under the
assumed model; it does not establish accurate, well-conditioned pose geometry.
The measurements show that trusting it here is premature. Spatial concentration,
planarity, solver conditioning, and the interaction with local optimization need
separate diagnosis before proposing a replacement; none is established as the
sole cause by this experiment. Increasing an arbitrary minimum draw count on
these evaluation pairs would hide the problem rather than validate a fix.

All 1,260 paired uniform-control matrices and masks were identical between
revisions. There were no measurement errors. The current runtime implementation
is retained as an experimental branch for reproducing this negative result;
**`cbf98d55` should not be merged as a validated PROSAC improvement**.


## Reproduce

Export ALIKED with the existing harness:

```bash
python benchmarks/geometry/ransac.py prepare \
  --data-root ../imc2021-simple/data/phototourism \
  --scenes sacre_coeur,reichstag,st_peters_square,british_museum,florence_cathedral_side,lincoln_memorial_statue,london_bridge \
  --cache aliked_lightglue=raw_matches_aliked_n2048.h5 \
  --pairs 10 --selection-seed 19 --npz /tmp/prosac-aliked-70.npz
```

Use its `names` fields to select the same images for RoMa; call the companion's
`sample_with_scores` as described above and add `roma` records with the same
calibrations to the NPZ. The corrected per-pair outputs from this run are kept
in `/tmp/prosac-roma-v2-pairs`; the combined input is
`/tmp/prosac-learned-140.npz`. Do not substitute the stale RoMa cache scores.

From each checkout, using the same explicit interpreter:

```bash
/path/to/python benchmarks/geometry/ransac.py run \
  --npz /tmp/prosac-learned-140.npz --device cpu --threads 1 \
  --batches 32,256 --sample-budget 4096 --scores msac --prosac on,off \
  --lo-iters 5 --seeds 0,1,2 --threshold 1 --confidence 0.999 \
  --timing-pairs 2 --min-run-time 0.05 --json /tmp/learned-results.json
```

Repeat with `--device cuda --batches 256`. Evaluate with the companion project's
`imc2021.metrics` through the harness's `evaluate` command, using the combined
NPZ. The JSON report retains every scene/seed quality cell, including failures.
These are diagnostic subsets at a fixed threshold, not feature rankings at each
method's independently tuned threshold or a statistical significance claim.
