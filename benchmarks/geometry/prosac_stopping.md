# PROSAC prefix stopping after #4902

**Follow-up:** [RoMa and ALIKED + LightGlue](prosac_learned.md) expose premature
stopping and substantial accuracy loss. The change measured here is experimental
and should not be merged as a validated PROSAC improvement. The branch restores
the merged #4902 runtime; see the [subsequent Pareto study](prosac_pareto.md).

Merged #4902 (`dfac11ab`) implements progressive sampling but always spends its
entire budget. Uniform RANSAC can stop early, so equal maximum budgets do not
mean equal work. This change adds prefix confidence stopping, leaving sampling,
scoring, solvers, local optimization, and defaults unchanged. `confidence=1`
retains the old full-budget behavior.

The original [PROSAC paper, section 2.2](https://cmp.felk.cvut.cz/~matas/papers/chum-prosac-cvpr05.pdf)
requires non-random support and enough draws within the candidate prefix. We use
the exact without-replacement all-inlier probability and a conservative Chernoff
bound for non-randomness (accidental-inlier probability 0.1, significance 0.05,
excluding the fitted minimal sample). A prefix qualifies only if its required
draw count fits before its growth interval ends. The sampler is not truncated.
These guarantees depend on the paper's assumptions about ranking and accidental
matches; they do not guarantee pose accuracy on repeated or degenerate geometry.

## Protocol

The existing `ransac.py run` and `evaluate` commands measured the public estimator
on identical ranked inputs at base and changed revisions. Each feature has 60
pairs and three seeds per dataset. The first dataset uses sacre_coeur, reichstag,
and st_peters_square (20 pairs per scene); held-out data uses british_museum,
florence_cathedral_side, lincoln_memorial_statue, and london_bridge (15 each).
SIFT uses 4k features and ratio 0.75; XFeat uses 512 and SMNN ratio 0.95.

Every run uses fundamental eight-point estimation, MSAC, batch 256, maximum 4096
minimal samples, confidence 0.999, five full-inlier LO iterations, and **one fixed
1-pixel threshold**. No threshold or stopping-parameter sweep was used. Pose mAA
averages accuracy at 1–10 degrees, first per scene/seed, then across cells.
Timing uses `common.time_us` with warmup, synchronization and repeated calls
(minimum 0.05 seconds), on two pairs per scene/feature for each seed. Tables give
the mean of those call medians, not timing over the entire quality dataset.

Hardware: i7-14700K, RTX 4090, WSL2; one PyTorch CPU thread, Python 3.11.14,
PyTorch 2.14.0+cu130, Kornia 0.9.0rc1. Base and changed runs were sequential in
separate processes; uniform-sampling timings indicate the remaining machine
noise. Input/source hashes and complete measurement metadata are in
`prosac_stopping.json`. Each harness run printed its imported Kornia path, which
was checked against the intended checkout.

## Results

### Initial scenes, CPU

| Revision | Feature | Sampler | mAA | Mean timed median (ms) | Mean sampled sets |
|---|---|---|---:|---:|---:|
| Base | sift | PROSAC | 0.473 | 39.98 | 4096 |
| Base | sift | Uniform | 0.438 | 12.95 | 1956 |
| Base | xfeat | PROSAC | 0.311 | 36.01 | 4096 |
| Base | xfeat | Uniform | 0.282 | 20.31 | 2580 |
| Changed | sift | PROSAC | 0.449 | 7.37 | 1461 |
| Changed | sift | Uniform | 0.438 | 11.57 | 1956 |
| Changed | xfeat | PROSAC | 0.300 | 15.97 | 2179 |
| Changed | xfeat | Uniform | 0.282 | 19.82 | 2580 |

### Initial scenes, CUDA

| Revision | Feature | Sampler | mAA | Mean timed median (ms) | Mean sampled sets |
|---|---|---|---:|---:|---:|
| Base | sift | PROSAC | 0.499 | 50.90 | 4096 |
| Base | sift | Uniform | 0.435 | 19.20 | 1953 |
| Base | xfeat | PROSAC | 0.312 | 55.54 | 4096 |
| Base | xfeat | Uniform | 0.251 | 35.47 | 2539 |
| Changed | sift | PROSAC | 0.469 | 10.44 | 1493 |
| Changed | sift | Uniform | 0.435 | 16.37 | 1953 |
| Changed | xfeat | PROSAC | 0.313 | 29.93 | 2165 |
| Changed | xfeat | Uniform | 0.251 | 35.63 | 2539 |

### Held-out scenes, CPU

| Revision | Feature | Sampler | mAA | Mean timed median (ms) | Mean sampled sets |
|---|---|---|---:|---:|---:|
| Base | sift | PROSAC | 0.276 | 37.35 | 4096 |
| Base | sift | Uniform | 0.277 | 10.98 | 1405 |
| Base | xfeat | PROSAC | 0.120 | 35.65 | 4096 |
| Base | xfeat | Uniform | 0.140 | 21.63 | 2421 |
| Changed | sift | PROSAC | 0.263 | 9.25 | 1146 |
| Changed | sift | Uniform | 0.277 | 10.86 | 1405 |
| Changed | xfeat | PROSAC | 0.103 | 16.40 | 2005 |
| Changed | xfeat | Uniform | 0.140 | 20.77 | 2421 |

On the initial scenes PROSAC with stopping beats uniform sampling in both measured
latency and mAA. On held-out scenes it is faster but less accurate than uniform,
especially for XFeat; full-budget PROSAC already trailed uniform there. The new
stopping rule does not resolve that accuracy limitation, and these results do
not justify making PROSAC the default. Uniform predictions and inlier masks were
bit-for-bit identical across base and changed runs in all three regimes.


Early stopping trades some full-budget accuracy for speed. Compare both accuracy
and latency against uniform sampling, rather than interpreting early stopping as
an accuracy improvement over the full-budget PROSAC control. These are diagnostic
subsets and three seeds, not a claim of statistical significance or a new default
configuration. CPU and CUDA generate different samples even with identical seeds.

## Reproduce

Prepare the two datasets with the preparation command documented in `ransac.py`,
using the scene lists and pair counts above and selection seed 19. Execute from
each checkout with the same explicit interpreter, checking the printed source:

```bash
/path/to/python benchmarks/geometry/ransac.py run \
  --npz /tmp/ransac-imc-120.npz --device cpu --threads 1 \
  --batches 256 --sample-budget 4096 --scores msac --prosac on,off \
  --lo-iters 5 --seeds 0,1,2 --threshold 1 --confidence 0.999 \
  --timing-pairs 2 --min-run-time 0.05 --json /tmp/prosac-results.json

# Use the environment containing imc2021-simple and its evaluation dependencies.
/path/to/evaluator-python benchmarks/geometry/ransac.py evaluate \
  --npz /tmp/ransac-imc-120.npz --predictions /tmp/prosac-results.json \
  --json /tmp/prosac-evaluated.json
```

Repeat with `--device cuda` for the first dataset, and with the held-out NPZ on
CPU. The fixed-budget control can also be obtained on the changed checkout with
`--confidence 1 --prosac on`; the reported baseline measurements actually run
merged #4902, rather than simulating it by changing a parameter.
