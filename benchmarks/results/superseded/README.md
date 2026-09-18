# Superseded benchmark snapshots

Runs kept for history but no longer published. `docs/generate_benchmarks.py` skips this tree, so
nothing here reaches the performance page, the landing-page chart or the llms digest. The files
keep their `<kornia-version>/<suite>--<machine>--<device>.json` layout, so the schema rules in
`benchmarks/results_schema.py` still apply and the tests still validate them.

**A snapshot is identified by its kornia version *and* its `git_commit`, never by its date.** One
version directory spans many commits: a run can carry the current version and a recent timestamp
and still describe an implementation that no longer exists. Move a run here when a merged change
alters the speed of the ops it measures and the machine cannot be re-measured; re-measure instead
whenever the hardware is available.

| File | Measured at | Superseded by |
| --- | --- | --- |
| `0.9.0rc1/filters--apple-m1--cpu.json` | `83725c6b`, 2026-08-08 | #4647 — box, median, Laplacian and compiled Gaussian filters were accelerated after this run; the Apple replacement is `filters--apple-m1-pro--*` |
| `0.9.0rc1/filters--apple-m1--mps.json` | `83725c6b`, 2026-08-08 | as above |
| `0.9.0rc1/filters--apple-m4--cpu.json` | `f0a06c70`, 2026-08-18 | as above; no post-#4647 M4 run exists yet |
| `0.9.0rc1/filters--apple-m4--mps.json` | `f0a06c70`, 2026-08-18 | as above |
| `0.9.0rc1/augmentation--apple-m1--cpu.json` | `83725c6b`, 2026-08-08 | #4647 — `RandomGaussianBlur` delegates to `kornia.filters.gaussian_blur2d`, so its rows predate the change; replaced by `augmentation--apple-m1-pro--*` |
| `0.9.0rc1/augmentation--apple-m1--mps.json` | `83725c6b`, 2026-08-08 | as above |
| `0.9.0rc1/augmentation--apple-m4--cpu.json` | `f0a06c70`, 2026-08-18 | as above; no post-#4647 M4 run exists yet |
| `0.9.0rc1/augmentation--apple-m4--mps.json` | `f0a06c70`, 2026-08-18 | as above |
| `0.9.0rc1/augmentation--apple-m1-pro--cpu.json` | `6d1da7b6`, 2026-09-18 | #4659 — fixes crop recompilation and shared augmentation compile caches; no post-fix M1 Pro run is available |
| `0.9.0rc1/augmentation--apple-m1-pro--mps.json` | `6d1da7b6`, 2026-09-18 | as above |

The M1 and M4 files are single warmed runs of the filters and augmentation flagships against the pre-#4647
implementations. Published beside a post-#4647 run they would read as a hardware difference — for
example `median_blur` at batch 32 measured 11 img/s on the M1 and 16 img/s on the M4 before the
change against 69 img/s on an M1 Pro after it, and `RandomGaussianBlur` at batch 32 measured
157 img/s on the M1 and 414 on the M4 against 3235 on an M1 Pro — which is a change in kornia, not
in the machines. The augmentation classes hold no filter implementation of their own; they call
`kornia.filters`, so a filters change moves their numbers too.

The M1 Pro augmentation pair includes the filter changes but predates #4659. Its compiled
`RandomResizedCrop` rows measured repeated compilation, not steady-state throughput. CPU/CUDA
augmentation snapshots on the Intel i7-14700K and RTX 4090 were re-measured after merging #4659;
the Apple augmentation pair is archived until that hardware can be re-measured. The M1 Pro
filter snapshots remain published because #4659 does not change those filters.
