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

These four were single warmed runs of the filters flagship against the pre-#4647 implementations.
Published beside a post-#4647 run they would read as a hardware difference — for example
`median_blur` at batch 32 measured 11 img/s on the M1 and 16 img/s on the M4 before the change
against 69 img/s on an M1 Pro after it — which is a change in kornia, not in the machines.
