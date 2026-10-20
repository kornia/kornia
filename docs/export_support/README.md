# Graph-capture survey behind the survey page

The [ONNX, torch.compile and torch.export survey](../source/get-started/index.rst) page of the
documentation is rendered from `docs/source/_data/export_support.json`, a committed snapshot of
what this directory measures. The page itself is generated at docs build time by
`docs/generate_export_support.py`; nothing here runs during a docs build.

## What is measured

Every *case* is one public callable (function or `nn.Module` instance) with concrete CPU float32
inputs, registered in one of the `cases_*.py` files:

| file | covers |
| --- | --- |
| `cases_aug.py` | `kornia.augmentation` (2D, 3D, containers, auto policies) |
| `cases_feature.py` | `kornia.feature` (detectors, descriptors, matchers) |
| `cases_geomA.py` | `kornia.geometry` transforms, conversions, cameras, calibration |
| `cases_geomB.py` | the rest of `kornia.geometry` (epipolar, Lie groups, boxes, ...) |
| `cases_misc.py` | `kornia.filters`, `color`, `enhance`, `morphology`, `image`, `losses`, `metrics`, `utils` |
| `cases_models.py` | `kornia.models`, `contrib`, `sensors`, `tracking`, `x` (weights are downloaded when they are not cached) |

For each case:

- **ONNX** (`harness.py`): `torch.onnx.export(dynamo=True, opset_version=18)`, `onnx.checker`,
  then an `onnxruntime` run compared against eager: the same dtype, and values within the case's tolerance.
- **torch.export** and **torch.compile** (`probe_compile.py`): `torch.export.export`, then
  `torch._dynamo.explain` for the graph-break count and a `torch.compile` run compared against
  eager. The count is the operator's own: the wrapper returns a tensor output without probing its
  attributes, which would be a break of its own after a resumed frame. An explain count of 0 is not
  proof of a single graph (Dynamo can run part of a frame eagerly without recording a break; #6194);
  the conformance test compiles with `fullgraph=True`.

Random operators cannot be compared value-for-value (the RNG stream differs inside the graph), so
they are reported as *ok-unverified* when the captured program runs and returns finite outputs of
the right shape.

The cases also drive the conformance tests of #5819. Every case the snapshot records as ONNX "ok"
in a stable-core package, and not tagged `pretrained`, is promised to `tests/onnx/test_conformance.py`,
which runs it through the same `harness.run_case`; every such case recorded as compile "ok" is promised
to `tests/test_inductor_conformance.py`, which runs it through `probe_compile.run_case(case,
export=False, fullgraph=True)`. Pull requests run both for the packages they touch and the packages
that import them. Renaming, skipping or dropping a promised case therefore fails those tests, and the
harness and the probe must not change the process they are imported into: the survey's own process
settings (quiet logging and warnings, four threads, the compile probe's per-case timeout and Dynamo
cache limit) are applied by `run_cases` and `probe_compile.main`, not at import.

## Regenerating the snapshot

```bash
pixi run -e default install-docs && pixi run -e default build-docs   # once, so operators get cross-referenced
python docs/export_support/run.py                         # ~2 h on 8 CPU cores; resumable
pixi run -e default build-docs
```

`run.py` writes per-group results and logs to `docs/export_support/results/` (git-ignored) and
then calls `merge.py`, which produces the snapshot. Rerunning `run.py` only probes cases that have
no record yet, so a worker that crashed (a segfault leaves a `crashed` marker) can just be started
again. Each probe also writes an `inventory_*.json` naming every case of its group and the kornia
revision and torch version it ran with; records left by another revision or torch are discarded
instead of resumed. The merge refuses to write the snapshot while a worker failed, a case has no
record, or the groups were measured at different revisions, so a partial run cannot silently
undercount the coverage; `--force` merges anyway. `run.py aug` restricts the run to one case group;
`run.py --merge-only` rebuilds the JSON from existing results. Run the probes with the same torch /
onnx / onnxruntime / onnxscript versions as the snapshot header unless the point of the rerun is
to move to new versions.

## Adding a case

Register it in the matching `cases_*.py` with `case(...)` (see the docstring in `harness.py`): a
unique name, the group used as the page section, the callable, the tensor inputs, Python-only
kwargs, and a note that shows up on the page. Variants of one operator use the `name[variant]`
convention so they group under the same row label.
