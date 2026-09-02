# Graph-capture survey behind the support page

The [ONNX, torch.compile and torch.export support](../source/get-started/index.rst) page of the
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
| `cases_misc.py` | `kornia.filters`, `color`, `enhance`, `morphology`, `contrib`, `losses`, `metrics`, `utils`, `sensors`, `io`, `nerf`, `tracking`, `x` |
| `cases_models.py` | `kornia.models` (weights are downloaded when they are not cached) |

For each case:

- **ONNX** (`harness.py`): `torch.onnx.export(dynamo=True, opset_version=18)`, `onnx.checker`,
  then an `onnxruntime` run compared against eager.
- **torch.export** and **torch.compile** (`probe_compile.py`): `torch.export.export`, then
  `torch._dynamo.explain` for the graph-break count and a `torch.compile` run compared against
  eager.

Random operators cannot be compared value-for-value (the RNG stream differs inside the graph), so
they are reported as *ok-unverified* when the captured program runs and returns finite outputs of
the right shape.

## Regenerating the snapshot

```bash
pixi run -e default install-docs && pixi run build-docs   # once, so operators get cross-referenced
python docs/export_support/run.py                         # ~2 h on 8 CPU cores; resumable
pixi run build-docs
```

`run.py` writes per-group results and logs to `docs/export_support/results/` (git-ignored) and
then calls `merge.py`, which produces the snapshot. Rerunning `run.py` only probes cases that have
no record yet, so a worker that crashed (a segfault leaves a `crashed` marker) can just be started
again. `run.py aug` restricts the run to one case group; `run.py --merge-only` rebuilds the JSON
from existing results. Run the probes with the same torch / onnx / onnxruntime / onnxscript
versions as the snapshot header unless the point of the rerun is to move to new versions.

## Adding a case

Register it in the matching `cases_*.py` with `case(...)` (see the docstring in `harness.py`): a
unique name, the group used as the page section, the callable, the tensor inputs, Python-only
kwargs, and a note that shows up on the page. Variants of one operator use the `name[variant]`
convention so they group under the same row label.
