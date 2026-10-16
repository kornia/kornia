# Flagship audit: CPU and CUDA optimizations

Measured Kornia **0.9.0rc1**, base `520f04ee1` against implementation `2d028497e`,
on an Intel i7-14700K / RTX 4090 under WSL2, Python 3.11.14,
PyTorch 2.14.0+cu130, float32, 256×256, batches 1/8/32, four PyTorch/OpenCV threads.
The same public-API flagship harness from `2d028497e` ran against both checkouts.
This includes the corrected bilinear torchvision baselines and the new singleton-angle row.

## Batch-32 results

Median milliseconds per complete batch, lower is better. Speedup is base / changed.
Focal loss includes forward **and backward**; the other rows measure forward calls.
These numbers describe the seeded workloads on this machine, not a guarantee for every input.
Raw JSON includes all batch sizes, medians, IQRs, library versions and backend failures.
Ratios are approximate: CUDA gamma/metrics baseline and rotation runs have sizable IQRs.
The large gains are clear; small rotation gains or regressions require repeated measurements.

### Eager

| Operation | CPU base → changed (ms) | Speedup | CUDA base → changed (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `adjust_gamma` | 12.112 → 12.413 | 0.98× | 0.225 → 0.052 | 4.32× |
| `mean_iou` | 5.275 → 4.480 | 1.18× | 3.689 → 0.382 | 9.65× |
| `distance_transform` | 3831.285 → 332.239 | 11.53× | 50.151 → 5.055 | 9.92× |
| `focal_loss` | 715.680 → 133.513 | 5.36× | 19.272 → 4.107 | 4.69× |
| `rotate` | 11.192 → 10.975 | 1.02× | 1.138 → 1.086 | 1.05× |
| `rotate_shared` | 11.357 → 2.823 | 4.02× | 1.165 → 1.110 | 1.05× |

### Compiled

| Operation | CPU base → changed (ms) | Speedup | CUDA base → changed (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `adjust_gamma` | 5.552 → 5.532 | 1.00× | 0.043 → 0.038 | 1.13× |
| `mean_iou` | 5.513 → 2.108 | 2.62× | 3.711 → 0.082 | 45.14× |
| `distance_transform` | 2894.156 → 2647.926 | 1.09× | 8.052 → 8.039 | 1.00× |
| `focal_loss` | 153.502 → 108.546 | 1.41× | 2.075 → 1.588 | 1.31× |
| `rotate` | 11.512 → 11.535 | 1.00× | 0.308 → 0.325 | 0.95× |
| `rotate_shared` | 11.648 → 10.547 | 1.10× | 0.232 → 0.224 | 1.04× |

The original per-image `rotate` row is retained: equal values in B independent parameters
are not treated as a shared parameter. `rotate_shared` explicitly supplies one angle.
CPU gamma at batch 32 is effectively unchanged. The distance-transform optimization is for
eager non-differentiated execution; compiled execution keeps the original fixed loop.
Small timing differences in unchanged compiled paths should not be interpreted as code gains.
The CUDA distance-transform cells use a longer two-second repeat: the initial changed compiled
run had an IQR of 1.76 ms on a 9.49 ms median. Repeating gives 8.05 → 8.04 ms with IQRs
below 0.05 ms. Both initial and repeat JSON files are retained.
CUDA rotation cells also use a two-second batch-32 repeat at `fe17a6dd6` (documentation-only
changes after `2d028497e`). The initial apparent shared compiled regression did not repeat:
0.232 → 0.224 ms, with IQRs 0.033 / 0.013 ms. Shared eager CUDA rotation is effectively
unchanged in the repeat; the clear shared-grid gain is on CPU. Both runs are retained.

## Reproduction and artifacts

Use the same explicit Python interpreter in both worktrees, with identical dependencies.
The flagship header prints the imported Kornia path; verify it points into the intended tree.
Each run pins seeds, warms the CPU, warms compiled calls outside the measured region, and
uses `common.time_us` (synchronized blocked autorange) with `--min-run-time 0.5`.
Very slow CPU distance-transform cells have fewer repetitions than the inexpensive cells.
Runs were serial with no test or other benchmark jobs competing on this machine.

```bash
/path/to/python benchmarks/enhance/flagship.py --ops adjust_gamma --device cpu \
  --batches 1,8,32 --size 256 --threads 4 --compile --min-run-time 0.5 --json /tmp/enhance.json
```

Repeat for CUDA and the following suite/operation selections:

| Suite | `--ops` | Raw base/changed CPU/CUDA JSON |
| --- | --- | --- |
| Enhance | `adjust_gamma` | [artifacts](enhance/flagship_audit_results/) |
| Metrics | `mean_iou` | [artifacts](metrics/flagship_audit_results/) |
| Contrib | `distance_transform` | [artifacts](contrib/flagship_audit_results/) |
| Losses | `focal_loss` | [artifacts](losses/flagship_audit_results/) |
| Geometry | `rotate,rotate_shared` | [artifacts](geometry/flagship_audit_results/) |

For the baseline geometry run, load the harness from `2d028497e` through stdin while keeping
`__file__` at the baseline checkout's harness path. This supplies the new benchmark row while
importing the baseline library and recording the baseline commit:

```bash
git show 2d028497e:benchmarks/geometry/flagship.py > /tmp/geometry-flagship.py
# From the baseline worktree root, using the same interpreter as the changed checkout:
/path/to/python - --ops rotate,rotate_shared --device cpu --batches 1,8,32 \
  --size 256 --threads 4 --compile --min-run-time 0.5 --json /tmp/geometry-base.json <<'PYTHON'
from pathlib import Path
import kornia
print(kornia.__file__)
__file__ = str(Path.cwd() / "benchmarks/geometry/flagship.py")
exec(compile(Path("/tmp/geometry-flagship.py").read_text(), __file__, "exec"))
PYTHON
```

Geometry and augmentation release snapshots on the available Intel/RTX machine are refreshed
with bilinear baselines. Apple snapshots using the mismatched interpolation are archived,
since that hardware was unavailable. Historical prose tables are marked accordingly.

## Scope and practical guidance

- **Enhance:** ordinary Python-float gamma/gain avoid device parameter allocations and validation synchronizations. Float32 parameters still round before power; tensor, half, exceptional and captured calls retain the tensor path. CLAHE remains a separate histogram/copy optimization project.
- **Metrics:** confusion matrices accumulate a fixed-size batched integer histogram, also accelerating mean IoU. Validation remains enabled and large counts are converted to float only after accumulation.
- **Contrib:** eager distance-transform inference checks for a settled cascade every eight rounds. Once no pixel advances, the remaining iterations are exact no-ops. Differentiation, forward AD, function transforms and graph capture retain the fixed loop. Runtime still depends on the mask; this does not correct the documented distance approximation.
- **Losses:** ordinary unweighted reduced focal loss gathers target-class terms, reducing intermediates and saved autograd memory. The existing mean divides by every class and pixel, including ignored pixels; that convention is preserved. Weighted, unreduced and unusual coefficient cases retain the dense path.
- **Geometry:** pass a singleton angle/matrix when the whole batch shares a transform. Trainable transform parameters retain the original reduction order; independent parameters remain independent. Bilinear interpolation is now explicit in torchvision rotation and random-affine benchmark baselines.
- **Filters and morphology:** benchmark compiled execution for stable workloads, particularly median, bilateral and unfold-heavy morphology. Existing selection-network and separable paths already address some obvious algorithmic waste. Do not silently compile inside public functions: first-call cost and shape specialization belong to the caller.
- **Augmentation:** time complete pipelines, including parameter sampling. Compare interpolation and randomness deliberately. A fixed transform shared across a batch is a different workload from independent sampled transforms; individual-op compile gains do not establish a pipeline gain.
- **Color:** Bayer demosaicing needs a separate quality and gradient-preserving kernel investigation. A direct stencil rewrite can alter nonfinite behavior and half overflow; the audit does not justify shipping one without that work.
- **Feature/models:** SIFT and learned feature pipelines need representative image sizes, feature counts and matching-quality measurements. These are algorithmic investigations rather than interchangeable kernels.
- **IO:** no implementation change from this audit. Decode format, output dtype/layout and transfer cost must match before choosing a backend.

Compilation is a per-operation, device, shape and dtype decision. The audit found substantial fusion opportunities alongside CPU resize regressions; it does not support enabling compilation globally. OpenCV/uint8 CPU timings and differentiable batched float tensor timings describe different application regimes.

## Validation

Focused tests cover values, gradients, empty batches, invalid arguments, noncontiguous inputs, scalar rounding, reduced loss semantics, early-stop parity, tracing, export and compiled execution. The distance-transform ONNX export/runtime case also passes. CUDA half tests run in isolated subprocesses. The full pre-commit suite and type checker passed. Combined float32/float64 runs with
compile coverage passed 1,748 CPU and 1,725 CUDA cases. The final isolated CUDA half
reduced-focal checks passed 38 cases; benchmark schema/timing/rendering checks passed 80.
