# CPU Gaussian blur and scale-pyramid comparison

The tensor slice implementation accelerates large eager inference calls to `gaussian_blur2d`, `GaussianBlur2d`, and `ScalePyramid` on CPU builds without oneDNN. It uses the same Gaussian coefficients, support, padding, and horizontal/vertical pass order. No approximate kernel, truncation change, or reduced precision is introduced.

The fast path requires contiguous CPU float32/float64 input, at least 65,536 pixels per channel and 131,072 total elements. Reverse-mode gradient tracking, autocast, graph capture/tracing, non-contiguous images, smaller inputs, other dtypes/devices, and oneDNN builds retain convolution. Tensor weights and out-of-place `addcmul` preserve forward-mode differentiation and vmap support. The generic scale pyramid reuses its cached kernels.

## Measured regime

Apple M1, macOS 26.5.1, Python 3.11.14, PyTorch 2.14.0, one CPU thread, fixed seed 0. Each cell is a warmed median from `common.time_us` with a one-second minimum and its IQR. The entire public call is timed, including Gaussian kernel generation and allocations; the two backward cases include backward. Pyramid cases use the default six levels per octave. These are base-versus-branch comparisons, not cross-library comparisons.

Base: `607869e2d0a179d8ee6d3890a702b04ba5517c11`. The changed implementation was measured from a working tree based on that commit; per-file SHA256 hashes in the raw JSON pin the exact implementation. Both runs use the same harness SHA256. The harness prints and checks the imported checkout and prints the Python interpreter.

Raw results: [before](gaussian_cpu_results/before-cpu.json), [after](gaussian_cpu_results/after-cpu.json). All measured cases, including losses and unchanged dispatch paths, are below. “Speedup” is the ratio of the two medians, not a guarantee across machines.

| Case | Base median ± IQR (ms) | Changed median ± IQR (ms) | Ratio | Max absolute output/gradient difference |
| --- | ---: | ---: | ---: | ---: |
| blur-128-c1-k7 | 0.132 ± 0.002 | 0.156 ± 0.001 | 0.84× | 0 |
| blur-256-c1-k7 | 0.419 ± 0.005 | 0.423 ± 0.054 | 0.99× | 0 |
| blur-256-c3-k3 | 0.408 ± 0.026 | 0.410 ± 0.064 | 1.00× | 1.8e-07 |
| blur-256-c3-k7 | 1.271 ± 0.125 | 0.732 ± 0.030 | 1.74× | 1.2e-07 |
| blur-256-c3-k15 | 2.704 ± 0.184 | 1.325 ± 0.097 | 2.04× | 1.8e-07 |
| blur-256-c3-k31 | 6.376 ± 0.283 | 2.447 ± 0.083 | 2.61× | 1.8e-07 |
| blur-512-c1-k3 | 0.830 ± 0.017 | 0.570 ± 0.030 | 1.45× | 1.8e-07 |
| blur-512-c1-k7 | 2.126 ± 0.210 | 0.970 ± 0.036 | 2.19× | 1.8e-07 |
| blur-512-c1-k15 | 5.577 ± 0.110 | 1.670 ± 0.093 | 3.34× | 1.8e-07 |
| blur-512-c1-k31 | 12.478 ± 0.998 | 3.099 ± 0.074 | 4.03× | 1.8e-07 |
| blur-batch | 17.681 ± 1.262 | 9.802 ± 3.145 | 1.80× | 1.8e-07 |
| blur-batched-sigma | 2.651 ± 0.470 | 1.699 ± 0.149 | 1.56× | 1.2e-07 |
| blur-float64 | 11.678 ± 0.867 | 4.987 ± 0.880 | 2.34× | 3.3e-16 |
| blur-channels-last | 1.460 ± 0.247 | 1.383 ± 0.503 | 1.06× | 0 |
| blur-strided | 1.123 ± 0.538 | 1.165 ± 0.588 | 0.96× | 0 |
| blur-input-backward | 4.272 ± 0.253 | 3.764 ± 0.266 | 1.14× | 0 |
| blur-sigma-backward | 5.387 ± 0.396 | 4.869 ± 0.257 | 1.11× | 0 |
| blur-signed | 3.008 ± 0.176 | 1.367 ± 0.146 | 2.20× | 8.9e-08 |
| blur-constant | 2.819 ± 0.244 | 1.363 ± 0.131 | 2.07× | 0 |
| blur-impulse | 2.793 ± 0.146 | 1.347 ± 0.143 | 2.07× | 0 |
| pyramid-gray | 46.209 ± 2.213 | 21.098 ± 2.357 | 2.19× | 1.8e-07 |
| pyramid-rgb | 24.444 ± 2.276 | 15.774 ± 1.595 | 1.55× | 1.8e-07 |
| pyramid-double | 46.491 ± 5.563 | 20.776 ± 1.966 | 2.24× | 1.8e-07 |
| pyramid-float64 | 96.687 ± 2.382 | 45.688 ± 4.029 | 2.12× | 3.3e-16 |

`blur-<size>-c<channels>-k<kernel>` uses batch one and square images/kernels. `blur-batch` is four RGB 256×256 images with a 15-tap kernel; `blur-batched-sigma` is two RGB 256×256 images with distinct sigma pairs and a 7-tap kernel. Backward cases are RGB 256×256, 7 taps. Pyramid inputs are 512×512 grayscale, 256×256 RGB, or 256×256 grayscale doubled internally. Full parameters live in each JSON row.

Large single-image blurs improve by 1.45–4.03×, except the 3-tap RGB case, which is effectively unchanged. Complete scale pyramids improve by 1.55–2.24×. Small, layout-fallback, and backward timings vary in both directions despite retaining convolution; for example, the unchanged 128×128 case is 0.84×. Do not interpret noise in these cells as an optimization. Some cases have wide IQRs (especially non-contiguous layouts and batched blur); reproduce on the target machine.

## Numerical agreement

The largest float32 output difference is 1.79e-7; the float64 maximum is 3.33e-16. These bounds include every repeated Gaussian level of the complete scale pyramids. Random [0,1], signed [-1,1], constant, and impulse inputs are included. Constant and impulse outputs were bitwise identical in this run. Reverse-mode outputs and gradients were bitwise identical because that path retains convolution. Raw `tensor_errors` also record mean absolute error, max error normalized by reference magnitude, and relative L2 error.

Independent tests cover anisotropic per-image sigma, all four padding modes, helper input/kernel gradients, public sigma forward-mode derivatives, vmap, and dispatch fallbacks. These measurements establish ordinary floating-point rounding differences for the tested values; they do not promise bitwise equality or bound downstream effects in discontinuous feature selection.

## Reproduction

Use the same Python environment for two disposable worktrees: base revision above and this PR branch. Copy `benchmarks/filters/gaussian_cpu.py` from the branch to the base worktree, where it is untracked. Run from each worktree root as a module so the local checkout shadows any editable install:

```bash
# In the base worktree; choose an absolute path in your task scratch directory.
"$task_python" -m benchmarks.filters.gaussian_cpu --device cpu \
  --json "$task_scratch/before-cpu.json" --save-reference "$task_scratch/cpu-reference.pt"
# In the changed worktree, using the same interpreter and scratch directory.
"$task_python" -m benchmarks.filters.gaussian_cpu --device cpu \
  --json "$task_scratch/after-cpu.json" --reference "$task_scratch/cpu-reference.pt"
```

The reference tensors are temporary local files, not committed artifacts. The JSON reports contain aggregate metadata and errors, not local paths. CUDA and MPS are supported by `--device` for regression measurements; CUDA timing synchronizes through `blocked_autorange`, MPS synchronizes explicitly, and MPS skips unsupported float64 cases. CUDA cuDNN TF32 is disabled for this comparison. Accelerator execution is unchanged by this optimization.

Intel/oneDNN performance, CUDA execution, larger batch/feature-map regimes, and end-to-end feature matching remain unverified here. The PR includes separate instructions for an opt-in Intel experiment that removes only the oneDNN guard while retaining the oneDNN convolution baseline. No automatic oneDNN fast path is enabled without that evidence.
