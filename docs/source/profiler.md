# Experimental representation profiler

This experimental branch builds on navya7821's [PR #3669](https://github.com/kornia/kornia/pull/3669)
and adds correctness fixes and regression tests for [issue #3626](https://github.com/kornia/kornia/issues/3626).
It is not an upstream release of `kornia.profiler`.

## Install this branch

Use Python 3.11 or 3.12 and a separate virtual environment. Installing the
published `kornia` package alone does not install this experimental module.

```bash
git clone --branch fix/profiler-metrics-and-tests https://github.com/eric9003/kornia.git
cd kornia
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS, or
`.venv\Scripts\Activate.ps1` in Windows PowerShell. Then run:

```bash
python -m pip install -e ".[profiler]"
python -m pip install pytest pytest-timeout
python -c "import kornia; print(kornia.__file__)"
```

The printed path should point into this checkout. The optional `profiler` extra
installs pandas for DataFrame, CSV and JSON report export. Dictionary results
and the CKA function do not require pandas. CPU is sufficient for the examples;
a CUDA-enabled PyTorch installation and NVIDIA GPU are optional.

## Call linear CKA directly

Save the following as `try_cka.py` and run `python try_cka.py`:

```python
import torch
from kornia.profiler.metrics.linear import linear_cka

torch.manual_seed(42)
x = torch.randn(32, 128)  # 32 corresponding samples, 128 features each

print("Self similarity:", linear_cka(x, x))
print("Uniform scaling:", linear_cka(x, 3 * x))
print("Added noise:", linear_cka(x, x + 0.5 * torch.randn_like(x)))
```

The first two scores should be approximately 1; adding noise usually reduces
the score. Each argument has shape `(N, D)`; feature dimensions can differ,
but sample counts, sample order and devices must match. At least two samples
and nonconstant centered features are required. The result is a Python float,
not a differentiable tensor for use as a training loss.

## Compare intermediate model features

Save this example as `try_profiler.py` and run `python try_profiler.py`:

```python
import torch
from torch import nn
from kornia.profiler import model_profile_under_input_changes

model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU()).eval()
images = torch.rand(16, 3, 32, 32)
report = model_profile_under_input_changes(
    model,
    images,
    images * 0.8,
    config={"layers": ["0", "1"], "processing": "none", "metrics": ["linear", "cosine", "l2", "gram"]},
)
print(report.results)  # List of dictionaries; does not require pandas.
# report.save_as_report("report.csv")  # Requires optional pandas dependency.
```

Results use keys such as `linear_0`, `cosine_1`, `l2_0` and `gram_1`,
where the suffix is the selected layer name. Inspect available layer names
with `print(dict(model.named_modules()).keys())`. To use CUDA, move both the
model and both input tensors to the same CUDA device before calling the API.

Higher `linear`, `cosine` and `gram` scores indicate more similar features;
lower `l2` indicates less change. Compare these measurements with prediction
accuracy or task metrics separately.

## Run the regression tests

From the repository root, using the same environment:

```bash
python -m pytest tests/profiler/test_profiler.py -v
```

The tests cover the independent CKA reference formula and invariances, invalid
inputs, per-layer Gram results, feature snapshots, hook cleanup, restoration
of training flags and different model input structures. The test file
parametrizes CPU and CUDA; unavailable CUDA cases are skipped. Inspect failures
and skips rather than interpreting every skip as a failure.

The recorded 46-test pass belongs to the v0.8.3 integration environment.
This current-main branch still needs compatibility checks and full pre-commit
checks before it is ready to merge. See [the validation notes](../../PROFILER_VALIDATION.md)
for the broader suite results and remaining work.

## Input and metric semantics

The generic input API accepts a single positional tensor, a positional tuple,
or a mapping of keyword arguments. The augmentation wrapper accepts images
and uses Kornia's RandomRotation, RandomGaussianBlur, and RandomBrightness.
For fixed strengths, use equal interval endpoints, for example sigma=(1,1).
Kornia RandomBrightness adds (brightness_factor - 1) to pixel values and clips
the output; it is not multiplication by brightness_factor. Transform raw
images before applying the model's input normalization.

`linear` is biased, sample-centered linear CKA. Inputs must contain at least
two corresponding samples. Feature dimensions may differ; sample order must
match. Constant centered features have undefined CKA and raise ValueError.
The implementation uses float64 and N-by-N sample Gram matrices: its memory
cost grows quadratically in the sample count. Do not average per-batch CKA
values and interpret that as CKA over the complete dataset.

`gram` computes channel Gram cosine similarity for 4D features; non-4D entries
are skipped by the profiler. `l2` is a mean per-sample relative distance and is
asymmetric in the reference and changed inputs. High CKA or Gram similarity
does not imply preserved predictions or give a causal explanation.

Captured tensors are detached snapshots, safe from subsequent in-place
operations. Initialization validates layer names before registering hooks.
The context removes hooks and restores each module's previous training flag,
even after an exception. It does not roll back model parameters, buffers, or
custom side effects from forward methods. Group storage intentionally retains
feature snapshots until explicitly cleared or the profiler is released. Repeated
calls that are meant to accumulate observations therefore consume memory.

`ModelProfiler.compute` compares corresponding stored group entries. Unequal
group lengths currently retain upstream behavior: warn and truncate to the
shorter group. Choose selected layers and modest sample counts rather than
capturing every layer of a large network.

Validation for this integration targets CPU and NVIDIA CUDA. The broader
v0.8.3 upstream runs were incomplete because of external fixture/checkpoint
availability. Current-main compatibility, other accelerators and arbitrary
structured layer outputs have not been validated. Only tensor layer outputs are collected.
