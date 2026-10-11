# Experimental representation profiler

This local integration reuses Kornia PR #3669 and applies focused correctness
fixes. It is not an upstream release of `kornia.profiler`.

```python
import torch
from torch import nn
from kornia.profiler import model_profile_under_input_changes

model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU()).eval()
images = torch.rand(16, 3, 32, 32)
report = model_profile_under_input_changes(
    model, images, images * 0.8,
    config={"layers": ["0", "1"], "processing": "none",
            "metrics": ["linear", "cosine", "l2", "gram"]},
)
print(report.results)  # Python dictionaries; does not require pandas.
# report.save_as_report("report.csv")  # Requires optional pandas dependency.
```

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

Validation for this integration targets CPU and NVIDIA CUDA. The entire
upstream test suite, other accelerators, and arbitrary structured layer outputs
have not been validated. Only tensor layer outputs are collected.
