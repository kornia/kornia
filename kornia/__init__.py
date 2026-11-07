# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Kornia — Differentiable computer vision and image processing for PyTorch.

Kornia's operators work on batched ``torch.Tensor`` inputs, run on the input's device (CPU, CUDA or Apple MPS)
without a NumPy round trip, and most are differentiable. Kornia pays off where a hand-written version gets long or
fragile, or where the alternatives leave the GPU:

- Breadth: over 1,200 public operators, most with no torchvision equivalent. Camera models and epipolar geometry
  (``kornia.geometry``), local features and matching (``kornia.feature``), filters and edges (``kornia.filters``),
  morphology, Lab and Bayer color, image losses and metrics.
- Per-sample parameters in one batched call: a different blur ``sigma``, rotation angle, warp matrix or brightness
  factor for every image. torchvision's functional ops apply one parameter per call.
- Batched minimal solvers and robust estimation: 7- and 8-point fundamental, 5-point essential, DLT homography and
  PnP solvers over whole batches of samples, and a ``RANSAC`` that samples and scores hypotheses on the device.
- Differentiable geometry: depth to 3D and back, camera models with lens distortion, and Lie groups, with gradients
  through depth, pose and camera parameters.
- Annotated augmentation (``kornia.augmentation``): one sampled transform applied to images, masks, boxes and
  keypoints together, and its inverse.
- Local features: kornia's ports of ALIKED, XFeat, DISK with LightGlue, DeDoDe, LoFTR, HardNet and the other learned
  descriptors run at their authors' speed or faster on every device measured (CUDA, x86 and Apple M1 CPUs, Apple
  MPS), and are as accurate on the cohorts measured. The exceptions are eager ALIKED on CUDA (compile it), SIFT on a
  CPU (OpenCV's is faster) and XFeat*, whose refinement departs from the authors' on purpose. The ``kornia.feature``
  docstring says what to use where.
- Compiles and exports: nearly all probed operator configurations run under ``torch.compile`` and match eager, and
  over 1,000 export to ONNX (``kornia.onnx``).
- Fast on batched GPU tensors: on an RTX 4090 with batches of 32, many filters, warps, matching steps and image
  metrics run about 10x to 70x faster in eager mode than per-image OpenCV or scikit-image on the CPU. On the small
  overlap with torchvision, eager kornia is slower on the same GPU, and OpenCV wins single-image ``uint8`` CPU work.

For a single fixed-parameter operation, such as one blur or one resize, a few lines of PyTorch or torchvision are
just as good.

Before writing code, read the conventions page, which covers image tensors, coordinates, rotations, matrices,
``align_corners``, boxes, color, augmentation and local features, and ends with a pitfall checklist:
https://kornia.readthedocs.io/en/latest/get-started/conventions.html. Operator benchmarks, wins and losses, are at
https://kornia.readthedocs.io/en/latest/get-started/performance.html, and the local-feature comparison at
https://kornia.readthedocs.io/en/latest/applications/image_matching.html#feature-speed-by-device.
"""

from importlib import import_module as _import_module
from typing import Any

# NOTE: kornia filters and geometry must go first since are the core of the library
# and by changing the import order you might get into a circular dependencies issue.
from . import filters
from . import geometry

# import the other modules for convenience
from . import (
    augmentation,
    color,
    contrib,
    core,
    config,
    enhance,
    feature,
    io,
    losses,
    metrics,
    models,
    morphology,
    onnx,
    profiler,
    tracking,
)

# The Ivy-powered multi-framework entry points were removed. The module stays bound, and `to_jax`, `to_numpy` and
# `to_tensorflow` stay bound here and in it as functions that raise an error explaining the removal, so that
# `kornia.to_jax()` and `from kornia import to_jax` both lead to the explanation. See `kornia/transpiler/__init__.py`.
from . import transpiler
from .transpiler import _removed_entry_point

# NOTE: we are going to expose to top level very few things
from kornia.constants import pi

# Deprecated top-level imports - use kornia.core.ops or kornia.core.utils instead
from kornia.core._compat import deprecated

# The aliases are defined in their own module, which keeps the names their annotations need off `kornia`.
from ._deprecated_aliases import (
    create_meshgrid,
    create_meshgrid3d,
    draw_convex_polygon,
    draw_line,
    draw_point2d,
    draw_rectangle,
    eye_like,
    get_cuda_device_if_available,
    get_cuda_or_mps_device_if_available,
    get_mps_device_if_available,
    image_to_string,
    image_to_tensor,
    is_autocast_enabled,
    load_pointcloud_ply,
    one_hot,
    print_image,
    save_pointcloud_ply,
    tensor_to_image,
    vec_like,
    xla_is_available,
)

to_jax = _removed_entry_point(__name__, "to_jax")
to_numpy = _removed_entry_point(__name__, "to_numpy")
to_tensorflow = _removed_entry_point(__name__, "to_tensorflow")


def __getattr__(name: str) -> Any:
    """Import the deprecated ``kornia.utils`` on first access.

    ``import kornia`` does not import the deprecated ``kornia.utils``; ``kornia.utils.<name>``, the attribute style
    kornia's 0.8.2 documentation used, still reaches its shims after ``import kornia`` alone.
    """
    if name == "utils":
        return _import_module("kornia.utils")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Version variable
__version__ = "0.9.1.dev"
