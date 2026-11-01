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

This package exposes core modules (filters, geometry, etc.) and provides
convenience imports at the top level.
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
