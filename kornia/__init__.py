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
    tracking,
)


# Multi-framework support using ivy
from .transpiler import to_jax, to_numpy, to_tensorflow

# NOTE: we are going to expose to top level very few things
from kornia.constants import pi

# Deprecated top-level imports - use kornia.core.ops or kornia.core.utils instead
from kornia.core._compat import deprecated

# Import the actual functions to wrap
from kornia.core.ops import eye_like as _eye_like, vec_like as _vec_like
from kornia.core.utils import (
    get_cuda_device_if_available as _get_cuda_device_if_available,
    get_cuda_or_mps_device_if_available as _get_cuda_or_mps_device_if_available,
    get_mps_device_if_available as _get_mps_device_if_available,
    is_autocast_enabled as _is_autocast_enabled,
    xla_is_available as _xla_is_available,
)
from kornia.geometry import (
    create_meshgrid as _create_meshgrid,
    create_meshgrid3d as _create_meshgrid3d,
    load_pointcloud_ply as _load_pointcloud_ply,
    save_pointcloud_ply as _save_pointcloud_ply,
)
from kornia.image import (
    draw_convex_polygon as _draw_convex_polygon,
    draw_line as _draw_line,
    draw_point2d as _draw_point2d,
    draw_rectangle as _draw_rectangle,
    image_to_string as _image_to_string,
    image_to_tensor as _image_to_tensor,
    print_image as _print_image,
    tensor_to_image as _tensor_to_image,
)
from kornia.losses import one_hot as _one_hot


@deprecated(replace_with="kornia.core.ops.eye_like", version="0.8.3")
def eye_like(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.core.ops.eye_like` instead."""
    return _eye_like(*args, **kwargs)


@deprecated(replace_with="kornia.core.ops.vec_like", version="0.8.3")
def vec_like(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.core.ops.vec_like` instead."""
    return _vec_like(*args, **kwargs)


@deprecated(replace_with="kornia.core.utils.xla_is_available", version="0.8.3")
def xla_is_available(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.core.utils.xla_is_available` instead."""
    return _xla_is_available(*args, **kwargs)


@deprecated(replace_with="kornia.geometry.create_meshgrid", version="0.8.3")
def create_meshgrid(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.geometry.create_meshgrid` instead."""
    return _create_meshgrid(*args, **kwargs)


@deprecated(replace_with="kornia.image.image_to_tensor", version="0.8.3")
def image_to_tensor(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.image.image_to_tensor` instead."""
    return _image_to_tensor(*args, **kwargs)


@deprecated(replace_with="kornia.image.tensor_to_image", version="0.8.3")
def tensor_to_image(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.image.tensor_to_image` instead."""
    return _tensor_to_image(*args, **kwargs)


@deprecated(replace_with="kornia.core.utils.get_cuda_device_if_available", version="0.8.3")
def get_cuda_device_if_available(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.core.utils.get_cuda_device_if_available` instead."""
    return _get_cuda_device_if_available(*args, **kwargs)


@deprecated(replace_with="kornia.core.utils.get_mps_device_if_available", version="0.8.3")
def get_mps_device_if_available(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.core.utils.get_mps_device_if_available` instead."""
    return _get_mps_device_if_available(*args, **kwargs)


@deprecated(replace_with="kornia.core.utils.get_cuda_or_mps_device_if_available", version="0.8.3")
def get_cuda_or_mps_device_if_available(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.core.utils.get_cuda_or_mps_device_if_available` instead."""
    return _get_cuda_or_mps_device_if_available(*args, **kwargs)


@deprecated(replace_with="kornia.core.utils.is_autocast_enabled", version="0.8.3")
def is_autocast_enabled(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.core.utils.is_autocast_enabled` instead."""
    return _is_autocast_enabled(*args, **kwargs)


@deprecated(
    replace_with="kornia.geometry.create_meshgrid3d",
    version="0.8.3",
    extra_reason=" Previously available as `kornia.utils.create_meshgrid3d`.",
)
def create_meshgrid3d(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.geometry.create_meshgrid3d` instead (previously `kornia.utils.create_meshgrid3d`)."""
    return _create_meshgrid3d(*args, **kwargs)


@deprecated(
    replace_with="kornia.geometry.load_pointcloud_ply",
    version="0.8.3",
    extra_reason=" Previously available as `kornia.utils.load_pointcloud_ply`.",
)
def load_pointcloud_ply(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.geometry.load_pointcloud_ply` instead (previously `kornia.utils.load_pointcloud_ply`)."""
    return _load_pointcloud_ply(*args, **kwargs)


@deprecated(
    replace_with="kornia.geometry.save_pointcloud_ply",
    version="0.8.3",
    extra_reason=" Previously available as `kornia.utils.save_pointcloud_ply`.",
)
def save_pointcloud_ply(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.geometry.save_pointcloud_ply` instead (previously `kornia.utils.save_pointcloud_ply`)."""
    return _save_pointcloud_ply(*args, **kwargs)


@deprecated(
    replace_with="kornia.image.draw_line",
    version="0.8.3",
    extra_reason=" Previously available as `kornia.utils.draw_line`.",
)
def draw_line(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.image.draw_line` instead (previously `kornia.utils.draw_line`)."""
    return _draw_line(*args, **kwargs)


@deprecated(
    replace_with="kornia.image.draw_rectangle",
    version="0.8.3",
    extra_reason=" Previously available as `kornia.utils.draw_rectangle`.",
)
def draw_rectangle(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.image.draw_rectangle` instead (previously `kornia.utils.draw_rectangle`)."""
    return _draw_rectangle(*args, **kwargs)


@deprecated(
    replace_with="kornia.image.draw_point2d",
    version="0.8.3",
    extra_reason=" Previously available as `kornia.utils.draw_point2d`.",
)
def draw_point2d(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.image.draw_point2d` instead (previously `kornia.utils.draw_point2d`)."""
    return _draw_point2d(*args, **kwargs)


@deprecated(
    replace_with="kornia.image.draw_convex_polygon",
    version="0.8.3",
    extra_reason=" Previously available as `kornia.utils.draw_convex_polygon`.",
)
def draw_convex_polygon(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.image.draw_convex_polygon` instead (previously `kornia.utils.draw_convex_polygon`)."""
    return _draw_convex_polygon(*args, **kwargs)


@deprecated(
    replace_with="kornia.image.image_to_string",
    version="0.8.3",
    extra_reason=" Previously available as `kornia.utils.image_to_string`.",
)
def image_to_string(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.image.image_to_string` instead (previously `kornia.utils.image_to_string`)."""
    return _image_to_string(*args, **kwargs)


@deprecated(
    replace_with="kornia.image.print_image",
    version="0.8.3",
    extra_reason=" Previously available as `kornia.utils.print_image`.",
)
def print_image(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.image.print_image` instead (previously `kornia.utils.print_image`)."""
    return _print_image(*args, **kwargs)


@deprecated(
    replace_with="kornia.losses.one_hot",
    version="0.8.3",
    extra_reason=" Previously available as `kornia.utils.one_hot`.",
)
def one_hot(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.losses.one_hot` instead (previously `kornia.utils.one_hot`)."""
    return _one_hot(*args, **kwargs)


# Version variable
__version__ = "0.8.2"
