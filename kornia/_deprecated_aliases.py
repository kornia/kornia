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


"""Deprecated top-level aliases such as ``kornia.create_meshgrid``; ``kornia/__init__.py`` binds them.

Each alias takes the signature of the function it forwards to and warns only outside TorchScript and outside
``torch.compile``/``torch.export`` graph capture, so code written against 0.8.2 scripts and compiles wherever the
function the alias forwards to does. They live in their own module so that the names their annotations need
(``torch``, ``os``, ``typing``) do not become attributes of ``kornia``.
"""

import os
from typing import Any, List, Optional, Union

import torch

from kornia.core._compat import _emit_deprecation_warning
from kornia.core.ops import eye_like as _eye_like
from kornia.core.ops import vec_like as _vec_like
from kornia.core.utils import (
    get_cuda_device_if_available as _get_cuda_device_if_available,
)
from kornia.core.utils import (
    get_cuda_or_mps_device_if_available as _get_cuda_or_mps_device_if_available,
)
from kornia.core.utils import (
    get_mps_device_if_available as _get_mps_device_if_available,
)
from kornia.core.utils import (
    is_autocast_enabled as _is_autocast_enabled,
)
from kornia.core.utils import (
    xla_is_available as _xla_is_available,
)
from kornia.geometry import (
    create_meshgrid as _create_meshgrid,
)
from kornia.geometry import (
    create_meshgrid3d as _create_meshgrid3d,
)
from kornia.geometry import (
    load_pointcloud_ply as _load_pointcloud_ply,
)
from kornia.geometry import (
    save_pointcloud_ply as _save_pointcloud_ply,
)
from kornia.image import (
    draw_convex_polygon as _draw_convex_polygon,
)
from kornia.image import (
    draw_line as _draw_line,
)
from kornia.image import (
    draw_point2d as _draw_point2d,
)
from kornia.image import (
    draw_rectangle as _draw_rectangle,
)
from kornia.image import (
    image_to_string as _image_to_string,
)
from kornia.image import (
    image_to_tensor as _image_to_tensor,
)
from kornia.image import (
    print_image as _print_image,
)
from kornia.image import (
    tensor_to_image as _tensor_to_image,
)
from kornia.losses.one_hot import one_hot as _one_hot

__all__ = [
    "create_meshgrid",
    "create_meshgrid3d",
    "draw_convex_polygon",
    "draw_line",
    "draw_point2d",
    "draw_rectangle",
    "eye_like",
    "get_cuda_device_if_available",
    "get_cuda_or_mps_device_if_available",
    "get_mps_device_if_available",
    "image_to_string",
    "image_to_tensor",
    "is_autocast_enabled",
    "load_pointcloud_ply",
    "one_hot",
    "print_image",
    "save_pointcloud_ply",
    "tensor_to_image",
    "vec_like",
    "xla_is_available",
]


def _warn(name: str, replace_with: str, previously_in_utils: bool = False) -> None:
    """Warn that ``kornia.<name>`` is deprecated in favor of ``replace_with``, at the line that used it."""
    extra_reason = f" Previously available as `kornia.utils.{name}`." if previously_in_utils else None
    _emit_deprecation_warning(name, replace_with, "0.8.3", extra_reason, stacklevel=4)


def eye_like(n: int, input: torch.Tensor, shared_memory: bool = False) -> torch.Tensor:
    """Deprecated: Use `kornia.core.ops.eye_like` instead."""
    if not torch.jit.is_scripting():
        _warn("eye_like", "kornia.core.ops.eye_like")
    return _eye_like(n, input, shared_memory)


def vec_like(n: int, tensor: torch.Tensor, shared_memory: bool = False) -> torch.Tensor:
    """Deprecated: Use `kornia.core.ops.vec_like` instead."""
    if not torch.jit.is_scripting():
        _warn("vec_like", "kornia.core.ops.vec_like")
    return _vec_like(n, tensor, shared_memory)


def xla_is_available() -> bool:
    """Deprecated: Use `kornia.core.utils.xla_is_available` instead."""
    if not torch.jit.is_scripting():
        _warn("xla_is_available", "kornia.core.utils.xla_is_available")
    return _xla_is_available()


def create_meshgrid(
    height: int,
    width: int,
    normalized_coordinates: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    align_corners: bool = True,
) -> torch.Tensor:
    """Deprecated: Use `kornia.geometry.create_meshgrid` instead."""
    if not torch.jit.is_scripting():
        _warn("create_meshgrid", "kornia.geometry.create_meshgrid")
    return _create_meshgrid(height, width, normalized_coordinates, device, dtype, align_corners)


def image_to_tensor(image: Any, keepdim: bool = True) -> torch.Tensor:
    """Deprecated: Use `kornia.image.image_to_tensor` instead."""
    if not torch.jit.is_scripting():
        _warn("image_to_tensor", "kornia.image.image_to_tensor")
    return _image_to_tensor(image, keepdim)


def tensor_to_image(tensor: torch.Tensor, keepdim: bool = False, force_contiguous: bool = False) -> Any:
    """Deprecated: Use `kornia.image.tensor_to_image` instead."""
    if not torch.jit.is_scripting():
        _warn("tensor_to_image", "kornia.image.tensor_to_image")
    return _tensor_to_image(tensor, keepdim, force_contiguous)


def get_cuda_device_if_available(index: int = 0) -> torch.device:
    """Deprecated: Use `kornia.core.utils.get_cuda_device_if_available` instead."""
    if not torch.jit.is_scripting():
        _warn("get_cuda_device_if_available", "kornia.core.utils.get_cuda_device_if_available")
    return _get_cuda_device_if_available(index)


def get_mps_device_if_available() -> torch.device:
    """Deprecated: Use `kornia.core.utils.get_mps_device_if_available` instead."""
    if not torch.jit.is_scripting():
        _warn("get_mps_device_if_available", "kornia.core.utils.get_mps_device_if_available")
    return _get_mps_device_if_available()


def get_cuda_or_mps_device_if_available() -> torch.device:
    """Deprecated: Use `kornia.core.utils.get_cuda_or_mps_device_if_available` instead."""
    if not torch.jit.is_scripting():
        _warn("get_cuda_or_mps_device_if_available", "kornia.core.utils.get_cuda_or_mps_device_if_available")
    return _get_cuda_or_mps_device_if_available()


def is_autocast_enabled(both: bool = True) -> bool:
    """Deprecated: Use `kornia.core.utils.is_autocast_enabled` instead."""
    if not torch.jit.is_scripting():
        _warn("is_autocast_enabled", "kornia.core.utils.is_autocast_enabled")
    return _is_autocast_enabled(both)


def create_meshgrid3d(
    depth: int,
    height: int,
    width: int,
    normalized_coordinates: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    align_corners: bool = True,
) -> torch.Tensor:
    """Deprecated: Use `kornia.geometry.create_meshgrid3d` instead (previously `kornia.utils.create_meshgrid3d`)."""
    if not torch.jit.is_scripting():
        _warn("create_meshgrid3d", "kornia.geometry.create_meshgrid3d", previously_in_utils=True)
    return _create_meshgrid3d(depth, height, width, normalized_coordinates, device, dtype, align_corners)


def load_pointcloud_ply(filename: str | os.PathLike[str], header_size: Optional[int] = None) -> torch.Tensor:
    """Deprecated: Use `kornia.geometry.load_pointcloud_ply` instead (previously `kornia.utils.load_pointcloud_ply`)."""
    if not torch.jit.is_scripting():
        _warn("load_pointcloud_ply", "kornia.geometry.load_pointcloud_ply", previously_in_utils=True)
    return _load_pointcloud_ply(filename, header_size)


def save_pointcloud_ply(filename: str | os.PathLike[str], pointcloud: torch.Tensor) -> None:
    """Deprecated: Use `kornia.geometry.save_pointcloud_ply` instead (previously `kornia.utils.save_pointcloud_ply`)."""
    if not torch.jit.is_scripting():
        _warn("save_pointcloud_ply", "kornia.geometry.save_pointcloud_ply", previously_in_utils=True)
    return _save_pointcloud_ply(filename, pointcloud)


def draw_line(image: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
    """Deprecated: Use `kornia.image.draw_line` instead (previously `kornia.utils.draw_line`)."""
    if not torch.jit.is_scripting():
        _warn("draw_line", "kornia.image.draw_line", previously_in_utils=True)
    return _draw_line(image, p1, p2, color)


def draw_rectangle(
    image: torch.Tensor,
    rectangle: torch.Tensor,
    color: Optional[torch.Tensor] = None,
    fill: Optional[bool] = None,
    line_width: int = 1,
) -> torch.Tensor:
    """Deprecated: Use `kornia.image.draw_rectangle` instead (previously `kornia.utils.draw_rectangle`)."""
    if not torch.jit.is_scripting():
        _warn("draw_rectangle", "kornia.image.draw_rectangle", previously_in_utils=True)
    return _draw_rectangle(image, rectangle, color, fill, line_width)


def draw_point2d(image: torch.Tensor, points: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
    """Deprecated: Use `kornia.image.draw_point2d` instead (previously `kornia.utils.draw_point2d`)."""
    if not torch.jit.is_scripting():
        _warn("draw_point2d", "kornia.image.draw_point2d", previously_in_utils=True)
    return _draw_point2d(image, points, color)


def draw_convex_polygon(
    images: torch.Tensor, polygons: Union[torch.Tensor, List[torch.Tensor]], colors: torch.Tensor
) -> torch.Tensor:
    """Deprecated: Use `kornia.image.draw_convex_polygon` instead (previously `kornia.utils.draw_convex_polygon`)."""
    if not torch.jit.is_scripting():
        _warn("draw_convex_polygon", "kornia.image.draw_convex_polygon", previously_in_utils=True)
    return _draw_convex_polygon(images, polygons, colors)


def image_to_string(image: torch.Tensor, max_width: int = 256) -> str:
    """Deprecated: Use `kornia.image.image_to_string` instead (previously `kornia.utils.image_to_string`)."""
    if not torch.jit.is_scripting():
        _warn("image_to_string", "kornia.image.image_to_string", previously_in_utils=True)
    return _image_to_string(image, max_width)


def print_image(image: Union[str, os.PathLike, torch.Tensor], max_width: int = 96) -> None:
    """Deprecated: Use `kornia.image.print_image` instead (previously `kornia.utils.print_image`)."""
    if not torch.jit.is_scripting():
        _warn("print_image", "kornia.image.print_image", previously_in_utils=True)
    return _print_image(image, max_width)


def one_hot(
    labels: torch.Tensor, num_classes: int, device: torch.device, dtype: torch.dtype, eps: float = 0.0
) -> torch.Tensor:
    """Deprecated: Use `kornia.losses.one_hot` instead (previously `kornia.utils.one_hot`)."""
    if not torch.jit.is_scripting():
        _warn("one_hot", "kornia.losses.one_hot", previously_in_utils=True)
    return _one_hot(labels, num_classes, device, dtype, eps)
