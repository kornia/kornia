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


"""Utils submodule for Kornia.

This module has been deprecated. Functions have been moved to their respective modules.
Import from the new locations instead (e.g., `kornia.image.draw_line` instead of `kornia.utils.draw_line`).

Each shim takes the signature of the function it forwards to (``torch_meshgrid`` and ``map_location_to_cpu`` keep their
0.8.2 signatures) and warns only outside TorchScript and outside ``torch.compile``/``torch.export`` graph capture, so
code written against 0.8.2 scripts and compiles wherever the function the shim forwards to does.
"""

import importlib
import os
import sys
from types import FrameType
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch

# ``deprecated`` was importable from here in 0.8.2, so it stays bound.
from kornia.core._compat import _emit_deprecation_warning, deprecated
from kornia.core.ops import (
    eye_like as _eye_like,
)
from kornia.core.ops import (
    vec_like as _vec_like,
)
from kornia.core.utils import T as _T
from kornia.core.utils import (
    batched_forward as _batched_forward,
)
from kornia.core.utils import (
    dataclass_to_dict as _dataclass_to_dict,
)
from kornia.core.utils import (
    dict_to_dataclass as _dict_to_dataclass,
)
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
    is_mps_tensor_safe as _is_mps_tensor_safe,
)
from kornia.core.utils import (
    safe_inverse_with_mask as _safe_inverse_with_mask,
)
from kornia.core.utils import (
    safe_solve_with_mask as _safe_solve_with_mask,
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
    ImageToTensor as _ImageToTensor,
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
    image_list_to_tensor as _image_list_to_tensor,
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
from kornia.io.sample import _IMAGE_URLS
from kornia.io.sample import get_sample_images as _get_sample_images
from kornia.losses.one_hot import one_hot as _one_hot


def _warn(name: str, module: str) -> None:
    """Warn that ``kornia.utils.<name>`` is deprecated in favor of ``<module>.<name>``, at the line that used it."""
    _emit_deprecation_warning(
        name,
        f"{module}.{name}",
        "0.8.3",
        f" The `kornia.utils` module is deprecated. Import from `{module}` instead.",
        stacklevel=4,
    )


# Re-export with deprecation warnings
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
        _warn("create_meshgrid", "kornia.geometry")
    return _create_meshgrid(height, width, normalized_coordinates, device, dtype, align_corners)


def create_meshgrid3d(
    depth: int,
    height: int,
    width: int,
    normalized_coordinates: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Deprecated: Use `kornia.geometry.create_meshgrid3d` instead."""
    if not torch.jit.is_scripting():
        _warn("create_meshgrid3d", "kornia.geometry")
    return _create_meshgrid3d(depth, height, width, normalized_coordinates, device, dtype)


def load_pointcloud_ply(filename: str | os.PathLike[str], header_size: Optional[int] = None) -> torch.Tensor:
    """Deprecated: Use `kornia.geometry.load_pointcloud_ply` instead."""
    if not torch.jit.is_scripting():
        _warn("load_pointcloud_ply", "kornia.geometry")
    return _load_pointcloud_ply(filename, header_size)


def save_pointcloud_ply(filename: str | os.PathLike[str], pointcloud: torch.Tensor) -> None:
    """Deprecated: Use `kornia.geometry.save_pointcloud_ply` instead."""
    if not torch.jit.is_scripting():
        _warn("save_pointcloud_ply", "kornia.geometry")
    return _save_pointcloud_ply(filename, pointcloud)


def draw_line(image: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
    """Deprecated: Use `kornia.image.draw_line` instead."""
    if not torch.jit.is_scripting():
        _warn("draw_line", "kornia.image")
    return _draw_line(image, p1, p2, color)


def draw_rectangle(
    image: torch.Tensor,
    rectangle: torch.Tensor,
    color: Optional[torch.Tensor] = None,
    fill: Optional[bool] = None,
    line_width: int = 1,
) -> torch.Tensor:
    """Deprecated: Use `kornia.image.draw_rectangle` instead."""
    if not torch.jit.is_scripting():
        _warn("draw_rectangle", "kornia.image")
    return _draw_rectangle(image, rectangle, color, fill, line_width)


def draw_point2d(image: torch.Tensor, points: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
    """Deprecated: Use `kornia.image.draw_point2d` instead."""
    if not torch.jit.is_scripting():
        _warn("draw_point2d", "kornia.image")
    return _draw_point2d(image, points, color)


def draw_convex_polygon(
    images: torch.Tensor, polygons: Union[torch.Tensor, List[torch.Tensor]], colors: torch.Tensor
) -> torch.Tensor:
    """Deprecated: Use `kornia.image.draw_convex_polygon` instead."""
    if not torch.jit.is_scripting():
        _warn("draw_convex_polygon", "kornia.image")
    return _draw_convex_polygon(images, polygons, colors)


def image_to_string(image: torch.Tensor, max_width: int = 256) -> str:
    """Deprecated: Use `kornia.image.image_to_string` instead."""
    if not torch.jit.is_scripting():
        _warn("image_to_string", "kornia.image")
    return _image_to_string(image, max_width)


def print_image(image: Union[str, os.PathLike, torch.Tensor], max_width: int = 96) -> None:
    """Deprecated: Use `kornia.image.print_image` instead."""
    if not torch.jit.is_scripting():
        _warn("print_image", "kornia.image")
    return _print_image(image, max_width)


def image_to_tensor(image: Any, keepdim: bool = True) -> torch.Tensor:
    """Deprecated: Use `kornia.image.image_to_tensor` instead."""
    if not torch.jit.is_scripting():
        _warn("image_to_tensor", "kornia.image")
    return _image_to_tensor(image, keepdim)


def tensor_to_image(tensor: torch.Tensor, keepdim: bool = False, force_contiguous: bool = False) -> Any:
    """Deprecated: Use `kornia.image.tensor_to_image` instead."""
    if not torch.jit.is_scripting():
        _warn("tensor_to_image", "kornia.image")
    return _tensor_to_image(tensor, keepdim, force_contiguous)


def one_hot(
    labels: torch.Tensor, num_classes: int, device: torch.device, dtype: torch.dtype, eps: float = 0.0
) -> torch.Tensor:
    """Deprecated: Use `kornia.losses.one_hot` instead."""
    if not torch.jit.is_scripting():
        _warn("one_hot", "kornia.losses")
    return _one_hot(labels, num_classes, device, dtype, eps)


# The 0.8.3 removal shipped without shims for these public names, breaking
# `kornia.utils.<name>` with no deprecation window. Restored here with the
# standard warning so the documented deprecation policy holds.


def eye_like(n: int, input: torch.Tensor, shared_memory: bool = False) -> torch.Tensor:
    """Deprecated: Use `kornia.core.ops.eye_like` instead."""
    if not torch.jit.is_scripting():
        _warn("eye_like", "kornia.core.ops")
    return _eye_like(n, input, shared_memory)


def vec_like(n: int, tensor: torch.Tensor, shared_memory: bool = False) -> torch.Tensor:
    """Deprecated: Use `kornia.core.ops.vec_like` instead."""
    if not torch.jit.is_scripting():
        _warn("vec_like", "kornia.core.ops")
    return _vec_like(n, tensor, shared_memory)


def safe_solve_with_mask(B: torch.Tensor, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deprecated: Use `kornia.core.utils.safe_solve_with_mask` instead."""
    if not torch.jit.is_scripting():
        _warn("safe_solve_with_mask", "kornia.core.utils")
    return _safe_solve_with_mask(B, A)


def safe_inverse_with_mask(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deprecated: Use `kornia.core.utils.safe_inverse_with_mask` instead."""
    if not torch.jit.is_scripting():
        _warn("safe_inverse_with_mask", "kornia.core.utils")
    return _safe_inverse_with_mask(A)


def is_mps_tensor_safe(x: torch.Tensor) -> bool:
    """Deprecated: Use `kornia.core.utils.is_mps_tensor_safe` instead."""
    if not torch.jit.is_scripting():
        _warn("is_mps_tensor_safe", "kornia.core.utils")
    return _is_mps_tensor_safe(x)


def dataclass_to_dict(obj: Any) -> Any:
    """Deprecated: Use `kornia.core.utils.dataclass_to_dict` instead."""
    if not torch.jit.is_scripting():
        _warn("dataclass_to_dict", "kornia.core.utils")
    return _dataclass_to_dict(obj)


def dict_to_dataclass(dict_obj: Dict[str, Any], dataclass_type: Type[_T]) -> _T:
    """Deprecated: Use `kornia.core.utils.dict_to_dataclass` instead."""
    if not torch.jit.is_scripting():
        _warn("dict_to_dataclass", "kornia.core.utils")
    return _dict_to_dataclass(dict_obj, dataclass_type)


def image_list_to_tensor(images: List[Any]) -> torch.Tensor:
    """Deprecated: Use `kornia.image.image_list_to_tensor` instead."""
    if not torch.jit.is_scripting():
        _warn("image_list_to_tensor", "kornia.image")
    return _image_list_to_tensor(images)


class ImageToTensor(_ImageToTensor):
    """Deprecated: Use `kornia.image.ImageToTensor` instead.

    A subclass of `kornia.image.ImageToTensor` that warns on construction, so the new class is not itself
    deprecated; `isinstance` checks and further subclassing keep working.
    """

    def __init__(self, keepdim: bool = False) -> None:
        _warn("ImageToTensor", "kornia.image")
        super().__init__(keepdim)


def get_sample_images(
    resize: Optional[Tuple[int, int]] = None,
    paths: List[str] = _IMAGE_URLS,
    download: bool = True,
    cache_dir: Optional[str] = None,
    as_list: Optional[bool] = None,
    divisible_factor: Optional[int] = None,
    **kwargs: Any,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Deprecated: Use `kornia.io.get_sample_images` instead."""
    if not torch.jit.is_scripting():
        _warn("get_sample_images", "kornia.io")
    return _get_sample_images(resize, paths, download, cache_dir, as_list, divisible_factor, **kwargs)


def batched_forward(
    model: torch.nn.Module, data: torch.Tensor, device: torch.device, batch_size: int = 128, **kwargs: Any
) -> torch.Tensor:
    """Deprecated: Use `kornia.core.utils.batched_forward` instead."""
    if not torch.jit.is_scripting():
        _warn("batched_forward", "kornia.core.utils")
    return _batched_forward(model, data, device, batch_size, **kwargs)


def torch_meshgrid(tensors: List[torch.Tensor], indexing: str):  # the 0.8.2 signature, return left unannotated
    """Deprecated: Use `torch.meshgrid` directly."""
    if not torch.jit.is_scripting():
        _emit_deprecation_warning(
            "torch_meshgrid",
            "torch.meshgrid",
            "0.8.3",
            " This compatibility wrapper predates torch 1.10; call `torch.meshgrid(tensors, indexing=...)`.",
        )
    return torch.meshgrid(tensors, indexing=indexing)


def _in_torch_or_kornia(frame: FrameType) -> bool:
    return str(frame.f_globals.get("__name__", "")).split(".", 1)[0] in ("torch", "kornia")


def _warn_map_location_to_cpu() -> None:
    """Warn at the line that called ``torch.load``.

    ``torch.load`` calls the hook from inside torch, once for every storage it restores. A fixed ``stacklevel`` names
    a torch frame, and the default filters hide a ``DeprecationWarning`` that is not attributed to ``__main__``, so the
    warning goes to the first frame outside torch and kornia instead. Every storage of a call then warns at the same
    line, so the warnings registry shows it once there under the default filters, as for any repeated warning.
    """
    level = 4  # counted from _emit_deprecation_warning: this function, map_location_to_cpu, then the hook's caller
    frame: Optional[FrameType] = sys._getframe(2)
    while frame is not None and _in_torch_or_kornia(frame):
        frame, level = frame.f_back, level + 1
    if frame is None:  # nothing outside torch and kornia: name the hook's caller
        level = 4
    _emit_deprecation_warning(
        "map_location_to_cpu",
        'torch.load(..., map_location="cpu")',
        "0.8.3",
        " This identity hook predates `map_location` accepting device strings.",
        stacklevel=level,
    )


def map_location_to_cpu(storage: Any, *args: Any, **kwargs: Any) -> Any:
    """Deprecated: Pass `map_location="cpu"` to `torch.load` instead."""
    _warn_map_location_to_cpu()
    return storage


def get_cuda_device_if_available(index: int = 0) -> torch.device:
    """Deprecated: Use `kornia.core.utils.get_cuda_device_if_available` instead."""
    if not torch.jit.is_scripting():
        _warn("get_cuda_device_if_available", "kornia.core.utils")
    return _get_cuda_device_if_available(index)


def get_mps_device_if_available() -> torch.device:
    """Deprecated: Use `kornia.core.utils.get_mps_device_if_available` instead."""
    if not torch.jit.is_scripting():
        _warn("get_mps_device_if_available", "kornia.core.utils")
    return _get_mps_device_if_available()


def get_cuda_or_mps_device_if_available() -> torch.device:
    """Deprecated: Use `kornia.core.utils.get_cuda_or_mps_device_if_available` instead."""
    if not torch.jit.is_scripting():
        _warn("get_cuda_or_mps_device_if_available", "kornia.core.utils")
    return _get_cuda_or_mps_device_if_available()


def is_autocast_enabled(both: bool = True) -> bool:
    """Deprecated: Use `kornia.core.utils.is_autocast_enabled` instead."""
    if not torch.jit.is_scripting():
        _warn("is_autocast_enabled", "kornia.core.utils")
    return _is_autocast_enabled(both)


def xla_is_available() -> bool:
    """Deprecated: Use `kornia.core.utils.xla_is_available` instead."""
    if not torch.jit.is_scripting():
        _warn("xla_is_available", "kornia.core.utils")
    return _xla_is_available()


def __getattr__(name: str) -> Any:
    # ``CachedDownloader`` is used through its classmethods without being constructed, so it cannot warn from
    # ``__init__`` the way ``ImageToTensor`` does: it resolves here instead, and warns on access.
    # ``kornia.utils.sample`` is the v0.8.2 submodule path of ``get_sample_images``.
    if name == "CachedDownloader":
        from kornia.onnx.download import CachedDownloader as _CachedDownloader

        _warn("CachedDownloader", "kornia.onnx.download")
        return _CachedDownloader
    if name == "sample":
        return importlib.import_module("kornia.utils.sample")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CachedDownloader",
    "ImageToTensor",
    "batched_forward",
    "create_meshgrid",
    "create_meshgrid3d",
    "dataclass_to_dict",
    "dict_to_dataclass",
    "draw_convex_polygon",
    "draw_line",
    "draw_point2d",
    "draw_rectangle",
    "eye_like",
    "get_cuda_device_if_available",
    "get_cuda_or_mps_device_if_available",
    "get_mps_device_if_available",
    "get_sample_images",
    "image_list_to_tensor",
    "image_to_string",
    "image_to_tensor",
    "is_autocast_enabled",
    "is_mps_tensor_safe",
    "load_pointcloud_ply",
    "map_location_to_cpu",
    "one_hot",
    "print_image",
    "safe_inverse_with_mask",
    "safe_solve_with_mask",
    "save_pointcloud_ply",
    "tensor_to_image",
    "torch_meshgrid",
    "vec_like",
    "xla_is_available",
]
