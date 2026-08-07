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
"""

from __future__ import annotations

from typing import Any

from kornia.core._compat import _emit_deprecation_warning, deprecated
from kornia.core.ops import (
    eye_like as _eye_like,
)
from kornia.core.ops import (
    vec_like as _vec_like,
)
from kornia.core.utils import (
    dataclass_to_dict as _dataclass_to_dict,
)
from kornia.core.utils import (
    dict_to_dataclass as _dict_to_dataclass,
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
from kornia.losses.one_hot import one_hot as _one_hot


# Re-export with deprecation warnings
@deprecated(
    replace_with="kornia.geometry.create_meshgrid",
    version="0.8.3",
    extra_reason=" The `kornia.utils` module has been removed. Import from `kornia.geometry` instead.",
)
def create_meshgrid(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.geometry.create_meshgrid` instead."""
    return _create_meshgrid(*args, **kwargs)


@deprecated(
    replace_with="kornia.geometry.create_meshgrid3d",
    version="0.8.3",
    extra_reason=" The `kornia.utils` module has been removed. Import from `kornia.geometry` instead.",
)
def create_meshgrid3d(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.geometry.create_meshgrid3d` instead."""
    return _create_meshgrid3d(*args, **kwargs)


@deprecated(
    replace_with="kornia.geometry.load_pointcloud_ply",
    version="0.8.3",
    extra_reason=" The `kornia.utils` module has been removed. Import from `kornia.geometry` instead.",
)
def load_pointcloud_ply(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.geometry.load_pointcloud_ply` instead."""
    return _load_pointcloud_ply(*args, **kwargs)


@deprecated(
    replace_with="kornia.geometry.save_pointcloud_ply",
    version="0.8.3",
    extra_reason=" The `kornia.utils` module has been removed. Import from `kornia.geometry` instead.",
)
def save_pointcloud_ply(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.geometry.save_pointcloud_ply` instead."""
    return _save_pointcloud_ply(*args, **kwargs)


@deprecated(
    replace_with="kornia.image.draw_line",
    version="0.8.3",
    extra_reason=" The `kornia.utils` module has been removed. Import from `kornia.image` instead.",
)
def draw_line(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.image.draw_line` instead."""
    return _draw_line(*args, **kwargs)


@deprecated(
    replace_with="kornia.image.draw_rectangle",
    version="0.8.3",
    extra_reason=" The `kornia.utils` module has been removed. Import from `kornia.image` instead.",
)
def draw_rectangle(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.image.draw_rectangle` instead."""
    return _draw_rectangle(*args, **kwargs)


@deprecated(
    replace_with="kornia.image.draw_point2d",
    version="0.8.3",
    extra_reason=" The `kornia.utils` module has been removed. Import from `kornia.image` instead.",
)
def draw_point2d(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.image.draw_point2d` instead."""
    return _draw_point2d(*args, **kwargs)


@deprecated(
    replace_with="kornia.image.draw_convex_polygon",
    version="0.8.3",
    extra_reason=" The `kornia.utils` module has been removed. Import from `kornia.image` instead.",
)
def draw_convex_polygon(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.image.draw_convex_polygon` instead."""
    return _draw_convex_polygon(*args, **kwargs)


@deprecated(
    replace_with="kornia.image.image_to_string",
    version="0.8.3",
    extra_reason=" The `kornia.utils` module has been removed. Import from `kornia.image` instead.",
)
def image_to_string(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.image.image_to_string` instead."""
    return _image_to_string(*args, **kwargs)


@deprecated(
    replace_with="kornia.image.print_image",
    version="0.8.3",
    extra_reason=" The `kornia.utils` module has been removed. Import from `kornia.image` instead.",
)
def print_image(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.image.print_image` instead."""
    return _print_image(*args, **kwargs)


@deprecated(
    replace_with="kornia.image.image_to_tensor",
    version="0.8.3",
    extra_reason=" The `kornia.utils` module has been removed. Import from `kornia.image` instead.",
)
def image_to_tensor(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.image.image_to_tensor` instead."""
    return _image_to_tensor(*args, **kwargs)


@deprecated(
    replace_with="kornia.image.tensor_to_image",
    version="0.8.3",
    extra_reason=" The `kornia.utils` module has been removed. Import from `kornia.image` instead.",
)
def tensor_to_image(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.image.tensor_to_image` instead."""
    return _tensor_to_image(*args, **kwargs)


@deprecated(
    replace_with="kornia.losses.one_hot",
    version="0.8.3",
    extra_reason=" The `kornia.utils` module has been removed. Import from `kornia.losses` instead.",
)
def one_hot(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.losses.one_hot` instead."""
    return _one_hot(*args, **kwargs)


# The 0.8.3 removal shipped without shims for these public names, breaking
# `kornia.utils.<name>` with no deprecation window. Restored here with the
# standard warning so the documented deprecation policy holds.


@deprecated(
    replace_with="kornia.core.ops.eye_like",
    version="0.8.3",
    extra_reason=" The `kornia.utils` module has been removed. Import from `kornia.core.ops` instead.",
)
def eye_like(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.core.ops.eye_like` instead."""
    return _eye_like(*args, **kwargs)


@deprecated(
    replace_with="kornia.core.ops.vec_like",
    version="0.8.3",
    extra_reason=" The `kornia.utils` module has been removed. Import from `kornia.core.ops` instead.",
)
def vec_like(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.core.ops.vec_like` instead."""
    return _vec_like(*args, **kwargs)


@deprecated(
    replace_with="kornia.core.utils.safe_solve_with_mask",
    version="0.8.3",
    extra_reason=" The `kornia.utils` module has been removed. Import from `kornia.core.utils` instead.",
)
def safe_solve_with_mask(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.core.utils.safe_solve_with_mask` instead."""
    return _safe_solve_with_mask(*args, **kwargs)


@deprecated(
    replace_with="kornia.core.utils.safe_inverse_with_mask",
    version="0.8.3",
    extra_reason=" The `kornia.utils` module has been removed. Import from `kornia.core.utils` instead.",
)
def safe_inverse_with_mask(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.core.utils.safe_inverse_with_mask` instead."""
    return _safe_inverse_with_mask(*args, **kwargs)


@deprecated(
    replace_with="kornia.core.utils.is_mps_tensor_safe",
    version="0.8.3",
    extra_reason=" The `kornia.utils` module has been removed. Import from `kornia.core.utils` instead.",
)
def is_mps_tensor_safe(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.core.utils.is_mps_tensor_safe` instead."""
    return _is_mps_tensor_safe(*args, **kwargs)


@deprecated(
    replace_with="kornia.core.utils.dataclass_to_dict",
    version="0.8.3",
    extra_reason=" The `kornia.utils` module has been removed. Import from `kornia.core.utils` instead.",
)
def dataclass_to_dict(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.core.utils.dataclass_to_dict` instead."""
    return _dataclass_to_dict(*args, **kwargs)


@deprecated(
    replace_with="kornia.core.utils.dict_to_dataclass",
    version="0.8.3",
    extra_reason=" The `kornia.utils` module has been removed. Import from `kornia.core.utils` instead.",
)
def dict_to_dataclass(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.core.utils.dict_to_dataclass` instead."""
    return _dict_to_dataclass(*args, **kwargs)


@deprecated(
    replace_with="kornia.image.image_list_to_tensor",
    version="0.8.3",
    extra_reason=" The `kornia.utils` module has been removed. Import from `kornia.image` instead.",
)
def image_list_to_tensor(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.image.image_list_to_tensor` instead."""
    return _image_list_to_tensor(*args, **kwargs)


class ImageToTensor(_ImageToTensor):
    """Deprecated: Use `kornia.image.ImageToTensor` instead.

    A real subclass (not the `@deprecated` decorator, which turns classes into plain
    callables) so `isinstance` checks and further subclassing keep working.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _emit_deprecation_warning(
            "ImageToTensor",
            "kornia.image.ImageToTensor",
            "0.8.3",
            " The `kornia.utils` module has been removed. Import from `kornia.image` instead.",
        )
        super().__init__(*args, **kwargs)


@deprecated(
    replace_with="kornia.io.get_sample_images",
    version="0.8.3",
    extra_reason=" The `kornia.utils` module has been removed. Import from `kornia.io` instead.",
)
def get_sample_images(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.io.get_sample_images` instead."""
    from kornia.io import get_sample_images as _get_sample_images

    return _get_sample_images(*args, **kwargs)


@deprecated(
    replace_with="kornia.core.utils.batched_forward",
    version="0.8.3",
    extra_reason=" The `kornia.utils` module has been removed. Import from `kornia.core.utils` instead.",
)
def batched_forward(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use `kornia.core.utils.batched_forward` instead."""
    from kornia.core.utils import batched_forward as _batched_forward

    return _batched_forward(*args, **kwargs)


@deprecated(
    replace_with="torch.meshgrid",
    version="0.8.3",
    extra_reason=" This compatibility wrapper predates torch 1.10; call `torch.meshgrid(tensors, indexing=...)`.",
)
def torch_meshgrid(tensors: Any, indexing: str) -> Any:
    """Deprecated: Use `torch.meshgrid` directly."""
    import torch

    return torch.meshgrid(tensors, indexing=indexing)


@deprecated(
    replace_with='torch.load(..., map_location="cpu")',
    version="0.8.3",
    extra_reason=" This identity hook predates `map_location` accepting device strings.",
)
def map_location_to_cpu(storage: Any, *args: Any, **kwargs: Any) -> Any:
    """Deprecated: Pass `map_location="cpu"` to `torch.load` instead."""
    return storage


def __getattr__(name: str) -> Any:
    # Lazy so that `import kornia.utils` never pulls in the onnx machinery.
    if name == "CachedDownloader":
        from kornia.onnx.download import CachedDownloader as _CachedDownloader

        _emit_deprecation_warning(
            "CachedDownloader",
            "kornia.onnx.download.CachedDownloader",
            "0.8.3",
            " The `kornia.utils` module has been removed. Import from `kornia.onnx.download` instead.",
        )
        return _CachedDownloader
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
    "get_sample_images",
    "image_list_to_tensor",
    "image_to_string",
    "image_to_tensor",
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
]
