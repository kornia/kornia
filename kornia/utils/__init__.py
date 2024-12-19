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

from ._compat import torch_meshgrid
from .download import CachedDownloader
from .draw import draw_convex_polygon, draw_line, draw_point2d, draw_rectangle
from .grid import create_meshgrid, create_meshgrid3d
from .helpers import (
    _extract_device_dtype,
    dataclass_to_dict,
    deprecated,
    dict_to_dataclass,
    get_cuda_device_if_available,
    get_cuda_or_mps_device_if_available,
    get_mps_device_if_available,
    is_autocast_enabled,
    is_mps_tensor_safe,
    map_location_to_cpu,
    safe_inverse_with_mask,
    safe_solve_with_mask,
    xla_is_available,
)
from .image import ImageToTensor, image_list_to_tensor, image_to_tensor, tensor_to_image
from .image_print import image_to_string, print_image
from .memory import batched_forward
from .misc import (
    differentiable_clipping,
    differentiable_polynomial_floor,
    differentiable_polynomial_rounding,
    eye_like,
    vec_like,
)
from .one_hot import one_hot
from .pointcloud_io import load_pointcloud_ply, save_pointcloud_ply
from .sample import get_sample_images

__all__ = [
    "CachedDownloader",
    "ImageToTensor",
    "_extract_device_dtype",
    "batched_forward",
    "create_meshgrid",
    "create_meshgrid3d",
    "dataclass_to_dict",
    "deprecated",
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
