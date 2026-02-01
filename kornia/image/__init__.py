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

"""Image submodule for Kornia.

This package provides image data structures and utilities, including image layout, size, pixel format,
and channel order definitions.
"""

from .base import ChannelsOrder, ImageLayout, ImageSize, PixelFormat
from .draw import draw_convex_polygon, draw_line, draw_point2d, draw_rectangle
from .image import Image
from .image_print import image_to_string, print_image
from .utils import (
    ImageToTensor,
    image_list_to_tensor,
    image_to_tensor,
    make_grid,
    perform_keep_shape_image,
    perform_keep_shape_video,
    tensor_to_image,
)

__all__ = [
    "ChannelsOrder",
    "Image",
    "ImageLayout",
    "ImageSize",
    "ImageToTensor",
    "PixelFormat",
    "draw_convex_polygon",
    "draw_line",
    "draw_point2d",
    "draw_rectangle",
    "image_list_to_tensor",
    "image_to_string",
    "image_to_tensor",
    "make_grid",
    "perform_keep_shape_image",
    "perform_keep_shape_video",
    "print_image",
    "tensor_to_image",
]
