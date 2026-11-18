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

"""Hold, draw on, convert and print images, and tile batches into grids.

kornia's image tensors are channels first, ``(C, H, W)`` or ``(B, C, H, W)``, and NumPy arrays channels last,
``(H, W, C)`` or ``(B, H, W, C)``. Points are :math:`(x, y)`, x the column, and sizes are ``(height, width)``. The
conversions and :func:`make_grid` do not rescale: a ``uint8`` image stays in :math:`[0, 255]`.

Which class to use:

- One image with its checked layout and color space: :class:`Image`. Its parts are :class:`ImageSize`,
  :class:`PixelFormat` with :class:`ColorSpace`, and :class:`ImageLayout` with :class:`ChannelsOrder`.

Which function to use:

- A NumPy array to a tensor that shares its memory: :func:`image_to_tensor`, or its module form
  :class:`ImageToTensor`. A list of same-shape arrays to one new batch: :func:`image_list_to_tensor`.
- A tensor to a NumPy array for display: :func:`tensor_to_image`.
- Draw into images in place: points with :func:`draw_point2d`, line segments with :func:`draw_line`, rectangle
  outlines or fills with :func:`draw_rectangle`, filled convex polygons with :func:`draw_convex_polygon`.
- A batch tiled into one image: :func:`make_grid`.
- An image in a terminal: :func:`image_to_string` returns the ANSI text, :func:`print_image` prints it and also takes
  a file path.
- Let a ``(B, C, H, W)`` function take any number of leading dimensions: :func:`perform_keep_shape_image`; for
  ``(B, C, D, H, W)`` volumes, :func:`perform_keep_shape_video`.
- Image files: :func:`~kornia.io.load_image` and :func:`~kornia.io.write_image`.
"""

from .base import ChannelsOrder, ColorSpace, ImageLayout, ImageSize, PixelFormat
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
    "ColorSpace",
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
