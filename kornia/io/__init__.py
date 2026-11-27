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

"""Load image files into tensors and write tensors to image files with kornia-rs.

Tensors are ``(C, H, W)`` in RGB order, alpha last. Files are decoded and encoded on the CPU, and the extension of the
path, in any letter case, picks the format.

Which function to use:

- A file to a tensor: :func:`load_image`, with :class:`ImageLoadType` choosing the channels, dtype and range.
- A tensor to a JPEG, PNG or TIFF file: :func:`write_image`.
- The sample images of kornia's examples, downloaded once and cached, or your own paths and URLs as a list or one
  resized batch: :func:`get_sample_images`.
- A file into an :class:`~kornia.image.Image` container: :meth:`~kornia.image.Image.from_file`.
"""

from .io import ImageLoadType, load_image, write_image
from .sample import get_sample_images

__all__ = ["ImageLoadType", "get_sample_images", "load_image", "write_image"]
