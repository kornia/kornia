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

"""Dilate, erode, open and close images by a structuring element, and reconstruct them from a seed.

Every function takes floating-point images ``(B, C, H, W)`` and applies one 2D ``kernel`` to every channel. The
non-zero cells of the ``(k_h, k_w)`` kernel form the neighborhood, and a ``structuring_element`` of the same shape
makes the operation non-flat. :func:`dilation` reflects the kernel and :func:`erosion` does not, as in
``scipy.ndimage``; the composite operations inherit both. :func:`reconstruction` takes a seed and a mask instead, an
optional ``kernel`` (``3 x 3`` by default) and no structuring element.

Which function to use:

- Grow bright regions, the maximum over the neighborhood: :func:`dilation`; shrink them, the minimum:
  :func:`erosion`.
- Remove the bright details the kernel does not fit in: :func:`opening`; fill the dark ones: :func:`closing`.
- Those details alone: the bright ones, :func:`top_hat`; the dark ones, :func:`bottom_hat`.
- Edge strength, the dilation minus the erosion: :func:`gradient`.
- Grow a seed image inside a mask until it stops changing, to fill holes or keep whole shapes:
  :func:`reconstruction`.
"""

from .morphology import *
