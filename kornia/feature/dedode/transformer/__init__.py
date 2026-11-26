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

"""Vision transformer backbones for DeDoDe: the frozen DINOv2 ViT-L of its G descriptor.

The backbone takes images ``(B, 3, H, W)`` whose ``H`` and ``W`` are multiples of its patch size, 14 for the DINOv2
weights, and none of its parameters requires gradients.

Which function to use:

- The DINOv2 ViT-L/14 architecture, to load the ``dinov2_vitl14_pretrain.pth`` weights into: :func:`vit_large`,
  with the keyword arguments its docstring gives. For keypoints and their descriptions, use
  :class:`~kornia.feature.DeDoDe`, which builds and loads it for its ``G`` descriptor.
"""

from .dinov2 import vit_large

__all__ = [
    "vit_large",
]
