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

from kornia.augmentation._2d.mix.transplantation import RandomTransplantation
from kornia.augmentation._3d.base import AugmentationBase3D

__all__ = ["RandomTransplantation3D"]


class RandomTransplantation3D(RandomTransplantation, AugmentationBase3D):
    """Paste a labelled region from the previous 3D volume, applied per sample with probability ``p`` (default 0.5).

    ``image_out, mask_out = RandomTransplantation3D(...)(input, mask)`` copies selected labelled regions
    between batched volumes and their segmentation masks. Use :class:`RandomTransplantation` for 2D images.

    See the Convention block on :class:`~kornia.augmentation.RandomTransplantation`.

    Convention:
        - called directly it behaves exactly like the 2D class; it exists so that
          :class:`~kornia.augmentation.container.AugmentationSequential` dispatches it as a 3D augmentation. Inside a
          container use it for ``(B, C, D, H, W)`` volumes and :class:`~kornia.augmentation.RandomTransplantation`
          for ``(B, C, H, W)`` images; the container rejects 4D input for this class and 5D input for the 2D one.

    See Also:
        - :class:`RandomTransplantation`: performs the corresponding operation on 2D images.

    """
