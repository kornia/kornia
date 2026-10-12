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


class RandomTransplantation3D(RandomTransplantation, AugmentationBase3D):  # type: ignore
    """RandomTransplantation3D augmentation.

    3D version of the :class:`kornia.augmentation.RandomTransplantation` augmentation intended to be used with
    :class:`kornia.augmentation.container.AugmentationSequential`. The interface is identical to the 2D version.

    See the Convention block on :class:`~kornia.augmentation.RandomTransplantation`.

    Convention:
        - called directly it behaves exactly like the 2D class; it exists so that
          :class:`~kornia.augmentation.container.AugmentationSequential` dispatches it as a 3D augmentation. Inside a
          container use it for ``(B, C, D, H, W)`` volumes and :class:`~kornia.augmentation.RandomTransplantation`
          for ``(B, C, H, W)`` images; the container rejects 4D input for this class and 5D input for the 2D one.
    """
