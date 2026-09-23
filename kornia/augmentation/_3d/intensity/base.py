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

from kornia.augmentation._3d.base import RigidAffineAugmentationBase3D


class IntensityAugmentationBase3D(RigidAffineAugmentationBase3D):
    r"""Base class for 3D intensity augmentations.

    See the Convention block on :class:`~kornia.augmentation.AugmentationBase3D`.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it to the batch
          form ``False``.

    Convention:
        - these augmentations leave voxel coordinates in place and record an identity ``(B, 4, 4)`` matrix.
          They still have no direct ``inverse`` method; a container skips them on its inverse path.
        - their image-value requirements are class-specific. In particular, :class:`RandomEqualize3D` documents
          ``[0, 1]`` as its intended range and qualifies its device-specific validation.

    """
