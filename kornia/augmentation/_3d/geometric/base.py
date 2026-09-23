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


class GeometricAugmentationBase3D(RigidAffineAugmentationBase3D):
    r"""Base class for 3D geometric augmentations.

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
        - subclasses record a source-to-destination ``(B, 4, 4)`` voxel-coordinate matrix. Coordinates are
          ordered ``(x, y, z)``, while volume tensor axes are ``(D, H, W)``. For :class:`RandomCrop3D`, the source
          frame is the padded volume: the matrix omits the translation introduced by pre-crop padding.
          Resampling follows the matrix in that source frame at ``align_corners=True``; at
          ``align_corners=False`` normalization can make the applied warp differ from the recorded matrix
          (`#4503 <https://github.com/kornia/kornia/issues/4503>`_).
          Matrices use the input dtype, so integer voxel translations are rounded when that dtype cannot represent
          them exactly.
        - a zero-parameter :class:`RandomAffine3D` or :class:`RandomRotation3D` warp reproduces its input up to
          roundoff in ``float32`` and ``float64``. In half precision the sampling grid is itself rounded, so the
          error grows with the volume size.

    """
