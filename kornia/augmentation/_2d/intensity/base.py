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

from typing import Any, Dict, Optional

from torch import Tensor

from kornia.augmentation._2d.base import RigidAffineAugmentationBase2D, _input_metadata_only
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints


class IntensityAugmentationBase2D(RigidAffineAugmentationBase2D):
    r"""IntensityAugmentationBase2D base class for customized intensity augmentation implementations.

    See the Convention block on :class:`~kornia.augmentation.AugmentationBase2D`.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
          to the batch form ``False``.

    Convention:
        - an intensity augmentation moves no image pixel, so this base supplies ``compute_transformation``
          itself and reports the identity matrix, built lazily on the first read of ``transform_matrix``. Its
          mask, box and keypoint handlers pass their inputs through unless a subclass overrides them
          (:class:`RandomErasing` also erases masks). It adds no ``inverse``:
          :class:`~kornia.augmentation.container.AugmentationSequential.inverse` skips 2D intensity children.
        - these classes assume the ``[0, 1]`` float image range stated under "Image tensors" in
          :doc:`/get-started/conventions`. The base does not validate it; each class page states what the class
          does outside that range, and a class whose page is silent does not clamp.
        - what this block and the class pages say about an output describes the samples the ``p`` gate
          transforms; every other sample comes back with its input values. Below ``p=1`` the transform is
          still computed for every sample and the gate then selects, so a skipped sample that fails a value
          check still makes the call raise -- :class:`RandomEqualize` and :class:`RandomClahe` raise for an
          out-of-range image even at ``p=0.0`` -- and a skipped sample's gradient can be NaN where the
          transform's derivative is infinite (`#4576 <https://github.com/kornia/kornia/issues/4576>`_).
        - the scalar factors a concrete class draws are per sample -- one value, or one per channel
          per sample where the class's own docstring says so -- and ``same_on_batch=True`` shares one draw.
          :class:`RandomMotionBlur` draws one kernel size for the whole batch; :class:`Normalize`,
          :class:`Denormalize` and :class:`RandomDissolving` hard-code ``same_on_batch=True``. A drawn
          whole-image field (``gaussian_noise``, ``gradient``, ``plasma``, and :class:`RandomSaltAndPepperNoise`'s
          ``mask_salt`` and ``mask_pepper``) is stored with the batched ``(B, C, H, W)`` input shape, even for a
          ``(C, H, W)`` input with ``keepdim=True``; :class:`RandomPlasmaShadow`'s single-channel map is
          ``(B, 1, H, W)``. With ``same_on_batch=True``, ``gaussian_noise`` is stored as
          ``(1, C, H, W)`` and the plasma classes store an expanded view that shares one map across the batch.
        - where a class documents bounds for a parameter, a range outside them usually raises at construction;
          the class page states a bound it does not enforce, or checks only on the forward pass. A scalar
          magnitude ``x`` usually means ``center ± x`` around a centre the class fixes -- the neutral value, for a
          factor such as ``contrast`` -- with its lower end floored at the bound, so ``contrast=1.5`` reads as
          ``[0, 2.5]``, while an upper end past the bound raises. :class:`RandomSharpness` reads a scalar as
          ``[0, x]`` and :class:`RandomPosterize` as ``[x, 8]``.

    .. warning::
        Several of these classes return an all-zero image, with no warning, for an input whose values are all
        negative, and :class:`RandomSolarize` does for one whose values are all at least ``1.5``. Tracked in
        `#4430 <https://github.com/kornia/kornia/issues/4430>`_.

    """

    # Intensity augmentations are pointwise: apply_transform ignores the transform matrix and
    # the matrix is always identity (mask/boxes/keypoints pass through). Defer building it until
    # `.transform_matrix` is read, saving an eye_like + blend on every forward.
    _compute_matrix_lazily = True

    @_input_metadata_only
    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        return self.identity_matrix(input)

    def apply_non_transform_mask(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return input

    def apply_transform_mask(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return input

    def apply_non_transform_box(
        self, input: Boxes, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Boxes:
        return input

    def apply_transform_box(
        self, input: Boxes, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Boxes:
        return input

    def apply_transform_keypoint(
        self, input: Keypoints, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Keypoints:
        return input

    def apply_transform_class(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return input
