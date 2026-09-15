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
          itself and reports the identity matrix. Its direct mask, box and keypoint handlers pass their inputs
          through, except where a subclass overrides them: :class:`RandomErasing` also erases masks.
          A subclass supplies ``apply_transform`` and, where it draws anything, its ``_param_generator``.
        - the matrix is built lazily, on the first read of ``transform_matrix``.
        - this base adds no ``inverse``. When
          :class:`~kornia.augmentation.container.AugmentationSequential.inverse` can run, it skips 2D intensity
          children and reverses supported geometric children.
        - these classes assume the library-wide ``[0, 1]`` float image value range, stated under
          "Image tensors" in :doc:`/get-started/conventions`. No base-class check validates it on the way
          in, so it is a precondition rather than a validated contract.
        - outside that range, concrete classes apply their documented policy: some clamp, rescale, or use a
          ``uint8`` conversion; :class:`RandomPlanckianJitter` clamps only the upper end; others do not
          clamp; and :class:`RandomEqualize` raises a ``RuntimeError`` where its value check runs (on MPS the
          check is skipped and a raw indexing error surfaces instead). The resulting values also depend on
          the sampled parameters and image contents, so these policies are not an exhaustive classification
          of every out-of-range input. :class:`RandomDissolving` is unmeasured because constructing it
          downloads a Stable Diffusion checkpoint. :class:`RandomClahe` and :class:`RandomJPEG` are not in
          ``kornia.augmentation.__all__`` and document their own behavior on their own pages.
        - the scalar factors a concrete class draws are per sample -- one value, or one per channel
          per sample where the class's own docstring says so. Several classes also draw a whole-image
          field -- ``gaussian_noise``, ``gradient``, ``plasma`` -- whose stored shape normally follows the
          original batched ``(B, C, H, W)`` input shape. A ``(C, H, W)`` input remains batched in
          those parameters even when ``keepdim=True``. ``gaussian_noise`` instead stores ``(1, C, H, W)``
          with ``same_on_batch=True``, then expands it at application time; ``RandomPlasmaShadow`` stores
          ``(B, 1, H, W)``. :class:`ColorJiggle` and :class:`ColorJitter` both draw an application ``order``;
          it is shared by the whole batch. Only :class:`ColorJitter` accepts a fixed ``order`` override.
        - where a class documents bounds for a parameter, an explicit range outside them raises at
          construction, with two exceptions. :class:`RandomGamma`'s non-negativity checks, for ``gamma``
          and for ``gain`` alike, live in :func:`kornia.enhance.adjust_gamma`, so they run on the
          forward pass; and :class:`RandomSolarize` admits ``additions`` at the closed bound ``0.5`` at
          construction, which :func:`kornia.enhance.solarize` then rejects on the forward pass. A scalar
          magnitude is a different case: several classes fit it to the bound instead of raising, tracked
          in `#4563 <https://github.com/kornia/kornia/issues/4563>`_.

    .. warning::
        Several of these classes can return an all-zero image for an input whose values are all negative,
        depending on their sampled parameters, with no warning. Tracked in
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
