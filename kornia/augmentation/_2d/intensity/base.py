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
          of every out-of-range input. :class:`RandomDissolving` is unmeasured because constructing it needs
          the optional ``diffusers`` package and, on a cold cache, downloads a Stable Diffusion checkpoint.
          :class:`RandomClahe` and :class:`RandomJPEG` are not in ``kornia.augmentation.__all__`` and document
          their own behavior on their own pages.
        - what this block and the class pages say about an output describes the samples the ``p`` gate
          transforms; every other sample comes back with its input values. Below ``p=1`` the transform is
          still computed for every sample and the gate then selects, so a skipped sample that fails a value
          check still makes the call raise -- :class:`RandomEqualize` and :class:`RandomClahe` raise for an
          out-of-range image even at ``p=0.0`` -- and a skipped sample's gradient can be NaN where the
          transform's derivative is infinite (`#4576 <https://github.com/kornia/kornia/issues/4576>`_).
        - the scalar factors a concrete class draws are per sample -- one value, or one per channel
          per sample where the class's own docstring says so. :class:`RandomMotionBlur` is the exception for
          its kernel size, of which one draw serves the whole batch, as its ``Args`` say;
          :class:`RandomClahe` draws ``clip_limit`` per sample but applies the first sample's to the whole
          batch (`#4572 <https://github.com/kornia/kornia/issues/4572>`_); and :class:`RandomDissolving`
          hard-codes ``same_on_batch=True``. Several classes also draw a whole-image field --
          ``gaussian_noise``, ``gradient``, ``plasma``, and :class:`RandomSaltAndPepperNoise`'s boolean
          ``mask_salt`` and ``mask_pepper`` -- whose stored shape normally follows the original batched
          ``(B, C, H, W)`` input shape. A ``(C, H, W)`` input remains batched in those parameters even when
          ``keepdim=True``. ``gaussian_noise`` instead stores ``(1, C, H, W)`` with ``same_on_batch=True``,
          then expands it at application time, while the plasma classes keep one map per sample even then
          (`#4570 <https://github.com/kornia/kornia/issues/4570>`_); ``RandomPlasmaShadow`` stores
          ``(B, 1, H, W)``. :class:`ColorJiggle` and :class:`ColorJitter` both draw an application ``order``;
          it is shared by the whole batch. Only :class:`ColorJitter` takes a fixed ``order`` constructor
          argument. Without one, on either class, an ``order`` tensor passed as a forward keyword, or
          replayed ``params``, replaces the drawn order for that call; a fixed order ignores both.
        - where a class documents bounds for a parameter, an explicit range outside them usually raises at
          construction. These checks run on the forward pass instead: :class:`RandomGamma`'s non-negativity checks
          on ``gamma`` and ``gain``, which live in :func:`kornia.enhance.adjust_gamma`;
          :class:`RandomSolarize`'s ``additions`` at the closed bounds ``-0.5`` and ``0.5``, and
          :class:`RandomGaussianBlur`'s ``sigma`` at ``0`` and even ``kernel_size``, which the constructors
          admit and :func:`kornia.enhance.solarize` and :func:`kornia.filters.gaussian_blur2d` reject;
          :class:`RandomMedianBlur`'s even ``kernel_size``, which raises a raw torch error the same way;
          :class:`RandomRain`'s drop-size bounds; a tuple ``kernel_size`` for :class:`RandomMotionBlur` whose
          drawn odd size is below ``3`` -- an even bound is rounded up to the next odd size rather than
          rejected, and that rounding can leave the requested range, so ``(4, 4)`` draws ``5`` and ``(2, 2)``
          draws ``3``, while ``(0, 2)`` raises because the odd size it rounds to is ``1``;
          and :class:`RandomChannelDropout`'s ``num_drop_channels`` against the input's channel count.
          :class:`RandomPlanckianJitter`'s ``select_from`` rejects an index past the table at construction
          but accepts a negative one, as Python indexing does. A scalar magnitude is a different case: several
          classes fit it to the bound instead of raising, tracked in
          `#4563 <https://github.com/kornia/kornia/issues/4563>`_.

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
