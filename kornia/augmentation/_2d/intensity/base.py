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

from typing import Any, Dict, Optional, Sequence

import torch
from torch import Tensor

from kornia.augmentation._2d.base import RigidAffineAugmentationBase2D, _input_metadata_only
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints


def _clamp_unit_range(input: Tensor) -> Tensor:
    """Clamp ``input`` into ``[0, 1]`` with an identity gradient on the closed range.

    ``torch.clamp``'s derivative at a bound is ``1`` on torch 2.5.1 and 2.9.1 and ``0`` on 2.14.0, and valid images
    often hold exact ``0`` and ``1`` pixels. The values inside ``[0, 1]`` are therefore taken from ``input`` itself,
    so they keep their gradient on every torch version; values outside get the bound and a zero gradient.

    The values always equal ``input.clamp(0, 1)``, ``NaN`` and ``-0.0`` included, so a call that records no
    gradient and carries no forward-mode tangent is that single clamp. Otherwise, select ``input`` where the
    clamp left it unchanged: one comparison against the clamped copy instead of two bound comparisons and their
    conjunction. Forward-mode tangents must keep the identity derivative even inside ``torch.no_grad()``.
    """
    clamped = input.clamp(0.0, 1.0)
    if (
        not (torch.is_grad_enabled() and input.requires_grad)
        and torch.autograd.forward_ad.unpack_dual(input).tangent is None
    ):
        return clamped
    return torch.where(clamped == input, input, clamped)


class _PicklableCompileMixin:
    """Keep a module picklable after its own ``compile()`` replaced callables with ``torch.compile``d ones.

    ``compile()`` calls :meth:`_record_compile` before compiling. Pickling (and ``copy.deepcopy``) then stores
    the uncompiled callables, which pickle, and unpickling compiles them again with the same arguments.
    """

    def _record_compile(self, names: Sequence[str], compile_kwargs: Dict[str, Any]) -> None:
        # Keep the first, uncompiled callables if compile() is called more than once.
        if self.__dict__.get("_uncompiled_fns") is None:
            self._uncompiled_fns = {name: getattr(self, name) for name in names}
        self._compile_kwargs = compile_kwargs

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()
        uncompiled = state.get("_uncompiled_fns")
        if uncompiled is not None:
            state.update(uncompiled)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super().__setstate__(state)
        compile_kwargs = state.get("_compile_kwargs")
        if compile_kwargs is not None:
            self._uncompiled_fns = None
            self.compile(**compile_kwargs)


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
        - classes whose arithmetic assumes unit-range images clamp the input into ``[0, 1]`` before
          the operation, and clamp the result too. That input clamp passes the gradient of a value in
          ``[0, 1]``, ``0`` and ``1`` included, unchanged, and gives a value outside a zero gradient, on every
          supported torch version. Range-independent classes do not clamp their input;
          their class pages state this explicitly. ``RandomBrightness`` and ``RandomContrast`` opt out of
          both clamps with ``clip_output=False``. Normalize after range-dependent augmentations: clamping
          an all-negative normalized image discards its information and may produce an all-zero output.
        - what this block and the class pages say about an output describes the samples the ``p`` gate
          transforms; every other sample comes back with its input values and, where they are finite, an
          identity gradient. Below ``p=1`` :class:`RandomEqualize`, :class:`RandomClahe` and :class:`RandomGamma`
          run their transform on 0.5 for the skipped samples, so those discarded computations do not
          affect the skipped image or its gradient.
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
