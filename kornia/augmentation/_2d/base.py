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

from typing import Any, Dict, Optional, Tuple

import torch
from torch import float16, float32, float64

from kornia.augmentation.base import _AugmentationBase
from kornia.augmentation.utils import _transform_input, _transform_input_by_shape, _validate_input_dtype
from kornia.core.ops import eye_like
from kornia.core.utils import is_autocast_enabled
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints


class AugmentationBase2D(_AugmentationBase):
    r"""AugmentationBase2D base class for customized augmentation implementations.

    AugmentationBase2D aims at offering a generic base class for a greater level of customization.
    If the subclass contains routined matrix-based transformations, `RigidAffineAugmentationBase2D`
    might be a better fit.

    This class is the anchor for the contract the 2D augmentations of ``kornia.augmentation`` inherit;
    they point here instead of restating it. The mix classes derive from
    :class:`~kornia.augmentation.MixAugmentationBaseV2` rather than from this base, and the 3D classes from
    :class:`~kornia.augmentation.AugmentationBase3D`, but the ``p`` / ``p_batch``, RNG, replay and
    serialization halves of this block are shared with them through ``_BasicAugmentationBase``.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it to the batch
          form ``False``.

    Convention:
        - the working layout is ``(B, C, H, W)`` float. A ``(C, H, W)`` input is promoted to ``(1, C, H, W)``
          and an ``(H, W)`` input to ``(1, 1, H, W)``; ``keepdim=True`` restores the input rank on the way out
          and never drops a real batch dimension. The dtype guard accepts ``float16``, ``bfloat16``, ``float32``
          and ``float64`` and raises ``TypeError`` naming those four on an integer tensor. The output keeps the
          input's device. It keeps the input's dtype too, with two exceptions:
          :class:`RandomPlanckianJitter` returns ``float32`` for a ``float16`` or ``bfloat16`` input and
          :class:`RandomMixUpV2` returns ``float32`` for a ``float16`` one. The five mix classes do not reach
          the output at all on ``bfloat16`` -- they raise ``KeyError`` downstream of the guard.
        - any other input rank raises ``ValueError`` naming the three accepted shapes. ``validate_tensor``
          is stricter than that -- it rejects the legal ``(C, H, W)`` rank with a ``RuntimeError`` -- and
          ``transform_tensor`` runs first, so ``forward`` never reaches it. Through a container the same
          user mistake surfaces as a ``RuntimeError`` with a different message instead.
        - ``p`` is a per-sample Bernoulli and ``p_batch`` a single Bernoulli per call that gates the whole
          batch, drawn before ``p``: with ``p=1.0, p_batch=0.0`` nothing is applied. ``same_on_batch=True``
          asks the batch to share one draw; the gate and the transform parameters follow it, while keys that
          index or pair up the batch stay per sample by construction, and :class:`RandomRain` draws a
          different drop count per sample regardless (tracked in
          `#4448 <https://github.com/kornia/kornia/issues/4448>`_).
          ``p_batch`` is part of this base signature and available to custom subclasses, but of the 69 concrete
          classes only :class:`RandomHorizontalFlip` and :class:`RandomVerticalFlip` name it: the rest raise
          ``TypeError`` on the keyword, except :class:`RandomDissolving`, whose ``**kwargs`` binds it and
          drops it without a signal.
        - the random parameters are drawn on the CPU whatever the input's device, and come back in
          ``torch.get_default_dtype()`` rather than in the input's dtype -- set ``torch.set_default_dtype``
          before the call to draw in another dtype. ``set_rng_device_and_dtype`` moves the ``p`` /
          ``p_batch`` gate, and on most classes nothing else, so it is not the way to move sampling to an
          accelerator.
        - reproducibility goes through torch's global CPU generator: ``torch.manual_seed`` before the call
          reproduces the draw bitwise. There is no per-instance generator; ``generator=`` raises at
          construction and is silently dropped by ``forward``, as any other unknown keyword is.
        - :doc:`/get-started/conventions` is the canonical statement of all of this: the seeding,
          ``DataLoader``-worker and consumption-order rules, the classes ``set_rng_device_and_dtype`` does
          move and the three it makes raise, and what a serialization round trip carries.
        - the last draw is kept in ``_params``; ``forward(x, params=...)`` replaces that dict wholesale rather
          than merging into it, stores the caller's dict by reference without adding or mutating a key, and
          replays the same output bitwise. The three ``RandomPlasma*`` classes are the exception: they draw
          their fractal noise while applying it, so replaying them needs the same global seed as well
          (tracked in `#4445 <https://github.com/kornia/kornia/issues/4445>`_).
        - an augmentation carries no learnable parameters, and the sampling-range buffers some classes expose
          in ``state_dict()`` are inert, so re-construct the augmentation to change what it samples.
          ``pickle`` and ``copy.deepcopy`` do carry the last ``_params``.
        - an empty batch is an empty output on the classes that accept one, but it is not a package-wide
          guarantee: a minority of the classes raise on ``B = 0``, in several unrelated exception families.
        - rotation-like parameters are in degrees, and a positive angle turns the image counter-clockwise as
          displayed (top-left origin, y pointing down), as :func:`~kornia.geometry.transform.rotate` documents.
          The ``*Affine*`` classes deviate and turn clockwise -- see :class:`RandomAffine` (tracked in
          `#4408 <https://github.com/kornia/kornia/issues/4408>`_).
        - ``torch.jit.script`` does not support these modules: the ``forward(*args, **kwargs)`` signature of the
          base is not scriptable. ``torch.compile`` works in its default, graph-break-tolerant mode;
          ``fullgraph=True`` fails wherever a drawn value reaches a Python-level size or branch: on
          :class:`RandomCrop`, :class:`RandomResizedCrop`, :class:`LongestMaxSize`,
          :class:`SmallestMaxSize` and :class:`RandomCrop3D`, on :class:`ColorJiggle` /
          :class:`ColorJitter` and :class:`RandomSnow`, on the mix classes and on the four sequential
          containers. :class:`Resize`, :class:`CenterCrop` and :class:`PadTo` compile whole.

    .. warning::
        One wrong input rank raises three different exception types depending on the entry point that sees it.
        Tracked in `#4424 <https://github.com/kornia/kornia/issues/4424>`_.

    .. warning::
        ``p_batch`` is documented on the bases but is named by only two concrete constructors, so the
        randomness model the base page describes is not the one most classes offer. Tracked in
        `#4425 <https://github.com/kornia/kornia/issues/4425>`_.

    .. warning::
        ``set_rng_device_and_dtype`` is documented as the way to change where and in what dtype the
        parameters are sampled, but on most classes it moves the ``p`` / ``p_batch`` gate and nothing else.
        Tracked in `#4426 <https://github.com/kornia/kornia/issues/4426>`_.

    .. warning::
        ``forward`` swallows ``generator=`` -- and every other unknown keyword -- without a warning, so a
        per-instance generator looks accepted and is ignored. Tracked in
        `#4427 <https://github.com/kornia/kornia/issues/4427>`_.

    .. warning::
        The ``_param_generator.*`` range buffers that reach ``state_dict()`` are inert, so a ``state_dict``
        round trip is a silent no-op. Tracked in `#4428 <https://github.com/kornia/kornia/issues/4428>`_.

    .. warning::
        ``B = 0`` raises on a minority of the concrete classes instead of returning an empty batch, which
        contradicts the library-wide empty-in/empty-out convention of
        `#4115 <https://github.com/kornia/kornia/issues/4115>`_. Tracked in
        `#4429 <https://github.com/kornia/kornia/issues/4429>`_.

    """

    def validate_tensor(self, input: torch.Tensor) -> None:
        """Check if the input torch.Tensor is formatted as expected."""
        _validate_input_dtype(input, accepted_dtypes=[torch.bfloat16, float16, float32, float64])
        if len(input.shape) != 4:
            raise RuntimeError(f"Expect (B, C, H, W). Got {input.shape}.")

    def transform_tensor(
        self, input: torch.Tensor, *, shape: Optional[torch.Tensor] = None, match_channel: bool = True
    ) -> torch.Tensor:
        """Convert any incoming (H, W), (C, H, W) and (B, C, H, W) into (B, C, H, W)."""
        _validate_input_dtype(input, accepted_dtypes=[torch.bfloat16, float16, float32, float64])

        if shape is None:
            return _transform_input(input)
        else:
            return _transform_input_by_shape(input, reference_shape=shape, match_channel=match_channel)


class RigidAffineAugmentationBase2D(AugmentationBase2D):
    r"""AugmentationBase2D base class for rigid/affine augmentation implementations.

    See the Convention block on :class:`~kornia.augmentation.AugmentationBase2D`.

    RigidAffineAugmentationBase2D enables routined transformation with given transformation matrices
    for different data types like masks, boxes, and keypoints.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it to the batch
          form ``False``.

    Convention:
        - a subclass implements :meth:`compute_transformation`, which returns the ``(B, 3, 3)`` matrix of the
          sampled transform; the base raises ``NotImplementedError`` on its own. That matrix is what drives the
          mask, box and keypoint paths, so a rigid subclass gets them for free.
        - the matrix of the last call is readable as ``transform_matrix`` and is built lazily: it is
          computed on first access for the subclasses whose ``apply_transform`` does not read it. ``pickle``
          and ``copy.deepcopy`` carry it along with ``_params``.
        - this base adds no ``inverse``. Among the 2D bases only
          :class:`~kornia.augmentation.GeometricAugmentationBase2D` has one.

    """

    _transform_matrix: Optional[torch.Tensor] = None
    # Set True on subclasses whose ``apply_transform`` ignores the transform matrix (e.g. flips):
    # the image output never reads it, so building it every forward is pure overhead. When True,
    # ``apply_func`` defers the matrix and ``transform_matrix`` computes it on first access.
    _compute_matrix_lazily: bool = False
    _lazy_matrix_args: Optional[Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]] = None

    @property
    def transform_matrix(self) -> Optional[torch.Tensor]:
        if self._transform_matrix is None and self._lazy_matrix_args is not None:
            in_tensor, params, flags = self._lazy_matrix_args
            self._transform_matrix = self.generate_transformation_matrix(in_tensor, params, flags)
            self._lazy_matrix_args = None
        return self._transform_matrix

    def identity_matrix(self, input: torch.Tensor) -> torch.Tensor:
        """Return 3x3 identity matrix."""
        return eye_like(3, input)

    def compute_transformation(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, Any]
    ) -> torch.Tensor:
        raise NotImplementedError

    def generate_transformation_matrix(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, Any]
    ) -> torch.Tensor:
        """Generate transformation matrices with the given input and param settings."""
        batch_prob = params["batch_prob"]
        to_apply = torch.atleast_1d(batch_prob > 0.5)

        in_tensor = self.transform_tensor(input)

        trans_matrix_applied = self.compute_transformation(in_tensor, params=params, flags=flags)

        if self.p == 1.0 and self.p_batch == 1.0:
            # Always applied (static probabilities): the blend selects the computed matrix
            # everywhere, so it equals `trans_matrix_applied`. Skip building the identity and
            # the `where` — this is a hot per-call cost (~40% of a flip's forward is the matrix
            # path) that the image output never needs. Mirrors the `transform_inputs` fast path.
            trans_matrix = trans_matrix_applied
            if is_autocast_enabled():
                trans_matrix = trans_matrix.type(input.dtype)
            return trans_matrix

        trans_matrix_identity = self.identity_matrix(in_tensor)

        if is_autocast_enabled():
            trans_matrix_applied = trans_matrix_applied.type(input.dtype)
            trans_matrix_identity = trans_matrix_identity.type(input.dtype)

        # If batch sizes line up, do the where-blend. Otherwise (e.g. VideoSequential
        # passes B-sized batch_prob into a B*T-sized input) fall back to all-or-nothing.
        if trans_matrix_applied.shape[0] == to_apply.shape[0] == trans_matrix_identity.shape[0]:
            to_apply_expanded = to_apply.view(-1, *([1] * (trans_matrix_applied.dim() - 1))).to(
                trans_matrix_applied.device
            )
            trans_matrix = torch.where(to_apply_expanded, trans_matrix_applied, trans_matrix_identity)
        else:
            trans_matrix = trans_matrix_applied if bool(to_apply.any()) else trans_matrix_identity

        return trans_matrix

    def inverse_inputs(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def inverse_masks(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def inverse_boxes(
        self,
        input: Boxes,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> Boxes:
        raise NotImplementedError

    def inverse_keypoints(
        self,
        input: Keypoints,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> Keypoints:
        raise NotImplementedError

    def inverse_classes(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def apply_func(
        self, in_tensor: torch.Tensor, params: Dict[str, torch.Tensor], flags: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        if flags is None:
            flags = self.flags

        if self._compute_matrix_lazily:
            # apply_transform ignores the matrix for these ops, so don't build it here; defer to
            # the first `.transform_matrix` access (e.g. AugmentationSequential propagating to
            # boxes/keypoints/masks). A standalone flip that never reads the matrix skips it.
            self._commit_state(transform_matrix=None, lazy_matrix_args=(in_tensor, params, flags))
            return self.transform_inputs(in_tensor, params, flags, None)

        trans_matrix = self.generate_transformation_matrix(in_tensor, params, flags)
        output = self.transform_inputs(in_tensor, params, flags, trans_matrix)
        self._commit_state(transform_matrix=trans_matrix, lazy_matrix_args=None)

        return output
