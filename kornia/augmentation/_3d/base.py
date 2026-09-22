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

import torch
from torch import float16, float32, float64

import kornia
from kornia.augmentation.base import _AugmentationBase
from kornia.augmentation.utils import _transform_input3d, _transform_input3d_by_shape, _validate_input_dtype
from kornia.geometry.boxes import Boxes3D
from kornia.geometry.keypoints import Keypoints3D


class AugmentationBase3D(_AugmentationBase):
    r"""AugmentationBase3D base class for customized augmentation implementations.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.

    Convention:
        - the working layout is ``(B, C, D, H, W)`` float. Bare 3D augmentations accept inputs of rank three
          and four and promote them by prepending batch and, for rank three, channel dimensions; ``keepdim=True``
          restores that original rank. Inside ``AugmentationSequential``, rank-4 input is rejected because
          ``(C, D, H, W)`` is ambiguous with the container's ``(B, C, H, W)`` image layout. The dtype guard
          accepts only ``float16``, ``bfloat16``, ``float32``, and ``float64``.
          :class:`~kornia.augmentation.RandomTransplantation3D` overrides ``forward`` and takes batched inputs
          only.
        - ``p`` gates samples and ``p_batch`` gates a whole call; ``same_on_batch=True`` shares the generated
          values. Of the concrete 3D constructors only :class:`~kornia.augmentation.RandomTransplantation3D`
          exposes ``p_batch`` (`#4425 <https://github.com/kornia/kornia/issues/4425>`_); :class:`CenterCrop3D` and
          :class:`RandomCrop3D` instead map their own ``p`` onto the call-wide gate. Parameters use the common
          augmentation RNG and ``forward(x, params=...)`` replays a complete generated dictionary. See
          :doc:`/get-started/conventions` for the canonical sampling, seeding, and serialization contract.
        - rigid subclasses expose the last sampled ``(B, 4, 4)`` ``transform_matrix``. The 3D bases do not
          implement ``inverse``: their subclasses have no ``inverse`` method, and a geometric 3D child makes an
          :class:`~kornia.augmentation.container.AugmentationSequential` inverse raise.
          :class:`~kornia.augmentation.RandomTransplantation3D` is the exception: it also derives from
          :class:`~kornia.augmentation.MixAugmentationBaseV2` and so carries that class's ``inverse``, which
          raises ``RuntimeError``.

    """

    def validate_tensor(self, input: torch.Tensor) -> None:
        """Check if the input torch.Tensor is formatted as expected."""
        _validate_input_dtype(input, accepted_dtypes=[torch.bfloat16, float16, float32, float64])
        if len(input.shape) != 5:
            raise RuntimeError(f"Expect (B, C, D, H, W). Got {input.shape}.")

    def transform_tensor(
        self, input: torch.Tensor, *, shape: Optional[torch.Tensor] = None, match_channel: bool = True
    ) -> torch.Tensor:
        """Convert any incoming (D, H, W), (C, D, H, W) and (B, C, D, H, W) into (B, C, D, H, W)."""
        _validate_input_dtype(input, accepted_dtypes=[torch.bfloat16, float16, float32, float64])
        if shape is None:
            return _transform_input3d(input)

        return _transform_input3d_by_shape(input, reference_shape=shape, match_channel=match_channel)

    def identity_matrix(self, input: torch.Tensor) -> torch.Tensor:
        """Return 4x4 identity matrix."""
        return kornia.core.ops.eye_like(4, input)


class RigidAffineAugmentationBase3D(AugmentationBase3D):
    r"""AugmentationBase3D base class for rigid/affine augmentation implementations.

    RigidAffineAugmentationBase3D enables routined transformation with given transformation matrices
    for different data types like masks, boxes, and keypoints.

    See the Convention block on :class:`~kornia.augmentation.AugmentationBase3D`.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it to the batch
          form ``False``.

    """

    _transform_matrix: Optional[torch.Tensor]

    @property
    def transform_matrix(self) -> Optional[torch.Tensor]:
        return self._transform_matrix

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
        trans_matrix_identity = self.identity_matrix(in_tensor)

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
        input: Boxes3D,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> Boxes3D:
        raise NotImplementedError

    def inverse_keypoints(
        self,
        input: Keypoints3D,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> Keypoints3D:
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

        trans_matrix = self.generate_transformation_matrix(in_tensor, params, flags)
        output = self.transform_inputs(in_tensor, params, flags, trans_matrix)
        self._transform_matrix = trans_matrix

        return output
