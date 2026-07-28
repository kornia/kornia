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

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it to the batch
          form ``False``.

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
            to_apply_expanded = to_apply.view(-1, *([1] * (trans_matrix_applied.dim() - 1)))
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
