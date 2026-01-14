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

"""Base classes for 2D augmentations with flattened hierarchy."""

from typing import Any, Dict, Optional

import torch
from torch import float16, float32, float64

from kornia.augmentation.base import AugmentationBase
from kornia.augmentation.utils import _transform_input, _transform_input_by_shape, _validate_input_dtype
from kornia.core.ops import eye_like


class AugmentationBase2D(AugmentationBase):
    r"""AugmentationBase2D base class for 2D augmentation implementations.

    This is the common base class for all 2D augmentations (both intensity and geometric).
    It provides 2D tensor validation and transformation.

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

    @property
    def transform_matrix(self) -> Optional[torch.Tensor]:
        return self._transform_matrix

    def identity_matrix(self, input: torch.Tensor) -> torch.Tensor:
        """Return 3x3 identity matrix."""
        return eye_like(3, input)

    def compute_transformation(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute transformation matrix. Override in geometric augmentations."""
        return self.identity_matrix(input)

    def validate_tensor(self, input: torch.Tensor) -> None:
        """Check if the input torch.tensor is formatted as expected."""
        _validate_input_dtype(input, accepted_dtypes=[float16, float32, float64])
        if len(input.shape) != 4:
            raise RuntimeError(f"Expect (B, C, H, W). Got {input.shape}.")

    def transform_tensor(
        self, input: torch.Tensor, *, shape: Optional[torch.Tensor] = None, match_channel: bool = True
    ) -> torch.Tensor:
        """Convert any incoming (H, W), (C, H, W) and (B, C, H, W) into (B, C, H, W)."""
        _validate_input_dtype(input, accepted_dtypes=[float16, float32, float64])

        if shape is None:
            return _transform_input(input)
        else:
            return _transform_input_by_shape(input, reference_shape=shape, match_channel=match_channel)

    def apply_func(
        self, in_tensor: torch.Tensor, params: Dict[str, torch.Tensor], flags: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        if flags is None:
            flags = self.flags

        # Set transformation matrix to identity for base 2D augmentations
        self._transform_matrix = self.identity_matrix(in_tensor)
        output = self.transform_inputs(in_tensor, params, flags, self._transform_matrix)

        return output


# Backward compatibility alias
RigidAffineAugmentationBase2D = AugmentationBase2D

__all__ = [
    "AugmentationBase2D",
    "RigidAffineAugmentationBase2D",
]
