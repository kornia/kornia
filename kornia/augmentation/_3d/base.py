from __future__ import annotations

from torch import Tensor, float16, float32, float64

import kornia
from kornia.augmentation.base import _AugmentationBase
from kornia.augmentation.utils import _transform_input3d, _validate_input_dtype


class AugmentationBase3D(_AugmentationBase):
    r"""AugmentationBase3D base class for customized augmentation implementations.

    For any augmentation, the implementation of "generate_parameters" and "apply_transform" are required while the
    "compute_transformation" is only required when passing "return_transform" as True.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.
    """

    def __check_batching__(self, input: Tensor):
        if isinstance(input, tuple):
            inp, mat = input
            if len(inp.shape) == 5:
                if len(mat.shape) != 3:
                    raise AssertionError('Input tensor is in batch mode ' 'but transformation matrix is not')
                if mat.shape[0] != inp.shape[0]:
                    raise AssertionError(
                        f"In batch dimension, input has {inp.shape[0]} but transformation matrix has {mat.shape[0]}"
                    )
            elif len(inp.shape) in (3, 4):
                if len(mat.shape) != 2:
                    raise AssertionError("Input tensor is in non-batch mode but transformation matrix is not")
            else:
                raise ValueError(f'Unrecognized output shape. Expected 3, 4 or 5, got {len(inp.shape)}')

    def transform_tensor(self, input: Tensor) -> Tensor:
        """Convert any incoming (D, H, W), (C, D, H, W) and (B, C, D, H, W) into (B, C, D, H, W)."""
        _validate_input_dtype(input, accepted_dtypes=[float16, float32, float64])
        return _transform_input3d(input)

    def identity_matrix(self, input) -> Tensor:
        """Return 4x4 identity matrix."""
        return kornia.eye_like(4, input)
