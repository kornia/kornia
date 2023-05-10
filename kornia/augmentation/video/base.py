from torch import float16, float32, float64

import kornia
from kornia.augmentation.base import _AugmentationBase
from kornia.augmentation.utils import _transform_input3d, _validate_input_dtype
from kornia.core import Tensor


class AugmentationBaseVideo(_AugmentationBase):
    r"""AugmentationBaseVideo base class for customized video augmentation implementations.

    AugmentationBaseVideo aims at offering a generic base class for a greater level of customization.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.
    """

    def __init__(
        self, p: float = 0.5, p_batch: float = 1.0, same_on_batch: bool = False, keepdim: bool = False
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)

    def validate_tensor(self, input: Tensor) -> None:
        """Check if the input tensor is formatted as expected."""
        _validate_input_dtype(input, accepted_dtypes=[float16, float32, float64])
        if len(input.shape) != 5:
            raise RuntimeError(f"Expect (B, C, D, H, W). Got {input.shape}.")

    def transform_tensor(self, input: Tensor) -> Tensor:
        """Convert any incoming (T, H, W), (C, T, H, W) and (B, C, T, H, W) into (B, C, T, H, W)."""
        _validate_input_dtype(input, accepted_dtypes=[float16, float32, float64])
        return _transform_input3d(input)

    def identity_matrix(self, input: Tensor) -> Tensor:
        """Return 4x4 identity matrix."""
        return kornia.eye_like(4, input)
