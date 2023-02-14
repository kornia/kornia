from typing import Any, Dict, Optional

from torch import Size, float16, float32, float64

import kornia
from kornia.augmentation.base import _AugmentationBase
from kornia.augmentation.utils import _transform_input3d, _validate_input_dtype
from kornia.core import Tensor, zeros
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
    """

    def _expand_batch_prob(self, batch_prob: Tensor) -> Tensor:
        return batch_prob[:, None, None, None, None]

    def validate_tensor(self, input: Tensor) -> None:
        """Check if the input tensor is formatted as expected."""
        _validate_input_dtype(input, accepted_dtypes=[float16, float32, float64])
        if len(input.shape) != 5:
            raise RuntimeError(f"Expect (B, C, D, H, W). Got {input.shape}.")

    def transform_tensor(self, input: Tensor) -> Tensor:
        """Convert any incoming (D, H, W), (C, D, H, W) and (B, C, D, H, W) into (B, C, D, H, W)."""
        _validate_input_dtype(input, accepted_dtypes=[float16, float32, float64])
        return _transform_input3d(input)

    def identity_matrix(self, input) -> Tensor:
        """Return 4x4 identity matrix."""
        return kornia.eye_like(4, input)


class RigidAffineAugmentationBase3D(AugmentationBase3D):
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
    @property
    def transform_matrix(self) -> Optional[Tensor]:
        if self._params is not None and "transform_matrix" in self._params:
            return self._params["transform_matrix"]
        return None

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        raise NotImplementedError

    def forward_parameters(self, batch_shape: Size) -> Dict[str, Tensor]:
        params = super().forward_parameters(batch_shape)
        transform_matrix = self.generate_transformation_matrix(zeros(batch_shape), params, self.flags)
        params.update({"transform_matrix": transform_matrix})
        return params

    def generate_transformation_matrix(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        """Generate transformation matrices with the given input and param settings."""
        batch_prob = params['batch_prob'][:, None, None]

        in_tensor = self.transform_tensor(input)

        trans_matrix = self.compute_transformation(in_tensor, params=params, flags=flags)

        return trans_matrix * batch_prob.round() + self.identity_matrix(in_tensor) * (1 - batch_prob.round())

    def inverse_inputs(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        raise NotImplementedError

    def inverse_masks(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        raise NotImplementedError

    def inverse_boxes(
        self, input: Boxes3D, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Boxes3D:
        raise NotImplementedError

    def inverse_keypoints(
        self, input: Keypoints3D, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Keypoints3D:
        raise NotImplementedError

    def inverse_classes(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        raise NotImplementedError

    def apply_func(
        self, in_tensor: Tensor, params: Dict[str, Tensor], flags: Optional[Dict[str, Any]] = None
    ) -> Tensor:
        if flags is None:
            flags = self.flags

        output = self.transform_inputs(in_tensor, params, flags)

        return output
