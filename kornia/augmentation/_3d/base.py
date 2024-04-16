from typing import Any, Dict, Optional

from torch import float16, float32, float64

import kornia
from kornia.augmentation.base import _AugmentationBase
from kornia.augmentation.utils import _transform_input3d, _transform_input3d_by_shape, _validate_input_dtype
from kornia.core import Tensor
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

    def validate_tensor(self, input: Tensor) -> None:
        """Check if the input tensor is formatted as expected."""
        _validate_input_dtype(input, accepted_dtypes=[float16, float32, float64])
        if len(input.shape) != 5:
            raise RuntimeError(f"Expect (B, C, D, H, W). Got {input.shape}.")

    def transform_tensor(self, input: Tensor, *, shape: Optional[Tensor] = None, match_channel: bool = True) -> Tensor:
        """Convert any incoming (D, H, W), (C, D, H, W) and (B, C, D, H, W) into (B, C, D, H, W)."""
        _validate_input_dtype(input, accepted_dtypes=[float16, float32, float64])
        if shape is None:
            return _transform_input3d(input)
        else:
            return _transform_input3d_by_shape(input, reference_shape=shape, match_channel=match_channel)

    def identity_matrix(self, input: Tensor) -> Tensor:
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

    _transform_matrix: Optional[Tensor]

    @property
    def transform_matrix(self) -> Optional[Tensor]:
        return self._transform_matrix

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        raise NotImplementedError

    def generate_transformation_matrix(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        """Generate transformation matrices with the given input and param settings."""
        batch_prob = params["batch_prob"]
        to_apply = batch_prob > 0.5  # NOTE: in case of Relaxed Distributions.
        in_tensor = self.transform_tensor(input)
        if not to_apply.any():
            trans_matrix = self.identity_matrix(in_tensor)
        elif to_apply.all():
            trans_matrix = self.compute_transformation(in_tensor, params=params, flags=flags)
        else:
            trans_matrix = self.identity_matrix(in_tensor)
            trans_matrix = trans_matrix.index_put(
                (to_apply,), self.compute_transformation(in_tensor[to_apply], params=params, flags=flags)
            )
        return trans_matrix

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

        trans_matrix = self.generate_transformation_matrix(in_tensor, params, flags)
        output = self.transform_inputs(in_tensor, params, flags, trans_matrix)
        self._transform_matrix = trans_matrix

        return output
