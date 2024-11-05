from typing import Any, Dict, Optional

from torch import float16, float32, float64

from kornia.augmentation.base import _AugmentationBase
from kornia.augmentation.utils import _transform_input, _transform_input_by_shape, _validate_input_dtype
from kornia.core import Tensor
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints
from kornia.utils import eye_like, is_autocast_enabled


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

    def validate_tensor(self, input: Tensor) -> None:
        """Check if the input tensor is formatted as expected."""
        _validate_input_dtype(input, accepted_dtypes=[float16, float32, float64])
        if len(input.shape) != 4:
            raise RuntimeError(f"Expect (B, C, H, W). Got {input.shape}.")

    def transform_tensor(self, input: Tensor, *, shape: Optional[Tensor] = None, match_channel: bool = True) -> Tensor:
        """Convert any incoming (H, W), (C, H, W) and (B, C, H, W) into (B, C, H, W)."""
        _validate_input_dtype(input, accepted_dtypes=[float16, float32, float64])

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

    _transform_matrix: Optional[Tensor]

    @property
    def transform_matrix(self) -> Optional[Tensor]:
        return self._transform_matrix

    def identity_matrix(self, input: Tensor) -> Tensor:
        """Return 3x3 identity matrix."""
        return eye_like(3, input)

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
            trans_matrix_A = self.identity_matrix(in_tensor)
            trans_matrix_B = self.compute_transformation(in_tensor[to_apply], params=params, flags=flags)

            if is_autocast_enabled():
                trans_matrix_A = trans_matrix_A.type(input.dtype)
                trans_matrix_B = trans_matrix_B.type(input.dtype)

            trans_matrix = trans_matrix_A.index_put((to_apply,), trans_matrix_B)

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
        self, input: Boxes, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Boxes:
        raise NotImplementedError

    def inverse_keypoints(
        self, input: Keypoints, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Keypoints:
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
