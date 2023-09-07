from typing import Any, Dict, Optional

from torch import float16, float32, float64

from kornia.augmentation.base import _AugmentationBase
from kornia.augmentation.utils import _validate_input_dtype
from kornia.core import Tensor
from kornia.geometry.boxes import Boxes3D
from kornia.geometry.keypoints import Keypoints3D
from kornia.utils import eye_like, is_autocast_enabled


class AugmentationBasePC(_AugmentationBase):
    r"""AugmentationBasePC base class for customized point cloud augmentation implementations.

    AugmentationBasePC aims at offering a generic base class for a greater level of customization.
    If the subclass contains routined matrix-based transformations, `RigidAffineAugmentationBasePC`
    might be a better fit.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.

    Note:
        By default, :math:`(B, N, 3)` represent xyz only, :math:`(B, N, 6)` represents xyz and normal,
        :math:`(B, N, 9)` represents xyz, normal, and rgb. Complex point cloud data may use our point cloud
        data type.
    """

    def __init__(self, p: float = 0.5, p_batch: float = 1.0, same_on_batch: bool = False) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=True)

    def validate_tensor(self, input: Tensor) -> None:
        """Check if the input tensor is formatted as expected."""
        _validate_input_dtype(input, accepted_dtypes=[float16, float32, float64])
        if len(input.shape) != 3 or input.size(-1) < 3:
            raise RuntimeError(f"Expect (B, N, C) and C >= 3. Got {input.shape}.")

    def transform_tensor(self, input: Tensor) -> Tensor:
        return input


class RigidAffineAugmentationBasePC(AugmentationBasePC):
    r"""AugmentationBasePC base class for rigid/affine augmentation implementations.

    RigidAffineAugmentationBasePC enables routined transformation with given transformation matrices
    for different data types like masks, boxes, and keypoints.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.
    """

    def __init__(self, p: float = 0.5, p_batch: float = 1.0, same_on_batch: bool = False) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch)
        self._transform_matrix: Optional[Tensor] = None

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

        batch_prob = params['batch_prob']
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
