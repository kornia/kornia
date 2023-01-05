from typing import Any, Dict, Optional, Union
from torch import Tensor, float16, float32, float64

from kornia.augmentation.base import _AugmentationBase
from kornia.augmentation.utils import _transform_input, _validate_input_dtype
from kornia.utils import eye_like
from kornia.geometry.boxes import Boxes


class AugmentationBase2D(_AugmentationBase):
    r"""AugmentationBase2D base class for customized augmentation implementations.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it to the batch
          form ``False``.
    """

    def validate_tensor(self, input: Tensor) -> bool:
        """Check if the input tensor is formated as expected."""
        _validate_input_dtype(input, accepted_dtypes=[float16, float32, float64])
        if len(input.shape) != 4:
            raise RuntimeError(f"Expect (B, C, H, W). Got {input.shape}.")

    def __check_batching__(self, input: Tensor):
        if isinstance(input, tuple):
            inp, mat = input
            if len(inp.shape) == 4:
                if len(mat.shape) != 3:
                    raise AssertionError('Input tensor is in batch mode ' 'but transformation matrix is not')
                if mat.shape[0] != inp.shape[0]:
                    raise AssertionError(
                        f"In batch dimension, input has {inp.shape[0]} but transformation matrix has {mat.shape[0]}"
                    )
            elif len(inp.shape) in (2, 3):
                if len(mat.shape) != 2:
                    raise AssertionError("Input tensor is in non-batch mode but transformation matrix is not")
            else:
                raise ValueError(f'Unrecognized output shape. Expected 2, 3, or 4, got {len(inp.shape)}')

    def transform_tensor(self, input: Tensor) -> Tensor:
        """Convert any incoming (H, W), (C, H, W) and (B, C, H, W) into (B, C, H, W)."""
        _validate_input_dtype(input, accepted_dtypes=[float16, float32, float64])
        return _transform_input(input)


class RigidAffineAugmentationBase2D(AugmentationBase2D):
    r"""AugmentationBase2D base class for rigid/affine augmentation implementations.

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
    def transform_matrix(self,) -> Optional[Tensor]:
        return self._transform_matrix

    def identity_matrix(self, input) -> Tensor:
        """Return 3x3 identity matrix."""
        return eye_like(3, input)

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        raise NotImplementedError

    def generate_transformation_matrix(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]
    ) -> Tensor:
        """Generate transformation matrices with the given input and param settings.
        """
        to_apply = params['batch_prob']
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
        self,
        input: Union[Tensor, Boxes],
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Boxes:
        raise NotImplementedError

    def inverse_keypoints(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
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
