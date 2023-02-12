from typing import Any, Dict, Optional

from torch import float16, float32, float64, Size

from kornia.augmentation.base import _AugmentationBase
from kornia.augmentation.utils import _transform_input, _validate_input_dtype
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

    def transform_tensor(self, input: Tensor) -> Tensor:
        """Convert any incoming (H, W), (C, H, W) and (B, C, H, W) into (B, C, H, W)."""
        _validate_input_dtype(input, accepted_dtypes=[float16, float32, float64])
        return _transform_input(input)


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

    @property
    def transform_matrix(self) -> Optional[Tensor]:
        if self._params is not None and "transform_matrix" in self._params:
            return self._params["transform_matrix"]
        return None

    def identity_matrix(self, input) -> Tensor:
        """Return 3x3 identity matrix."""
        return eye_like(3, input)

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        raise NotImplementedError

    def generate_transformation_matrix(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        """Generate transformation matrices with the given input and param settings."""

        batch_prob = params['batch_prob'][:, None, None, None]

        in_tensor = self.transform_tensor(input)

        trans_matrix = self.compute_transformation(in_tensor, params=params, flags=flags)

        return trans_matrix * batch_prob.round() + self.identity_matrix(in_tensor) * (1 - batch_prob.round())

    def inverse_inputs(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]
    ) -> Tensor:
        raise NotImplementedError

    def inverse_masks(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]
    ) -> Tensor:
        raise NotImplementedError

    def inverse_boxes(
        self, input: Boxes, params: Dict[str, Tensor], flags: Dict[str, Any]
    ) -> Boxes:
        raise NotImplementedError

    def inverse_keypoints(
        self, input: Keypoints, params: Dict[str, Tensor], flags: Dict[str, Any]
    ) -> Keypoints:
        raise NotImplementedError

    def inverse_classes(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]
    ) -> Tensor:
        raise NotImplementedError

    def forward_parameters(self, batch_shape: Size) -> Dict[str, Tensor]:
        params = super().forward_parameters(batch_shape)
        transform_matrix = self.generate_transformation_matrix(batch_shape, params, self.flags)
        params.update({"transform_matrix": transform_matrix})
        return params

    def forward(
        self,
        input: Tensor,
        params: Optional[Dict[str, Tensor]] = None,
        flags: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Tensor:
        """Perform forward operations.

        Args:
            input: the input tensor.
            params: the corresponding parameters for an operation.
                If None, a new parameter suite will be generated.
            **kwargs: key-value pairs to override the parameters and flags.

        Note:
            By default, all the overwriting parameters in kwargs will not be recorded
            as in ``self._params``. If you wish it to be recorded, you may pass
            ``save_kwargs=True`` additionally.
        """
        input_shape = input.shape
        in_tensor = self.transform_tensor(input)

        if flags is None:
            flags = self.flags

        if params is None:
            params = self.forward_parameters(in_tensor)
            self._params = params

        params, flags = self._process_kwargs_to_params_and_flags(params, flags, **kwargs)

        output = self.transform_inputs(in_tensor, params, flags, transform=params["transform_matrix"])

        return self.transform_output_tensor(output, input_shape) if self.keepdim else output
