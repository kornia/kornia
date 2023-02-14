from typing import Any, Dict, Optional

from kornia.augmentation._2d.base import RigidAffineAugmentationBase2D
from kornia.constants import Resample
from kornia.core import Tensor, as_tensor
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints
from kornia.utils.helpers import _torch_inverse_cast


class GeometricAugmentationBase2D(RigidAffineAugmentationBase2D):
    r"""GeometricAugmentationBase2D base class for customized geometric augmentation implementations.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
          to the batch form ``False``.
    """

    def inverse_transform(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        """By default, the exact transformation as ``apply_transform`` will be used."""
        raise NotImplementedError

    def compute_inverse_transformation(self, transform: Tensor) -> Tensor:
        """Compute the inverse transform of given transformation matrices."""
        return _torch_inverse_cast(transform)

    def inverse_parameters(self, params: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Update the value for `transform_matrix_inv` key."""
        transform = params["transform_matrix"]
        transform = self.compute_inverse_transformation(transform)
        params.update({"transform_matrix_inv": transform})
        return params

    def get_transformation_matrix(
        self, input: Tensor, params: Optional[Dict[str, Tensor]] = None, flags: Optional[Dict[str, Any]] = None
    ) -> Tensor:
        """Obtain transformation matrices.

        Return the current transformation matrix if existed. Generate a new one, otherwise.
        """
        flags = self.flags if flags is None else flags
        if params is not None and "transform_matrix" in params:
            transform = params["transform_matrix"]
        elif params is not None:
            transform = self.generate_transformation_matrix(input, params, flags)
        else:
            params = self.forward_parameters(input.shape)
            transform = params["transform_matrix"]
        return as_tensor(transform, device=input.device, dtype=input.dtype)

    def apply_non_transform_mask(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]
    ) -> Tensor:
        """Process masks corresponding to the inputs that are no transformation applied."""
        return input

    def apply_transform_mask(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]
    ) -> Tensor:
        """Process masks corresponding to the inputs that are transformed.

        Note:
            Convert "resample" arguments to "nearest" by default.
        """
        resample_method: Optional[Resample]
        if "resample" in flags:
            resample_method = flags["resample"]
            flags["resample"] = Resample.get("nearest")
        output = self.apply_transform(input, params, flags)
        if resample_method is not None:
            flags["resample"] = resample_method
        return output

    def apply_non_transform_box(
        self, input: Boxes, params: Dict[str, Tensor], flags: Dict[str, Any]
    ) -> Boxes:
        """Process boxes corresponding to the inputs that are no transformation applied."""
        return input

    def apply_transform_box(
        self, input: Boxes, params: Dict[str, Tensor], flags: Dict[str, Any]
    ) -> Boxes:
        """Process boxes corresponding to the inputs that are transformed."""
        transform = params["transform_matrix"]
        input = self.apply_non_transform_box(input, params, flags)
        return input.transform_boxes_(transform)

    def apply_non_transform_keypoint(
        self, input: Keypoints, params: Dict[str, Tensor], flags: Dict[str, Any]
    ) -> Keypoints:
        """Process keypoints corresponding to the inputs that are no transformation applied."""
        return input

    def apply_transform_keypoint(
        self, input: Keypoints, params: Dict[str, Tensor], flags: Dict[str, Any]
    ) -> Keypoints:
        """Process keypoints corresponding to the inputs that are transformed."""
        transform = params["transform_matrix"]
        input = self.apply_non_transform_keypoint(input, params, flags)
        return input.transform_keypoints_(transform)

    def apply_non_transform_class(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]
    ) -> Tensor:
        """Process class tags corresponding to the inputs that are no transformation applied."""
        return input

    def apply_transform_class(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]
    ) -> Tensor:
        """Process class tags corresponding to the inputs that are transformed."""
        return input

    def inverse_inputs(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], **kwargs,
    ) -> Tensor:
        in_tensor = self.transform_tensor(input)
        batch_prob = params['batch_prob'][:, None, None, None]

        params, flags = self._process_kwargs_to_params_and_flags(
            self._params if params is None else params, flags, **kwargs
        )

        if "transform_matrix_inv" not in params:
            params = self.inverse_parameters(params)

        inversed = self.inverse_transform(in_tensor, params=params, flags=flags)

        # No need to permute according to batch_prob
        if self._param_generator is not None and self._param_generator.has_fit_batch_prob:
            return inversed
        return inversed * batch_prob.round() + input * (1 - batch_prob.round())

    def inverse_masks(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], **kwargs,
    ) -> Tensor:
        resample_method: Optional[Resample] = None
        if "resample" in flags:
            resample_method = flags["resample"]
            flags["resample"] = Resample.get("nearest")
        output = self.inverse_inputs(input, params, flags, **kwargs)
        if resample_method is not None:
            flags["resample"] = resample_method
        return output

    def inverse_boxes(
        self, input: Boxes, params: Dict[str, Tensor], flags: Dict[str, Any], **kwargs,
    ) -> Boxes:
        batch_prob = params['batch_prob'][:, None, None]

        params, flags = self._process_kwargs_to_params_and_flags(
            self._params if params is None else params, flags, **kwargs
        )

        if "transform_matrix_inv" not in params:
            params = self.inverse_parameters(params)
        transform = params["transform_matrix_inv"]

        # No need to permute according to batch_prob
        if self._param_generator is not None and self._param_generator.has_fit_batch_prob:
            return input.transform_boxes(transform)
        return input.transform_boxes(transform) * batch_prob.round() + input * (1 - batch_prob.round())

    def inverse_keypoints(
        self, input: Keypoints, params: Dict[str, Tensor], flags: Dict[str, Any], **kwargs,
    ) -> Keypoints:
        """Inverse the transformation on keypoints.

        Args:
            input: input keypoints tensor or object.
            params: the corresponding parameters for an operation.
            flags: static parameters.
            transform: the inverse tansformation matrix
        """
        batch_prob = params['batch_prob'][:, None, None]

        params, flags = self._process_kwargs_to_params_and_flags(
            self._params if params is None else params, flags, **kwargs
        )

        if "transform_matrix_inv" not in params:
            params = self.inverse_parameters(params)
        transform = params["transform_matrix_inv"]

        # No need to permute according to batch_prob
        if self._param_generator is not None and self._param_generator.has_fit_batch_prob:
            return input.transform_keypoints(transform)
        return input.transform_keypoints(transform) * batch_prob.round() + input * (1 - batch_prob.round())

    def inverse_classes(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], **kwargs,
    ) -> Tensor:
        return input

    def inverse(self, input: Tensor, params: Optional[Dict[str, Tensor]] = None, **kwargs) -> Tensor:
        """Perform inverse operations.

        Args:
            input: the input tensor.
            params: the corresponding parameters for an operation.
                If None, a new parameter suite will be generated.
            **kwargs: key-value pairs to override the parameters and flags.
        """
        input_shape = input.shape
        in_tensor = self.transform_tensor(input)

        params, flags = self._process_kwargs_to_params_and_flags(
            self._params if params is None else params, self.flags, **kwargs
        )

        if params is None:
            params = self._params

        output = self.inverse_inputs(in_tensor, params, flags)

        if self.keepdim:
            return self.transform_output_tensor(output, input_shape)

        return output
