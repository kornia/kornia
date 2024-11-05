from typing import Any, Dict, Optional, Tuple

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

    def inverse_transform(
        self,
        input: Tensor,
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """By default, the exact transformation as ``apply_transform`` will be used."""
        raise NotImplementedError

    def compute_inverse_transformation(self, transform: Tensor) -> Tensor:
        """Compute the inverse transform of given transformation matrices."""
        return _torch_inverse_cast(transform)

    def get_transformation_matrix(
        self, input: Tensor, params: Optional[Dict[str, Tensor]] = None, flags: Optional[Dict[str, Any]] = None
    ) -> Tensor:
        """Obtain transformation matrices.

        Return the current transformation matrix if existed. Generate a new one, otherwise.
        """
        flags = self.flags if flags is None else flags
        if params is not None:
            transform = self.generate_transformation_matrix(input, params, flags)
        elif self.transform_matrix is None:
            params = self.forward_parameters(input.shape)
            transform = self.generate_transformation_matrix(input, params, flags)
        else:
            transform = self.transform_matrix
        return as_tensor(transform, device=input.device, dtype=input.dtype)

    def apply_non_transform_mask(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        """Process masks corresponding to the inputs that are no transformation applied."""
        return input

    def apply_transform_mask(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        """Process masks corresponding to the inputs that are transformed.

        Note:
            Convert "resample" arguments to "nearest" by default.
        """
        resample_method: Optional[Resample]
        if "resample" in flags:
            resample_method = flags["resample"]
            flags["resample"] = Resample.get("nearest")
        output = self.apply_transform(input, params, flags, transform)
        if resample_method is not None:
            flags["resample"] = resample_method
        return output

    def apply_non_transform_box(
        self, input: Boxes, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Boxes:
        """Process boxes corresponding to the inputs that are no transformation applied."""
        return input

    def apply_transform_box(
        self, input: Boxes, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Boxes:
        """Process boxes corresponding to the inputs that are transformed."""
        if transform is None:
            if self.transform_matrix is None:
                raise RuntimeError("No valid transformation matrix found. Please either pass one or forward one first.")
            transform = self.transform_matrix
        input = self.apply_non_transform_box(input, params, flags, transform)
        return input.transform_boxes_(transform)

    def apply_non_transform_keypoint(
        self, input: Keypoints, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Keypoints:
        """Process keypoints corresponding to the inputs that are no transformation applied."""
        return input

    def apply_transform_keypoint(
        self, input: Keypoints, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Keypoints:
        """Process keypoints corresponding to the inputs that are transformed."""
        if transform is None:
            if self.transform_matrix is None:
                raise RuntimeError("No valid transformation matrix found. Please either pass one or forward one first.")
            transform = self.transform_matrix
        input = self.apply_non_transform_keypoint(input, params, flags, transform)
        return input.transform_keypoints_(transform)

    def apply_non_transform_class(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        """Process class tags corresponding to the inputs that are no transformation applied."""
        return input

    def apply_transform_class(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        """Process class tags corresponding to the inputs that are transformed."""
        return input

    def inverse_inputs(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        in_tensor = self.transform_tensor(input)
        output = in_tensor.clone()
        batch_prob = params["batch_prob"]
        to_apply = batch_prob > 0.5  # NOTE: in case of Relaxed Distributions.

        params, flags = self._process_kwargs_to_params_and_flags(
            self._params if params is None else params, flags, **kwargs
        )

        size: Optional[Tuple[int, int]] = None
        if "forward_input_shape" in params:
            # Majorly for cropping functions
            _size = params["forward_input_shape"].tolist()
            size = (_size[-2], _size[-1])

        # if no augmentation needed
        if not to_apply.any():
            output = in_tensor
        # if all data needs to be augmented
        elif to_apply.all():
            output = self.inverse_transform(in_tensor, flags=flags, transform=transform, size=size)
        else:
            output[to_apply] = self.inverse_transform(
                in_tensor[to_apply],
                transform=transform[to_apply] if transform is not None else transform,
                size=size,
                flags=flags,
            )
        return output

    def inverse_masks(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        resample_method: Optional[Resample] = None
        if "resample" in flags:
            resample_method = flags["resample"]
            flags["resample"] = Resample.get("nearest")
        output = self.inverse_inputs(input, params, flags, transform, **kwargs)
        if resample_method is not None:
            flags["resample"] = resample_method
        return output

    def inverse_boxes(
        self,
        input: Boxes,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Boxes:
        output = input.clone()
        batch_prob = params["batch_prob"]
        to_apply = batch_prob > 0.5  # NOTE: in case of Relaxed Distributions.

        if transform is None:
            raise RuntimeError("`transform` has to be a tensor. Got None.")

        params, flags = self._process_kwargs_to_params_and_flags(
            self._params if params is None else params, flags, **kwargs
        )

        # if no augmentation needed
        if not to_apply.any():
            output = input
        # if all data needs to be augmented
        elif to_apply.all():
            output = input.transform_boxes_(transform)
        else:
            output[to_apply] = input[to_apply].transform_boxes_(transform[to_apply])

        return output

    def inverse_keypoints(
        self,
        input: Keypoints,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Keypoints:
        """Inverse the transformation on keypoints.

        Args:
            input: input keypoints tensor or object.
            params: the corresponding parameters for an operation.
            flags: static parameters.
            transform: the inverse transformation matrix
        """
        output = input.clone()
        batch_prob = params["batch_prob"]
        to_apply = batch_prob > 0.5  # NOTE: in case of Relaxed Distributions.

        if transform is None:
            raise RuntimeError("`transform` has to be a tensor. Got None.")

        params, flags = self._process_kwargs_to_params_and_flags(
            self._params if params is None else params, flags, **kwargs
        )

        # if no augmentation needed
        if not to_apply.any():
            output = input
        # if all data needs to be augmented
        elif to_apply.all():
            output = input.transform_keypoints_(transform)
        else:
            output[to_apply] = input[to_apply].transform_keypoints_(transform[to_apply])

        return output

    def inverse_classes(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        return input

    def inverse(self, input: Tensor, params: Optional[Dict[str, Tensor]] = None, **kwargs: Any) -> Tensor:
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
        transform = self.get_transformation_matrix(in_tensor, params=params, flags=flags)

        transform = self.compute_inverse_transformation(transform)
        output = self.inverse_inputs(in_tensor, params, flags, transform)

        if self.keepdim:
            return self.transform_output_tensor(output, input_shape)

        return output
