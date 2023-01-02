import warnings

from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union, cast
import torch

from kornia.augmentation._2d.base import _AugmentationBase, RigidAffineAugmentationBase2D
from kornia.augmentation.utils import override_parameters
from kornia.core import Module, Tensor, as_tensor
from kornia.geometry.boxes import Boxes
from kornia.utils.helpers import _torch_inverse_cast
from kornia.geometry.bbox import transform_bbox
from kornia.geometry.linalg import transform_points
from kornia.testing import KORNIA_UNWRAP
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

    def compute_inverse_transformation(self, transform: Tensor):
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
            transform = self.compute_transformation(input[params['batch_prob']], params=params, flags=flags)
        elif self.transform_matrix is None:
            params = self.forward_parameters(input.shape)
            transform = self.generate_transformation_matrix(input, params, flags)
        else:
            transform = self.transform_matrix
        return as_tensor(transform, device=input.device, dtype=input.dtype)

    def apply_non_transform_mask(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        """Process masks corresponding to the inputs that are no transformation applied.
        """
        return input

    def apply_transform_mask(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        """Process masks corresponding to the inputs that are transformed.

        Note:
            Convert "resample" arguments to "nearest" by default.
        """
        resample_method: Optional[str]
        if "resample" in flags:
            resample_method = flags["resample"]
            flags["resample"] = "nearest"
        output = self.apply_transform(input, params, flags, transform)
        if resample_method is not None:
            flags["resample"] = resample_method
        return output

    def apply_non_transform_box(
        self, input: Boxes, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Boxes:
        """Process boxes corresponding to the inputs that are no transformation applied.
        """
        padding_size = None
        if "padding_size" in params:
            # Mostly for operations like RandomCrop.
            padding_size = params["padding_size"]
        return input.pad(padding_size)

    def apply_transform_box(
        self, input: Boxes, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Boxes:
        """Process boxes corresponding to the inputs that are transformed.
        """
        raise NotImplementedError

    def apply_non_transform_keypoint(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        """Process keypoints corresponding to the inputs that are no transformation applied.
        """
        return input

    def apply_transform_keypoint(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        """Process keypoints corresponding to the inputs that are transformed.
        """
        raise NotImplementedError

    def apply_non_transform_class(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        """Process class tags corresponding to the inputs that are no transformation applied.
        """
        return input

    def apply_transform_class(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        """Process class tags corresponding to the inputs that are transformed.
        """
        return input

    def inverse_inputs(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Tensor
    ) -> Tensor:
        in_tensor = self.transform_tensor(input)
        output = in_tensor.clone()
        to_apply = params['batch_prob']

        size = None
        if "forward_input_shape" in params:
            # Majorly for cropping functions
            size = params['forward_input_shape'].numpy().tolist()
            size = (size[-2], size[-1])

        # if no augmentation needed
        if not to_apply.any():
            output = in_tensor
        # if all data needs to be augmented
        elif to_apply.all():
            transform = self.compute_inverse_transformation(transform)
            output = self.inverse_transform(in_tensor, flags=flags, transform=transform, size=size)
        else:
            transform[to_apply] = self.compute_inverse_transformation(transform[to_apply])
            output[to_apply] = self.inverse_transform(
                in_tensor[to_apply], transform=transform[to_apply], size=size, flags=flags
            )
        return output

    def inverse_masks(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Tensor
    ) -> Tensor:
        
        resample_method: Optional[str]
        if "resample" in flags:
            resample_method = flags["resample"]
            flags["resample"] = "nearest"
        output = self.inverse_inputs(input, params, flags, transform)
        if resample_method is not None:
            flags["resample"] = resample_method
        return output

    def inverse_boxes(
        self,
        input: Union[Tensor, Boxes],
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Boxes:
        padding_size = None
        if "padding_size" in params:
            # Mostly for operations like RandomCrop.
            padding_size = params["padding_size"]
        return input.unpad(padding_size)

    def inverse_keypoints(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        raise NotImplementedError

    def inverse_classes(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return input

    def inverse(
        self,
        input: Tensor,
        params: Optional[Dict[str, Tensor]] = None,
        **kwargs,
    ) -> Tensor:
        """Perform inverse operations.

        Args:
            input: the input tensor.
            params: the corresponding parameters for an operation.
                If None, a new parameter suite will be generated.
            **kwargs: key-value pairs to override the parameters and flags.
        """
        input_shape = input.shape
        in_tensor = self.transform_tensor(input)
        batch_shape = input.shape

        if len(kwargs.keys()) != 0:
            _src_params = self._params if params is None else params
            params = override_parameters(_src_params, kwargs, in_place=False)
            flags = override_parameters(self.flags, kwargs, in_place=False)
        else:
            flags = self.flags

        if params is None:
            params = self._params
        transform = self.get_transformation_matrix(in_tensor, params=params, flags=flags)

        if 'batch_prob' not in params:
            params['batch_prob'] = as_tensor([True] * batch_shape[0])
            warnings.warn("`batch_prob` is not found in params. Will assume applying on all data.")

        output = self.inverse_inputs(in_tensor, params, flags, transform)

        if self.keepdim:
            return self.transform_output_tensor(output, input_shape)

        return output
