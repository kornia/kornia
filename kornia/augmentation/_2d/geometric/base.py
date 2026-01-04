# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Any, Dict, Optional, Tuple

import torch

from kornia.augmentation._2d.base import RigidAffineAugmentationBase2D
from kornia.constants import Resample
from kornia.core.utils import _torch_inverse_cast
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints


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
        input: torch.Tensor,
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """By default, the exact transformation as ``apply_transform`` will be used."""
        raise NotImplementedError

    def compute_inverse_transformation(self, transform: torch.Tensor) -> torch.Tensor:
        """Compute the inverse transform of given transformation matrices."""
        return _torch_inverse_cast(transform)

    def get_transformation_matrix(
        self,
        input: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]] = None,
        flags: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
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
        return torch.as_tensor(transform, device=input.device, dtype=input.dtype)

    def apply_non_transform_mask(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process masks corresponding to the inputs that are no transformation applied."""
        return input

    def apply_transform_mask(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        self,
        input: Boxes,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> Boxes:
        """Process boxes corresponding to the inputs that are no transformation applied."""
        return input

    def apply_transform_box(
        self,
        input: Boxes,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> Boxes:
        """Process boxes corresponding to the inputs that are transformed."""
        if transform is None:
            if self.transform_matrix is None:
                raise RuntimeError("No valid transformation matrix found. Please either pass one or forward one first.")
            transform = self.transform_matrix
        input = self.apply_non_transform_box(input, params, flags, transform)
        return input.transform_boxes_(transform)

    def apply_non_transform_keypoint(
        self,
        input: Keypoints,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> Keypoints:
        """Process keypoints corresponding to the inputs that are no transformation applied."""
        return input

    def apply_transform_keypoint(
        self,
        input: Keypoints,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> Keypoints:
        """Process keypoints corresponding to the inputs that are transformed."""
        if transform is None:
            if self.transform_matrix is None:
                raise RuntimeError("No valid transformation matrix found. Please either pass one or forward one first.")
            transform = self.transform_matrix
        input = self.apply_non_transform_keypoint(input, params, flags, transform)
        return input.transform_keypoints_(transform)

    def apply_non_transform_class(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process class tags corresponding to the inputs that are no transformation applied."""
        return input

    def apply_transform_class(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process class tags corresponding to the inputs that are transformed."""
        return input

    def inverse_inputs(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
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
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        resample_method: Optional[Resample] = None
        align_corners_value: Optional[bool] = None
        if "resample" in flags:
            resample_method = flags["resample"]
            flags["resample"] = Resample.get("nearest")
        # Preserve align_corners from extra_args (kwargs) if provided
        # This ensures masks use the same align_corners setting in inverse as in forward
        if "align_corners" in kwargs:
            align_corners_value = flags.get("align_corners")
            flags["align_corners"] = kwargs["align_corners"]
        output = self.inverse_inputs(input, params, flags, transform, **kwargs)
        if resample_method is not None:
            flags["resample"] = resample_method
        # Restore align_corners if it was modified (mirror the modification condition)
        # This ensures complete state restoration even if the original value was None
        if "align_corners" in kwargs:
            flags["align_corners"] = align_corners_value
        return output

    def inverse_boxes(
        self,
        input: Boxes,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Boxes:
        output = input.clone()
        batch_prob = params["batch_prob"]
        to_apply = batch_prob > 0.5  # NOTE: in case of Relaxed Distributions.

        if transform is None:
            raise RuntimeError("`transform` has to be a torch.tensor. Got None.")

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
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Keypoints:
        """Inverse the transformation on keypoints.

        Args:
            input: input keypoints torch.tensor or object.
            params: the corresponding parameters for an operation.
            flags: static parameters.
            transform: the inverse transformation matrix
            kwargs: additional arguments

        """
        output = input.clone()
        batch_prob = params["batch_prob"]
        to_apply = batch_prob > 0.5  # NOTE: in case of Relaxed Distributions.

        if transform is None:
            raise RuntimeError("`transform` has to be a torch.tensor. Got None.")

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
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return input

    def inverse(
        self, input: torch.Tensor, params: Optional[Dict[str, torch.Tensor]] = None, **kwargs: Any
    ) -> torch.Tensor:
        """Perform inverse operations.

        Args:
            input: the input torch.tensor.
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
