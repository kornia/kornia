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

from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.distributions import Bernoulli, Distribution

from kornia.augmentation.utils import (
    _adapted_sampling,
    _transform_output_shape,
    override_parameters,
)
from kornia.core.utils import is_autocast_enabled
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints

TensorWithTransformMat = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def _apply_transform_unimplemented(
    self: nn.Module,
    input: torch.Tensor,
    params: Dict[str, torch.Tensor],
    flags: Dict[str, Any],
    transform: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Define the computation performed at every call.

    Should be overridden by all subclasses.
    """
    raise NotImplementedError(f'nn.Module [{type(self).__name__}] is missing the required "apply_transform" function')


class AugmentationBase(nn.Module):
    r"""AugmentationBase base class for customized augmentation implementations.

    This is the unified base class that combines core augmentation logic with
    multi-datatype transform support (images, masks, boxes, keypoints).

    By default, the random computations will happen on CPU with ``torch.get_default_dtype()``.
    To change this behaviour, please use ``set_rng_device_and_dtype``.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation
          probabilities element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls
          the augmentation probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
          to the batch form ``False``.

    """

    ONNX_EXPORTABLE = False

    def __init__(
        self,
        p: float = 0.5,
        p_batch: float = 1.0,
        same_on_batch: bool = False,
        keepdim: bool = False,
    ) -> None:
        super().__init__()
        self.p = p
        self.p_batch = p_batch
        self.same_on_batch = same_on_batch
        self.keepdim = keepdim
        self._params: Dict[str, torch.Tensor] = {}
        self._p_gen: Distribution
        self._p_batch_gen: Distribution
        if p != 0.0 or p != 1.0:
            self._p_gen = Bernoulli(self.p)
        if p_batch != 0.0 or p_batch != 1.0:
            self._p_batch_gen = Bernoulli(self.p_batch)
        self.flags: Dict[str, Any] = {}
        self.set_rng_device_and_dtype(torch.device("cpu"), torch.get_default_dtype())

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the transformation to the input."""
        raise NotImplementedError(
            f'nn.Module [{type(self).__name__}] is missing the required "apply_transform" function'
        )

    def to(self, *args: Any, **kwargs: Any) -> "AugmentationBase":
        r"""Set the device and dtype for the random number generator."""
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.set_rng_device_and_dtype(device, dtype)
        return super().to(*args, **kwargs)

    def __repr__(self) -> str:
        txt = f"p={self.p}, p_batch={self.p_batch}, same_on_batch={self.same_on_batch}"
        for k, v in self.flags.items():
            if isinstance(v, Enum):
                txt += f", {k}={v.name.lower()}"
            else:
                txt += f", {k}={v}"
        return f"{self.__class__.__name__}({txt})"

    def __unpack_input__(self, input: torch.Tensor) -> torch.Tensor:
        return input

    def transform_tensor(
        self,
        input: torch.Tensor,
        *,
        shape: Optional[torch.Tensor] = None,
        match_channel: bool = True,
    ) -> torch.Tensor:
        """Standardize input tensors."""
        raise NotImplementedError

    def validate_tensor(self, input: torch.Tensor) -> None:
        """Check if the input torch.tensor is formatted as expected."""
        raise NotImplementedError

    def transform_output_tensor(self, output: torch.Tensor, output_shape: Tuple[int, ...]) -> torch.Tensor:
        """Standardize output tensors."""
        return _transform_output_shape(output, output_shape) if self.keepdim else output

    def generate_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        """Generate random parameters for the augmentation. Override in subclasses."""
        return {}

    def set_rng_device_and_dtype(self, device: torch.device, dtype: torch.dtype) -> None:
        """Change the random generation device and dtype."""
        self.device = device
        self.dtype = dtype

    def __batch_prob_generator__(
        self,
        batch_shape: Tuple[int, ...],
        p: float,
        p_batch: float,
        same_on_batch: bool,
    ) -> torch.Tensor:
        batch_prob: torch.Tensor
        if p_batch == 1:
            batch_prob = torch.zeros(1) + 1
        elif p_batch == 0:
            batch_prob = torch.zeros(1)
        else:
            batch_prob = _adapted_sampling((1,), self._p_batch_gen, same_on_batch)

        if batch_prob.sum() == 1:
            elem_prob: torch.Tensor
            if p == 1:
                elem_prob = torch.zeros(batch_shape[0]) + 1
            elif p == 0:
                elem_prob = torch.zeros(batch_shape[0])
            else:
                elem_prob = _adapted_sampling((batch_shape[0],), self._p_gen, same_on_batch)
            batch_prob = batch_prob * elem_prob
        else:
            batch_prob = batch_prob.repeat(batch_shape[0])
        if len(batch_prob.shape) == 2:
            return batch_prob[..., 0]
        return batch_prob

    def _process_kwargs_to_params_and_flags(
        self,
        params: Optional[Dict[str, torch.Tensor]] = None,
        flags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        save_kwargs = kwargs.get("save_kwargs", False)

        params = self._params if params is None else params
        flags = self.flags if flags is None else flags

        if save_kwargs:
            params = override_parameters(params, kwargs, in_place=True)
            self._params = params
        else:
            self._params = params
            params = override_parameters(params, kwargs, in_place=False)

        flags = override_parameters(flags, kwargs, in_place=False)
        return params, flags

    def forward_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        batch_prob = self.__batch_prob_generator__(batch_shape, self.p, self.p_batch, self.same_on_batch)
        to_apply = batch_prob > 0.5
        _params = self.generate_parameters(torch.Size((int(to_apply.sum().item()), *batch_shape[1:])))
        if _params is None:
            _params = {}
        _params["batch_prob"] = batch_prob
        input_size = torch.tensor(batch_shape, dtype=torch.long)
        _params.update({"forward_input_shape": input_size})
        return _params

    # --- Multi-datatype transform methods ---

    def apply_non_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply to images that are skipped from transformation (batch_prob == False)."""
        return input

    def transform_inputs(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        params, flags = self._process_kwargs_to_params_and_flags(
            self._params if params is None else params, flags, **kwargs
        )

        batch_prob = params["batch_prob"]
        to_apply = batch_prob > 0.5
        ori_shape = input.shape
        in_tensor = self.transform_tensor(input)

        self.validate_tensor(in_tensor)
        if to_apply.all():
            output = self.apply_transform(in_tensor, params, flags, transform=transform)
        elif not to_apply.any():
            output = self.apply_non_transform(in_tensor, params, flags, transform=transform)
        else:
            output = self.apply_non_transform(in_tensor, params, flags, transform=transform)
            applied = self.apply_transform(
                in_tensor[to_apply],
                params,
                flags,
                transform=transform if transform is None else transform[to_apply],
            )

            if is_autocast_enabled():
                output = output.type(input.dtype)
                applied = applied.type(input.dtype)
            output = output.index_put((to_apply,), applied)

        output = _transform_output_shape(output, ori_shape) if self.keepdim else output

        if is_autocast_enabled():
            output = output.type(input.dtype)
        return output

    def transform_masks(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        params, flags = self._process_kwargs_to_params_and_flags(
            self._params if params is None else params, flags, **kwargs
        )

        batch_prob = params["batch_prob"]
        to_apply = batch_prob > 0.5
        ori_shape = input.shape

        shape = params["forward_input_shape"]
        in_tensor = self.transform_tensor(input, shape=shape, match_channel=False)

        self.validate_tensor(in_tensor)
        if to_apply.all():
            output = self.apply_transform_mask(in_tensor, params, flags, transform=transform)
        elif not to_apply.any():
            output = self.apply_non_transform_mask(in_tensor, params, flags, transform=transform)
        else:
            output = self.apply_non_transform_mask(in_tensor, params, flags, transform=transform)
            applied = self.apply_transform_mask(
                in_tensor[to_apply],
                params,
                flags,
                transform=transform if transform is None else transform[to_apply],
            )
            output = output.index_put((to_apply,), applied)
        output = _transform_output_shape(output, ori_shape, reference_shape=shape) if self.keepdim else output
        return output

    def transform_boxes(
        self,
        input: Boxes,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Boxes:
        if not isinstance(input, Boxes):
            raise RuntimeError(f"Only `Boxes` is supported. Got {type(input)}.")

        params, flags = self._process_kwargs_to_params_and_flags(
            self._params if params is None else params, flags, **kwargs
        )

        batch_prob = params["batch_prob"]
        to_apply = batch_prob > 0.5
        output: Boxes
        if to_apply.bool().all():
            output = self.apply_transform_box(input, params, flags, transform=transform)
        elif not to_apply.any():
            output = self.apply_non_transform_box(input, params, flags, transform=transform)
        else:
            output = self.apply_non_transform_box(input, params, flags, transform=transform)
            applied = self.apply_transform_box(
                input[to_apply],
                params,
                flags,
                transform=transform if transform is None else transform[to_apply],
            )
            if is_autocast_enabled():
                output = output.type(input.dtype)
                applied = applied.type(input.dtype)

            output = output.index_put((to_apply,), applied)
        return output

    def transform_keypoints(
        self,
        input: Keypoints,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Keypoints:
        if not isinstance(input, Keypoints):
            raise RuntimeError(f"Only `Keypoints` is supported. Got {type(input)}.")

        params, flags = self._process_kwargs_to_params_and_flags(
            self._params if params is None else params, flags, **kwargs
        )

        batch_prob = params["batch_prob"]
        to_apply = batch_prob > 0.5
        if to_apply.all():
            output = self.apply_transform_keypoint(input, params, flags, transform=transform)
        elif not to_apply.any():
            output = self.apply_non_transform_keypoint(input, params, flags, transform=transform)
        else:
            output = self.apply_non_transform_keypoint(input, params, flags, transform=transform)
            applied = self.apply_transform_keypoint(
                input[to_apply],
                params,
                flags,
                transform=transform if transform is None else transform[to_apply],
            )
            if is_autocast_enabled():
                output = output.type(input.dtype)
                applied = applied.type(input.dtype)
            output = output.index_put((to_apply,), applied)
        return output

    def transform_classes(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        params, flags = self._process_kwargs_to_params_and_flags(
            self._params if params is None else params, flags, **kwargs
        )

        batch_prob = params["batch_prob"]
        to_apply = batch_prob > 0.5
        if to_apply.all():
            output = self.apply_transform_class(input, params, flags, transform=transform)
        elif not to_apply.any():
            output = self.apply_non_transform_class(input, params, flags, transform=transform)
        else:
            output = self.apply_non_transform_class(input, params, flags, transform=transform)
            applied = self.apply_transform_class(
                input[to_apply],
                params,
                flags,
                transform=transform if transform is None else transform[to_apply],
            )
            output = output.index_put((to_apply,), applied)
        return output

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
        """Process masks corresponding to the inputs that are transformed."""
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def apply_func(
        self,
        in_tensor: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        if flags is None:
            flags = self.flags

        output = self.transform_inputs(in_tensor, params, flags)

        return output

    def forward(
        self, input: torch.Tensor, params: Optional[Dict[str, torch.Tensor]] = None, **kwargs: Any
    ) -> torch.Tensor:
        """Perform forward operations.

        Args:
            input: the input torch.tensor.
            params: the corresponding parameters for an operation.
                If None, a new parameter suite will be generated.
            **kwargs: key-value pairs to override the parameters and flags.

        """
        in_tensor = self.__unpack_input__(input)
        input_shape = in_tensor.shape
        in_tensor = self.transform_tensor(in_tensor)
        batch_shape = in_tensor.shape
        if params is None:
            params = self.forward_parameters(batch_shape)

        if "batch_prob" not in params:
            params["batch_prob"] = torch.tensor([True] * batch_shape[0])

        params, flags = self._process_kwargs_to_params_and_flags(params, self.flags, **kwargs)

        output = self.apply_func(in_tensor, params, flags)
        return self.transform_output_tensor(output, input_shape) if self.keepdim else output


# Backward compatibility aliases
_BasicAugmentationBase = AugmentationBase
_AugmentationBase = AugmentationBase
