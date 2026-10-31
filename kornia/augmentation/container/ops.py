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

from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, Generic, List, Optional, ParamSpec, Type, TypeVar, Union

import torch
from torch import nn

import kornia.augmentation as K
from kornia.augmentation.base import _AugmentationBase
from kornia.augmentation.utils.helpers import deepcopy_dict, override_parameters
from kornia.constants import DataKey
from kornia.core.utils import is_exporting
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints

from .params import ParamItem

DataType = Union[torch.Tensor, List[torch.Tensor], Boxes, Keypoints]

# NOTE: shouldn't this SequenceDataType alias be equals to List[DataType]?
SequenceDataType = Union[List[torch.Tensor], List[List[torch.Tensor]], List[Boxes], List[Keypoints]]

T = TypeVar("T")


class SequentialOpsInterface(Generic[T], metaclass=ABCMeta):
    """Abstract interface for applying and inversing transformations."""

    @classmethod
    def get_instance_module_param(cls, param: ParamItem) -> Dict[str, torch.Tensor]:
        """Extract per-module parameter dict from a :class:`ParamItem`.

        Args:
            param: Parameter wrapper produced by sequential containers.

        Returns:
            Dictionary of tensor parameters for one module call.

        Raises:
            TypeError: ``param.data`` is not a dictionary.
        """
        if isinstance(param, ParamItem) and isinstance(param.data, dict):
            _params = param.data
        else:
            raise TypeError(f"Expected param (ParamItem.data) be a dictionary. Got {param}.")
        return _params

    @classmethod
    def get_transform_matrix(
        cls, module: nn.Module, param: ParamItem, input: Any, extra_args: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Get the matrix recorded by these params, not by the module's most recent image call."""
        params = cls.get_instance_module_param(param)
        # The image call records the params dict it was given, and its values and their versions, next to the matrix
        # it computed. The matrix is reused while the dict holds the same values unedited: the annotations of the same
        # draw share it, and a replay after a caller replaced or edited a value is recomputed from the values.
        matches = getattr(module, "_matrix_matches_params", None)
        if not is_exporting() and matches is not None and matches(params):
            transform = getattr(module, "transform_matrix", None)
            if transform is not None:
                return transform

        forward_input_shape = params.get("forward_input_shape")
        if forward_input_shape is None:
            raise ValueError("`forward_input_shape` is required to replay geometric annotations without an image.")

        if isinstance(input, list):
            input = input[0] if input else None
        if isinstance(input, (Boxes, Keypoints)):
            input = input.data
        reference = input if isinstance(input, torch.Tensor) else None
        device = reference.device if reference is not None else forward_input_shape.device
        dtype = reference.dtype if reference is not None and torch.is_floating_point(reference) else torch.float32
        input_shape = tuple(forward_input_shape.tolist())
        padding_size = params.get("padding_size")
        if (
            not is_exporting()
            and isinstance(padding_size, torch.Tensor)
            and padding_size.shape[0] > 0
            and len(input_shape) >= 2
        ):
            left, right, top, bottom = padding_size[0].tolist()
            input_shape = (*input_shape[:-2], input_shape[-2] - top - bottom, input_shape[-1] - left - right)
        matrix_input = torch.empty((), device=device, dtype=dtype).expand(input_shape)
        # Compute from a copy of the params, as the image call does (its ``data_keys`` override copies the dict). Under
        # export the caller's tensors can be graph constants, and a child that copies one as-is into its matrix, as
        # ``RandomAffine`` does with ``center``, makes torch 2.9's dynamo ONNX exporter rename that constant and fail.
        flags = module.flags if extra_args is None else override_parameters(module.flags, extra_args, in_place=False)
        return module.generate_transformation_matrix(matrix_input, deepcopy_dict(params), flags)

    @classmethod
    def get_sequential_module_param(cls, param: ParamItem) -> List[ParamItem]:
        """Extract nested sequential parameters from a :class:`ParamItem`.

        Args:
            param: Parameter wrapper produced by sequential containers.

        Returns:
            List of :class:`ParamItem` values for nested modules.

        Raises:
            TypeError: ``param.data`` is not a list.
        """
        if isinstance(param, ParamItem) and isinstance(param.data, list):
            _params = param.data
        else:
            raise TypeError(f"Expected param (ParamItem.data) be a list. Got {param}.")
        return _params

    @classmethod
    @abstractmethod
    def transform(cls, input: T, module: nn.Module, param: ParamItem, extra_args: Optional[Dict[str, Any]] = None) -> T:
        """Apply a transformation with respect to the parameters.

        Args:
            input: the input torch.Tensor.
            module: any torch nn.Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
            extra_args: Optional dictionary of extra arguments with specific options for different input types.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def inverse(cls, input: T, module: nn.Module, param: ParamItem, extra_args: Optional[Dict[str, Any]] = None) -> T:
        """Inverse a transformation with respect to the parameters.

        Args:
            input: the input torch.Tensor.
            module: any torch nn.Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
            extra_args: Optional dictionary of extra arguments with specific options for different input types.
        """
        raise NotImplementedError


class AugmentationSequentialOps:
    """Implement the operational logic for the Augmentation Sequential container.

    This class manages how data keys (e.g., IMAGE, MASK, BBOX) are handled
    during the execution of an augmentation pipeline.

    Args:
        data_keys: A list of :class:`DataKey` defining the types of data to process.
    """

    def __init__(self, data_keys: Optional[List[DataKey]]) -> None:
        self._data_keys = data_keys

    @property
    def data_keys(self) -> Optional[List[DataKey]]:
        """Return currently configured data keys."""
        return self._data_keys

    @data_keys.setter
    def data_keys(self, data_keys: Optional[Union[List[DataKey], List[str], List[int]]]) -> None:
        if data_keys:
            self._data_keys = [DataKey.get(inp) for inp in data_keys]
        else:
            self._data_keys = None

    def preproc_datakeys(self, data_keys: Optional[Union[List[str], List[int], List[DataKey]]] = None) -> List[DataKey]:
        """Normalize user-provided data keys into :class:`DataKey` values.

        Args:
            data_keys: Optional keys passed by the caller. If omitted, this method
                uses ``self.data_keys``.

        Returns:
            Normalized list of data keys.

        Raises:
            ValueError: Neither argument nor instance-level keys are available.
        """
        if data_keys is None:
            if isinstance(self.data_keys, list):
                return self.data_keys
            raise ValueError("nn.Sequential ops needs data keys to be able to process.")
        return [DataKey.get(inp) for inp in data_keys]

    def _get_op(self, data_key: DataKey) -> Type[SequentialOpsInterface[Any]]:
        """Return the corresponding operation given a data key."""
        if data_key == DataKey.INPUT:
            return InputSequentialOps
        if data_key == DataKey.MASK:
            return MaskSequentialOps
        if data_key in {DataKey.BBOX, DataKey.BBOX_XYWH, DataKey.BBOX_XYXY}:
            return BoxSequentialOps
        if data_key == DataKey.KEYPOINTS:
            return KeypointSequentialOps
        if data_key == DataKey.CLASS:
            return ClassSequentialOps
        raise RuntimeError(f"Operation for `{data_key.name}` is not found.")

    def transform(
        self,
        *arg: DataType,
        module: nn.Module,
        param: ParamItem,
        extra_args: Dict[DataKey, Dict[str, Any]],
        data_keys: Optional[Union[List[str], List[int], List[DataKey]]] = None,
    ) -> Union[DataType, SequenceDataType]:
        """Apply one module to all inputs according to their data keys.

        Args:
            *arg: Inputs to transform (image, mask, boxes, keypoints, ...).
            module: Module to execute.
            param: Parameters associated with ``module``.
            extra_args: Optional runtime overrides keyed by :class:`DataKey`.
            data_keys: Optional key override for this call.

        Returns:
            Transformed output(s). A single value is returned for one input,
            otherwise a list is returned.
        """
        _data_keys = self.preproc_datakeys(data_keys)

        if isinstance(module, K.RandomTransplantation):
            # This child reads the mask while drawing params, before the list-mask handler can reject it.
            for inp, dcate in zip(arg, _data_keys):
                if dcate == DataKey.MASK and isinstance(inp, list):
                    raise NotImplementedError(f"{type(module).__name__} does not support mask lists.")
            # For transforms which require the full input to calculate the parameters (e.g. RandomTransplantation)
            param = ParamItem(
                name=param.name,
                data=module.params_from_input(
                    *arg,
                    data_keys=_data_keys,
                    params=param.data,
                    extra_args=extra_args,
                ),
            )

        # Rigid annotation handlers read the matrix produced by the image call. Process that input first even
        # when the caller supplies data keys in another order, then restore the caller's output order.
        order = list(range(len(arg)))
        if (
            isinstance(module, K.RigidAffineAugmentationBase2D)
            and not isinstance(module, K.IntensityAugmentationBase2D)
            and DataKey.INPUT in _data_keys
        ):
            image_index = _data_keys.index(DataKey.INPUT)
            order.insert(0, order.pop(image_index))

        outputs = list(arg)
        for index in order:
            inp, dcate = arg[index], _data_keys[index]
            op = self._get_op(dcate)
            extra_arg = extra_args.get(dcate, {})
            if dcate.name == "MASK" and isinstance(inp, list):
                outputs[index] = MaskSequentialOps.transform_list(inp, module, param=param, extra_args=extra_arg)
            else:
                outputs[index] = op.transform(inp, module, param=param, extra_args=extra_arg)
        if len(outputs) == 1 and isinstance(outputs, (list, tuple)):
            return outputs[0]
        return outputs

    def inverse(
        self,
        *arg: DataType,
        module: nn.Module,
        param: ParamItem,
        extra_args: Dict[DataKey, Dict[str, Any]],
        data_keys: Optional[Union[List[str], List[int], List[DataKey]]] = None,
    ) -> Union[DataType, SequenceDataType]:
        """Apply inverse transformation dispatch for one module step.

        Args:
            *arg: Inputs to invert.
            module: Module used in the forward pass.
            param: Parameters captured for ``module`` during forward.
            extra_args: Optional runtime overrides keyed by :class:`DataKey`.
            data_keys: Optional key override for this call.

        Returns:
            Inverse-transformed output(s). A single value is returned for one
            input, otherwise a list is returned.
        """
        _data_keys = self.preproc_datakeys(data_keys)
        outputs = []
        for inp, dcate in zip(arg, _data_keys):
            op = self._get_op(dcate)
            extra_arg = extra_args.get(dcate, {})
            if dcate.name == "MASK" and isinstance(inp, list):
                # Mirror ``transform``: a list of masks is inverted element by element.
                outputs.append(MaskSequentialOps.inverse_list(inp, module, param=param, extra_args=extra_arg))
            else:
                outputs.append(op.inverse(inp, module, param=param, extra_args=extra_arg))
        if len(outputs) == 1 and isinstance(outputs, (list, tuple)):
            return outputs[0]
        return outputs


P = ParamSpec("P")


def make_input_only_sequential(module: "K.container.ImageSequentialBase") -> Callable[P, torch.Tensor]:
    """Disable all other additional inputs (e.g. ) for ImageSequential."""

    def f(*args: P.args, **kwargs: P.kwargs) -> torch.Tensor:
        return module(*args, **kwargs)

    return f


def get_geometric_only_param(module: "K.container.ImageSequentialBase", param: List[ParamItem]) -> List[ParamItem]:
    """Return geometry param."""
    named_modules = module.get_forward_sequence(param)

    res: List[ParamItem] = []
    for (_, mod), p in zip(named_modules, param):
        if isinstance(mod, (K.GeometricAugmentationBase2D, K.GeometricAugmentationBase3D)):
            res.append(p)
    return res


class InputSequentialOps(SequentialOpsInterface[torch.Tensor]):
    """Implement the operations for processing input tensors within a sequential container.

    This class provides class methods to apply transformations and manage the
    flow of data through the augmentation pipeline.
    """

    @classmethod
    def transform(
        cls, input: torch.Tensor, module: nn.Module, param: ParamItem, extra_args: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Apply one module step to an image tensor.

        Args:
            input: Input image tensor.
            module: Module to execute.
            param: Parameters for ``module``.
            extra_args: Optional runtime overrides.

        Returns:
            Transformed tensor.

        Raises:
            AssertionError: A non-augmentation module receives non-empty params.
        """
        if extra_args is None:
            extra_args = {}
        if isinstance(module, (_AugmentationBase, K.MixAugmentationBaseV2)):
            input = module(input, params=cls.get_instance_module_param(param), data_keys=[DataKey.INPUT], **extra_args)
        elif isinstance(module, (K.container.ImageSequentialBase,)):
            input = module.transform_inputs(input, params=cls.get_sequential_module_param(param), extra_args=extra_args)
        elif isinstance(module, (K.auto.operations.OperationBase,)):
            input = module(input, params=cls.get_instance_module_param(param))
        else:
            if param.data is not None:
                raise AssertionError(f"Non-augmentaion operation {param.name} require empty parameters. Got {param}.")
            input = module(input)
        return input

    @classmethod
    def inverse(
        cls, input: torch.Tensor, module: nn.Module, param: ParamItem, extra_args: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Apply one inverse module step to an image tensor.

        Args:
            input: Tensor to invert.
            module: Module used in the forward path.
            param: Forward parameters for ``module``.
            extra_args: Optional runtime overrides.

        Returns:
            Inverse-transformed tensor.

        Raises:
            NotImplementedError: Inverse for 3D geometric ops is not supported.
        """
        if extra_args is None:
            extra_args = {}
        if isinstance(module, K.GeometricAugmentationBase2D):
            input = module.inverse(input, params=cls.get_instance_module_param(param), **extra_args)
        elif isinstance(module, K.MixAugmentationBaseV2):
            input = module.inverse(**extra_args)
        elif isinstance(module, (K.GeometricAugmentationBase3D,)):
            raise NotImplementedError(
                "The support for 3d inverse operations are not yet supported. You are welcome to file a PR in our repo."
            )
        elif isinstance(module, (K.auto.operations.OperationBase,)):
            return InputSequentialOps.inverse(input, module=module.op, param=param, extra_args=extra_args)
        elif isinstance(module, K.container.ImageSequentialBase):
            input = module.inverse_inputs(input, params=cls.get_sequential_module_param(param), extra_args=extra_args)
        return input


class ClassSequentialOps(SequentialOpsInterface[torch.Tensor]):
    """Apply and inverse transformations for class labels if needed."""

    @classmethod
    def transform(
        cls, input: torch.Tensor, module: nn.Module, param: ParamItem, extra_args: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Apply class-label handling for one module step.

        Args:
            input: Class-label tensor.
            module: Module to execute.
            param: Parameters for ``module``.
            extra_args: Optional runtime overrides.

        Returns:
            Class labels after transformation handling.

        Raises:
            NotImplementedError: Label-changing mix ops are not supported yet.
        """
        if isinstance(module, K.MixAugmentationBaseV2):
            raise NotImplementedError(
                "The support for class labels for mix augmentations that change the class label is not yet supported."
            )
        return input

    @classmethod
    def inverse(
        cls, input: torch.Tensor, module: nn.Module, param: ParamItem, extra_args: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Return class labels unchanged during inverse dispatch.

        Note:
            Class labels do not have a geometric inverse in this pipeline.

        Args:
            input: Class-label tensor.
            module: Module used in forward (unused).
            param: Forward parameters (unused).
            extra_args: Optional runtime overrides (unused).

        Returns:
            Unmodified class labels.
        """
        return input


class MaskSequentialOps(SequentialOpsInterface[torch.Tensor]):
    """Apply and inverse transformations for mask tensors."""

    @staticmethod
    def _mask_extra_args(module: Any, extra_args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        extra_args = {} if extra_args is None else extra_args
        # A per-channel image fill has no unambiguous meaning for a semantic mask, whose channel count
        # commonly differs from the image. Default mask padding to background while preserving an explicit
        # per-mask override and the legacy behavior of scalar fills.
        fill = module.flags.get("fill")
        if fill is not None and not isinstance(fill, (int, float)):
            extra_args = dict(extra_args)
            extra_args.setdefault("fill", 0.0)
        # The warp-based augmentations (RandomAffine) store their fill as the ``fill_value`` tensor, a scalar as a
        # 0-d tensor. Apply the same rule: a per-channel fill pads the mask with background, one element applies to
        # both (#6021).
        fill_value = module.flags.get("fill_value")
        if isinstance(fill_value, torch.Tensor) and fill_value.numel() > 1:
            extra_args = dict(extra_args)
            extra_args.setdefault("fill_value", torch.zeros((), device=fill_value.device, dtype=fill_value.dtype))
        return extra_args

    @classmethod
    def transform(
        cls, input: torch.Tensor, module: nn.Module, param: ParamItem, extra_args: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Apply a transformation with respect to the parameters.

        Args:
            input: the input torch.Tensor.
            module: any torch nn.Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
            extra_args: Optional dictionary of extra arguments with specific options for different input types.
        """
        if extra_args is None:
            extra_args = {}

        if isinstance(module, K.RigidAffineAugmentationBase2D) and not isinstance(
            module, K.IntensityAugmentationBase2D
        ):
            extra_args = cls._mask_extra_args(module, extra_args)
            input = module.transform_masks(
                input,
                params=cls.get_instance_module_param(param),
                flags=module.flags,
                transform=cls.get_transform_matrix(module, param, input),
                **extra_args,
            )

        elif isinstance(module, (K.GeometricAugmentationBase3D,)):
            raise NotImplementedError(
                "The support for 3d mask operations are not yet supported. You are welcome to file a PR in our repo."
            )

        elif isinstance(module, K.RandomTransplantation):
            output = module(input, params=cls.get_instance_module_param(param), data_keys=[DataKey.MASK], **extra_args)
            # A (B, H, W) mask leaves the transplant as (B, 1, H, W), like any other child without keepdim.
            input = output.unsqueeze(1) if output.ndim == 3 and not module.keepdim else output

        elif isinstance(module, K.MixAugmentationBaseV2):
            # Dispatch to the mix child's own mask handler. Unsupported children
            # raise NotImplementedError from apply_transform_mask (matching a
            # direct call) instead of silently returning the mask unchanged.
            input = module.transform_mask(input, cls.get_instance_module_param(param), module.flags)

        elif isinstance(module, (_AugmentationBase)):
            input = module.transform_masks(
                input, params=cls.get_instance_module_param(param), flags=module.flags, **extra_args
            )

        elif isinstance(module, K.container.ImageSequentialBase):
            input = module.transform_masks(input, params=cls.get_sequential_module_param(param), extra_args=extra_args)

        elif isinstance(module, (K.auto.operations.OperationBase,)):
            input = MaskSequentialOps.transform(input, module=module.op, param=param, extra_args=extra_args)

        return input

    @staticmethod
    def _sample_params(params: Dict[str, torch.Tensor], index: int) -> Dict[str, torch.Tensor]:
        """Select one sample's gate and tensor parameters without mutating the saved draw."""
        batch = params["batch_prob"].numel()
        selected = {}
        for name, value in params.items():
            if name == "forward_input_shape":
                value = value.clone()
                value[0] = 1
            elif value.ndim > 0 and value.shape[0] == batch:
                value = value[index : index + 1]
            selected[name] = value
        return selected

    @classmethod
    def transform_list(
        cls, input: List[torch.Tensor], module: nn.Module, param: ParamItem, extra_args: Optional[Dict[str, Any]] = None
    ) -> List[torch.Tensor]:
        """Transform one mask per image sample, allowing different channel counts.

        Args:
            input: Per-sample mask tensors in image order.
            module: Augmentation or nested image sequence to apply.
            param: The full image batch's saved parameters.
            extra_args: Mask interpolation and other runtime overrides.

        Returns:
            One transformed mask per sample, in the same order.
        """
        if extra_args is None:
            extra_args = {}
        if isinstance(module, K.GeometricAugmentationBase3D):
            raise NotImplementedError("3D mask lists are not supported.")
        if isinstance(module, (K.VideoSequential, K.MixAugmentationBaseV2)):
            raise NotImplementedError(f"{type(module).__name__} does not support mask lists.")
        if isinstance(module, K.PatchSequential):
            # Its own handler: masks pass through an intensity-only pipeline, a geometric one raises.
            seq_params = cls.get_sequential_module_param(param)
            return [module.transform_masks(inp, params=seq_params, extra_args=extra_args) for inp in input]
        if isinstance(module, K.container.ImageSequentialBase):
            for child_param in cls.get_sequential_module_param(param):
                input = cls.transform_list(input, module.get_submodule(child_param.name), child_param, extra_args)
            return input
        if isinstance(module, K.auto.operations.OperationBase):
            return cls.transform_list(input, module.op, param, extra_args)
        if isinstance(module, _AugmentationBase):
            params = cls.get_instance_module_param(param)
            transform = None
            if isinstance(module, K.RigidAffineAugmentationBase2D) and not isinstance(
                module, K.IntensityAugmentationBase2D
            ):
                extra_args = cls._mask_extra_args(module, extra_args)
                transform = cls.get_transform_matrix(module, param, input)
            output = [
                module.transform_masks(
                    inp,
                    params=cls._sample_params(params, i),
                    flags=module.flags,
                    transform=transform[i : i + 1] if transform is not None else None,
                    **extra_args,
                )
                for i, inp in enumerate(input)
            ]
            # Each call recorded one sample's slice; keep the batch draw as the child's params, as a tensor mask does.
            module._commit_state(params=params)
            return output
        return input

    @classmethod
    def inverse(
        cls, input: torch.Tensor, module: nn.Module, param: ParamItem, extra_args: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Inverse a transformation with respect to the parameters.

        Args:
            input: the input torch.Tensor.
            module: any torch nn.Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
            extra_args: Optional dictionary of extra arguments with specific options for different input types.
        """
        if extra_args is None:
            extra_args = {}

        if isinstance(module, (K.GeometricAugmentationBase2D,)):
            extra_args = cls._mask_extra_args(module, extra_args)
            transform = module.compute_inverse_transformation(cls.get_transform_matrix(module, param, input))
            input = module.inverse_masks(
                input,
                params=cls.get_instance_module_param(param),
                flags=module.flags,
                transform=transform,
                **extra_args,
            )

        elif isinstance(module, (K.GeometricAugmentationBase3D,)):
            raise NotImplementedError(
                "The support for 3d mask operations are not yet supported. You are welcome to file a PR in our repo."
            )

        elif isinstance(module, K.container.ImageSequentialBase):
            input = module.inverse_masks(input, params=cls.get_sequential_module_param(param), extra_args=extra_args)

        elif isinstance(module, (K.auto.operations.OperationBase,)):
            input = MaskSequentialOps.inverse(input, module=module.op, param=param, extra_args=extra_args)

        return input

    @classmethod
    def inverse_list(
        cls, input: List[torch.Tensor], module: nn.Module, param: ParamItem, extra_args: Optional[Dict[str, Any]] = None
    ) -> List[torch.Tensor]:
        """Invert each sample's mask with its own forward parameters and matrix.

        Args:
            input: Per-sample masks returned by the forward pass.
            module: Augmentation or nested image sequence to invert.
            param: The full image batch's saved forward parameters.
            extra_args: Mask interpolation and other runtime overrides.

        Returns:
            One inverse-transformed mask per sample, in the same order.
        """
        if extra_args is None:
            extra_args = {}
        if isinstance(module, K.GeometricAugmentationBase2D):
            extra_args = cls._mask_extra_args(module, extra_args)
            transform = module.compute_inverse_transformation(cls.get_transform_matrix(module, param, input))
            params = cls.get_instance_module_param(param)
            output = [
                module.inverse_masks(
                    inp,
                    params=cls._sample_params(params, i),
                    flags=module.flags,
                    transform=transform[i : i + 1],
                    **extra_args,
                )
                for i, inp in enumerate(input)
            ]
            module._commit_state(params=params)
            return output
        if isinstance(module, K.GeometricAugmentationBase3D):
            raise NotImplementedError("3D mask lists are not supported.")
        if isinstance(module, (K.VideoSequential, K.MixAugmentationBaseV2)):
            raise NotImplementedError(f"{type(module).__name__} does not support mask lists.")
        if isinstance(module, K.PatchSequential):
            seq_params = cls.get_sequential_module_param(param)
            return [module.inverse_masks(inp, params=seq_params, extra_args=extra_args) for inp in input]
        if isinstance(module, K.container.ImageSequentialBase):
            for child_param in reversed(cls.get_sequential_module_param(param)):
                input = cls.inverse_list(input, module.get_submodule(child_param.name), child_param, extra_args)
            return input
        if isinstance(module, K.auto.operations.OperationBase):
            return cls.inverse_list(input, module.op, param, extra_args)
        return input


class BoxSequentialOps(SequentialOpsInterface[Boxes]):
    """Apply and inverse transformations for bounding box tensors.

    This is for transform boxes in the format (B, N, 4, 2).
    """

    @classmethod
    def transform(
        cls, input: Boxes, module: nn.Module, param: ParamItem, extra_args: Optional[Dict[str, Any]] = None
    ) -> Boxes:
        """Apply a transformation with respect to the parameters.

        Args:
            input: the input torch.Tensor, (B, N, 4, 2) or (B, 4, 2).
            module: any torch nn.Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
            extra_args: Optional dictionary of extra arguments with specific options for different input types.
        """
        if extra_args is None:
            extra_args = {}
        _input = input.clone()

        if isinstance(module, K.RigidAffineAugmentationBase2D) and not isinstance(
            module, K.IntensityAugmentationBase2D
        ):
            _input = module.transform_boxes(
                _input,
                cls.get_instance_module_param(param),
                module.flags,
                transform=cls.get_transform_matrix(module, param, _input),
                **extra_args,
            )

        elif isinstance(module, (K.GeometricAugmentationBase3D,)):
            raise NotImplementedError(
                "The support for 3d box operations are not yet supported. You are welcome to file a PR in our repo."
            )

        elif isinstance(module, K.MixAugmentationBaseV2):
            # Dispatch to the mix child's own box handler (e.g. RandomMosaic).
            # Unsupported children raise NotImplementedError from
            # apply_transform_boxes, matching a direct call.
            _input = module.transform_boxes(_input, cls.get_instance_module_param(param), module.flags)

        elif isinstance(module, K.IntensityAugmentationBase2D):
            _input = module.transform_boxes(_input, cls.get_instance_module_param(param), module.flags, **extra_args)

        elif isinstance(module, K.container.ImageSequentialBase):
            _input = module.transform_boxes(
                _input, params=cls.get_sequential_module_param(param), extra_args=extra_args
            )

        elif isinstance(module, (K.auto.operations.OperationBase,)):
            return BoxSequentialOps.transform(input, module=module.op, param=param, extra_args=extra_args)

        return _input

    @classmethod
    def inverse(
        cls, input: Boxes, module: nn.Module, param: ParamItem, extra_args: Optional[Dict[str, Any]] = None
    ) -> Boxes:
        """Inverse a transformation with respect to the parameters.

        Args:
            input: the input torch.Tensor.
            module: any torch nn.Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
            extra_args: Optional dictionary of extra arguments with specific options for different input types.
        """
        if extra_args is None:
            extra_args = {}
        _input = input.clone()

        if isinstance(module, (K.GeometricAugmentationBase2D,)):
            transform = module.compute_inverse_transformation(cls.get_transform_matrix(module, param, _input))
            _input = module.inverse_boxes(
                _input,
                param.data,
                module.flags,
                transform=transform,
                **extra_args,
            )

        elif isinstance(module, (K.GeometricAugmentationBase3D,)):
            raise NotImplementedError(
                "The support for 3d box operations are not yet supported. You are welcome to file a PR in our repo."
            )

        elif isinstance(module, K.container.ImageSequentialBase):
            _input = module.inverse_boxes(_input, params=cls.get_sequential_module_param(param), extra_args=extra_args)

        elif isinstance(module, (K.auto.operations.OperationBase,)):
            return BoxSequentialOps.inverse(input, module=module.op, param=param, extra_args=extra_args)
        return _input


class KeypointSequentialOps(SequentialOpsInterface[Keypoints]):
    """Apply and inverse transformations for keypoints tensors.

    This is for transform keypoints in the format (B, N, 2).
    """

    @classmethod
    def transform(
        cls, input: Keypoints, module: nn.Module, param: ParamItem, extra_args: Optional[Dict[str, Any]] = None
    ) -> Keypoints:
        """Apply a transformation with respect to the parameters.

        Args:
            input: the input torch.Tensor, (B, N, 4, 2) or (B, 4, 2).
            module: any torch nn.Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
            extra_args: Optional dictionary of extra arguments with specific options for different input types.
        """
        if extra_args is None:
            extra_args = {}
        _input = input.clone()

        if isinstance(module, K.RigidAffineAugmentationBase2D) and not isinstance(
            module, K.IntensityAugmentationBase2D
        ):
            _input = module.transform_keypoints(
                _input,
                cls.get_instance_module_param(param),
                module.flags,
                transform=cls.get_transform_matrix(module, param, _input),
                **extra_args,
            )

        elif isinstance(module, (K.GeometricAugmentationBase3D,)):
            raise NotImplementedError(
                "The support for 3d keypoint operations are not yet supported. "
                "You are welcome to file a PR in our repo."
            )

        elif isinstance(module, K.MixAugmentationBaseV2):
            # Dispatch to the mix child's own keypoint handler. Unsupported
            # children raise NotImplementedError from apply_transform_keypoint
            # (matching a direct call) instead of silently returning keypoints.
            out = module.transform_keypoint(_input.data, cls.get_instance_module_param(param), module.flags)
            _input = Keypoints(out, raise_if_not_floating_point=False)

        elif isinstance(module, K.IntensityAugmentationBase2D):
            _input = module.transform_keypoints(
                _input, cls.get_instance_module_param(param), module.flags, **extra_args
            )

        elif isinstance(module, K.container.ImageSequentialBase):
            _input = module.transform_keypoints(
                _input, params=cls.get_sequential_module_param(param), extra_args=extra_args
            )

        elif isinstance(module, (K.auto.operations.OperationBase,)):
            return KeypointSequentialOps.transform(input, module=module.op, param=param, extra_args=extra_args)

        return _input

    @classmethod
    def inverse(
        cls, input: Keypoints, module: nn.Module, param: ParamItem, extra_args: Optional[Dict[str, Any]] = None
    ) -> Keypoints:
        """Inverse a transformation with respect to the parameters.

        Args:
            input: Input keypoints. Coordinates are conceptually stored as
                ``(B, N, 2)``, where the last dimension stores ``(x, y)``.
            module: any torch nn.Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
            extra_args: Optional dictionary of extra arguments with specific options for different input types.

        Returns:
            Keypoints after inverse transformation.
        """
        if extra_args is None:
            extra_args = {}
        _input = input.clone()

        if isinstance(module, (K.GeometricAugmentationBase2D,)):
            transform = module.compute_inverse_transformation(cls.get_transform_matrix(module, param, _input))
            _input = module.inverse_keypoints(
                _input, cls.get_instance_module_param(param), module.flags, transform=transform, **extra_args
            )

        elif isinstance(module, (K.GeometricAugmentationBase3D,)):
            raise NotImplementedError(
                "The support for 3d keypoint operations are not yet supported. "
                "You are welcome to file a PR in our repo."
            )

        elif isinstance(module, K.container.ImageSequentialBase):
            _input = module.inverse_keypoints(
                _input, params=cls.get_sequential_module_param(param), extra_args=extra_args
            )

        elif isinstance(module, (K.auto.operations.OperationBase,)):
            return KeypointSequentialOps.inverse(input, module=module.op, param=param, extra_args=extra_args)

        return _input
