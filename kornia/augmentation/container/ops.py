import copy
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union

from typing_extensions import ParamSpec

import kornia.augmentation as K
from kornia.augmentation.base import _AugmentationBase
from kornia.constants import DataKey
from kornia.core import Module, Tensor
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints

from .params import ParamItem

DataType = Union[Tensor, List[Tensor], Boxes, Keypoints]

# NOTE: shouldn't this SequenceDataType alias be equals to List[DataType]?
SequenceDataType = Union[List[Tensor], List[List[Tensor]], List[Boxes], List[Keypoints]]

T = TypeVar("T")


class SequentialOpsInterface(Generic[T], metaclass=ABCMeta):
    """Abstract interface for applying and inversing transformations."""

    @classmethod
    def get_instance_module_param(cls, param: ParamItem) -> Dict[str, Tensor]:
        if isinstance(param, ParamItem) and isinstance(param.data, dict):
            _params = param.data
        else:
            raise TypeError(f"Expected param (ParamItem.data) be a dictionary. Gotcha {param}.")
        return _params

    @classmethod
    def get_sequential_module_param(cls, param: ParamItem) -> List[ParamItem]:
        if isinstance(param, ParamItem) and isinstance(param.data, list):
            _params = param.data
        else:
            raise TypeError(f"Expected param (ParamItem.data) be a list. Gotcha {param}.")
        return _params

    @classmethod
    @abstractmethod
    def transform(cls, input: T, module: Module, param: ParamItem, extra_args: Dict[str, Any] = {}) -> T:
        """Apply a transformation with respect to the parameters.

        Args:
            input: the input tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def inverse(cls, input: T, module: Module, param: ParamItem, extra_args: Dict[str, Any] = {}) -> T:
        """Inverse a transformation with respect to the parameters.

        Args:
            input: the input tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
        """
        raise NotImplementedError


class AugmentationSequentialOps:
    def __init__(self, data_keys: Optional[List[DataKey]]) -> None:
        self._data_keys = data_keys

    @property
    def data_keys(self) -> Optional[List[DataKey]]:
        return self._data_keys

    @data_keys.setter
    def data_keys(self, data_keys: Optional[Union[List[DataKey], List[str], List[int]]]) -> None:
        if data_keys:
            self._data_keys = [DataKey.get(inp) for inp in data_keys]
        else:
            self._data_keys = None

    def preproc_datakeys(self, data_keys: Optional[Union[List[str], List[int], List[DataKey]]] = None) -> List[DataKey]:
        if data_keys is None:
            if isinstance(self.data_keys, list):
                return self.data_keys
            raise ValueError("Sequential ops needs data keys to be able to process.")
        else:
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
        module: Module,
        param: ParamItem,
        extra_args: Dict[DataKey, Dict[str, Any]],
        data_keys: Optional[Union[List[str], List[int], List[DataKey]]] = None,
    ) -> Union[DataType, SequenceDataType]:
        _data_keys = self.preproc_datakeys(data_keys)

        if isinstance(module, K.RandomTransplantation):
            # For transforms which require the full input to calculate the parameters (e.g. RandomTransplantation)
            param = ParamItem(
                name=param.name,
                data=module.params_from_input(
                    *arg,  # type: ignore[arg-type]
                    data_keys=_data_keys,
                    params=param.data,  # type: ignore[arg-type]
                    extra_args=extra_args,
                ),
            )

        outputs = []
        for inp, dcate in zip(arg, _data_keys):
            op = self._get_op(dcate)
            extra_arg = extra_args.get(dcate, {})
            if dcate.name == "MASK" and isinstance(inp, list):
                outputs.append(MaskSequentialOps.transform_list(inp, module, param=param, extra_args=extra_arg))
            else:
                outputs.append(op.transform(inp, module, param=param, extra_args=extra_arg))
        if len(outputs) == 1 and isinstance(outputs, (list, tuple)):
            return outputs[0]
        return outputs

    def inverse(
        self,
        *arg: DataType,
        module: Module,
        param: ParamItem,
        extra_args: Dict[DataKey, Dict[str, Any]],
        data_keys: Optional[Union[List[str], List[int], List[DataKey]]] = None,
    ) -> Union[DataType, SequenceDataType]:
        _data_keys = self.preproc_datakeys(data_keys)
        outputs = []
        for inp, dcate in zip(arg, _data_keys):
            op = self._get_op(dcate)
            extra_arg = extra_args[dcate] if dcate in extra_args else {}
            outputs.append(op.inverse(inp, module, param=param, extra_args=extra_arg))
        if len(outputs) == 1 and isinstance(outputs, (list, tuple)):
            return outputs[0]
        return outputs


P = ParamSpec("P")


def make_input_only_sequential(module: "K.container.ImageSequentialBase") -> Callable[P, Tensor]:
    """Disable all other additional inputs (e.g. ) for ImageSequential."""

    def f(*args: P.args, **kwargs: P.kwargs) -> Tensor:
        return module(*args, **kwargs)

    return f


def get_geometric_only_param(module: "K.container.ImageSequentialBase", param: List[ParamItem]) -> List[ParamItem]:
    named_modules = module.get_forward_sequence(param)

    res: List[ParamItem] = []
    for (_, mod), p in zip(named_modules, param):
        if isinstance(mod, (K.GeometricAugmentationBase2D, K.GeometricAugmentationBase3D)):
            res.append(p)
    return res


class InputSequentialOps(SequentialOpsInterface[Tensor]):
    @classmethod
    def transform(cls, input: Tensor, module: Module, param: ParamItem, extra_args: Dict[str, Any] = {}) -> Tensor:
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
    def inverse(cls, input: Tensor, module: Module, param: ParamItem, extra_args: Dict[str, Any] = {}) -> Tensor:
        if isinstance(module, K.GeometricAugmentationBase2D):
            input = module.inverse(input, params=cls.get_instance_module_param(param), **extra_args)
        elif isinstance(module, (K.GeometricAugmentationBase3D,)):
            raise NotImplementedError(
                "The support for 3d inverse operations are not yet supported. You are welcome to file a PR in our repo."
            )
        elif isinstance(module, (K.auto.operations.OperationBase,)):
            return InputSequentialOps.inverse(input, module=module.op, param=param, extra_args=extra_args)
        elif isinstance(module, K.ImageSequential) and not module.is_intensity_only():
            input = module.inverse_inputs(input, params=cls.get_sequential_module_param(param), extra_args=extra_args)
        elif isinstance(module, K.container.ImageSequentialBase):
            input = module.inverse_inputs(input, params=cls.get_sequential_module_param(param), extra_args=extra_args)
        return input


class ClassSequentialOps(SequentialOpsInterface[Tensor]):
    """Apply and inverse transformations for class labels if needed."""

    @classmethod
    def transform(cls, input: Tensor, module: Module, param: ParamItem, extra_args: Dict[str, Any] = {}) -> Tensor:
        if isinstance(module, K.MixAugmentationBaseV2):
            raise NotImplementedError(
                "The support for class labels for mix augmentations that change the class label is not yet supported."
            )
        return input

    @classmethod
    def inverse(cls, input: Tensor, module: Module, param: ParamItem, extra_args: Dict[str, Any] = {}) -> Tensor:
        return input


class MaskSequentialOps(SequentialOpsInterface[Tensor]):
    """Apply and inverse transformations for mask tensors."""

    @classmethod
    def transform(cls, input: Tensor, module: Module, param: ParamItem, extra_args: Dict[str, Any] = {}) -> Tensor:
        """Apply a transformation with respect to the parameters.

        Args:
            input: the input tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
        """

        if isinstance(module, (K.GeometricAugmentationBase2D,)):
            input = module.transform_masks(
                input,
                params=cls.get_instance_module_param(param),
                flags=module.flags,
                transform=module.transform_matrix,
                **extra_args,
            )

        elif isinstance(module, (K.GeometricAugmentationBase3D,)):
            raise NotImplementedError(
                "The support for 3d mask operations are not yet supported. You are welcome to file a PR in our repo."
            )

        elif isinstance(module, K.RandomTransplantation):
            input = module(input, params=cls.get_instance_module_param(param), data_keys=[DataKey.MASK], **extra_args)

        elif isinstance(module, (_AugmentationBase)):
            input = module.transform_masks(
                input, params=cls.get_instance_module_param(param), flags=module.flags, **extra_args
            )

        elif isinstance(module, K.ImageSequential) and not module.is_intensity_only():
            input = module.transform_masks(input, params=cls.get_sequential_module_param(param), extra_args=extra_args)

        elif isinstance(module, K.container.ImageSequentialBase):
            input = module.transform_masks(input, params=cls.get_sequential_module_param(param), extra_args=extra_args)

        elif isinstance(module, (K.auto.operations.OperationBase,)):
            input = MaskSequentialOps.transform(input, module=module.op, param=param, extra_args=extra_args)

        return input

    @classmethod
    def transform_list(
        cls, input: List[Tensor], module: Module, param: ParamItem, extra_args: Dict[str, Any] = {}
    ) -> List[Tensor]:
        """Apply a transformation with respect to the parameters.

        Args:
            input: list of input tensors.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
        """
        if isinstance(module, (K.GeometricAugmentationBase2D,)):
            tfm_input = []
            params = cls.get_instance_module_param(param)
            params_i = copy.deepcopy(params)
            for i, inp in enumerate(input):
                params_i["batch_prob"] = params["batch_prob"][i]
                tfm_inp = module.transform_masks(
                    inp, params=params_i, flags=module.flags, transform=module.transform_matrix, **extra_args
                )
                tfm_input.append(tfm_inp)
            input = tfm_input

        elif isinstance(module, (K.GeometricAugmentationBase3D,)):
            raise NotImplementedError(
                "The support for 3d mask operations are not yet supported. You are welcome to file a PR in our repo."
            )

        elif isinstance(module, (_AugmentationBase)):
            tfm_input = []
            params = cls.get_instance_module_param(param)
            params_i = copy.deepcopy(params)
            for i, inp in enumerate(input):
                params_i["batch_prob"] = params["batch_prob"][i]
                tfm_inp = module.transform_masks(inp, params=params_i, flags=module.flags, **extra_args)
                tfm_input.append(tfm_inp)
            input = tfm_input

        elif isinstance(module, K.ImageSequential) and not module.is_intensity_only():
            tfm_input = []
            seq_params = cls.get_sequential_module_param(param)
            for inp in input:
                tfm_inp = module.transform_masks(inp, params=seq_params, extra_args=extra_args)
                tfm_input.append(tfm_inp)
            input = tfm_input

        elif isinstance(module, K.container.ImageSequentialBase):
            tfm_input = []
            seq_params = cls.get_sequential_module_param(param)
            for inp in input:
                tfm_inp = module.transform_masks(inp, params=seq_params, extra_args=extra_args)
                tfm_input.append(tfm_inp)
            input = tfm_input

        elif isinstance(module, (K.auto.operations.OperationBase,)):
            raise NotImplementedError(
                "The support for list of masks under auto operations are not yet supported. You are welcome to file a"
                " PR in our repo."
            )
        return input

    @classmethod
    def inverse(cls, input: Tensor, module: Module, param: ParamItem, extra_args: Dict[str, Any] = {}) -> Tensor:
        """Inverse a transformation with respect to the parameters.

        Args:
            input: the input tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
        """

        if isinstance(module, (K.GeometricAugmentationBase2D,)):
            if module.transform_matrix is None:
                raise ValueError(f"No valid transformation matrix found in {module.__class__}.")
            transform = module.compute_inverse_transformation(module.transform_matrix)
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


class BoxSequentialOps(SequentialOpsInterface[Boxes]):
    """Apply and inverse transformations for bounding box tensors.

    This is for transform boxes in the format (B, N, 4, 2).
    """

    @classmethod
    def transform(cls, input: Boxes, module: Module, param: ParamItem, extra_args: Dict[str, Any] = {}) -> Boxes:
        """Apply a transformation with respect to the parameters.

        Args:
            input: the input tensor, (B, N, 4, 2) or (B, 4, 2).
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
        """
        _input = input.clone()

        if isinstance(module, (K.GeometricAugmentationBase2D,)):
            _input = module.transform_boxes(
                _input,
                cls.get_instance_module_param(param),
                module.flags,
                transform=module.transform_matrix,
                **extra_args,
            )

        elif isinstance(module, (K.GeometricAugmentationBase3D,)):
            raise NotImplementedError(
                "The support for 3d box operations are not yet supported. You are welcome to file a PR in our repo."
            )

        elif isinstance(module, K.ImageSequential) and not module.is_intensity_only():
            _input = module.transform_boxes(
                _input, params=cls.get_sequential_module_param(param), extra_args=extra_args
            )

        elif isinstance(module, K.container.ImageSequentialBase):
            _input = module.transform_boxes(
                _input, params=cls.get_sequential_module_param(param), extra_args=extra_args
            )

        elif isinstance(module, (K.auto.operations.OperationBase,)):
            return BoxSequentialOps.transform(input, module=module.op, param=param, extra_args=extra_args)

        return _input

    @classmethod
    def inverse(cls, input: Boxes, module: Module, param: ParamItem, extra_args: Dict[str, Any] = {}) -> Boxes:
        """Inverse a transformation with respect to the parameters.

        Args:
            input: the input tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
        """
        _input = input.clone()

        if isinstance(module, (K.GeometricAugmentationBase2D,)):
            if module.transform_matrix is None:
                raise ValueError(f"No valid transformation matrix found in {module.__class__}.")
            transform = module.compute_inverse_transformation(module.transform_matrix)
            _input = module.inverse_boxes(
                _input,
                param.data,  # type: ignore[arg-type]
                module.flags,
                transform=transform,
                **extra_args,
            )

        elif isinstance(module, (K.GeometricAugmentationBase3D,)):
            raise NotImplementedError(
                "The support for 3d box operations are not yet supported. You are welcome to file a PR in our repo."
            )

        elif isinstance(module, K.ImageSequential) and not module.is_intensity_only():
            _input = module.inverse_boxes(_input, params=cls.get_sequential_module_param(param), extra_args=extra_args)

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
        cls, input: Keypoints, module: Module, param: ParamItem, extra_args: Dict[str, Any] = {}
    ) -> Keypoints:
        """Apply a transformation with respect to the parameters.

        Args:
            input: the input tensor, (B, N, 4, 2) or (B, 4, 2).
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
        """
        _input = input.clone()

        if isinstance(module, (K.GeometricAugmentationBase2D,)):
            _input = module.transform_keypoints(
                _input,
                cls.get_instance_module_param(param),
                module.flags,
                transform=module.transform_matrix,
                **extra_args,
            )

        elif isinstance(module, (K.GeometricAugmentationBase3D,)):
            raise NotImplementedError(
                "The support for 3d keypoint operations are not yet supported. "
                "You are welcome to file a PR in our repo."
            )

        elif isinstance(module, K.ImageSequential) and not module.is_intensity_only():
            _input = module.transform_keypoints(
                _input, params=cls.get_sequential_module_param(param), extra_args=extra_args
            )

        elif isinstance(module, K.container.ImageSequentialBase):
            _input = module.transform_keypoints(
                _input, params=cls.get_sequential_module_param(param), extra_args=extra_args
            )

        elif isinstance(module, (K.auto.operations.OperationBase,)):
            return KeypointSequentialOps.transform(input, module=module.op, param=param, extra_args=extra_args)

        return _input

    @classmethod
    def inverse(cls, input: Keypoints, module: Module, param: ParamItem, extra_args: Dict[str, Any] = {}) -> Keypoints:
        """Inverse a transformation with respect to the parameters.

        Args:
            input: the input tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
        """
        _input = input.clone()

        if isinstance(module, (K.GeometricAugmentationBase2D,)):
            if module.transform_matrix is None:
                raise ValueError(f"No valid transformation matrix found in {module.__class__}.")
            transform = module.compute_inverse_transformation(module.transform_matrix)
            _input = module.inverse_keypoints(
                _input, cls.get_instance_module_param(param), module.flags, transform=transform, **extra_args
            )

        elif isinstance(module, (K.GeometricAugmentationBase3D,)):
            raise NotImplementedError(
                "The support for 3d keypoint operations are not yet supported. "
                "You are welcome to file a PR in our repo."
            )

        elif isinstance(module, K.ImageSequential) and not module.is_intensity_only():
            _input = module.inverse_keypoints(
                _input, params=cls.get_sequential_module_param(param), extra_args=extra_args
            )

        elif isinstance(module, K.container.ImageSequentialBase):
            _input = module.inverse_keypoints(
                _input, params=cls.get_sequential_module_param(param), extra_args=extra_args
            )

        elif isinstance(module, (K.auto.operations.OperationBase,)):
            return KeypointSequentialOps.inverse(input, module=module.op, param=param, extra_args=extra_args)

        return _input
