from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Callable, cast, Dict, Iterator, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

import kornia  # lazy loading for circular dependencies
from kornia.augmentation.base import (
    _AugmentationBase,
    GeometricAugmentationBase2D,
    MixAugmentationBase,
    TensorWithTransformMat,
)
from kornia.constants import DataKey
from kornia.geometry.bbox import transform_bbox
from kornia.geometry.linalg import transform_points
from kornia.utils.helpers import _torch_inverse_cast

from .base import ParamItem


def _get_geometric_only_param(
    module: "kornia.augmentation.container.ImageSequential", param: List[ParamItem]
) -> List[ParamItem]:
    named_modules: Iterator[Tuple[str, nn.Module]] = module.get_forward_sequence(param)

    res: List[ParamItem] = []
    for (_, mod), p in zip(named_modules, param):
        if isinstance(mod, (GeometricAugmentationBase2D,)):
            res.append(p)
    return res


class ApplyInverseInterface(metaclass=ABCMeta):
    """Abstract interface for applying and inversing transformations."""

    @classmethod
    @abstractmethod
    def apply_trans(
        cls,
        input: torch.Tensor,
        label: Optional[torch.Tensor],
        module: nn.Module,
        param: ParamItem,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply a transformation with respect to the parameters.

        Args:
            input: the input tensor.
            label: the optional label tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def inverse(
        cls,
        input: torch.Tensor,
        module: nn.Module,
        param: Optional[ParamItem] = None
    ) -> torch.Tensor:
        """Inverse a transformation with respect to the parameters.

        Args:
            input: the input tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
        """
        raise NotImplementedError


class ApplyInverseImpl(ApplyInverseInterface):
    """Standard matrix apply and inverse methods."""

    apply_func: Callable

    @classmethod
    def apply_trans(
        cls, input: torch.Tensor, label: Optional[torch.Tensor], module: nn.Module, param: ParamItem
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply a transformation with respect to the parameters.

        Args:
            input: the input tensor.
            label: the optional label tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
        """

        mat: Optional[torch.Tensor] = cls._get_transformation(input, module, param)
        to_apply = None
        if isinstance(module, _AugmentationBase):
            to_apply = param.data['batch_prob']  # type: ignore

        # If any inputs need to be transformed.
        if mat is not None and to_apply is not None and to_apply.sum() != 0:
            input[to_apply] = cls.apply_func(mat, input[to_apply])

        return input, label

    @classmethod
    def inverse(
        cls, input: torch.Tensor, module: nn.Module, param: Optional[ParamItem] = None
    ) -> torch.Tensor:
        """Inverse a transformation with respect to the parameters.

        Args:
            input: the input tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
        """
        mat: Optional[torch.Tensor] = cls._get_transformation(input, module, param)

        if mat is not None:
            transform: torch.Tensor = cls._get_inverse_transformation(mat)
            input = cls.apply_func(torch.as_tensor(transform, device=input.device, dtype=input.dtype), input)
        return input

    @classmethod
    def _get_transformation(
        cls, input: torch.Tensor, module: nn.Module, param: Optional[ParamItem] = None
    ) -> Optional[torch.Tensor]:

        if isinstance(module, (
            GeometricAugmentationBase2D,
            kornia.augmentation.container.ImageSequential,
        )) and param is None:
            raise ValueError(f"Parameters of transformation matrix for {module} has not been computed.")

        if isinstance(module, GeometricAugmentationBase2D):
            _param = cast(Dict[str, torch.Tensor], param.data)  # type: ignore
            mat = module.get_transformation_matrix(input, _param)
        elif isinstance(module, kornia.augmentation.container.ImageSequential) and not module.is_intensity_only():
            _param = cast(List[ParamItem], param.data)  # type: ignore
            mat = module.get_transformation_matrix(input, _param)  # type: ignore
        else:
            return None  # No need to update anything
        return mat

    @classmethod
    def _get_inverse_transformation(cls, transform: torch.Tensor) -> torch.Tensor:
        return _torch_inverse_cast(transform)


class InputApplyInverse(ApplyInverseImpl):
    """Apply and inverse transformations for (image) input tensors."""

    @classmethod
    def apply_trans(  # type: ignore
        cls,
        input: TensorWithTransformMat,
        label: Optional[torch.Tensor],
        module: nn.Module,
        param: ParamItem,
    ) -> Tuple[TensorWithTransformMat, Optional[torch.Tensor]]:
        """Apply a transformation with respect to the parameters.

        Args:
            input: the input tensor.
            label: the optional label tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
        """
        if isinstance(module, (MixAugmentationBase,)):
            input, label = module(input, label, params=param.data)
        elif isinstance(module, (_AugmentationBase,)):
            input = module(input, params=param.data)
        elif isinstance(module, kornia.augmentation.container.ImageSequential):
            temp = module.apply_inverse_func
            temp2 = module.return_label
            module.apply_inverse_func = InputApplyInverse
            module.return_label = True
            input, label = module(input, label, param.data)
            module.apply_inverse_func = temp
            module.return_label = temp2
        else:
            if param.data is not None:
                raise AssertionError(f"Non-augmentaion operation {param.name} require empty parameters. Got {param}.")
            # In case of return_transform = True
            if isinstance(input, (tuple, list)):
                input = (module(input[0]), input[1])
            else:
                input = module(input)
        return input, label

    @classmethod
    def inverse(cls, input: torch.Tensor, module: nn.Module, param: Optional[ParamItem] = None) -> torch.Tensor:
        """Inverse a transformation with respect to the parameters.

        Args:
            input: the input tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
        """
        if isinstance(module, GeometricAugmentationBase2D):
            input = module.inverse(input, None if param is None else cast(Dict, param.data))
        elif isinstance(module, kornia.augmentation.container.ImageSequential):
            temp = module.apply_inverse_func
            module.apply_inverse_func = InputApplyInverse
            input = module.inverse(input, None if param is None else cast(List, param.data))
            module.apply_inverse_func = temp
        return input


class MaskApplyInverse(ApplyInverseImpl):
    """Apply and inverse transformations for mask tensors."""

    @classmethod
    def make_input_only_sequential(cls, module: "kornia.augmentation.container.ImageSequential") -> Callable:
        """Disable all other additional inputs (e.g. ) for ImageSequential."""
        def f(*args, **kwargs):
            if_return_trans = module.return_transform
            if_return_label = module.return_label
            module.return_transform = False
            module.return_label = False
            out = module(*args, **kwargs)
            module.return_transform = if_return_trans
            module.return_label = if_return_label
            return out
        return f

    @classmethod
    def apply_trans(
        cls, input: torch.Tensor, label: Optional[torch.Tensor], module: nn.Module, param: Optional[ParamItem] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply a transformation with respect to the parameters.

        Args:
            input: the input tensor.
            label: the optional label tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
        """
        if param is not None:
            _param = param.data
        else:
            _param = None  # type: ignore

        if isinstance(module, GeometricAugmentationBase2D):
            _param = cast(Dict[str, torch.Tensor], _param)
            input = module(input, _param, return_transform=False)
        elif isinstance(module, kornia.augmentation.container.ImageSequential) and not module.is_intensity_only():
            _param = cast(List[ParamItem], _param)
            temp = module.apply_inverse_func
            module.apply_inverse_func = MaskApplyInverse
            geo_param: List[ParamItem] = _get_geometric_only_param(module, _param)
            input = cls.make_input_only_sequential(module)(input, None, geo_param)
            module.apply_inverse_func = temp
        else:
            pass  # No need to update anything
        return input, label

    @classmethod
    def inverse(
        cls, input: torch.Tensor, module: nn.Module, param: Optional[ParamItem] = None
    ) -> torch.Tensor:
        """Inverse a transformation with respect to the parameters.

        Args:
            input: the input tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
        """
        if isinstance(module, GeometricAugmentationBase2D):
            input = module.inverse(input, None if param is None else cast(Dict, param.data))
        elif isinstance(module, kornia.augmentation.container.ImageSequential):
            temp = module.apply_inverse_func
            module.apply_inverse_func = MaskApplyInverse
            input = module.inverse(input, None if param is None else cast(List, param.data))
            module.apply_inverse_func = temp
        return input


class BBoxXYXYApplyInverse(ApplyInverseImpl):
    """Apply and inverse transformations for bounding box tensors.

    This is for transform boxes in the format [xmin, ymin, xmax, ymax].
    """

    apply_func = partial(transform_bbox, mode="xyxy")


class BBoxXYWHApplyInverse(ApplyInverseImpl):
    """Apply and inverse transformations for bounding box tensors.

    This is for transform boxes in the format [xmin, ymin, width, height].
    """

    apply_func = partial(transform_bbox, mode="xywh")


class KeypointsApplyInverse(ApplyInverseImpl):
    """Apply and inverse transformations for keypoints tensors."""

    apply_func = transform_points


class ApplyInverse:
    """Apply and inverse transformations for any tensors (e.g. mask, box, points)."""

    @classmethod
    def _get_func_by_key(cls, dcate: Union[str, int, DataKey]) -> Type[ApplyInverseInterface]:
        if DataKey.get(dcate) == DataKey.INPUT:
            return InputApplyInverse
        if DataKey.get(dcate) in [DataKey.MASK]:
            return MaskApplyInverse
        if DataKey.get(dcate) in [DataKey.BBOX, DataKey.BBOX_XYXY]:
            return BBoxXYXYApplyInverse
        if DataKey.get(dcate) in [DataKey.BBOX_XYHW]:
            return BBoxXYWHApplyInverse
        if DataKey.get(dcate) in [DataKey.KEYPOINTS]:
            return KeypointsApplyInverse
        raise NotImplementedError(f"input type of {dcate} is not implemented.")

    @classmethod
    def apply_by_key(
        cls,
        input: TensorWithTransformMat,
        label: Optional[torch.Tensor],
        module: nn.Module,
        param: ParamItem,
        dcate: Union[str, int, DataKey] = DataKey.INPUT,
    ) -> Tuple[TensorWithTransformMat, Optional[torch.Tensor]]:
        """Apply a transformation with respect to the parameters.

        Args:
            input: the input tensor.
            label: the optional label tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
            dcate: data category. 'input', 'mask', 'bbox', 'bbox_xyxy', 'bbox_xyhw', 'keypoints'.
                By default, it is set to 'input'.
        """
        func: Type[ApplyInverseInterface] = cls._get_func_by_key(dcate)

        if isinstance(input, (tuple,)):
            # If the input is a tuple with (input, mat) or something else
            return (func.apply_trans(input[0], label, module, param), *input[1:])  # type: ignore
        return func.apply_trans(input, label, module=module, param=param)

    @classmethod
    def inverse_by_key(
        cls,
        input: torch.Tensor,
        module: nn.Module,
        param: Optional[ParamItem] = None,
        dcate: Union[str, int, DataKey] = DataKey.INPUT,
    ) -> torch.Tensor:
        """Inverse a transformation with respect to the parameters.

        Args:
            input: the input tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
            dcate: data category. 'input', 'mask', 'bbox', 'bbox_xyxy', 'bbox_xyhw', 'keypoints'.
                By default, it is set to 'input'.
        """
        func: Type[ApplyInverseInterface] = cls._get_func_by_key(dcate)
        return func.inverse(input, module, param)
