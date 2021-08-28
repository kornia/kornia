from typing import Callable, List, cast, Dict, Optional, Tuple, Union, Iterator

import torch
import torch.nn as nn

from kornia.augmentation.base import (
    _AugmentationBase,
    GeometricAugmentationBase2D,
    TensorWithTransformMat,
)

from kornia.augmentation.base import _AugmentationBase, MixAugmentationBase
from kornia.constants import DataKey
from kornia.geometry.bbox import transform_bbox
from kornia.geometry.linalg import transform_points
import kornia.augmentation.container as CTN  # lazy loading for circular dependencies

from .base import ParamItem


def get_geometric_only_param(module: "CTN.ImageSequential", param: List[ParamItem]) -> List[ParamItem]:
    named_modules = module.get_forward_sequence(param)

    res = []
    for (_, module), p in zip(named_modules, param):
        if isinstance(module, (GeometricAugmentationBase2D,)):
            res.append(p)
    return res


def make_input_only_sequential(module: "CTN.ImageSequential") -> Callable:
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


class InputApplyInverse:

    def apply_to_input(
        self,
        input: TensorWithTransformMat,
        label: Optional[torch.Tensor],
        module: nn.Module,
        param: ParamItem,
    ) -> Tuple[TensorWithTransformMat, Optional[torch.Tensor]]:
        if isinstance(module, (MixAugmentationBase,)):
            input, label = module(input, label, params=param.data)
        elif isinstance(module, (_AugmentationBase,)):
            input = module(input, params=param.data)
        else:
            if param.data is not None:
                raise AssertionError(f"Non-augmentaion operation {param.name} require empty parameters. Got {param}.")
            # In case of return_transform = True
            if isinstance(input, (tuple, list)):
                input = (module(input[0]), input[1])
            else:
                input = module(input)
        return input, label

    def inverse_input(self, input: torch.Tensor, module: nn.Module, param: Optional[ParamItem] = None) -> torch.Tensor:
        if isinstance(module, GeometricAugmentationBase2D):
            input = module.inverse(input, None if param is None else cast(Dict, param.data))
        elif isinstance(module, CTN.ImageSequential) and not module.is_intensity_only():
            raise NotImplementedError
        return input


class MaskApplyInverse:

    def apply_to_mask(self, input: torch.Tensor, module: nn.Module, param: Optional[ParamItem] = None) -> torch.Tensor:
        if param is not None:
            _param = cast(Dict[str, torch.Tensor], param.data)
        else:
            _param = None  # type: ignore

        if isinstance(module, GeometricAugmentationBase2D) and _param is None:
            input = module(input, return_transform=False)
        elif isinstance(module, GeometricAugmentationBase2D) and _param is not None:
            input = module(input, _param, return_transform=False)
        elif isinstance(module, CTN.ImageSequential) and not module.is_intensity_only():
            geo_param = get_geometric_only_param(module, _param)
            input = make_input_only_sequential(module)(input, None, geo_param)
        else:
            pass  # No need to update anything
        return input

    def inverse_mask(self, input: torch.Tensor, module: nn.Module, param: Optional[ParamItem] = None) -> torch.Tensor:
        if isinstance(module, GeometricAugmentationBase2D):
            input = module.inverse(input, None if param is None else cast(Dict, param.data))
        elif isinstance(module, CTN.ImageSequential) and not module.is_intensity_only():
            raise NotImplementedError
        return input


class BBoxApplyInverse:

    def apply_to_bbox(
        self, input: torch.Tensor, module: nn.Module, param: Optional[ParamItem] = None, mode: str = "xyxy"
    ) -> torch.Tensor:
        if param is not None:
            _param = cast(Dict[str, torch.Tensor], param.data)
        else:
            _param = None  # type: ignore

        if isinstance(module, GeometricAugmentationBase2D) and _param is None:
            raise ValueError(f"Transformation matrix for {module} has not been computed.")
        if (
            isinstance(module, GeometricAugmentationBase2D) or
            (isinstance(module, CTN.ImageSequential) and not module.is_intensity_only())
        ):
            input = transform_bbox(module.get_transformation_matrix(input, _param), input, mode)
        else:
            pass  # No need to update anything
        return input

    def inverse_bbox(
        self, input: torch.Tensor, module: nn.Module, param: Optional[ParamItem] = None, mode: str = "xyxy"
    ) -> torch.Tensor:
        if isinstance(module, GeometricAugmentationBase2D):
            transform = module.compute_inverse_transformation(
                module.get_transformation_matrix(input, None if param is None else cast(Dict, param.data))
            )
            input = transform_bbox(torch.as_tensor(transform, device=input.device, dtype=input.dtype), input, mode)
        elif isinstance(module, CTN.ImageSequential) and not module.is_intensity_only():
            raise NotImplementedError
        return input


class KeypointsApplyInverse:

    def apply_to_keypoints(
        self, input: torch.Tensor, module: nn.Module, param: Optional[ParamItem] = None
    ) -> torch.Tensor:
        if param is not None:
            _param = cast(Dict[str, torch.Tensor], param.data)
        else:
            _param = None  # type: ignore

        if isinstance(module, GeometricAugmentationBase2D) and _param is None:
            raise ValueError(f"Transformation matrix for {module} has not been computed.")
        if (
            isinstance(module, GeometricAugmentationBase2D) or
            (isinstance(module, CTN.ImageSequential) and not module.is_intensity_only())
        ):
            input = transform_points(module.get_transformation_matrix(input, _param), input)
        else:
            pass  # No need to update anything
        return input

    def inverse_keypoints(
        self, input: torch.Tensor, module: nn.Module, param: Optional[ParamItem] = None
    ) -> torch.Tensor:
        if isinstance(module, GeometricAugmentationBase2D):
            transform = module.compute_inverse_transformation(
                module.get_transformation_matrix(input, None if param is None else cast(Dict, param.data))
            )
            input = transform_points(torch.as_tensor(transform, device=input.device, dtype=input.dtype), input)
        elif isinstance(module, CTN.ImageSequential) and not module.is_intensity_only():
            raise NotImplementedError
        return input


class ApplyInverse(
    InputApplyInverse,
    MaskApplyInverse,
    BBoxApplyInverse,
    KeypointsApplyInverse,
):

    def apply_by_key(
        self,
        input: TensorWithTransformMat,
        label: Optional[torch.Tensor],
        module: Optional[nn.Module],
        param: ParamItem,
        dcate: Union[str, int, DataKey] = DataKey.INPUT,
    ) -> Tuple[TensorWithTransformMat, Optional[torch.Tensor]]:
        if module is None:
            module = self.get_submodule(param.name)
        if DataKey.get(dcate) in [DataKey.INPUT]:
            return self.apply_to_input(input, label, module=module, param=param)
        if DataKey.get(dcate) in [DataKey.MASK]:
            if isinstance(input, (tuple,)):
                return (self.apply_to_mask(input[0], module, param), *input[1:]), None
            return self.apply_to_mask(input, module, param), None
        if DataKey.get(dcate) in [DataKey.BBOX, DataKey.BBOX_XYXY]:
            if isinstance(input, (tuple,)):
                return (self.apply_to_bbox(input[0], module, param, mode='xyxy'), *input[1:]), None
            return self.apply_to_bbox(input, module, param, mode='xyxy'), None
        if DataKey.get(dcate) in [DataKey.BBOX_XYHW]:
            if isinstance(input, (tuple,)):
                return (self.apply_to_bbox(input[0], module, param, mode='xyhw'), *input[1:]), None
            return self.apply_to_bbox(input, module, param, mode='xyhw'), None
        if DataKey.get(dcate) in [DataKey.KEYPOINTS]:
            if isinstance(input, (tuple,)):
                return (self.apply_to_keypoints(input[0], module, param), *input[1:]), None
            return self.apply_to_keypoints(input, module, param), None
        raise NotImplementedError(f"input type of {dcate} is not implemented.")

    def inverse_by_key(
        self,
        input: torch.Tensor,
        module: nn.Module,
        param: Optional[ParamItem] = None,
        dcate: Union[str, int, DataKey] = DataKey.INPUT,
    ) -> torch.Tensor:
        if DataKey.get(dcate) in [DataKey.INPUT, DataKey.MASK]:
            return self.inverse_input(input, module, param)
        if DataKey.get(dcate) in [DataKey.BBOX, DataKey.BBOX_XYXY]:
            return self.inverse_bbox(input, module, param, mode='xyxy')
        if DataKey.get(dcate) in [DataKey.BBOX_XYHW]:
            return self.inverse_bbox(input, module, param, mode='xyhw')
        if DataKey.get(dcate) in [DataKey.KEYPOINTS]:
            return self.inverse_keypoints(input, module, param)
        raise NotImplementedError(f"input type of {dcate} is not implemented.")
