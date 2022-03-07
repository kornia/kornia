import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Type, Union, cast

import torch
import torch.nn as nn

import kornia  # lazy loading for circular dependencies
from kornia.augmentation import GeometricAugmentationBase2D, MixAugmentationBase, RandomCrop, RandomErasing
from kornia.augmentation.base import _AugmentationBase
from kornia.augmentation.container.base import ParamItem
from kornia.constants import DataKey
from kornia.geometry.bbox import transform_bbox
from kornia.geometry.linalg import transform_points
from kornia.utils.helpers import _torch_inverse_cast


def _get_geometric_only_param(
    module: "kornia.augmentation.ImageSequential", param: List[ParamItem]
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
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def inverse(cls, input: torch.Tensor, module: nn.Module, param: Optional[ParamItem] = None) -> torch.Tensor:
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
        mat: Optional[torch.Tensor]
        if hasattr(module, "_transform_matrix") and module._transform_matrix is not None:
            mat = cast(torch.Tensor, module._transform_matrix)
        else:
            mat = cls._get_transformation(input, module, param)
        mat = torch.as_tensor(mat, device=input.device, dtype=input.dtype)
        to_apply = None
        if isinstance(module, _AugmentationBase):
            to_apply = param.data['batch_prob']  # type: ignore
        if isinstance(module, kornia.augmentation.ImageSequential):
            to_apply = torch.ones(input.shape[0], device=input.device, dtype=input.dtype).bool()

        # If any inputs need to be transformed.
        if mat is not None and to_apply is not None and to_apply.sum() != 0:
            input[to_apply] = cls.apply_func(mat, input[to_apply])

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
        mat: Optional[torch.Tensor]
        if hasattr(module, "_transform_matrix") and module._transform_matrix is not None:
            mat = cast(torch.Tensor, module._transform_matrix)
        else:
            mat = cls._get_transformation(input, module, param)
        mat = torch.as_tensor(mat, device=input.device, dtype=input.dtype)

        if mat is not None:
            transform: torch.Tensor = cls._get_inverse_transformation(mat)
            input = cls.apply_func(torch.as_tensor(transform, device=input.device, dtype=input.dtype), input)
        return input

    @classmethod
    def _get_transformation(
        cls, input: torch.Tensor, module: nn.Module, param: Optional[ParamItem] = None
    ) -> Optional[torch.Tensor]:

        if (
            isinstance(module, (GeometricAugmentationBase2D, kornia.augmentation.ImageSequential))
            and param is None
        ):
            raise ValueError(f"Parameters of transformation matrix for {module} has not been computed.")

        if isinstance(module, GeometricAugmentationBase2D):
            _param = cast(Dict[str, torch.Tensor], param.data)  # type: ignore
            mat = module.get_transformation_matrix(input, _param)
        elif isinstance(module, kornia.augmentation.ImageSequential) and not module.is_intensity_only():
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
    data_key = DataKey.INPUT

    @classmethod
    def apply_trans(  # type: ignore
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
        if isinstance(module, (MixAugmentationBase,)):
            input, label = module(input, label=label, params=param.data)
        elif isinstance(module, (_AugmentationBase,)):
            input = module(input, params=param.data)
        elif isinstance(module, kornia.augmentation.ImageSequential):
            temp = module.apply_inverse_func
            temp2 = module.return_label
            module.apply_inverse_func = InputApplyInverse
            module.return_label = True
            if isinstance(module, kornia.augmentation.AugmentationSequential):
                input, label = module(input, label=label, params=param.data, data_keys=[cls.data_key])
            else:
                input, label = module(input, label=label, params=param.data)
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
            input = module.inverse(input, params=None if param is None else cast(Dict, param.data))
        elif isinstance(module, kornia.augmentation.ImageSequential):
            temp = module.apply_inverse_func
            module.apply_inverse_func = InputApplyInverse
            input = module.inverse(input, params=None if param is None else cast(List, param.data))
            module.apply_inverse_func = temp
        return input


class MaskApplyInverse(ApplyInverseImpl):
    """Apply and inverse transformations for mask tensors."""
    data_key = DataKey.MASK

    @classmethod
    def make_input_only_sequential(cls, module: "kornia.augmentation.ImageSequential") -> Callable:
        """Disable all other additional inputs (e.g. ) for ImageSequential."""

        def f(*args, **kwargs):
            if_return_label = module.return_label
            module.return_label = False
            out = module(*args, **kwargs)
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

        if isinstance(module, (GeometricAugmentationBase2D, RandomErasing)):
            _param = cast(Dict[str, torch.Tensor], _param).copy()
            # TODO: Parametrize value to pad with across the board for different keys
            if 'values' in _param:
                _param['values'] = torch.zeros_like(_param['values'])  # Always pad with zeros
            input = module(input, params=_param)
        elif isinstance(module, kornia.augmentation.ImageSequential) and not module.is_intensity_only():
            _param = cast(List[ParamItem], _param)
            temp = module.apply_inverse_func
            module.apply_inverse_func = MaskApplyInverse
            geo_param: List[ParamItem] = _get_geometric_only_param(module, _param)
            input = cls.make_input_only_sequential(module)(input, label=None, params=geo_param)
            module.apply_inverse_func = temp
        else:
            pass  # No need to update anything
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
            input = module.inverse(input, params=None if param is None else cast(Dict, param.data))
        elif isinstance(module, kornia.augmentation.ImageSequential):
            temp = module.apply_inverse_func
            module.apply_inverse_func = MaskApplyInverse
            input = module.inverse(input, params=None if param is None else cast(List, param.data))
            module.apply_inverse_func = temp
        return input


class BBoxApplyInverse(ApplyInverseImpl):
    """Apply and inverse transformations for bounding box tensors.

    This is for transform boxes in the format (B, N, 4, 2).
    """

    @classmethod
    def _get_padding_size(cls, module: nn.Module, param: Optional[ParamItem]) -> Optional[torch.Tensor]:
        if isinstance(module, RandomCrop):
            _param = cast(Dict[str, torch.Tensor], param.data)  # type: ignore
            return _param.get("padding_size")
        return None

    @classmethod
    def pad(cls, input: torch.Tensor, padding_size: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: (B, N, 4, 2)
            padding_size: (B, 4)
        """
        if len(input.shape) not in (3, 4,):
            raise AssertionError(input.shape)

        if len(padding_size.shape) != 2:
            raise AssertionError(padding_size.shape)

        _input = input.clone()

        if input.dim() == 3:
            # B,4,2 to B,1,4,2
            _input = _input[:, None]

        _input[..., 0] += padding_size[..., None, :1]  # left padding
        _input[..., 1] += padding_size[..., None, 2:3]  # top padding

        if input.dim() == 3:
            _input = _input[:, 0]  # squeeze back

        return _input

    @classmethod
    def unpad(cls, input: torch.Tensor, padding_size: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: (B, N, 4, 2)
            padding_size: (B, 4)
        """
        if len(input.shape) not in (3, 4,):
            raise AssertionError(input.shape)

        if len(padding_size.shape) != 2:
            raise AssertionError(padding_size.shape)

        _input = input.clone()

        if input.dim() == 3:
            # B,4,2 to B,1,4,2
            _input = _input[:, None]

        _input[..., 0] -= padding_size[..., None, :1]  # left padding
        _input[..., 1] -= padding_size[..., None, 2:3]  # top padding

        if input.dim() == 3:
            _input = _input[:, 0]  # squeeze back

        return _input

    apply_func = partial(transform_bbox, mode="xyxy", restore_coordinates=True)

    @classmethod
    def apply_trans(
        cls, input: torch.Tensor, label: Optional[torch.Tensor], module: nn.Module, param: ParamItem
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply a transformation with respect to the parameters.

        Args:
            input: the input tensor, (B, N, 4, 2) or (B, 4, 2).
            label: the optional label tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
        """
        _input = input.clone()

        padding_size = cls._get_padding_size(module, param)
        if padding_size is not None:
            _input = cls.pad(_input, padding_size.to(_input))

        _input, label = super().apply_trans(_input, label, module, param)

        # TODO: Filter/crop boxes outside crop (with negative or larger than crop size coords)?

        return _input, label

    @classmethod
    def inverse(cls, input: torch.Tensor, module: nn.Module, param: Optional[ParamItem] = None) -> torch.Tensor:
        """Inverse a transformation with respect to the parameters.

        Args:
            input: the input tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
        """
        _input = input.clone()

        _input = super().inverse(_input, module, param)

        padding_size = cls._get_padding_size(module, param)
        if padding_size is not None:
            _input = cls.unpad(_input, padding_size.to(input))

        return _input


class BBoxXYXYApplyInverse(BBoxApplyInverse):
    """Apply and inverse transformations for bounding box tensors.

    This is for transform boxes in the format [xmin, ymin, xmax, ymax].
    """

    apply_func = partial(transform_bbox, mode="xyxy", restore_coordinates=True)

    @classmethod
    def pad(cls, input, padding_size):
        _padding_size = padding_size.to(input)
        for i in range(len(_padding_size)):
            input[i, :, 0::2] += _padding_size[i][0]  # left padding
            input[i, :, 1::2] += _padding_size[i][2]  # top padding
        return input

    @classmethod
    def unpad(cls, input, padding_size):
        _padding_size = padding_size.to(input)
        for i in range(len(_padding_size)):
            input[i, :, 0::2] -= _padding_size[i][0]  # left padding
            input[i, :, 1::2] -= _padding_size[i][2]  # top padding
        return input

    @classmethod
    def apply_trans(
        cls, input: torch.Tensor, label: Optional[torch.Tensor], module: nn.Module, param: ParamItem
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        warnings.warn("BBoxXYXYApplyInverse is no longer maintained. Please use BBoxApplyInverse instead.")
        return super().apply_trans(input, label=label, module=module, param=param)

    @classmethod
    def inverse(cls, input: torch.Tensor, module: nn.Module, param: Optional[ParamItem] = None) -> torch.Tensor:
        warnings.warn("BBoxXYXYApplyInverse is no longer maintained. Please use BBoxApplyInverse instead.")
        return super().inverse(input, module=module, param=param)


class BBoxXYWHApplyInverse(BBoxXYXYApplyInverse):
    """Apply and inverse transformations for bounding box tensors.

    This is for transform boxes in the format [xmin, ymin, width, height].
    """

    apply_func = partial(transform_bbox, mode="xywh", restore_coordinates=True)

    @classmethod
    def pad(cls, input, padding_size):
        _padding_size = padding_size.to(input)
        # pad only xy, not wh
        for i in range(len(_padding_size)):
            input[i, :, 0] += _padding_size[i][0]  # left padding
            input[i, :, 1] += _padding_size[i][2]  # top padding
        return input

    @classmethod
    def unpad(cls, input, padding_size):
        _padding_size = padding_size.to(input)
        # unpad only xy, not wh
        for i in range(len(_padding_size)):
            input[i, :, 0] -= _padding_size[i][0]  # left padding
            input[i, :, 1] -= _padding_size[i][2]  # top padding
        return input


class KeypointsApplyInverse(BBoxApplyInverse):
    """Apply and inverse transformations for keypoints tensors.

    This is for transform keypoints in the format (B, N, 2).
    """

    # Hot fix for the typing mismatching
    apply_func = partial(transform_points)

    @classmethod
    def pad(cls, input: torch.Tensor, padding_size: torch.Tensor) -> torch.Tensor:

        if len(input.shape) not in (2, 3,):
            raise AssertionError(input.shape)

        if len(padding_size.shape) != 2:
            raise AssertionError(padding_size.shape)

        _input = input.clone()

        if input.dim() == 2:
            # B,2 to B,1,2
            _input = _input[:, None]

        _input[..., 0] += padding_size[..., :1]  # left padding
        _input[..., 1] += padding_size[..., 2:3]  # top padding

        if input.dim() == 2:
            _input = _input[:, 0]  # squeeze back

        return _input

    @classmethod
    def unpad(cls, input: torch.Tensor, padding_size: torch.Tensor) -> torch.Tensor:

        if len(input.shape) not in (2, 3,):
            raise AssertionError(input.shape)
        if len(padding_size.shape) != 2:
            raise AssertionError(padding_size.shape)

        _input = input.clone()

        if input.dim() == 2:
            # B,2 to B,1,2
            _input = _input[:, None]

        # unpad only xy, not wh
        _input[..., 0] -= padding_size[..., :1]  # left padding
        _input[..., 1] -= padding_size[..., 2:3]  # top padding

        if input.dim() == 2:
            _input = _input[:, 0]  # squeeze back

        return _input


class ApplyInverse:
    """Apply and inverse transformations for any tensors (e.g. mask, box, points)."""

    @classmethod
    def _get_func_by_key(cls, dcate: Union[str, int, DataKey]) -> Type[ApplyInverseInterface]:
        if DataKey.get(dcate) == DataKey.INPUT:
            return InputApplyInverse
        if DataKey.get(dcate) == DataKey.MASK:
            return MaskApplyInverse
        if DataKey.get(dcate) in [DataKey.BBOX, DataKey.BBOX_XYXY, DataKey.BBOX_XYWH]:
            # We are converting to (B, 4, 2) internally for all formats.
            return BBoxApplyInverse
        if DataKey.get(dcate) in [DataKey.KEYPOINTS]:
            return KeypointsApplyInverse
        raise NotImplementedError(f"input type of {dcate} is not implemented.")

    @classmethod
    def apply_by_key(
        cls,
        input: torch.Tensor,
        label: Optional[torch.Tensor],
        module: nn.Module,
        param: ParamItem,
        dcate: Union[str, int, DataKey] = DataKey.INPUT,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
