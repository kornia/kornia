import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type, Union, cast

import torch

import kornia  # lazy loading for circular dependencies
from kornia.augmentation import GeometricAugmentationBase2D, MixAugmentationBaseV2, RandomCrop, RandomErasing
from kornia.augmentation.base import _AugmentationBase
from kornia.augmentation.container.base import ParamItem
from kornia.augmentation.utils import override_parameters
from kornia.constants import DataKey
from kornia.core import Module, Tensor, as_tensor
from kornia.geometry.bbox import transform_bbox
from kornia.geometry.linalg import transform_points
from kornia.testing import KORNIA_UNWRAP
from kornia.utils.helpers import _torch_inverse_cast


def _get_geometric_only_param(module: 'kornia.augmentation.ImageSequential', param: List[ParamItem]) -> List[ParamItem]:
    named_modules: Iterator[Tuple[str, Module]] = module.get_forward_sequence(param)

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
        cls, input: Tensor, label: Optional[Tensor], module: Module, param: ParamItem, extra_args: Dict[str, Any] = {}
    ) -> Tuple[Tensor, Optional[Tensor]]:
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
        cls, input: Tensor, module: Module, param: Optional[ParamItem] = None, extra_args: Dict[str, Any] = {}
    ) -> Tensor:
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

    apply_func: Callable[[Tensor, Tensor], Tensor]

    @classmethod
    def apply_trans(
        cls, input: Tensor, label: Optional[Tensor], module: Module, param: ParamItem, extra_args: Dict[str, Any] = {}
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Apply a transformation with respect to the parameters.

        Args:
            input: the input tensor.
            label: the optional label tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
        """
        mat: Optional[Tensor]
        if hasattr(module, "transform_matrix") and module.transform_matrix is not None:
            mat = cast(Tensor, module.transform_matrix)
        else:
            mat = cls._get_transformation(input, module, param, extra_args=extra_args)
        mat = as_tensor(mat, device=input.device, dtype=input.dtype)

        to_apply = None
        if isinstance(module, _AugmentationBase) and isinstance(param.data, dict):
            to_apply = param.data['batch_prob']
        elif isinstance(module, kornia.augmentation.ImageSequential):
            to_apply = torch.ones(input.shape[0], device=input.device, dtype=input.dtype).bool()

        # If any inputs need to be transformed.
        if mat is not None and to_apply is not None and to_apply.sum() != 0 and input.numel() > 0:
            input[to_apply] = cls.apply_func(mat[to_apply], input[to_apply])

        return input, label

    @classmethod
    def inverse(
        cls, input: Tensor, module: Module, param: Optional[ParamItem] = None, extra_args: Dict[str, Any] = {}
    ) -> Tensor:
        """Inverse a transformation with respect to the parameters.

        Args:
            input: the input tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
        """
        mat: Optional[Tensor]
        if hasattr(module, "transform_matrix") and module.transform_matrix is not None:
            mat = cast(Tensor, module.transform_matrix)
        else:
            mat = cls._get_transformation(input, module, param, extra_args=extra_args)
        mat = as_tensor(mat, device=input.device, dtype=input.dtype)

        if mat is not None:
            transform: Tensor = cls._get_inverse_transformation(mat)
            input = cls.apply_func(as_tensor(transform, device=input.device, dtype=input.dtype), input)
        return input

    @classmethod
    def _get_transformation(
        cls, input: Tensor, module: Module, maybe_param: Optional[ParamItem] = None, extra_args: Dict[str, Any] = {}
    ) -> Optional[Tensor]:

        if (
            isinstance(module, (GeometricAugmentationBase2D, kornia.augmentation.ImageSequential))
            and maybe_param is None
        ):
            raise ValueError(f"Parameters of transformation matrix for {module} has not been computed.")

        maybe_mat: Optional[Tensor] = None
        param = KORNIA_UNWRAP(maybe_param, ParamItem)
        if isinstance(module, GeometricAugmentationBase2D):
            param_data = KORNIA_UNWRAP(param.data, Dict[str, Tensor])
            flags = override_parameters(module.flags, extra_args)
            maybe_mat = module.get_transformation_matrix(input, param_data, flags=flags)
        elif isinstance(module, kornia.augmentation.ImageSequential) and not module.is_intensity_only():
            param_data = KORNIA_UNWRAP(param.data, List[ParamItem])
            maybe_mat = module.get_transformation_matrix(input, param_data, recompute=False, extra_args=extra_args)
        else:
            pass  # No need to update anything
        return maybe_mat

    @classmethod
    def _get_inverse_transformation(cls, transform: Tensor) -> Tensor:
        return _torch_inverse_cast(transform)


class InputApplyInverse(ApplyInverseImpl):
    """Apply and inverse transformations for (image) input tensors."""

    data_key = DataKey.INPUT

    @classmethod
    def apply_trans(
        cls, input: Tensor, label: Optional[Tensor], module: Module, param: ParamItem, extra_args: Dict[str, Any] = {}
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Apply a transformation with respect to the parameters.

        Args:
            input: the input tensor.
            label: the optional label tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
        """
        if isinstance(module, (_AugmentationBase, MixAugmentationBaseV2)):
            input = module(input, params=param.data, **extra_args)
        elif isinstance(module, kornia.augmentation.ImageSequential):
            temp = module.apply_inverse_func
            temp2 = module.return_label
            module.apply_inverse_func = InputApplyInverse
            module.return_label = True
            if isinstance(module, kornia.augmentation.AugmentationSequential):
                input, label = module(input, label=label, params=param.data, data_keys=[cls.data_key])
            else:
                input, label = module(input, label=label, params=param.data, extra_args=extra_args)
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
    def inverse(
        cls, input: Tensor, module: Module, param: Optional[ParamItem] = None, extra_args: Dict[str, Any] = {}
    ) -> Tensor:
        """Inverse a transformation with respect to the parameters.

        Args:
            input: the input tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
        """
        if isinstance(module, GeometricAugmentationBase2D):
            if param is None:
                _params_geo = None
            elif isinstance(param, ParamItem) and isinstance(param.data, dict):
                _params_geo = param.data
            else:
                raise TypeError(f'Expected param (ParamItem.data) be a dictionary. Gotcha {type(param.data)}')

            input = module.inverse(input, params=_params_geo, extra_args=extra_args)

        elif isinstance(module, kornia.augmentation.ImageSequential):
            temp = module.apply_inverse_func
            module.apply_inverse_func = InputApplyInverse
            if param is None:
                _params = None
            elif isinstance(param, ParamItem) and isinstance(param.data, list):
                _params = param.data
            else:
                raise TypeError(f'Expected param (ParamItem.data) be a list. Gotcha {type(param.data)}')

            if isinstance(module, kornia.augmentation.AugmentationSequential):
                _ret = module.inverse(input, params=_params, data_keys=[cls.data_key])
                if isinstance(_ret, Tensor):
                    input = _ret
                else:
                    raise TypeError(
                        f'The return of the method inverse from {module} should be a Tensor. Gotcha {type(_ret)}'
                    )
            else:
                input = module.inverse(input, params=_params, extra_args=extra_args)

            module.apply_inverse_func = temp
        return input


class MaskApplyInverse(ApplyInverseImpl):
    """Apply and inverse transformations for mask tensors."""

    data_key = DataKey.MASK

    @classmethod
    def make_input_only_sequential(cls, module: 'kornia.augmentation.ImageSequential') -> Callable[..., torch.Tensor]:
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
        cls,
        input: Tensor,
        label: Optional[Tensor],
        module: Module,
        param: Optional[ParamItem] = None,
        extra_args: Dict[str, Any] = {},
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Apply a transformation with respect to the parameters.

        Args:
            input: the input tensor.
            label: the optional label tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
        """
        if isinstance(module, (GeometricAugmentationBase2D, RandomErasing)):
            if isinstance(param, ParamItem) and isinstance(param.data, dict):
                _param = param.data.copy()
                # TODO: Parametrize value to pad with across the board for different keys
                if 'values' in _param:
                    _param['values'] = torch.zeros_like(_param['values'])  # Always pad with zeros
            elif param is None:
                _param = None
            else:
                raise TypeError(f'Expected param be None or ParamItem.data as a dict. Gotcha {type(_param)}')

            input = module(input, params=_param, **extra_args)

        elif isinstance(module, kornia.augmentation.ImageSequential) and not module.is_intensity_only():
            if param is None:
                geo_param = None
            elif isinstance(param, ParamItem) and isinstance(param.data, list):
                geo_param = _get_geometric_only_param(module, param.data)
            else:
                raise TypeError(f'Expected param be None or ParamItem.data as a list. Gotcha {type(geo_param)}')

            temp = module.apply_inverse_func
            module.apply_inverse_func = MaskApplyInverse
            if isinstance(module, kornia.augmentation.AugmentationSequential):
                input = cls.make_input_only_sequential(module)(
                    input, label=None, params=geo_param, data_keys=[cls.data_key]
                )
            else:
                input = cls.make_input_only_sequential(module)(
                    input, label=None, params=geo_param, extra_args=extra_args
                )
            module.apply_inverse_func = temp
        else:
            pass  # No need to update anything
        return input, label

    @classmethod
    def inverse(
        cls, input: Tensor, module: Module, param: Optional[ParamItem] = None, extra_args: Dict[str, Any] = {}
    ) -> Tensor:
        """Inverse a transformation with respect to the parameters.

        Args:
            input: the input tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
        """

        if isinstance(module, GeometricAugmentationBase2D):
            if param is None:
                _params_geo = None
            elif isinstance(param, ParamItem) and isinstance(param.data, dict):
                _params_geo = param.data
            else:
                raise TypeError(f'Expected param (ParamItem.data) be a dict. Gotcha {type(param.data)}')

            input = module.inverse(input, params=_params_geo, **extra_args)

        elif isinstance(module, kornia.augmentation.ImageSequential):
            temp = module.apply_inverse_func
            module.apply_inverse_func = MaskApplyInverse

            if param is None:
                _params = None
            elif isinstance(param, ParamItem) and isinstance(param.data, list):
                _params = param.data
            else:
                raise TypeError(f'Expected param (ParamItem.data) be a list. Gotcha {type(param.data)}')

            if isinstance(module, kornia.augmentation.AugmentationSequential):
                _ret = module.inverse(input, params=_params, data_keys=[cls.data_key])
                if isinstance(_ret, Tensor):
                    input = _ret
                else:
                    raise TypeError(
                        f'The return of the method inverse from {module} should be a Tensor. Gotcha {type(_ret)}'
                    )
            else:
                input = module.inverse(input, params=_params, extra_args=extra_args)

            module.apply_inverse_func = temp
        return input


class BBoxApplyInverse(ApplyInverseImpl):
    """Apply and inverse transformations for bounding box tensors.

    This is for transform boxes in the format (B, N, 4, 2).
    """

    @classmethod
    def _get_padding_size(cls, module: Module, param: Optional[ParamItem]) -> Optional[Tensor]:
        if isinstance(module, RandomCrop) and param is not None and isinstance(param.data, dict):
            return param.data["padding_size"]
        elif param is not None and isinstance(param.data, list):
            return next((p_.data['padding_size'] for p_ in param.data if 'padding_size' in p_.data), None)
        return None

    @classmethod
    def pad(cls, input: Tensor, padding_size: Tensor) -> Tensor:
        """
        Args:
            input: (B, N, 4, 2)
            padding_size: (B, 4)
        """
        if len(input.shape) not in (3, 4):
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
    def unpad(cls, input: Tensor, padding_size: Tensor) -> Tensor:
        """
        Args:
            input: (B, N, 4, 2)
            padding_size: (B, 4)
        """
        if len(input.shape) not in (3, 4):
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
        cls, input: Tensor, label: Optional[Tensor], module: Module, param: ParamItem, extra_args: Dict[str, Any] = {}
    ) -> Tuple[Tensor, Optional[Tensor]]:
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

        _input, label = super().apply_trans(_input, label, module, param, extra_args=extra_args)

        # TODO: Filter/crop boxes outside crop (with negative or larger than crop size coords)?

        return _input, label

    @classmethod
    def inverse(
        cls, input: Tensor, module: Module, param: Optional[ParamItem] = None, extra_args: Dict[str, Any] = {}
    ) -> Tensor:
        """Inverse a transformation with respect to the parameters.

        Args:
            input: the input tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
        """
        _input = input.clone()

        _input = super().inverse(_input, module, param, extra_args=extra_args)

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
        cls, input: Tensor, label: Optional[Tensor], module: Module, param: ParamItem, extra_args: Dict[str, Any] = {}
    ) -> Tuple[Tensor, Optional[Tensor]]:
        warnings.warn("BBoxXYXYApplyInverse is no longer maintained. Please use BBoxApplyInverse instead.")
        return super().apply_trans(input, label=label, module=module, param=param, extra_args=extra_args)

    @classmethod
    def inverse(
        cls, input: Tensor, module: Module, param: Optional[ParamItem] = None, extra_args: Dict[str, Any] = {}
    ) -> Tensor:
        warnings.warn("BBoxXYXYApplyInverse is no longer maintained. Please use BBoxApplyInverse instead.")
        return super().inverse(input, module=module, param=param, extra_args=extra_args)


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
    def pad(cls, input: Tensor, padding_size: Tensor) -> Tensor:
        if len(input.shape) not in (2, 3):
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
    def unpad(cls, input: Tensor, padding_size: Tensor) -> Tensor:
        if len(input.shape) not in (2, 3):
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
        input: Tensor,
        label: Optional[Tensor],
        module: Module,
        param: ParamItem,
        dcate: Union[str, int, DataKey] = DataKey.INPUT,
        extra_args: Dict[str, Any] = {},
    ) -> Tuple[Tensor, Optional[Tensor]]:
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
            return (func.apply_trans(input[0], label, module, param, extra_args), *input[1:])
        return func.apply_trans(input, label, module=module, param=param, extra_args=extra_args)

    @classmethod
    def inverse_by_key(
        cls,
        input: Tensor,
        module: Module,
        param: Optional[ParamItem] = None,
        dcate: Union[str, int, DataKey] = DataKey.INPUT,
        extra_args: Dict[str, Any] = {},
    ) -> Tensor:
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
        return func.inverse(input, module, param, extra_args=extra_args)
