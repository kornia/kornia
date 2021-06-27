import warnings
from itertools import zip_longest
from typing import cast, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from kornia.augmentation.base import _AugmentationBase, GeometricAugmentationBase2D, IntensityAugmentationBase2D
from kornia.constants import DataKey
from kornia.geometry import transform_boxes, transform_points

from .image import ImageSequential, ParamItem
from .patch import PatchSequential

__all__ = ["AugmentationSequential"]


class AugmentationSequential(ImageSequential):
    r"""AugmentationSequential for handling multiple input types like inputs, masks, keypoints at once.

    .. image:: https://kornia-tutorials.readthedocs.io/en/latest/_images/data_augmentation_sequential_5_1.png
        :width: 49 %
    .. image:: https://kornia-tutorials.readthedocs.io/en/latest/_images/data_augmentation_sequential_7_2.png
        :width: 49 %

    Args:
        *args: a list of kornia augmentation modules.
        data_keys: the input type sequential for applying augmentations.
            Accepts "input", "mask", "bbox", "bbox_xyxy", "bbox_xywh", "keypoints".
        same_on_batch: apply the same transformation across the batch.
            If None, it will not overwrite the function-wise settings.
        return_transform: if ``True`` return the matrix describing the transformation
            applied to each. If None, it will not overwrite the function-wise settings.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
            to the batch form (False). If None, it will not overwrite the function-wise settings.
        random_apply: randomly select a sublist (order agnostic) of args to
            apply transformation.
            If int, a fixed number of transformations will be selected.
            If (a,), x number of transformations (a <= x <= len(args)) will be selected.
            If (a, b), x number of transformations (a <= x <= b) will be selected.
            If True, the whole list of args will be processed as a sequence in a random order.
            If False, the whole list of args will be processed as a sequence in original order.

    Return:
        List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]: the tensor (, and the transformation matrix)
            has been sequentially modified by the args.

    Examples:
        >>> import kornia
        >>> input = torch.randn(2, 3, 5, 6)
        >>> bbox = torch.tensor([[
        ...     [1., 1.],
        ...     [2., 1.],
        ...     [2., 2.],
        ...     [1., 2.],
        ... ]]).expand(2, -1, -1)
        >>> points = torch.tensor([[[1., 1.]]]).expand(2, -1, -1)
        >>> aug_list = AugmentationSequential(
        ...     kornia.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...     kornia.augmentation.RandomAffine(360, p=1.0),
        ... data_keys=["input", "mask", "bbox", "keypoints"],
        ... return_transform=False,
        ... same_on_batch=False,
        ... random_apply=10,
        ... )
        >>> out = aug_list(input, input, bbox, points)
        >>> [o.shape for o in out]
        [torch.Size([2, 3, 5, 6]), torch.Size([2, 3, 5, 6]), torch.Size([2, 4, 2]), torch.Size([2, 1, 2])]
        >>> out_inv = aug_list.inverse(*out)
        >>> [o.shape for o in out_inv]
        [torch.Size([2, 3, 5, 6]), torch.Size([2, 3, 5, 6]), torch.Size([2, 4, 2]), torch.Size([2, 1, 2])]
    """

    def __init__(
        self,
        *args: Union[_AugmentationBase, ImageSequential],
        data_keys: List[Union[str, int, DataKey]] = [DataKey.INPUT],
        same_on_batch: Optional[bool] = None,
        return_transform: Optional[bool] = None,
        keepdim: Optional[bool] = None,
        random_apply: Union[int, bool, Tuple[int, int]] = False,
    ) -> None:
        super(AugmentationSequential, self).__init__(
            *args,
            same_on_batch=same_on_batch,
            return_transform=return_transform,
            keepdim=keepdim,
            random_apply=random_apply,
        )

        self.data_keys = [DataKey.get(inp) for inp in data_keys]

        assert all(
            in_type in DataKey for in_type in self.data_keys
        ), f"`data_keys` must be in {DataKey}. Got {data_keys}."

        if self.data_keys[0] != DataKey.INPUT:
            raise NotImplementedError(f"The first input must be {DataKey.INPUT}.")

        for arg in args:
            if isinstance(arg, PatchSequential) and not arg.is_intensity_only():
                warnings.warn("Geometric transformation detected in PatchSeqeuntial, which would break bbox, mask.")

    def apply_to_mask(self, input: torch.Tensor, module: nn.Module, param: Optional[ParamItem] = None) -> torch.Tensor:
        if param is not None:
            _param = cast(Dict[str, torch.Tensor], param.data)
        else:
            _param = None  # type: ignore

        if isinstance(module, GeometricAugmentationBase2D) and _param is None:
            input = module(input, return_transform=False)
        elif isinstance(module, GeometricAugmentationBase2D) and _param is not None:
            input = module(input, _param, return_transform=False)
        else:
            pass  # No need to update anything
        return input

    def apply_to_bbox(
        self, input: torch.Tensor, module: nn.Module, param: Optional[ParamItem] = None, mode: str = "xyxy"
    ) -> torch.Tensor:
        if param is not None:
            _param = cast(Dict[str, torch.Tensor], param.data)
        else:
            _param = None  # type: ignore

        if isinstance(module, GeometricAugmentationBase2D) and _param is None:
            raise ValueError(f"Transformation matrix for {module} has not been computed.")
        if isinstance(module, GeometricAugmentationBase2D) and _param is not None:
            input = transform_boxes(module.get_transformation_matrix(input, _param), input, mode)
        else:
            pass  # No need to update anything
        return input

    def apply_to_keypoints(
        self, input: torch.Tensor, module: nn.Module, param: Optional[ParamItem] = None
    ) -> torch.Tensor:
        if param is not None:
            _param = cast(Dict[str, torch.Tensor], param.data)
        else:
            _param = None  # type: ignore

        if isinstance(module, GeometricAugmentationBase2D) and _param is None:
            raise ValueError(f"Transformation matrix for {module} has not been computed.")
        if isinstance(module, GeometricAugmentationBase2D) and _param is not None:
            input = transform_points(module.get_transformation_matrix(input, _param), input)
        else:
            pass  # No need to update anything
        return input

    def apply_by_key(
        self,
        input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        module_name: str,
        module: Optional[nn.Module] = None,
        param: Optional[ParamItem] = None,
        dcate: Union[str, int, DataKey] = DataKey.INPUT,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if param is not None:
            assert module_name == param.name
        if module is None:
            # TODO (jian): double check why typing is crashing
            module = self.get_submodule(module_name)  # type: ignore
        if DataKey.get(dcate) in [DataKey.INPUT]:
            return self.apply_to_input(input, module_name, module, param)
        if DataKey.get(dcate) in [DataKey.MASK]:
            if isinstance(input, (tuple,)):
                return (self.apply_to_mask(input[0], module, param), *input[1:])
            return self.apply_to_mask(input, module, param)
        if DataKey.get(dcate) in [DataKey.BBOX, DataKey.BBOX_XYXY]:
            if isinstance(input, (tuple,)):
                return (self.apply_to_bbox(input[0], module, param, mode='xyxy'), *input[1:])
            return self.apply_to_bbox(input, module, param, mode='xyxy')
        if DataKey.get(dcate) in [DataKey.BBOX_XYHW]:
            if isinstance(input, (tuple,)):
                return (self.apply_to_bbox(input[0], module, param, mode='xyhw'), *input[1:])
            return self.apply_to_bbox(input, module, param, mode='xyhw')
        if DataKey.get(dcate) in [DataKey.KEYPOINTS]:
            if isinstance(input, (tuple,)):
                return (self.apply_to_keypoints(input[0], module, param), *input[1:])
            return self.apply_to_keypoints(input, module, param)
        raise NotImplementedError(f"input type of {dcate} is not implemented.")

    def inverse_input(
        self, input: torch.Tensor, module: nn.Module, param: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        if isinstance(module, GeometricAugmentationBase2D) and param is None:
            input = module.inverse(input)
        elif isinstance(module, GeometricAugmentationBase2D) and param is not None:
            input = module.inverse(input, param)
        else:
            pass  # No need to update anything
        return input

    def inverse_bbox(
        self,
        input: torch.Tensor,
        module: nn.Module,
        param: Optional[Dict[str, torch.Tensor]] = None,
        mode: str = "xyxy",
    ) -> torch.Tensor:
        if isinstance(module, GeometricAugmentationBase2D):
            transform = module.compute_inverse_transformation(module.get_transformation_matrix(input, param))
            input = transform_boxes(torch.as_tensor(transform, device=input.device, dtype=input.dtype), input, mode)
        return input

    def inverse_keypoints(
        self, input: torch.Tensor, module: nn.Module, param: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        if isinstance(module, GeometricAugmentationBase2D):
            transform = module.compute_inverse_transformation(module.get_transformation_matrix(input, param))
            input = transform_points(torch.as_tensor(transform, device=input.device, dtype=input.dtype), input)
        return input

    def inverse_by_key(
        self,
        input: torch.Tensor,
        module: nn.Module,
        param: Optional[Dict[str, torch.Tensor]] = None,
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

    def inverse(
        self,
        *args: torch.Tensor,
        params: Optional[List[ParamItem]] = None,
        data_keys: Optional[List[Union[str, int, DataKey]]] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Reverse the transformation applied.

        Number of input tensors must align with the number of``data_keys``. If ``data_keys``
        is not set, use ``self.data_keys`` by default.
        """
        if data_keys is None:
            data_keys = cast(List[Union[str, int, DataKey]], self.data_keys)
        assert len(args) == len(data_keys), (
            "The number of inputs must align with the number of data_keys, " f"Got {len(args)} and {len(data_keys)}."
        )
        if params is None and self._params is None:
            raise ValueError("No parameters avaliable for inversing.")
        else:
            params = self._params

        outputs = []
        for input, dcate in zip(args, data_keys):
            if dcate == DataKey.INPUT and isinstance(input, (tuple, list)):
                input, _ = input  # ignore the transformation matrix whilst inverse
            for (name, module), param in zip_longest(list(self.get_forward_sequence(params))[::-1], params[::-1]):
                if isinstance(module, _AugmentationBase):
                    # Check if a param recorded
                    param = self._params[name] if name in self._params else None
                    # Check if a param provided. If provided, it will overwrite the recorded ones.
                    param = params[name] if name in params else param
                else:
                    param = None
                if isinstance(module, GeometricAugmentationBase2D) and dcate in DataKey:
                    # Waiting for #1013 to specify the geometric and intensity augmentations.
                    input = self.inverse_by_key(input, module, param, dcate)
                elif isinstance(module, IntensityAugmentationBase2D) and dcate in DataKey:
                    pass  # Do nothing
                elif isinstance(module, PatchSequential) and module.is_intensity_only() and dcate in DataKey:
                    pass  # Do nothing
                else:
                    raise NotImplementedError(f"data_key {dcate} is not implemented for {module}.")
            outputs.append(input)

        if len(outputs) == 1:
            return outputs[0]

        return outputs

    def forward(  # type: ignore
        self,
        *args: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        params: Optional[List[ParamItem]] = None,
        data_keys: Optional[List[Union[str, int, DataKey]]] = None,
    ) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]
    ]:
        """Compute multiple tensors simultaneously according to ``self.data_keys``."""
        if data_keys is None:
            data_keys = cast(List[Union[str, int, DataKey]], self.data_keys)

        assert len(args) == len(
            data_keys
        ), f"The number of inputs must align with the number of data_keys. Got {len(args)} and {len(data_keys)}."

        outputs = []
        self._params = []
        named_modules = list(self.get_forward_sequence(params))
        for input, dcate in zip(args, data_keys):
            # use the parameter if the first round has been finished.
            params = self._params if params is None or len(params) == 0 else params
            for param, (name, module) in zip_longest(params, named_modules):
                if dcate == DataKey.INPUT:
                    input = self.apply_to_input(input, name, module, param)
                elif isinstance(module, GeometricAugmentationBase2D) and dcate in DataKey:
                    input = self.apply_by_key(input, name, module, param, dcate)
                elif isinstance(module, IntensityAugmentationBase2D) and dcate in DataKey:
                    pass  # Do nothing
                elif isinstance(module, PatchSequential) and module.is_intensity_only() and dcate in DataKey:
                    pass  # Do nothing
                else:
                    raise NotImplementedError(f"data_key {dcate} is not implemented for {module}.")
            outputs.append(input)
        if len(outputs) == 1:
            return outputs[0]

        return outputs
