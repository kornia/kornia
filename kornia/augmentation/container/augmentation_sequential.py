from typing import Tuple, Union, Optional, List, Dict

import torch
from torch.autograd.grad_mode import F
import torch.nn as nn

from kornia.geometry import transform_points, transform_boxes
from kornia.augmentation.base import (
    _AugmentationBase,
    IntensityAugmentationBase2D,
    GeometricAugmentationBase2D,
)
from kornia.constants import InputType
from .sequential import Sequential


class AugmentationSequential(Sequential):
    r"""AugmentationSequential for handling multiple input types like inputs, masks, keypoints at once.

    Args:
        *args (_AugmentationBase): a list of augmentation module.
        input_types (List[str]): the input type sequential for applying augmentations.
            Accepts "input", "mask", "bbox", "bbox_xyxy", "bbox_xywh", "keypoints".
        same_on_batch (bool, optional): apply the same transformation across the batch.
            If None, it will not overwrite the function-wise settings. Default: None.
        return_transform (bool, optional): if ``True`` return the matrix describing the transformation
            applied to each. If None, it will not overwrite the function-wise settings. Default: None.
        keepdim (bool, optional): whether to keep the output shape the same as input (True) or broadcast it
            to the batch form (False). If None, it will not overwrite the function-wise settings. Default: None.

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
        >>> aug_list = AugmentationSequential([
        ...         kornia.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...         kornia.augmentation.RandomAffine(360, p=1.0)
        ...     ],
        ...     input_types=["input", "mask", "bbox", "keypoints"],
        ...     return_transform=False,
        ...     same_on_batch=False,
        ... )
        >>> out = aug_list(input, input, bbox, points)
        >>> [o.shape for o in out]
        [torch.Size([2, 3, 5, 6]), torch.Size([2, 3, 5, 6]), torch.Size([2, 4, 2]), torch.Size([2, 1, 2])]
        >>> out_inv = aug_list.inverse(*out)
        >>> [o.shape for o in out_inv]
        [torch.Size([2, 3, 5, 6]), torch.Size([2, 3, 5, 6]), torch.Size([2, 4, 2]), torch.Size([2, 1, 2])]
    """

    def __init__(
        self, augmentation_list: List[_AugmentationBase], input_types: List[Union[str, int, InputType]] = [InputType.INPUT],
        same_on_batch: Optional[bool] = None,
        return_transform: Optional[bool] = None, keepdim: Optional[bool] = None
    ) -> None:
        super(AugmentationSequential, self).__init__(
            *augmentation_list, same_on_batch=same_on_batch, return_transform=return_transform, keepdim=keepdim)

        self.input_types = [InputType.get(inp) for inp in input_types]

        assert all([in_type in InputType for in_type in self.input_types]), \
            f"`input_types` must be in {InputType}. Got {input_types}."

        if self.input_types[0] != InputType.INPUT:
            raise NotImplementedError(f"The first input must be {InputType.INPUT}.")

    def apply_to_mask(
        self, input: torch.Tensor, item: nn.Module, param: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        if isinstance(item, GeometricAugmentationBase2D) and param is None:
            input = item(input)
        elif isinstance(item, GeometricAugmentationBase2D) and param is not None:
            input = item(input, param)
        else:
            pass  # No need to update anything
        return input

    def apply_to_bbox(
        self, input: torch.Tensor, item: nn.Module, param: Optional[Dict[str, torch.Tensor]] = None,
        mode: str = "xyxy"
    ) -> torch.Tensor:
        if isinstance(item, GeometricAugmentationBase2D) and param is None:
            raise ValueError(f"Transformation matrix for {item} has not been computed.")
        elif isinstance(item, GeometricAugmentationBase2D) and param is not None:
            input = transform_boxes(
                torch.as_tensor(item._transform_matrix, device=input.device, dtype=input.dtype), input, mode)
        else:
            pass  # No need to update anything
        return input

    def apply_to_keypoints(
        self, input: torch.Tensor, item: nn.Module, param: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        if isinstance(item, GeometricAugmentationBase2D) and param is None:
            raise ValueError(f"Transformation matrix for {item} has not been computed.")
        elif isinstance(item, GeometricAugmentationBase2D) and param is not None:
            input = transform_points(
                torch.as_tensor(item._transform_matrix, device=input.device, dtype=input.dtype), input)
        else:
            pass  # No need to update anything
        return input

    def apply_by_input_type(
        self, input: torch.Tensor, item: nn.Module, param: Optional[Dict[str, torch.Tensor]] = None,
        itype: Union[str, int, InputType] = InputType.INPUT
    ) -> torch.Tensor:
        if itype in [InputType.INPUT]:
            return self.apply_to_input(input, item, param)
        if itype in [InputType.MASK]:
            return self.apply_to_mask(input, item, param)
        if itype in [InputType.BBOX, InputType.BBOX_XYXY]:
            return self.apply_to_bbox(input, item, param, mode='xyxy')
        if itype in [InputType.BBOX_XYHW]:
            return self.apply_to_bbox(input, item, param, mode='xyhw')
        if itype in [InputType.KEYPOINTS]:
            return self.apply_to_keypoints(input, item, param)
        raise NotImplementedError(f"input type of {itype} is not implemented.")

    def inverse_input(
        self, input: torch.Tensor, item: nn.Module, param: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if isinstance(item, GeometricAugmentationBase2D) and param is None:
            input = item.inverse(input)
        elif isinstance(item, GeometricAugmentationBase2D) and param is not None:
            input = item.inverse(input, param)
        else:
            pass  # No need to update anything
        return input

    def inverse_bbox(
        self, input: torch.Tensor, item: nn.Module, mode: str = "xyxy"
    ) -> torch.Tensor:
        if isinstance(item, GeometricAugmentationBase2D):
            transform = item.compute_inverse_transformation(item._transform_matrix)
            input = transform_boxes(torch.as_tensor(transform, device=input.device, dtype=input.dtype), input, mode)
        return input

    def inverse_keypoints(
        self, input: torch.Tensor, item: nn.Module
    ) -> torch.Tensor:
        if isinstance(item, GeometricAugmentationBase2D):
            transform = item.compute_inverse_transformation(item._transform_matrix)
            input = transform_points(torch.as_tensor(transform, device=input.device, dtype=input.dtype), input)
        return input

    def inverse_by_input_type(
        self, input: torch.Tensor, item: nn.Module, param: Optional[Dict[str, torch.Tensor]] = None,
        itype: Union[str, int, InputType] = InputType.INPUT
    ) -> torch.Tensor:
        if itype in [InputType.INPUT, InputType.MASK]:
            return self.inverse_input(input, item, param)
        if itype in [InputType.BBOX, InputType.BBOX_XYXY]:
            return self.inverse_bbox(input, item, mode='xyxy')
        if itype in [InputType.BBOX_XYHW]:
            return self.inverse_bbox(input, item, mode='xyhw')
        if itype in [InputType.KEYPOINTS]:
            return self.inverse_keypoints(input, item)
        raise NotImplementedError(f"input type of {itype} is not implemented.")

    def inverse(
        self, *args: torch.Tensor, params: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        input_types: Optional[List[Union[str, int, InputType]]] = None
    ) -> Union[torch.Tensor, List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]]:
        """Reverse the transformation applied.

        Number of input tensors must align with the number of``input_types``. If ``input_types``
        is not set, use ``self.input_types`` by default.
        """
        if input_types is None:
            input_types = self.input_types
        assert len(args) == len(input_types), (
            "The number of inputs must align with the number of input_types, "
            f"Got {len(args)} and {len(input_types)}."
        )
        self._params = {}
        params = params if params is not None else {}

        outputs = []
        for input, itype in zip(args, input_types):
            for item in list(self.children())[::-1]:
                if isinstance(item, _AugmentationBase):
                    func_name = item.__class__.__name__
                    # Check if a param recorded
                    param = self._params[func_name] if func_name in self._params else None
                    # Check if a param provided. If provided, it will overwrite the recorded ones.
                    param = params[func_name] if func_name in params else param
                else:
                    param = None
                if isinstance(item, GeometricAugmentationBase2D) and itype in InputType:
                    # Waiting for #1013 to specify the geometric and intensity augmentations.
                    input = self.inverse_by_input_type(input, item, param, itype)
                elif isinstance(item, IntensityAugmentationBase2D) and itype in InputType:
                    pass  # Do nothing
                else:
                    raise NotImplementedError(f"input_type {itype} is not implemented for {item}.")
            outputs.append(input)

        if len(outputs) == 1:
            return outputs[0]

        return tuple(outputs)

    def forward(
        self, *args: torch.Tensor, params: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    ) -> Union[torch.Tensor, List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]]:
        """Compute multiple tensors simultaneously according to ``self.input_types``.
        """
        assert len(args) == len(self.input_types) and self.input_types[0] in [InputType.INPUT], (
            "The number of inputs must align with the number of input_types, "
            f"and the first element must be input. Got {len(args)} and {len(self.input_types)}."
        )
        self._params = {}
        params = params if params is not None else {}

        outputs = []
        for input, itype in zip(args, self.input_types):
            for item in self.children():
                if isinstance(item, _AugmentationBase):
                    func_name = item.__class__.__name__
                    # Check if a param recorded
                    param = self._params[func_name] if func_name in self._params else None
                    # Check if a param provided. If provided, it will overwrite the recorded ones.
                    param = params[func_name] if func_name in params else param
                else:
                    param = None
                if itype == InputType.INPUT:
                    input = self.apply_to_input(input, item, param)
                elif isinstance(item, GeometricAugmentationBase2D) and itype in InputType:
                    input = self.apply_by_input_type(input, item, param, itype)
                elif isinstance(item, IntensityAugmentationBase2D) and itype in InputType:
                    pass  # Do nothing
                else:
                    raise NotImplementedError(f"input_type {itype} is not implemented for {item}.")
            outputs.append(input)
        if len(outputs) == 1:
            return outputs[0]

        return tuple(outputs)
