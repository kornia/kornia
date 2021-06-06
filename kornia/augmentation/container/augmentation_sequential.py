from typing import Tuple, Union, Optional, List, Dict, cast

import torch
import torch.nn as nn

from kornia.geometry import transform_points, transform_boxes
from kornia.augmentation.base import _AugmentationBase, IntensityAugmentationBase2D, GeometricAugmentationBase2D
from kornia.constants import DataCategory
from .sequential import Sequential


class AugmentationSequential(Sequential):
    r"""AugmentationSequential for handling multiple input types like inputs, masks, keypoints at once.

    Args:
        *args (_AugmentationBase): a list of kornia augmentation modules.
        data_cates (List[str]): the input type sequential for applying augmentations.
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
        >>> aug_list = AugmentationSequential(
        ...     kornia.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...     kornia.augmentation.RandomAffine(360, p=1.0),
        ... data_cates=["input", "mask", "bbox", "keypoints"],
        ... return_transform=False,
        ... same_on_batch=False,
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
        *args: _AugmentationBase,
        data_cates: List[Union[str, int, DataCategory]] = [DataCategory.INPUT],
        same_on_batch: Optional[bool] = None,
        return_transform: Optional[bool] = None,
        keepdim: Optional[bool] = None,
    ) -> None:
        super(AugmentationSequential, self).__init__(
            *args, same_on_batch=same_on_batch, return_transform=return_transform, keepdim=keepdim
        )

        self.data_cates = [DataCategory.get(inp) for inp in data_cates]

        assert all(
            [in_type in DataCategory for in_type in self.data_cates]
        ), f"`data_cates` must be in {DataCategory}. Got {data_cates}."

        if self.data_cates[0] != DataCategory.INPUT:
            raise NotImplementedError(f"The first input must be {DataCategory.INPUT}.")

    def apply_to_mask(
        self, input: torch.Tensor, module: nn.Module, param: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        if isinstance(module, GeometricAugmentationBase2D) and param is None:
            input = module(input, return_transform=False)
        elif isinstance(module, GeometricAugmentationBase2D) and param is not None:
            input = module(input, param, return_transform=False)
        else:
            pass  # No need to update anything
        return input

    def apply_to_bbox(
        self, input: torch.Tensor, module: nn.Module, param: Optional[Dict[str, torch.Tensor]] = None,
        mode: str = "xyxy"
    ) -> torch.Tensor:
        if isinstance(module, GeometricAugmentationBase2D) and param is None:
            raise ValueError(f"Transformation matrix for {module} has not been computed.")
        elif isinstance(module, GeometricAugmentationBase2D) and param is not None:
            input = transform_boxes(
                module.get_transformation_matrix(input, param),
                input,
                mode,
            )
        else:
            pass  # No need to update anything
        return input

    def apply_to_keypoints(
        self, input: torch.Tensor, module: nn.Module, param: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        if isinstance(module, GeometricAugmentationBase2D) and param is None:
            raise ValueError(f"Transformation matrix for {module} has not been computed.")
        elif isinstance(module, GeometricAugmentationBase2D) and param is not None:
            input = transform_points(
                module.get_transformation_matrix(input, param),
                input,
            )
        else:
            pass  # No need to update anything
        return input

    def apply_by_data_cate(
        self,
        input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        module: nn.Module,
        param: Optional[Dict[str, torch.Tensor]] = None,
        dcate: Union[str, int, DataCategory] = DataCategory.INPUT,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if dcate in [DataCategory.INPUT]:
            return self.apply_to_input(input, module, param)
        if dcate in [DataCategory.MASK]:
            if isinstance(input, (tuple,)):
                return (self.apply_to_mask(input[0], module, param), *input[1:])
            return self.apply_to_mask(input, module, param)
        if dcate in [DataCategory.BBOX, DataCategory.BBOX_XYXY]:
            if isinstance(input, (tuple,)):
                return (self.apply_to_bbox(input[0], module, param, mode='xyxy'), *input[1:])
            return self.apply_to_bbox(input, module, param, mode='xyxy')
        if dcate in [DataCategory.BBOX_XYHW]:
            if isinstance(input, (tuple,)):
                return (self.apply_to_bbox(input[0], module, param, mode='xyhw'), *input[1:])
            return self.apply_to_bbox(input, module, param, mode='xyhw')
        if dcate in [DataCategory.KEYPOINTS]:
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
        self, input: torch.Tensor, module: nn.Module, param: Optional[Dict[str, torch.Tensor]] = None,
        mode: str = "xyxy"
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

    def inverse_by_data_category(
        self,
        input: torch.Tensor,
        module: nn.Module,
        param: Optional[Dict[str, torch.Tensor]] = None,
        dcate: Union[str, int, DataCategory] = DataCategory.INPUT,
    ) -> torch.Tensor:
        if dcate in [DataCategory.INPUT, DataCategory.MASK]:
            return self.inverse_input(input, module, param)
        if dcate in [DataCategory.BBOX, DataCategory.BBOX_XYXY]:
            return self.inverse_bbox(input, module, param, mode='xyxy')
        if dcate in [DataCategory.BBOX_XYHW]:
            return self.inverse_bbox(input, module, param, mode='xyhw')
        if dcate in [DataCategory.KEYPOINTS]:
            return self.inverse_keypoints(input, module, param)
        raise NotImplementedError(f"input type of {dcate} is not implemented.")

    def inverse(
        self,
        *args: torch.Tensor,
        params: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        data_cates: Optional[List[Union[str, int, DataCategory]]] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Reverse the transformation applied.

        Number of input tensors must align with the number of``data_cates``. If ``data_cates``
        is not set, use ``self.data_cates`` by default.
        """
        if data_cates is None:
            data_cates = cast(List[Union[str, int, DataCategory]], self.data_cates)
        assert len(args) == len(data_cates), (
            "The number of inputs must align with the number of data_cates, "
            f"Got {len(args)} and {len(data_cates)}."
        )
        self._params = {}
        params = params if params is not None else {}

        outputs = []
        for input, dcate in zip(args, data_cates):
            for module in list(self.children())[::-1]:
                if isinstance(module, _AugmentationBase):
                    func_name = module.__class__.__name__
                    # Check if a param recorded
                    param = self._params[func_name] if func_name in self._params else None
                    # Check if a param provided. If provided, it will overwrite the recorded ones.
                    param = params[func_name] if func_name in params else param
                else:
                    param = None
                if isinstance(module, GeometricAugmentationBase2D) and dcate in DataCategory:
                    # Waiting for #1013 to specify the geometric and intensity augmentations.
                    input = self.inverse_by_data_category(input, module, param, dcate)
                elif isinstance(module, IntensityAugmentationBase2D) and dcate in DataCategory:
                    pass  # Do nothing
                else:
                    raise NotImplementedError(f"data_cate {dcate} is not implemented for {module}.")
            outputs.append(input)

        if len(outputs) == 1:
            return outputs[0]

        return outputs

    def forward(  # type: ignore
        self,
        *args: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        params: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]
    ]:
        """Compute multiple tensors simultaneously according to ``self.data_cates``."""
        assert len(args) == len(self.data_cates) and self.data_cates[0] in [DataCategory.INPUT], (
            "The number of inputs must align with the number of data_cates, "
            f"and the first element must be input. Got {len(args)} and {len(self.data_cates)}."
        )
        self._params = {}
        params = params if params is not None else {}

        outputs = []
        for input, dcate in zip(args, self.data_cates):
            for module in self.children():
                func_name = module.__class__.__name__
                # Check if a param recorded
                param = self._params[func_name] if func_name in self._params else None
                # Check if a param provided. If provided, it will overwrite the recorded ones.
                param = params[func_name] if func_name in params else param

                if dcate == DataCategory.INPUT:
                    input = self.apply_to_input(input, module, param)
                elif isinstance(module, GeometricAugmentationBase2D) and dcate in DataCategory:
                    input = self.apply_by_data_cate(input, module, param, dcate)
                elif isinstance(module, IntensityAugmentationBase2D) and dcate in DataCategory:
                    pass  # Do nothing
                else:
                    raise NotImplementedError(f"data_cate {dcate} is not implemented for {module}.")
            outputs.append(input)
        if len(outputs) == 1:
            return outputs[0]

        return outputs
