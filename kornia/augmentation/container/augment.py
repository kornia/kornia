import warnings
from typing import cast, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from kornia.augmentation.base import _AugmentationBase, GeometricAugmentationBase2D, IntensityAugmentationBase2D
from kornia.constants import DataKey
from kornia.geometry import transform_boxes, transform_points

from .image import ImageSequential
from .patch import PatchSequential


class AugmentationSequential(ImageSequential):
    r"""AugmentationSequential for handling multiple input types like inputs, masks, keypoints at once.

    Args:
        *args (_AugmentationBase): a list of kornia augmentation modules.
        data_keys (List[str]): the input type sequential for applying augmentations.
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
        ... data_keys=["input", "mask", "bbox", "keypoints"],
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
        *args: Union[_AugmentationBase, ImageSequential],
        data_keys: List[Union[str, int, DataKey]] = [DataKey.INPUT],
        same_on_batch: Optional[bool] = None,
        return_transform: Optional[bool] = None,
        keepdim: Optional[bool] = None,
    ) -> None:
        super(AugmentationSequential, self).__init__(
            *args, same_on_batch=same_on_batch, return_transform=return_transform, keepdim=keepdim
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
        self,
        input: torch.Tensor,
        module: nn.Module,
        param: Optional[Dict[str, torch.Tensor]] = None,
        mode: str = "xyxy",
    ) -> torch.Tensor:
        if isinstance(module, GeometricAugmentationBase2D) and param is None:
            raise ValueError(f"Transformation matrix for {module} has not been computed.")
        if isinstance(module, GeometricAugmentationBase2D) and param is not None:
            input = transform_boxes(module.get_transformation_matrix(input, param), input, mode)
        else:
            pass  # No need to update anything
        return input

    def apply_to_keypoints(
        self, input: torch.Tensor, module: nn.Module, param: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        if isinstance(module, GeometricAugmentationBase2D) and param is None:
            raise ValueError(f"Transformation matrix for {module} has not been computed.")
        if isinstance(module, GeometricAugmentationBase2D) and param is not None:
            input = transform_points(module.get_transformation_matrix(input, param), input)
        else:
            pass  # No need to update anything
        return input

    def apply_by_key(
        self,
        input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        module: nn.Module,
        param: Optional[Dict[str, torch.Tensor]] = None,
        dcate: Union[str, int, DataKey] = DataKey.INPUT,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if DataKey.get(dcate) in [DataKey.INPUT]:
            return self.apply_to_input(input, module, param)
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
        params: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
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
        self._params = {}
        params = params if params is not None else {}

        outputs = []
        for input, dcate in zip(args, data_keys):
            for module in list(self.children())[::-1]:
                if isinstance(module, _AugmentationBase):
                    func_name = module.__class__.__name__
                    # Check if a param recorded
                    param = self._params[func_name] if func_name in self._params else None
                    # Check if a param provided. If provided, it will overwrite the recorded ones.
                    param = params[func_name] if func_name in params else param
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
        params: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        data_keys: Optional[List[Union[str, int, DataKey]]] = None,
    ) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]
    ]:
        """Compute multiple tensors simultaneously according to ``self.data_keys``."""
        if data_keys is None:
            self._params = {}
            data_keys = cast(List[Union[str, int, DataKey]], self.data_keys)

        assert len(args) == len(
            data_keys
        ), f"The number of inputs must align with the number of data_keys. Got {len(args)} and {len(data_keys)}."
        params = params if params is not None else {}

        outputs = []
        for input, dcate in zip(args, data_keys):
            for module in self.children():
                if isinstance(module, (ImageSequential,)):
                    # Avoid same naming for sequential
                    func_name = f"{module.__class__.__name__}-{hex(id(module))}"
                else:
                    func_name = module.__class__.__name__
                # Check if a param recorded
                param = self._params[func_name] if func_name in self._params else None
                # Check if a param provided. If provided, it will overwrite the recorded ones.
                param = params[func_name] if func_name in params else param

                if dcate == DataKey.INPUT:
                    input = self.apply_to_input(input, module, param)
                elif isinstance(module, GeometricAugmentationBase2D) and dcate in DataKey:
                    input = self.apply_by_key(input, module, param, dcate)
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
