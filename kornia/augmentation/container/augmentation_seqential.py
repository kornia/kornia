from typing import Tuple, Union, Optional, List, Dict

import torch
import torch.nn as nn

from kornia.augmentation.base import _AugmentationBase
from .sequential import Sequential


class AugmentationSequential(Sequential):
    r"""AugmentationSequential for handling multiple input types like inputs, masks, keypoints at once.

    Args:
        *args (_AugmentationBase): a list of augmentation module.
        input_types (List[str]): the input type sequential for applying augmentations.
            Accepts "input", "mask", "bbox", "keypoints".
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
        >>> aug_list = AugmentationSequential([
        ...     kornia.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...     kornia.augmentation.RandomAffine(360, p=1.0),
        ...     kornia.enhance.Invert()
        ... ],
        ... input_types=["input", "input"],
        ... return_transform=False,
        ... same_on_batch=False,
        ... )
        >>> out = aug_list(input, input)
        >>> out[0].shape, out[1].shape, torch.equal(out[0], out[1])
        (torch.Size([2, 3, 5, 6]), torch.Size([2, 3, 5, 6]), True)

        Reproduce with provided params.
        >>> out2 = aug_list(input, input, params=aug_list._params)
        >>> torch.equal(out[0], out2[0]), torch.equal(out[1], out2[1])
        (True, True)
    """

    def __init__(
        self, augmentation_list: List[_AugmentationBase], input_types: List[str] = ["input"], same_on_batch: Optional[bool] = None,
        return_transform: Optional[bool] = None, keepdim: Optional[bool] = None
    ) -> None:
        super(AugmentationSequential, self).__init__(
            *augmentation_list, same_on_batch=same_on_batch, return_transform=return_transform, keepdim=keepdim)

        SUPPORTED_INPUT_TYPES = ["input", "mask", "bbox", "keypoints"]
        assert all([in_type in SUPPORTED_INPUT_TYPES for in_type in input_types]), \
            f"`input_types` must be in {SUPPORTED_INPUT_TYPES}. Got {input_types}."

        if not (len(set(input_types)) == 1 and input_types[0] == 'input'):
            for aug in augmentation_list:
                if isinstance(aug, _AugmentationBase):
                    raise NotImplementedError(
                        f"For input_type other than `input`, only kornia augmenatations are supported. Got {aug}.")
        self.input_types = input_types

    def apply_to_mask(
        self, input, item: nn.Module, param: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        func_name = item.__class__.__name__
        if isinstance(item, _AugmentationBase) and param is None:
            input = item(input)
            self._params.update({func_name: item._params})
        elif isinstance(item, _AugmentationBase) and param is not None:
            input = item(input, param)
            self._params.update({func_name: param})
        else:
            # In case of return_transform = True
            if isinstance(input, (tuple, list)):
                input = (item(input[0]), input[1])
            else:
                input = item(input)
        return input

    def apply_to_bbox(
        self, input, item: nn.Module, param: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def apply_to_keypoints(
        self, input, item: nn.Module, param: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self, *args: torch.Tensor, params: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    ) -> List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        assert len(args) == len(self.input_types), (
            "The number of inputs must align with the number of input_types. "
            f"Got {len(args)} and {len(self.input_types)}."
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
                if itype == "input":
                    input = self.apply_to_input(input, item, param)
                # elif itype == "mask":
                #     # Waiting for #1013 to specify the geometric and intensity augmentations.
                #     if isinstance(item, _AugmentationBase):  
                #         continue
                #     input = self.apply_to_mask(input, item, param)
                else:
                    raise NotImplementedError(f"input_type {itype} is not implemented.")
            outputs.append(input)

        if len(outputs) == 1:
            return outputs[0]

        return tuple(outputs)
