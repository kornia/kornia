from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from kornia.augmentation.base import _AugmentationBase


class Sequential(nn.Sequential):
    r"""Sequential for creating kornia image processing pipeline.

    Args:
        *args (nn.Module): a list of kornia augmentation and image operation modules.
        same_on_batch (bool, optional): apply the same transformation across the batch.
            If None, it will not overwrite the function-wise settings. Default: None.
        return_transform (bool, optional): if ``True`` return the matrix describing the transformation
            applied to each. If None, it will not overwrite the function-wise settings. Default: None.
        keepdim (bool, optional): whether to keep the output shape the same as input (True) or broadcast it
            to the batch form (False). If None, it will not overwrite the function-wise settings. Default: None.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: the tensor (, and the transformation matrix)
            has been sequentially modified by the args.

    Examples:
        >>> import kornia
        >>> input = torch.randn(2, 3, 5, 6)
        >>> aug_list = Sequential(
        ...     kornia.color.BgrToRgb(),
        ...     kornia.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...     kornia.filters.MedianBlur((3, 3)),
        ...     kornia.augmentation.RandomAffine(360, p=1.0),
        ...     kornia.enhance.Invert(),
        ... return_transform=True,
        ... same_on_batch=True,
        ... )
        >>> out = aug_list(input)
        >>> out[0].shape, out[1].shape
        (torch.Size([2, 3, 5, 6]), torch.Size([2, 3, 3]))

        Reproduce with provided params.
        >>> out2 = aug_list(input, params=aug_list._params)
        >>> torch.equal(out[0], out2[0]), torch.equal(out[1], out2[1])
        (True, True)
    """

    def __init__(
        self,
        *args: nn.Module,
        same_on_batch: Optional[bool] = None,
        return_transform: Optional[bool] = None,
        keepdim: Optional[bool] = None,
    ) -> None:
        super(Sequential, self).__init__(*args)
        self.same_on_batch = same_on_batch
        self.return_transform = return_transform
        self.keepdim = keepdim
        for arg in args:
            if not isinstance(arg, nn.Module):
                raise NotImplementedError(f"Only nn.Module are supported at this moment. Got {arg}.")
            if isinstance(arg, _AugmentationBase):
                if same_on_batch is not None:
                    arg.same_on_batch = same_on_batch
                if return_transform is not None:
                    arg.return_transform = return_transform
                if keepdim is not None:
                    arg.keepdim = keepdim
        self._params = {}

    def apply_to_input(self, input, item: nn.Module, param: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
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

    def forward(
        self, input: torch.Tensor, params: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if params is None:
            params = {}
        self._params = {}
        for item in self.children():
            func_name = item.__class__.__name__
            param = params[func_name] if func_name in params else None
            input = self.apply_to_input(input, item, param)
        return input
