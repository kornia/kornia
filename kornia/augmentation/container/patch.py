from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from kornia.augmentation.augmentation import ColorJitter
from kornia.augmentation.base import _AugmentationBase, IntensityAugmentationBase2D
from kornia.contrib.extract_patches import extract_tensor_patches

from .image import ImageSequential


class PatchSequential(ImageSequential):
    r"""Container for performing patch-level image processing.

    PatchSequential breaks input images into patches by a given grid size, which will be resembled back
    afterwards. Different image processing and augmentation methods will be performed on each patch region.

    Args:
        *args (nn.Module): a list of processing modules.
        same_on_batch (bool, optional): apply the same transformation across the batch.
            If None, it will not overwrite the function-wise settings. Default: None.
        keepdim (bool, optional): whether to keep the output shape the same as input (True) or broadcast it
            to the batch form (False). If None, it will not overwrite the function-wise settings. Default: None.
        patchwise_apply (bool, optional): apply image processing args will be applied patch-wisely.
            if ``True``, the number of args must be equal to grid number.
            if ``False``, the image processing args will be applied as a whole to all patches. Default: False.

    Return:
        List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]: the tensor (, and the transformation matrix)
            has been sequentially modified by the args.

    Examples:
        >>> import kornia.augmentation as K
        >>> input = torch.randn(2, 3, 224, 224)
        >>> seq = PatchSequential(
        ...     ImageSequential(
        ...         K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.5),
        ...         K.RandomPerspective(0.2, p=0.5),
        ...         K.RandomSolarize(0.1, 0.1, p=0.5),
        ...     ),
        ...     K.RandomAffine(360, p=1.0),
        ...     ImageSequential(
        ...         K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.5),
        ...         K.RandomPerspective(0.2, p=0.5),
        ...         K.RandomSolarize(0.1, 0.1, p=0.5),
        ...     ),
        ...     K.RandomSolarize(0.1, 0.1, p=0.1),
        ... grid_size=(2,2),
        ... patchwise_apply=False,
        ... same_on_batch=True,
        ... )
        >>> seq(input).shape
        torch.Size([2, 3, 224, 224])
    """

    def __init__(
        self,
        *args: nn.Module,
        grid_size: Tuple[int, int] = (4, 4),
        padding: str = "same",
        same_on_batch: Optional[bool] = None,
        keepdim: Optional[bool] = None,
        patchwise_apply: bool = False,
    ) -> None:
        super(PatchSequential, self).__init__(
            *args, same_on_batch=same_on_batch, return_transform=False, keepdim=keepdim
        )
        assert padding in ["same", "valid"], f"`padding` must be either `same` or `valid`. Got {padding}."
        self.grid_size = grid_size
        self.padding = padding
        self.patchwise_apply = patchwise_apply
        if patchwise_apply:
            assert len(args) == grid_size[0] * grid_size[1], (
                "The number of processing modules must be equal with grid size."
                f"Got {len(args)} and {grid_size[0] * grid_size[1]}."
            )

    def is_intensity_only(self) -> bool:
        """Check if all transformations are intensity-based.

        Note: patch processing would break the continuity of labels (e.g. bbounding boxes, masks).
        """
        for arg in self.children():
            if isinstance(arg, (ImageSequential,)):
                for _arg in arg.children():
                    if not isinstance(_arg, IntensityAugmentationBase2D):
                        return False
            elif not isinstance(_arg, IntensityAugmentationBase2D):
                return False
        return True

    def __repeat_param_across_patches__(self, param: torch.Tensor, patch_num: int) -> torch.Tensor:
        """Repeat parameters across patches.

        The input is shaped as (B, ...), while to output (B * patch_num, ...), which
        to guarentee that the same transformation would happen for each patch index.

        (B1, B2, ..., Bn) => (B1, ... Bn, B1, ..., Bn, ..., B1, ..., Bn)
                              | pt_size | | pt_size |  ..., | pt_size |
        """
        repeated = torch.cat([param] * patch_num, dim=0)
        return repeated

    def compute_padding(
        self, input: torch.Tensor, padding: str, grid_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[int, int, int, int]:
        if grid_size is None:
            grid_size = self.grid_size
        if padding == "valid":
            ph, pw = input.size(-2) // grid_size[0], input.size(-1) // grid_size[1]
            return (-pw // 2, pw // 2 - pw, -ph // 2, ph // 2 - ph)
        elif padding == 'same':
            ph = input.size(-2) - input.size(-2) // grid_size[0] * grid_size[0]
            pw = input.size(-1) - input.size(-1) // grid_size[1] * grid_size[1]
            return (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
        else:
            raise NotImplementedError(f"Expect `padding` as either 'valid' or 'same'. Got {padding}.")

    def extract_patches(
        self,
        input: torch.Tensor,
        grid_size: Optional[Tuple[int, int]] = None,
        pad: Optional[Tuple[int, int, int, int]] = None,
    ) -> torch.Tensor:
        """Extract patches from tensor.

        Example:
            >>> import kornia.augmentation as K
            >>> pas = PatchSequential(K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0))
            >>> pas.extract_patches(torch.arange(16).view(1, 1, 4, 4), grid_size=(2, 2))
            tensor([[[[[ 0,  1],
                       [ 4,  5]]],
            <BLANKLINE>
            <BLANKLINE>
                     [[[ 2,  3],
                       [ 6,  7]]],
            <BLANKLINE>
            <BLANKLINE>
                     [[[ 8,  9],
                       [12, 13]]],
            <BLANKLINE>
            <BLANKLINE>
                     [[[10, 11],
                       [14, 15]]]]])
            >>> pas.extract_patches(torch.arange(54).view(1, 1, 6, 9), grid_size=(2, 2), pad=(-1, -1, -2, -2))
            tensor([[[[[19, 20, 21]]],
            <BLANKLINE>
            <BLANKLINE>
                     [[[22, 23, 24]]],
            <BLANKLINE>
            <BLANKLINE>
                     [[[28, 29, 30]]],
            <BLANKLINE>
            <BLANKLINE>
                     [[[31, 32, 33]]]]])
        """
        if pad is not None:
            input = torch.nn.functional.pad(input, list(pad))
        if grid_size is None:
            grid_size = self.grid_size
        window_size = (input.size(-2) // grid_size[-2], input.size(-1) // grid_size[-1])
        stride = window_size
        return extract_tensor_patches(input, window_size, stride)

    def restore_from_patches(
        self,
        patches: torch.Tensor,
        grid_size: Tuple[int, int] = (4, 4),
        pad: Optional[Tuple[int, int, int, int]] = None,
    ) -> torch.Tensor:
        """Restore input from patches.

        Example:
            >>> import kornia.augmentation as K
            >>> pas = PatchSequential(K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0))
            >>> out = pas.extract_patches(torch.arange(16).view(1, 1, 4, 4), grid_size=(2, 2))
            >>> pas.restore_from_patches(out, grid_size=(2, 2))
            tensor([[[[ 0,  1,  2,  3],
                      [ 4,  5,  6,  7],
                      [ 8,  9, 10, 11],
                      [12, 13, 14, 15]]]])
        """
        if grid_size is None:
            grid_size = self.grid_size
        patches_tensor = patches.view(-1, grid_size[0], grid_size[1], *patches.shape[-3:])
        restored_tensor = torch.cat(torch.chunk(patches_tensor, grid_size[0], dim=1), -2).squeeze(1)
        restored_tensor = torch.cat(torch.chunk(restored_tensor, grid_size[1], dim=1), -1).squeeze(1)

        if pad is not None:
            restored_tensor = torch.nn.functional.pad(restored_tensor, [-i for i in pad])
        return restored_tensor

    def forward_patchwise(
        self, input: torch.Tensor, params: Optional[List[Optional[Dict[str, Dict[str, torch.Tensor]]]]] = None
    ) -> torch.Tensor:  # NOTE: return_transform is always False here.
        if params is None:
            assert input.size(1) == len(list(self.children()))
            params = [None] * input.size(1)
        else:
            assert input.size(1) == len(list(self.children())) == len(params)
        out = [
            proc(inp, param) for inp, proc, param in zip(input.permute(1, 0, 2, 3, 4), list(self.children()), params)
        ]
        input = torch.stack(out, dim=1)
        return input

    def forward_batchwise(
        self, input: torch.Tensor, params: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    ) -> torch.Tensor:  # NOTE: return_transform is always False here.
        if self.same_on_batch:
            batch_shape = (input.size(1), *input.shape[-3:])
            patch_num = input.size(0)
        else:
            batch_shape = (input.size(0) * input.size(1), *input.shape[-3:])

        if params is None:
            params = {}
            for aug in self.children():
                func_name = aug.__class__.__name__
                if isinstance(aug, _AugmentationBase):
                    aug.same_on_batch = False
                    param = aug.forward_parameters(batch_shape)
                    if self.same_on_batch:
                        for k, v in param.items():
                            # TODO: revise colorjitter order param in the future to align the standard.
                            if not (k == "order" and isinstance(aug, ColorJitter)):
                                param.update({k: self.__repeat_param_across_patches__(v, patch_num)})
                    aug.same_on_batch = True
                else:
                    param = None
                params.update({func_name: param})

        input = super().forward(input.view(-1, *input.shape[-3:]), params)  # type: ignore

        return input

    def forward(  # type: ignore
        self,
        input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        params: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # NOTE: return_transform is always False here.
        """Input transformation will be returned if input is a tuple."""
        if isinstance(input, (tuple,)):
            pad = self.compute_padding(input[0], self.padding)
            input = self.extract_patches(input[0], self.grid_size, pad), input[1]
        else:
            pad = self.compute_padding(input, self.padding)
            input = self.extract_patches(input, self.grid_size, pad)

        if not self.patchwise_apply:
            if isinstance(input, (tuple,)):
                input = self.forward_batchwise(input[0], params), input[1]
            else:
                input = self.forward_batchwise(input, params)
        else:
            assert params is None, "Passing params to patchwise forward is currently not supported."
            if isinstance(input, (tuple,)):
                input = self.forward_patchwise(input[0]), input[1]
            else:
                input = self.forward_patchwise(input)

        if isinstance(input, (tuple,)):
            input = (self.restore_from_patches(input[0], self.grid_size, pad=pad), input[1])
        else:
            input = self.restore_from_patches(input, self.grid_size, pad=pad)
        return input
