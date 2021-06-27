from itertools import zip_longest
from typing import cast, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from kornia.augmentation.augmentation import ColorJitter
from kornia.augmentation.base import _AugmentationBase, IntensityAugmentationBase2D
from kornia.contrib.extract_patches import extract_tensor_patches

from .image import ImageSequential, ParamItem

__all__ = ["PatchSequential"]


class PatchSequential(ImageSequential):
    r"""Container for performing patch-level image processing.

    .. image:: https://kornia-tutorials.readthedocs.io/en/latest/_images/data_patch_sequential_5_1.png

    PatchSequential breaks input images into patches by a given grid size, which will be resembled back
    afterwards. Different image processing and augmentation methods will be performed on each patch region.

    Args:
        *args: a list of processing modules.
        grid_size: controls the grid board seperation.
        padding: same or valid padding. If same padding, it will pad to include all pixels if the input
            tensor cannot be divisible by grid_size. If valid padding, the redundent border will be removed.
        same_on_batch: apply the same transformation across the batch.
            If None, it will not overwrite the function-wise settings.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
            to the batch form (False). If None, it will not overwrite the function-wise settings.
        patchwise_apply: apply image processing args will be applied patch-wisely.
            if ``True``, the number of args must be equal to grid number.
            if ``False``, the image processing args will be applied as a sequence to all patches. Default: False.
        random_apply: randomly select a sublist (order agnostic) of args to
            apply transformation.
            If ``int`` (batchwise mode only), a fixed number of transformations will be selected.
            If ``(a,)`` (batchwise mode only), x number of transformations (a <= x <= len(args)) will be selected.
            If ``(a, b)`` (batchwise mode only), x number of transformations (a <= x <= b) will be selected.
            If ``True``, the whole list of args will be processed in a random order.
            If ``False``, the whole list of args will be processed in original order.

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
        ... random_apply=True,
        ... )
        >>> out = seq(input)
        >>> out.shape
        torch.Size([2, 3, 224, 224])
        >>> out1 = seq(input, seq._params)
        >>> torch.equal(out, out1)
        True
    """

    def __init__(
        self,
        *args: nn.Module,
        grid_size: Tuple[int, int] = (4, 4),
        padding: str = "same",
        same_on_batch: Optional[bool] = None,
        keepdim: Optional[bool] = None,
        patchwise_apply: bool = False,
        random_apply: Union[int, bool, Tuple[int, int]] = False,
    ) -> None:
        _random_apply: Optional[Union[int, Tuple[int, int]]]
        if patchwise_apply and random_apply is True:
            _random_apply = (grid_size[0] * grid_size[1], grid_size[0] * grid_size[1])
        elif patchwise_apply and random_apply is False:
            assert len(args) == grid_size[0] * grid_size[1], (
                "The number of processing modules must be equal with grid size."
                f"Got {len(args)} and {grid_size[0] * grid_size[1]}."
            )
            _random_apply = random_apply
        elif patchwise_apply and isinstance(random_apply, (int, tuple)):
            raise ValueError(f"Only boolean value allowed when `patchwise_apply` is set to True. Got {random_apply}.")
        else:
            _random_apply = random_apply
        super(PatchSequential, self).__init__(
            *args, same_on_batch=same_on_batch, return_transform=False, keepdim=keepdim, random_apply=_random_apply
        )
        assert padding in ["same", "valid"], f"`padding` must be either `same` or `valid`. Got {padding}."
        self.grid_size = grid_size
        self.padding = padding
        self.patchwise_apply = patchwise_apply

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
        self, input: torch.Tensor, params: Optional[List[List[ParamItem]]] = None
    ) -> torch.Tensor:  # NOTE: return_transform is always False here.
        if params is None:
            params = [[]] * input.size(1)
            auglist = [self.get_forward_sequence() for _ in range(input.size(1))]
        else:
            auglist = [self.get_forward_sequence(p) for p in params]
            assert input.size(0) == len(auglist) == len(params)

        out = []
        self._params = []
        # TODO: This will need an optimization later.
        for inp, proc, param in zip(input, auglist, params):
            o = []
            p = []
            for inp_pat, (proc_name, proc_pat), _param in zip_longest(inp, proc, param):
                if isinstance(proc_pat, (_AugmentationBase, ImageSequential)):
                    o.append(proc_pat(inp_pat[None], _param.data if _param is not None else None))
                    p.append(ParamItem(proc_name, proc_pat._params))
                else:
                    o.append(proc_pat(inp_pat[None]))
                    p.append(ParamItem(proc_name, {}))
            out.append(torch.cat(o, dim=0))
            self._params.append(p)

        input = torch.stack(out, dim=0)
        return input

    def forward_batchwise(
        self, input: torch.Tensor, params: Optional[List[ParamItem]] = None
    ) -> torch.Tensor:  # NOTE: return_transform is always False here.
        if self.same_on_batch:
            batch_shape = (input.size(1), *input.shape[-3:])
            patch_num = input.size(0)
        else:
            batch_shape = (input.size(0) * input.size(1), *input.shape[-3:])

        if params is None:
            params = []
            for name, aug in self.get_forward_sequence():
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
                params.append(ParamItem(name, param))

        input = super().forward(input.view(-1, *input.shape[-3:]), params)  # type: ignore

        return input

    def forward(  # type: ignore
        self,
        input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        params: Optional[Union[List[ParamItem], List[List[ParamItem]]]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # NOTE: return_transform is always False here.
        """Input transformation will be returned if input is a tuple."""
        # BCHW -> B(patch)CHW
        if isinstance(input, (tuple,)):
            pad = self.compute_padding(input[0], self.padding)
            input = self.extract_patches(input[0], self.grid_size, pad), input[1]
        else:
            pad = self.compute_padding(input, self.padding)
            input = self.extract_patches(input, self.grid_size, pad)

        if not self.patchwise_apply:
            params = cast(List[ParamItem], params)
            if isinstance(input, (tuple,)):
                input = self.forward_batchwise(input[0], params), input[1]
            else:
                input = self.forward_batchwise(input, params)
        else:
            params = cast(List[List[ParamItem]], params)
            if isinstance(input, (tuple,)):
                input = self.forward_patchwise(input[0], params), input[1]
            else:
                input = self.forward_patchwise(input, params)

        if isinstance(input, (tuple,)):
            input = (self.restore_from_patches(input[0], self.grid_size, pad=pad), input[1])
        else:
            input = self.restore_from_patches(input, self.grid_size, pad=pad)
        return input
