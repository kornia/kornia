from typing import List, Optional, Tuple, Union, NamedTuple, cast
from itertools import chain, cycle, islice

import torch
import torch.nn as nn

from kornia.augmentation.base import (
    MixAugmentationBase,
    TensorWithTransMat,
    _AugmentationBase,
    IntensityAugmentationBase2D
)
from kornia.contrib.extract_patches import extract_tensor_patches
from kornia.constants import ConstantBase, Enum, EnumMetaFlags

from .image import ImageSequential, ParamItem

__all__ = ["PatchSequential"]


class PatchParamItem(NamedTuple):
    indices: torch.Tensor
    param: ParamItem


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
            If ``False`` and not ``patchwise_apply``, the whole list of args will be processed in original order.
            If ``False`` and ``patchwise_apply``, the whole list of args will be processed in original order
            location-wisely.

    Return:
        List[TensorWithTransMat]: the tensor (, and the transformation matrix)
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
        ... patchwise_apply=True,
        ... same_on_batch=True,
        ... random_apply=False,
        ... )
        >>> out = seq(input)
        >>> aa = seq._params
        >>> out.shape
        torch.Size([2, 3, 224, 224])
        >>> out1 = seq(input, params=seq._params)
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
        patchwise_apply: bool = None,
        random_apply: Union[int, bool, Tuple[int, int]] = False,
    ) -> None:
        _random_apply: Optional[Union[int, Tuple[int, int]]]
        # TODO: deprecate patchwise_apply
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

    def get_multiple_forward_sequence(self, sequence_num: int) -> List[List[Tuple[str, nn.Module]]]:
        """Get mulitple forward sequence but maximumly one mix augmentation in between."""
        if not self.same_on_batch and self.random_apply:
            with_mix = False
            for _ in range(sequence_num):
                seq, mix_added = self.__sample_forward_indices__(with_mix=with_mix)
                with_mix = mix_added
                yield seq
        elif not self.same_on_batch and not self.random_apply:
            # TODO: The mixup shall happen location-wisely
            for nchild in islice(cycle(self.named_children()), sequence_num):
                yield [nchild]
        elif self.same_on_batch and not self.random_apply:
            # TODO: The mixup shall happen image-wisely
            for nchild in islice(cycle(self.named_children()), sequence_num):
                yield [nchild]
        else:
            nchildren = list(self.named_children())
            for idx in torch.randperm(len(nchildren)):
                yield [nchildren[idx]]

    def get_parameter_sequence(self, sequence_num: int) -> List[ParamItem]:
        """Get mulitple forward sequence but maximumly one mix augmentation in between."""
        if not self.same_on_batch and self.random_apply:
            with_mix = False
            for _ in range(sequence_num):
                seq, mix_added = self.__sample_forward_indices__(with_mix=with_mix)
                with_mix = mix_added
                for s in seq:
                    yield s
        elif not self.same_on_batch and not self.random_apply:
            # TODO: The mixup shall happen location-wisely
            for nchild in islice(cycle(self.named_children()), sequence_num):
                yield [nchild]
        elif self.same_on_batch and not self.random_apply:
            # TODO: The mixup shall happen image-wisely
            for nchild in islice(cycle(self.named_children()), sequence_num):
                yield nchild
        else:
            nchildren = list(self.named_children())
            for idx in torch.randperm(len(nchildren)):
                yield nchildren[idx]

    def _compose_param(self):
        pass

    def _unique_module(
        self, sequence: List[Optional[Tuple[str, nn.Module]]], params: List[Optional[ParamItem]]
    ) -> Tuple[List[Tuple[str, nn.Module]], List[Optional[ParamItem]]]:
        names = []
        modules = []
        ps = []
        for item, p in zip(sequence, params):
            if item is not None:
                name, module = item
                names.append(name)
                modules.append(module)
                ps.append(p)
        return list(zip(list(set(names)), list(set(modules)))), list(set(ps))

    def read_batchwise_sequence(self, params):
        pass

    def read_locationwise_sequence(self, params):
        pass

    def read_patchwise_sequence(self, params):
        pass

    def split_batchwise_sequence(
        self, sequence: List[Tuple[str, nn.Module]],
        params: Optional[List[Optional[ParamItem]]] = None
    ) -> Tuple[
        Tuple[List[Tuple[str, nn.Module]], List[Optional[ParamItem]]],
        Tuple[Optional[Tuple[str, nn.Module]], Optional[ParamItem]],
        Tuple[List[Tuple[str, nn.Module]], List[Optional[ParamItem]]],
    ]:
        """Split sequence to be pre-mix modules, mix module, and post-mix modules.

        Example:
            >>> import kornia.augmentation as K
            >>> ps = PatchSequential(K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0))
            >>> output = ps.split_batchwise_sequence([
            ...     ('RandomEqualize', K.RandomEqualize()),
            ...     ('RandomEqualize_2', K.RandomEqualize()),
            ... ], None)
        """
        if params is None:
            params = []

        index = None
        for idx, (_, module) in enumerate(sequence):
            if isinstance(module, (MixAugmentationBase,)):
                index = idx
        if index is None:
            return (sequence, params), (None, None), ([], [])
        return (
            (sequence[:index], params[:index]),
            (sequence[index], params[index]),
            (sequence[index + 1:], params[index + 1:])
        )

    def split_locationwise_sequence(
        self, sequence: List[List[Tuple[str, nn.Module]]], params: Optional[List[List[Optional[ParamItem]]]] = None,
    ) -> Tuple[
        Tuple[List[List[Tuple[str, nn.Module]]], List[List[Optional[ParamItem]]]],
        Tuple[List[Optional[Tuple[str, nn.Module]]], List[Optional[ParamItem]]],
        Tuple[List[List[Tuple[str, nn.Module]]], List[List[Optional[ParamItem]]]],
    ]:
        """Split sequence to be pre-mix modules, mix module, and post-mix modules.

        Input sequence shall be a list of module sequence, which has the same size as grid size.

        Example:
            >>> import kornia.augmentation as K
            >>> ps = PatchSequential(K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0))
            >>> output = ps.split_locationwise_sequence([[
            ...     ('RandomEqualize', K.RandomEqualize()),
            ...     ('RandomEqualize_2', K.RandomEqualize()),
            ... ], [
            ...     ('RandomEqualize', K.RandomEqualize()),
            ...     ('RandomEqualize_2', K.RandomEqualize()),
            ... ]], None)
        """
        if params is None:
            params = [[]] * len(sequence)

        a1, a2, b1, b2, c1, c2 = [], [], [], [], [], []
        for seq, param in zip(sequence, params):
            a, b, c = self.split_batchwise_sequence(seq, param)
            a1.append(a[0])
            a2.append(a[1])
            b1.append(b[0])
            b2.append(b[1])
            c1.append(c[0])
            c2.append(c[1])
        return (a1, a2), (b1, b2), (c1, c2)

    def split_patchwise_sequence(
        self, sequence: List[List[List[Tuple[str, nn.Module]]]],
        params: Optional[List[List[List[Optional[ParamItem]]]]] = None
    ) -> Tuple[
        Tuple[List[List[List[Tuple[str, nn.Module]]]], List[List[List[Optional[ParamItem]]]]],
        Tuple[List[List[Optional[Tuple[str, nn.Module]]]], List[List[Optional[ParamItem]]]],
        Tuple[List[List[List[Tuple[str, nn.Module]]]], List[List[List[Optional[ParamItem]]]]],
    ]:
        """Split sequence to be pre-mix modules, mix module, and post-mix modules.

        Example:
            >>> import kornia.augmentation as K
            >>> ps = PatchSequential(K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0))
            >>> output = ps.split_patchwise_sequence([[[
            ...     ('RandomEqualize', K.RandomEqualize()),
            ...     ('RandomEqualize_2', K.RandomEqualize()),
            ... ], [
            ...     ('RandomEqualize', K.RandomEqualize()),
            ...     ('RandomEqualize_2', K.RandomEqualize()),
            ... ]], [[
            ...     ('RandomEqualize', K.RandomEqualize()),
            ...     ('RandomEqualize_2', K.RandomEqualize()),
            ... ], [
            ...     ('RandomEqualize', K.RandomEqualize()),
            ...     ('RandomEqualize_2', K.RandomEqualize()),
            ... ]]], None)
        """
        if params is None:
            params = [[]] * len(sequence)

        a1, a2, b1, b2, c1, c2 = [], [], [], [], [], []
        for seqs, param in zip(sequence, params):
            a, b, c = self.split_locationwise_sequence(seqs, param)
            a1.append(a[0])
            a2.append(a[1])
            b1.append(b[0])
            b2.append(b[1])
            c1.append(c[0])
            c2.append(c[1])
        return (a1, a2), (b1, b2), (c1, c2)

    def apply_elementwise(
        self,
        input: torch.Tensor,
        sequence: List[List[Tuple[str, nn.Module]]],
        params: List[List[Optional[ParamItem]]]
    ) -> Tuple[torch.Tensor, List[List[ParamItem]]]:
        # TODO: update parameter
        assert input.size(0) == len(sequence)
        out_param = []
        for idx, (seq, param) in enumerate(zip(sequence, params)):
            out_p = []
            for (name, module), p in zip(seq, param):
                if p is not None:  # Implicitly indicating is an augmentation or imageSequential
                    input[idx] = module(input[idx], params=p.data)
                    out_p.append(ParamItem(name, module._params))  # type: ignore
                else:
                    input[idx] = module(input[idx])
                    out_p.append(ParamItem(name, None))
            out_param.append(out_p)
        return input, out_param

    def apply_batchwise(
        self, input: torch.Tensor, sequence: List[Tuple[str, nn.Module]], params: List[Optional[ParamItem]]
    ) -> Tuple[torch.Tensor, List[ParamItem]]:
        # TODO: update parameter
        assert input.size(0) == len(sequence)
        out_param = []
        for (name, module), p in zip(sequence, params):
            if p is not None:  # Implicitly indicating is an augmentation or imageSequential
                input = module(input, params=p.data)
                out_param.append(ParamItem(name, module._params))  # type: ignore
            else:
                input = module(input)
                out_param.append(ParamItem(name, None))
        return input, out_param

    def apply_mix_augment(
        self, input: torch.Tensor, label: Optional[torch.Tensor],
        module: Tuple[str, nn.Module], params: Optional[ParamItem] = None
    ) -> Tuple[Tuple[torch.Tensor, Optional[torch.Tensor]], ParamItem]:
        # TODO: update parameter
        return (
            module[1](input, label, params=params),
            ParamItem(module[0], cast(MixAugmentationBase, module[1])._params)
        )

    def forward_batchwise(
        self, input: torch.Tensor, label: Optional[torch.Tensor] = None,
        params: Optional[List[ParamItem]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """All patches will be processed as a batch.
        """
        if params is None:
            aug_seq = self.get_forward_sequence()
        else:
            aug_seq = self.get_forward_sequence(params)
        pre_list, mix_aug, post_list = self.split_batchwise_sequence(list(aug_seq))

        in_shape = input.shape
        input = input.reshape(-1, *input.shape[-3:])
        input, p1 = self.apply_batchwise(input, *pre_list)
        if mix_aug[0] is not None:
            (input, label), p2 = self.apply_mix_augment(input, label, mix_aug[0], mix_aug[1])
        input, p3 = self.apply_batchwise(input, *post_list)
        return input.reshape(in_shape), label

    def forward_locationwise(
        self, input: torch.Tensor, label: Optional[torch.Tensor] = None,
        params: Optional[List[ParamItem]] = None, location_wise_mix: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """All patches in the same location will be processed as a batch.
        """
        if params is None:
            aug_seq = list(self.get_multiple_forward_sequence(input.size(1)))
        else:
            aug_seq = list(self.get_multiple_forward_sequence(input.size(1)))
            # pre_list, mix_aug, post_list = self.read_locationwise_sequence(params)
        pre_list, mix_aug, post_list = self.split_locationwise_sequence(list(aug_seq), params)

        input = input.permute(1, 0, 2, 3, 4)
        input, p1 = self.apply_elementwise(input, *pre_list)

        if location_wise_mix:
            out_label = []
            p2 = []
            for idx, aug, param in zip(range(input.size(0)), *mix_aug):
                if aug is not None:
                    (inp, lab), p = self.apply_mix_augment(input, label, aug, param)
                    input[idx] = inp
                    out_label.append(lab)
                    p2.append(p)
                else:
                    # TODO
                    pass
        else:
            out_module, out_param = self._unique_module(*mix_aug)
            assert len(out_module) == 1 and len(out_param) == 1
            for om, op in zip(out_module, out_param):
                if om is not None:
                    (input, label), p2 = self.apply_mix_augment(input, label, om, op)
        input, p3 = self.apply_elementwise(input, *post_list)
        return input.permute(1, 0, 2, 3, 4), label

    def forward_patchwise(
        self, input: torch.Tensor, label: Optional[torch.Tensor] = None,
        params: Optional[List[List[List[Optional[ParamItem]]]]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """All patches will be processed individually.

        Note:
            It can be slow without much optimizations for now.
        """
        if params is None:
            _aug_seq = list(self.get_multiple_forward_sequence(input.size(0) * input.size(1)))
            aug_seq = [_aug_seq[i:i + input.size(1)] for i in range(0, len(_aug_seq), input.size(1))]
        else:
            aug_seq = self.read_patchwise_sequence(params)

        pre_list, mix_aug, post_list = self.split_patchwise_sequence(aug_seq, params)

        in_shape = input.shape
        input = input.reshape(-1, *input.shape[-3:])
        input, p1 = self.apply_elementwise(input, list(chain(*pre_list[0])), list(chain(*pre_list[1])))
        out_module, out_param = self._unique_module(list(chain(*mix_aug[0])), list(chain(*mix_aug[1])))
        assert len(out_module) == 1
        if out_module[0] is not None:
            (input, label), p2 = self.apply_mix_augment(input, label, out_module[0], out_param[0])
        input, p3 = self.apply_elementwise(input, list(chain(*post_list[0])), list(chain(*post_list[1])))
        return input.reshape(in_shape), label

    def apply_by_param(
        self, input: torch.Tensor, label: Optional[torch.Tensor], params: PatchParamItem
    ) -> Tuple[TensorWithTransMat, Optional[torch.Tensor], ParamItem]:
        input = input[params.indices]
        if label is not None:
            label = label[params.indices]
        return self._apply_operation(
            input, label, params.param.name, self.get_submodule(params.param.name), params.param.data)

    def forward_parameters(
        self, batch_shape: torch.Size, params: Optional[List[ParamItem]], strategy: str
    ) -> List[PatchParamItem]:
        if params is None:
            if strategy == "batchwise":
                params = self.get_forward_sequence()
            elif strategy == "patchwise":
                params = self.get_parameter_sequence(batch_shape[0] * batch_shape[1])
            elif strategy == "loc":
                params = self.get_parameter_sequence(batch_shape[1])
        
        

    def auto_forward(
        self,
        input: TensorWithTransMat,
        label: Optional[torch.Tensor] = None
    ) -> Union[TensorWithTransMat, Tuple[TensorWithTransMat, Optional[torch.Tensor]]]:
        if not self.patchwise_apply:
            if isinstance(input, (tuple,)):
                in_trans = input[1]
                input, label = self.forward_batchwise(input[0], label)
                input = (input, in_trans)
            else:
                input, label = self.forward_batchwise(input, label)
        else:
            if isinstance(input, (tuple,)):
                in_trans = input[1]
                if not self.same_on_batch:
                    input, label = self.forward_patchwise(input[0], label)
                else:
                    input, label = self.forward_locationwise(input[0], label)
                input = (input, in_trans)
            else:
                if not self.same_on_batch:
                    input, label = self.forward_patchwise(input, label)
                else:
                    input, label = self.forward_locationwise(input, label)
        return input, label

    def forward(  # type: ignore
        self,
        input: TensorWithTransMat,
        label: Optional[torch.Tensor] = None,
        params: Optional[List[List[ParamItem]]] = None,
    ) -> Union[TensorWithTransMat, Tuple[TensorWithTransMat, torch.Tensor]]:
        """Input transformation will be returned if input is a tuple."""
        self.clear_state()
        # BCHW -> B(patch)CHW
        if isinstance(input, (tuple,)):
            pad = self.compute_padding(input[0], self.padding)
            input = self.extract_patches(input[0], self.grid_size, pad), input[1]
        else:
            pad = self.compute_padding(input, self.padding)
            input = self.extract_patches(input, self.grid_size, pad)

        if params is None:
            input, label = self.auto_forward(input, label)
        else:
            input, label = self.forward_by_params(input, label, params)

        if isinstance(input, (tuple,)):
            input = (self.restore_from_patches(input[0], self.grid_size, pad=pad), input[1])
        else:
            input = self.restore_from_patches(input, self.grid_size, pad=pad)
        return self.__packup_output__(input, label)
