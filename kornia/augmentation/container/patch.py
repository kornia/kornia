from itertools import cycle, islice
from typing import Iterator, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn

from kornia.augmentation.base import (
    _AugmentationBase,
    IntensityAugmentationBase2D,
    MixAugmentationBase,
    TensorWithTransformMat,
)
from kornia.augmentation.container.base import SequentialBase
from kornia.contrib.extract_patches import extract_tensor_patches

from .image import ImageSequential, ParamItem

__all__ = ["PatchSequential"]


class PatchParamItem(NamedTuple):
    indices: List[int]
    param: ParamItem


class PatchSequential(ImageSequential):
    r"""Container for performing patch-level image data augmentation.

    .. image:: https://kornia-tutorials.readthedocs.io/en/latest/_images/data_patch_sequential_5_1.png

    PatchSequential breaks input images into patches by a given grid size, which will be resembled back
    afterwards.

    Different image processing and augmentation methods will be performed on each patch region as
    in :cite:`lin2021patch`.

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
            if ``False``, the image processing args will be applied as a sequence to all patches.
        random_apply: randomly select a sublist (order agnostic) of args to
            apply transformation.
            If ``int`` (batchwise mode only), a fixed number of transformations will be selected.
            If ``(a,)`` (batchwise mode only), x number of transformations (a <= x <= len(args)) will be selected.
            If ``(a, b)`` (batchwise mode only), x number of transformations (a <= x <= b) will be selected.
            If ``True``, the whole list of args will be processed in a random order.
            If ``False`` and not ``patchwise_apply``, the whole list of args will be processed in original order.
            If ``False`` and ``patchwise_apply``, the whole list of args will be processed in original order
            location-wisely.

    .. note::
        Transformation matrix returned only considers the transformation applied in ``kornia.augmentation`` module.
        Those transformations in ``kornia.geometry`` will not be taken into account.

    .. note::
        See a working example `here <https://kornia-tutorials.readthedocs.io/en/
        latest/data_patch_sequential.html>`__.

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
        patchwise_apply: bool = True,
        random_apply: Union[int, bool, Tuple[int, int]] = False,
    ) -> None:
        _random_apply: Optional[Union[int, Tuple[int, int]]]

        if patchwise_apply and random_apply is True:
            # will only apply [1, 4] augmentations per patch
            _random_apply = (1, 4)
        elif patchwise_apply and random_apply is False:
            assert len(args) == grid_size[0] * grid_size[1], (
                "The number of processing modules must be equal with grid size."
                f"Got {len(args)} and {grid_size[0] * grid_size[1]}. "
                "Please set random_apply = True or patchwise_apply = False."
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

    def contains_label_operations(self, params: List[PatchParamItem]) -> bool:  # type: ignore
        for param in params:
            if param.param.name.startswith("RandomMixUp") or param.param.name.startswith("RandomCutMix"):
                return True
        return False

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
            >>> pas = PatchSequential(K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0), patchwise_apply=False)
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
            >>> pas = PatchSequential(K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0), patchwise_apply=False)
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

    def forward_parameters(self, batch_shape: torch.Size) -> List[PatchParamItem]:  # type: ignore
        out_param: List[PatchParamItem] = []
        if not self.patchwise_apply:
            params = self.generate_parameters(torch.Size([1, batch_shape[0] * batch_shape[1], *batch_shape[2:]]))
            indices = torch.arange(0, batch_shape[0] * batch_shape[1])
            [out_param.append(PatchParamItem(indices.tolist(), p)) for p, _ in params]  # type: ignore
            # "append" of "list" does not return a value
        elif not self.same_on_batch:
            params = self.generate_parameters(torch.Size([batch_shape[0] * batch_shape[1], 1, *batch_shape[2:]]))
            [out_param.append(PatchParamItem([i], p)) for p, i in params]  # type: ignore
            # "append" of "list" does not return a value
        else:
            params = self.generate_parameters(torch.Size([batch_shape[1], batch_shape[0], *batch_shape[2:]]))
            indices = torch.arange(0, batch_shape[0] * batch_shape[1], step=batch_shape[1])
            [out_param.append(PatchParamItem((indices + i).tolist(), p)) for p, i in params]  # type: ignore
            # "append" of "list" does not return a value
        return out_param

    def generate_parameters(self, batch_shape: torch.Size) -> Iterator[Tuple[ParamItem, int]]:
        """Get mulitple forward sequence but maximumly one mix augmentation in between.

        Args:
            batch_shape: 5-dim shape arranged as :math:``(N, B, C, H, W)``, in which N represents
                the number of sequence.
        """
        if not self.same_on_batch and self.random_apply:
            # diff_on_batch and random_apply => patch-wise augmentation
            with_mix = False
            for i in range(batch_shape[0]):
                seq, mix_added = self.get_random_forward_sequence(with_mix=with_mix)
                with_mix = mix_added
                for s in seq:
                    if isinstance(s[1], (_AugmentationBase, MixAugmentationBase, SequentialBase)):
                        yield ParamItem(s[0], s[1].forward_parameters(torch.Size(batch_shape[1:]))), i
                    else:
                        yield ParamItem(s[0], None), i
        elif not self.same_on_batch and not self.random_apply:
            for i, nchild in enumerate(self.named_children()):
                if isinstance(nchild[1], (_AugmentationBase, MixAugmentationBase, SequentialBase)):
                    yield ParamItem(nchild[0], nchild[1].forward_parameters(torch.Size(batch_shape[1:]))), i
                else:
                    yield ParamItem(nchild[0], None), i
        elif not self.random_apply:
            # same_on_batch + not random_apply => location-wise augmentation
            for i, nchild in enumerate(islice(cycle(self.named_children()), batch_shape[0])):
                if isinstance(nchild[1], (_AugmentationBase, MixAugmentationBase, SequentialBase)):
                    yield ParamItem(nchild[0], nchild[1].forward_parameters(torch.Size(batch_shape[1:]))), i
                else:
                    yield ParamItem(nchild[0], None), i
        else:
            # same_on_batch + random_apply => location-wise augmentation
            with_mix = False
            for i in range(batch_shape[0]):
                seq, mix_added = self.get_random_forward_sequence(with_mix=with_mix)
                with_mix = mix_added
                for s in seq:
                    if isinstance(s[1], (_AugmentationBase, MixAugmentationBase, SequentialBase)):
                        yield ParamItem(s[0], s[1].forward_parameters(torch.Size(batch_shape[1:]))), i
                    else:
                        yield ParamItem(s[0], None), i

    def apply_by_param(
        self, input: TensorWithTransformMat, label: Optional[torch.Tensor], params: PatchParamItem
    ) -> Tuple[TensorWithTransformMat, Optional[torch.Tensor], PatchParamItem]:
        _input: TensorWithTransformMat
        if isinstance(input, (tuple,)):
            in_shape = input[0].shape
            _input = (input[0][params.indices], input[1][params.indices])
        else:
            in_shape = input.shape
            _input = input[params.indices]

        _label: Optional[torch.Tensor]
        if label is not None:
            _label = label[params.indices]
        else:
            _label = label

        module = self.get_submodule(params.param.name)
        output, out_label = self.apply_to_input(_input, _label, module, params.param)

        if isinstance(module, (_AugmentationBase, MixAugmentationBase, SequentialBase)):
            out_param = ParamItem(params.param.name, module._params)
        else:
            out_param = ParamItem(params.param.name, None)

        if isinstance(output, (tuple,)) and isinstance(input, (tuple,)):
            input[0][params.indices] = output[0]
            input[1][params.indices] = output[1]
        elif isinstance(output, (tuple,)) and not isinstance(input, (tuple,)):
            input[params.indices] = output[0]
            input = (input, output[1])
        elif not isinstance(output, (tuple,)) and isinstance(input, (tuple,)):
            input[0][params.indices] = output
        elif not isinstance(output, (tuple,)) and not isinstance(input, (tuple,)):
            input[params.indices] = output

        # TODO: this label handling is naive that may not be able to handle complex cases.
        _label = None
        if label is not None and out_label is not None:
            if len(out_label.shape) == 1:
                # Wierd the mypy error though it is as same as in the next block
                _label = (
                    torch.ones(  # type: ignore
                        in_shape[0] * in_shape[1], device=out_label.device, out_label=label.dtype
                    )
                    * -1
                )
                _label = label
            else:
                _label = (
                    torch.ones(in_shape[0], *out_label.shape[1:], device=out_label.device, dtype=out_label.dtype) * -1
                )
                _label[:, 0] = label
            label[params.indices] = out_label
        elif label is None and out_label is not None:
            if len(out_label.shape) == 1:
                _label = torch.ones(in_shape[0] * in_shape[1], device=out_label.device, dtype=out_label.dtype) * -1
            else:
                _label = (
                    torch.ones(in_shape[0], *out_label.shape[1:], device=out_label.device, dtype=out_label.dtype) * -1
                )
            _label[params.indices] = out_label

        return input, label, PatchParamItem(params.indices, param=out_param)

    def forward_by_params(
        self, input: torch.Tensor, label: Optional[torch.Tensor], params: List[PatchParamItem]
    ) -> Union[TensorWithTransformMat, Tuple[TensorWithTransformMat, Optional[torch.Tensor]]]:
        _input: TensorWithTransformMat
        in_shape = input.shape
        _input = input.reshape(-1, *in_shape[-3:])

        if label is not None:
            label = torch.cat([label] * in_shape[1], dim=0)

        self.clear_state()
        for patch_param in params:
            _input, label, out_param = self.apply_by_param(_input, label, params=patch_param)
            self.update_params(out_param)
        if isinstance(_input, (tuple,)):
            _input = (_input[0].reshape(in_shape), _input[1])
        else:
            _input = _input.reshape(in_shape)
        return _input, label

    def forward(  # type: ignore
        self, input: torch.Tensor, label: Optional[torch.Tensor] = None, params: Optional[List[PatchParamItem]] = None
    ) -> Union[TensorWithTransformMat, Tuple[TensorWithTransformMat, torch.Tensor]]:
        """Input transformation will be returned if input is a tuple."""
        # BCHW -> B(patch)CHW
        if isinstance(input, (tuple,)):
            raise ValueError("tuple input is not currently supported.")
        _input: TensorWithTransformMat

        pad = self.compute_padding(input, self.padding)
        input = self.extract_patches(input, self.grid_size, pad)
        if label is not None:
            assert label.dim() == 1
            # repeat label as the same number as input patches.
            label = torch.stack([label] * self.grid_size[0] * self.grid_size[1]).reshape(-1)
        if params is None:
            params = self.forward_parameters(input.shape)

        _input, label = self.forward_by_params(input, label, params)

        if isinstance(_input, (tuple,)):
            _input = (self.restore_from_patches(_input[0], self.grid_size, pad=pad), _input[1])
        else:
            _input = self.restore_from_patches(_input, self.grid_size, pad=pad)

        self.return_label = label is not None or self.contains_label_operations(params)

        return self.__packup_output__(_input, label)
