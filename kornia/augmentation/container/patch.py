from itertools import cycle, islice
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch

import kornia.augmentation as K
from kornia.augmentation.base import _AugmentationBase
from kornia.contrib.extract_patches import extract_tensor_patches
from kornia.core import Module, Tensor, concatenate
from kornia.core import pad as fpad
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints

from .base import SequentialBase
from .image import ImageSequential
from .ops import InputSequentialOps
from .params import ParamItem, PatchParamItem

__all__ = ["PatchSequential"]


class PatchSequential(ImageSequential):
    r"""Container for performing patch-level image data augmentation.

    .. image:: _static/img/PatchSequential.png

    PatchSequential breaks input images into patches by a given grid size, which will be resembled back
    afterwards.

    Different image processing and augmentation methods will be performed on each patch region as
    in :cite:`lin2021patch`.

    Args:
        *args: a list of processing modules.
        grid_size: controls the grid board separation.
        padding: same or valid padding. If same padding, it will pad to include all pixels if the input
            tensor cannot be divisible by grid_size. If valid padding, the redundant border will be removed.
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
        See a working example `here <https://kornia.github.io/tutorials/nbs/data_patch_sequential.html>`__.

    Examples:
        >>> import kornia.augmentation as K
        >>> input = torch.randn(2, 3, 224, 224)
        >>> seq = PatchSequential(
        ...     ImageSequential(
        ...         K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.5),
        ...         K.RandomPerspective(0.2, p=0.5),
        ...         K.RandomSolarize(0.1, 0.1, p=0.5),
        ...     ),
        ...     K.RandomAffine(360, p=1.0),
        ...     ImageSequential(
        ...         K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.5),
        ...         K.RandomPerspective(0.2, p=0.5),
        ...         K.RandomSolarize(0.1, 0.1, p=0.5),
        ...     ),
        ...     K.RandomSolarize(0.1, 0.1, p=0.1),
        ...     grid_size=(2,2),
        ...     patchwise_apply=True,
        ...     same_on_batch=True,
        ...     random_apply=False,
        ... )
        >>> out = seq(input)
        >>> out.shape
        torch.Size([2, 3, 224, 224])
        >>> out1 = seq(input, params=seq._params)
        >>> torch.equal(out, out1)
        True

    Perform ``OneOf`` transformation with ``random_apply=1`` and ``random_apply_weights`` in ``PatchSequential``.

        >>> import kornia
        >>> input = torch.randn(2, 3, 224, 224)
        >>> seq = PatchSequential(
        ...     ImageSequential(
        ...         K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.5),
        ...         K.RandomPerspective(0.2, p=0.5),
        ...         K.RandomSolarize(0.1, 0.1, p=0.5),
        ...     ),
        ...     K.RandomAffine(360, p=1.0),
        ...     K.RandomSolarize(0.1, 0.1, p=0.1),
        ...     grid_size=(2,2),
        ...     patchwise_apply=False,
        ...     random_apply=1,
        ...     random_apply_weights=[0.5, 0.3, 0.8]
        ... )
        >>> out = seq(input)
        >>> out.shape
        torch.Size([2, 3, 224, 224])
    """

    def __init__(
        self,
        *args: Module,
        grid_size: Tuple[int, int] = (4, 4),
        padding: str = "same",
        same_on_batch: Optional[bool] = None,
        keepdim: Optional[bool] = None,
        patchwise_apply: bool = True,
        random_apply: Union[int, bool, Tuple[int, int]] = False,
        random_apply_weights: Optional[List[float]] = None,
    ) -> None:
        _random_apply: Optional[Union[int, Tuple[int, int]]]

        if patchwise_apply and random_apply is True:
            # will only apply [1, 4] augmentations per patch
            _random_apply = (1, 4)
        elif patchwise_apply and random_apply is False:
            if len(args) != grid_size[0] * grid_size[1]:
                raise ValueError(
                    "The number of processing modules must be equal with grid size."
                    f"Got {len(args)} and {grid_size[0] * grid_size[1]}. "
                    "Please set random_apply = True or patchwise_apply = False."
                )
            _random_apply = random_apply
        elif patchwise_apply and isinstance(random_apply, (int, tuple)):
            raise ValueError(f"Only boolean value allowed when `patchwise_apply` is set to True. Got {random_apply}.")
        else:
            _random_apply = random_apply
        super().__init__(
            *args,
            same_on_batch=same_on_batch,
            keepdim=keepdim,
            random_apply=_random_apply,
            random_apply_weights=random_apply_weights,
        )
        if padding not in ("same", "valid"):
            raise ValueError(f"`padding` must be either `same` or `valid`. Got {padding}.")
        self.grid_size = grid_size
        self.padding = padding
        self.patchwise_apply = patchwise_apply
        self._params: Optional[List[PatchParamItem]]  # type: ignore[assignment]

    def compute_padding(
        self, input: Tensor, padding: str, grid_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[int, int, int, int]:
        if grid_size is None:
            grid_size = self.grid_size
        if padding == "valid":
            ph, pw = input.size(-2) // grid_size[0], input.size(-1) // grid_size[1]
            return (-pw // 2, pw // 2 - pw, -ph // 2, ph // 2 - ph)
        if padding == "same":
            ph = input.size(-2) - input.size(-2) // grid_size[0] * grid_size[0]
            pw = input.size(-1) - input.size(-1) // grid_size[1] * grid_size[1]
            return (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
        raise NotImplementedError(f"Expect `padding` as either 'valid' or 'same'. Got {padding}.")

    def extract_patches(
        self,
        input: Tensor,
        grid_size: Optional[Tuple[int, int]] = None,
        pad: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tensor:
        """Extract patches from tensor.

        Example:
            >>> import kornia.augmentation as K
            >>> pas = PatchSequential(K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0), patchwise_apply=False)
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
            input = fpad(input, list(pad))
        if grid_size is None:
            grid_size = self.grid_size
        window_size = (input.size(-2) // grid_size[-2], input.size(-1) // grid_size[-1])
        stride = window_size
        return extract_tensor_patches(input, window_size, stride)

    def restore_from_patches(
        self, patches: Tensor, grid_size: Tuple[int, int] = (4, 4), pad: Optional[Tuple[int, int, int, int]] = None
    ) -> Tensor:
        """Restore input from patches.

        Example:
            >>> import kornia.augmentation as K
            >>> pas = PatchSequential(K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0), patchwise_apply=False)
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
        restored_tensor = concatenate(torch.chunk(patches_tensor, grid_size[0], 1), -2).squeeze(1)
        restored_tensor = concatenate(torch.chunk(restored_tensor, grid_size[1], 1), -1).squeeze(1)

        if pad is not None:
            restored_tensor = fpad(restored_tensor, [-i for i in pad])
        return restored_tensor

    def forward_parameters(self, batch_shape: torch.Size) -> List[PatchParamItem]:  # type: ignore[override]
        out_param: List[PatchParamItem] = []
        if not self.patchwise_apply:
            params = self.generate_parameters(torch.Size([1, batch_shape[0] * batch_shape[1], *batch_shape[2:]]))
            indices = torch.arange(0, batch_shape[0] * batch_shape[1])
            out_param = [PatchParamItem(indices.tolist(), p) for p, _ in params]
            # "append" of "list" does not return a value
        elif not self.same_on_batch:
            params = self.generate_parameters(torch.Size([batch_shape[0] * batch_shape[1], 1, *batch_shape[2:]]))
            out_param = [PatchParamItem([i], p) for p, i in params]
            # "append" of "list" does not return a value
        else:
            params = self.generate_parameters(torch.Size([batch_shape[1], batch_shape[0], *batch_shape[2:]]))
            indices = torch.arange(0, batch_shape[0] * batch_shape[1], step=batch_shape[1])
            out_param = [PatchParamItem((indices + i).tolist(), p) for p, i in params]
            # "append" of "list" does not return a value
        return out_param

    def generate_parameters(self, batch_shape: torch.Size) -> Iterator[Tuple[ParamItem, int]]:
        """Get multiple forward sequence but maximumly one mix augmentation in between.

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
                    if isinstance(s[1], (_AugmentationBase, SequentialBase, K.MixAugmentationBaseV2)):
                        yield ParamItem(s[0], s[1].forward_parameters(torch.Size(batch_shape[1:]))), i
                    else:
                        yield ParamItem(s[0], None), i
        elif not self.same_on_batch and not self.random_apply:
            for i, nchild in enumerate(self.named_children()):
                if isinstance(nchild[1], (_AugmentationBase, SequentialBase, K.MixAugmentationBaseV2)):
                    yield ParamItem(nchild[0], nchild[1].forward_parameters(torch.Size(batch_shape[1:]))), i
                else:
                    yield ParamItem(nchild[0], None), i
        elif not self.random_apply:
            # same_on_batch + not random_apply => location-wise augmentation
            for i, nchild in enumerate(islice(cycle(self.named_children()), batch_shape[0])):
                if isinstance(nchild[1], (_AugmentationBase, SequentialBase, K.MixAugmentationBaseV2)):
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
                    if isinstance(s[1], (_AugmentationBase, SequentialBase, K.MixAugmentationBaseV2)):
                        yield ParamItem(s[0], s[1].forward_parameters(torch.Size(batch_shape[1:]))), i
                    else:
                        yield ParamItem(s[0], None), i

    def forward_by_params(self, input: Tensor, params: List[PatchParamItem]) -> Tensor:
        in_shape = input.shape
        input = input.reshape(-1, *in_shape[-3:])

        for patch_param in params:
            # input, out_param = self.apply_by_param(input, params=patch_param)
            module = self.get_submodule(patch_param.param.name)
            _input = input[patch_param.indices]
            output = InputSequentialOps.transform(_input, module, patch_param.param, extra_args={})
            input[patch_param.indices] = output

        return input.reshape(in_shape)

    def transform_inputs(  # type: ignore[override]
        self, input: Tensor, params: List[PatchParamItem], extra_args: Dict[str, Any] = {}
    ) -> Tensor:
        pad = self.compute_padding(input, self.padding)
        input = self.extract_patches(input, self.grid_size, pad)
        input = self.forward_by_params(input, params)
        input = self.restore_from_patches(input, self.grid_size, pad=pad)

        return input

    def inverse_inputs(  # type: ignore[override]
        self, input: Tensor, params: List[PatchParamItem], extra_args: Dict[str, Any] = {}
    ) -> Tensor:
        if self.is_intensity_only():
            return input

        raise NotImplementedError("PatchSequential inverse cannot be used with geometric transformations.")

    def transform_masks(  # type: ignore[override]
        self, input: Tensor, params: List[PatchParamItem], extra_args: Dict[str, Any] = {}
    ) -> Tensor:
        if self.is_intensity_only():
            return input

        raise NotImplementedError("PatchSequential for boxes cannot be used with geometric transformations.")

    def inverse_masks(  # type: ignore[override]
        self, input: Tensor, params: List[PatchParamItem], extra_args: Dict[str, Any] = {}
    ) -> Tensor:
        if self.is_intensity_only():
            return input

        raise NotImplementedError("PatchSequential inverse cannot be used with geometric transformations.")

    def transform_boxes(  # type: ignore[override]
        self, input: Boxes, params: List[PatchParamItem], extra_args: Dict[str, Any] = {}
    ) -> Boxes:
        if self.is_intensity_only():
            return input

        raise NotImplementedError("PatchSequential for boxes cannot be used with geometric transformations.")

    def inverse_boxes(  # type: ignore[override]
        self, input: Boxes, params: List[PatchParamItem], extra_args: Dict[str, Any] = {}
    ) -> Boxes:
        if self.is_intensity_only():
            return input

        raise NotImplementedError("PatchSequential inverse cannot be used with geometric transformations.")

    def transform_keypoints(  # type: ignore[override]
        self, input: Keypoints, params: List[PatchParamItem], extra_args: Dict[str, Any] = {}
    ) -> Keypoints:
        if self.is_intensity_only():
            return input

        raise NotImplementedError("PatchSequential for keypoints cannot be used with geometric transformations.")

    def inverse_keypoints(  # type: ignore[override]
        self, input: Keypoints, params: List[PatchParamItem], extra_args: Dict[str, Any] = {}
    ) -> Keypoints:
        if self.is_intensity_only():
            return input

        raise NotImplementedError("PatchSequential inverse cannot be used with geometric transformations.")

    def inverse(  # type: ignore[override]
        self, input: Tensor, params: Optional[List[PatchParamItem]] = None, extra_args: Dict[str, Any] = {}
    ) -> Tensor:
        """Inverse transformation.

        Used to inverse a tensor according to the performed transformation by a forward pass, or with respect to
        provided parameters.
        """
        if self.is_intensity_only():
            return input

        raise NotImplementedError("PatchSequential inverse cannot be used with geometric transformations.")

    def forward(self, input: Tensor, params: Optional[List[PatchParamItem]] = None) -> Tensor:  # type: ignore[override]
        """Input transformation will be returned if input is a tuple."""
        # BCHW -> B(patch)CHW
        if isinstance(input, (tuple,)):
            raise ValueError("tuple input is not currently supported.")

        if params is None:
            params = self.forward_parameters(input.shape)

        output = self.transform_inputs(input, params=params)

        self._params = params

        return output
