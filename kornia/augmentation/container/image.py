from itertools import zip_longest
from typing import Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from kornia.augmentation.base import _AugmentationBase, MixAugmentationBase, TensorWithTransformMat

from .base import ParamItem, SequentialBase

__all__ = ["ImageSequential"]


# TODO: Add forward_parameters for ImageSequential
class ImageSequential(SequentialBase):
    r"""Sequential for creating kornia image processing pipeline.

    Args:
        *args : a list of kornia augmentation and image operation modules.
        same_on_batch: apply the same transformation across the batch.
            If None, it will not overwrite the function-wise settings.
        return_transform: if ``True`` return the matrix describing the transformation
            applied to each. If None, it will not overwrite the function-wise settings.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
            to the batch form (False). If None, it will not overwrite the function-wise settings.
        random_apply: randomly select a sublist (order agnostic) of args to
            apply transformation.
            If int, a fixed number of transformations will be selected.
            If (a,), x number of transformations (a <= x <= len(args)) will be selected.
            If (a, b), x number of transformations (a <= x <= b) will be selected.
            If True, the whole list of args will be processed as a sequence in a random order.
            If False, the whole list of args will be processed as a sequence in original order.

    Returns:
        TensorWithTransformMat: the tensor (, and the transformation matrix)
            has been sequentially modified by the args.

    Examples:
        >>> _ = torch.manual_seed(77)
        >>> import kornia
        >>> input, label = torch.randn(2, 3, 5, 6), torch.tensor([0, 1])
        >>> aug_list = ImageSequential(
        ...     kornia.color.BgrToRgb(),
        ...     kornia.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...     kornia.filters.MedianBlur((3, 3)),
        ...     kornia.augmentation.RandomAffine(360, p=1.0),
        ...     kornia.enhance.Invert(),
        ...     kornia.augmentation.RandomMixUp(p=1.0),
        ... return_transform=True,
        ... same_on_batch=True,
        ... random_apply=10,
        ... )
        >>> out, lab = aug_list(input, label=label)
        >>> lab
        tensor([[0.0000, 0.0000, 0.2746],
                [1.0000, 1.0000, 0.1576]])
        >>> out[0].shape, out[1].shape
        (torch.Size([2, 3, 5, 6]), torch.Size([2, 3, 3]))

        Reproduce with provided params.
        >>> out2, lab2 = aug_list(input, label=label, params=aug_list._params)
        >>> torch.equal(out[0], out2[0]), torch.equal(out[1], out2[1]), torch.equal(lab[1], lab2[1])
        (True, True, True)

    Note:
        Transformation matrix returned only considers the transformation applied in ``kornia.augmentation`` module.
        Those transformations in ``kornia.geometry`` will not be taken into account.
    """

    def __init__(
        self,
        *args: nn.Module,
        same_on_batch: Optional[bool] = None,
        return_transform: Optional[bool] = None,
        keepdim: Optional[bool] = None,
        random_apply: Union[int, bool, Tuple[int, int]] = False,
    ) -> None:
        super(ImageSequential, self).__init__(
            *args, same_on_batch=same_on_batch, return_transform=return_transform, keepdim=keepdim
        )

        self.random_apply: Union[Tuple[int, int], bool] = self._read_random_apply(random_apply, len(args))

    def _read_random_apply(
        self, random_apply: Union[int, bool, Tuple[int, int]], max_length: int
    ) -> Union[Tuple[int, int], bool]:
        if isinstance(random_apply, (bool,)) and random_apply is False:
            random_apply = False
        elif isinstance(random_apply, (bool,)) and random_apply is True:
            random_apply = (max_length, max_length + 1)
        elif isinstance(random_apply, (int,)):
            random_apply = (random_apply, random_apply + 1)
        elif (
            isinstance(random_apply, (tuple,))
            and len(random_apply) == 2
            and isinstance(random_apply[0], (int,))
            and isinstance(random_apply[1], (int,))
        ):
            random_apply = (random_apply[0], random_apply[1] + 1)
        elif isinstance(random_apply, (tuple,)) and len(random_apply) == 1 and isinstance(random_apply[0], (int,)):
            random_apply = (random_apply[0], max_length + 1)
        else:
            raise ValueError(f"Non-readable random_apply. Got {random_apply}.")
        if random_apply is not False:
            assert (
                isinstance(random_apply, (tuple,))
                and len(random_apply) == 2
                and isinstance(random_apply[0], (int,))
                and isinstance(random_apply[0], (int,))
            ), f"Expect a tuple of (int, int). Got {random_apply}."
        return random_apply

    def get_random_forward_sequence(self, with_mix: bool = True) -> Tuple[Iterator[Tuple[str, nn.Module]], bool]:
        num_samples = int(torch.randint(*self.random_apply, (1,)).item())  # type: ignore
        multinomial_weights = torch.ones((len(self),))
        # Mix augmentation can only be applied once per forward
        mix_indices = self.get_mix_augmentation_indices(self.named_children())
        # kick out the mix augmentations
        multinomial_weights[mix_indices] = 0
        indices = torch.multinomial(
            multinomial_weights,
            num_samples,
            # enable replacement if non-mix augmentation is less than required
            replacement=True if num_samples > multinomial_weights.sum() else False,
        )

        mix_added = False
        if with_mix and len(mix_indices) != 0:
            # Make the selection fair.
            if (torch.rand(1) < ((len(mix_indices) + len(indices)) / len(self))).item():
                indices[-1] = torch.multinomial((~multinomial_weights.bool()).float(), 1)
                indices = indices[torch.randperm(len(indices))]
                mix_added = True

        return self.get_children_by_indices(indices), mix_added

    def get_mix_augmentation_indices(self, named_modules: Iterator[Tuple[str, nn.Module]]) -> List[int]:
        indices = []
        for idx, (_, child) in enumerate(named_modules):
            if isinstance(child, (MixAugmentationBase,)):
                indices.append(idx)
        return indices

    def get_forward_sequence(self, params: Optional[List[ParamItem]] = None) -> Iterator[Tuple[str, nn.Module]]:
        if params is None:
            # Mix augmentation can only be applied once per forward
            mix_indices = self.get_mix_augmentation_indices(self.named_children())

            if self.random_apply:
                return self.get_random_forward_sequence()[0]

            if len(mix_indices) > 1:
                raise ValueError(
                    "Multiple mix augmentation is prohibited without enabling random_apply."
                    f"Detected {len(mix_indices)}."
                )

            return self.named_children()
        else:
            named_modules = self.get_children_by_params(params)
        return named_modules

    def _apply_operation(
        self, input: TensorWithTransformMat, label: Optional[torch.Tensor], module: nn.Module, param: ParamItem
    ) -> Tuple[TensorWithTransformMat, Optional[torch.Tensor], ParamItem]:
        if isinstance(module, (MixAugmentationBase,)):
            input, label = module(input, label, params=param.data)
            out_param = ParamItem(param.name, module._params)
        elif isinstance(module, (_AugmentationBase, ImageSequential)):
            input = module(input, params=param.data)
            out_param = ParamItem(param.name, module._params)
        else:
            assert param.data is None, f"Non-augmentaion operation {param.name} require empty parameters. Got {param}."
            # In case of return_transform = True
            if isinstance(input, (tuple, list)):
                input = (module(input[0]), input[1])
            else:
                input = module(input)
            out_param = ParamItem(param.name, None)
        return input, label, out_param

    def apply_to_input(
        self,
        input: TensorWithTransformMat,
        label: Optional[torch.Tensor],
        module: Optional[nn.Module],
        param: ParamItem,
    ) -> Tuple[TensorWithTransformMat, Optional[torch.Tensor]]:
        if module is None:
            module = self.get_submodule(param.name)

        input, label, out_param = self._apply_operation(input, label, module, param)
        self._params.append(out_param)

        return input, label

    def __packup_output__(
        self, output: TensorWithTransformMat, label: Optional[torch.Tensor] = None
    ) -> Union[TensorWithTransformMat, Tuple[TensorWithTransformMat, torch.Tensor]]:
        if self.return_label:
            return output, label  # type: ignore
            # Implicitly indicating the label cannot be optional since there is a mix aug
        return output

    def forward(  # type: ignore
        self,
        input: TensorWithTransformMat,
        label: Optional[torch.Tensor] = None,
        params: Optional[List[ParamItem]] = None,
    ) -> Union[TensorWithTransformMat, Tuple[TensorWithTransformMat, torch.Tensor]]:
        self.clear_state()
        named_modules = list(self.get_forward_sequence(params))
        if params is None:
            params = list(self.get_params_by_module(iter(named_modules)))
        self.return_label = self.get_mix_augmentation_indices(iter(named_modules))
        for (_, module), param in zip_longest(named_modules, params):
            input, label = self.apply_to_input(input, label, module=module, param=param)
        return self.__packup_output__(input, label)
