from collections import OrderedDict
from itertools import zip_longest
from typing import Any, Iterator, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn

from kornia.augmentation.base import _AugmentationBase

__all__ = ["ImageSequential"]


class ParamItem(NamedTuple):
    name: str
    data: Union[dict, list]


class ImageSequential(nn.Sequential):
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
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: the tensor (, and the transformation matrix)
            has been sequentially modified by the args.

    Examples:
        >>> import kornia
        >>> input = torch.randn(2, 3, 5, 6)
        >>> aug_list = ImageSequential(
        ...     kornia.color.BgrToRgb(),
        ...     kornia.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...     kornia.filters.MedianBlur((3, 3)),
        ...     kornia.augmentation.RandomAffine(360, p=1.0),
        ...     kornia.enhance.Invert(),
        ... return_transform=True,
        ... same_on_batch=True,
        ... random_apply=10,
        ... )
        >>> out = aug_list(input)
        >>> out[0].shape, out[1].shape
        (torch.Size([2, 3, 5, 6]), torch.Size([2, 3, 3]))

        Reproduce with provided params.
        >>> out2 = aug_list(input, params=aug_list._params)
        >>> torch.equal(out[0], out2[0]), torch.equal(out[1], out2[1])
        (True, True)

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
        self.same_on_batch = same_on_batch
        self.return_transform = return_transform
        self.keepdim = keepdim
        # To name the modules properly
        _args = OrderedDict()
        for idx, arg in enumerate(args):
            if not isinstance(arg, nn.Module):
                raise NotImplementedError(f"Only nn.Module are supported at this moment. Got {arg}.")
            if isinstance(arg, _AugmentationBase):
                if same_on_batch is not None:
                    arg.same_on_batch = same_on_batch
                if return_transform is not None:
                    arg.return_transform = return_transform
                if keepdim is not None:
                    arg.keepdim = keepdim
            _args.update({f"{arg.__class__.__name__}_{idx}": arg})
        super(ImageSequential, self).__init__(_args)

        self._params: List[Any] = []
        self.random_apply: Union[Tuple[int, int], bool]
        if random_apply:
            if isinstance(random_apply, (bool,)) and random_apply is True:
                self.random_apply = (len(args), len(args) + 1)
            elif isinstance(random_apply, (int,)):
                self.random_apply = (random_apply, random_apply + 1)
            elif (
                isinstance(random_apply, (tuple,))
                and len(random_apply) == 2
                and isinstance(random_apply[0], (int,))
                and isinstance(random_apply[1], (int,))
            ):
                self.random_apply = (random_apply[0], random_apply[1] + 1)
            elif isinstance(random_apply, (tuple,)) and len(random_apply) == 1 and isinstance(random_apply[0], (int,)):
                self.random_apply = (random_apply[0], len(args) + 1)
            else:
                raise ValueError(f"Non-readable random_apply. Got {random_apply}.")
            assert (
                isinstance(self.random_apply, (tuple,))
                and len(self.random_apply) == 2
                and isinstance(self.random_apply[0], (int,))
                and isinstance(self.random_apply[0], (int,))
            ), f"Expect a tuple of (int, int). Got {self.random_apply}."
        else:
            self.random_apply = False

    def _get_child_sequence(self) -> Iterator[Tuple[str, nn.Module]]:
        if self.random_apply:
            # random_apply will not be boolean here.
            num_samples = int(torch.randint(*self.random_apply, (1,)).item())  # type: ignore
            indices = torch.multinomial(
                torch.ones((len(self),)), num_samples, replacement=True if num_samples > len(self) else False
            )
            return self._get_children_by_indices(indices)
        return self.named_children()

    def _get_children_by_indices(self, indices: torch.Tensor) -> Iterator[Tuple[str, nn.Module]]:
        modules = list(self.named_children())
        for idx in indices:
            yield modules[idx]

    def _get_children_by_module_names(self, names: List[str]) -> Iterator[Tuple[str, nn.Module]]:
        modules = list(self.named_children())
        for name in names:
            yield modules[list(dict(self.named_children()).keys()).index(name)]

    def get_forward_sequence(self, params: Optional[List[ParamItem]] = None) -> Iterator[Tuple[str, nn.Module]]:
        if params is None:
            named_modules = self._get_child_sequence()
        else:
            named_modules = self._get_children_by_module_names([p.name for p in params])
        return named_modules

    def apply_to_input(
        self,
        input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        module_name: str,
        module: Optional[nn.Module] = None,
        param: Optional[ParamItem] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if module is None:
            # TODO (jian): double check why typing is crashing
            module = self.get_submodule(module_name)  # type: ignore
        if param is not None:
            assert module_name == param.name
            _param = param.data
        else:
            _param = None  # type: ignore

        if isinstance(module, (_AugmentationBase, ImageSequential)) and _param is None:
            input = module(input)
            self._params.append(ParamItem(module_name, module._params))
        elif isinstance(module, (_AugmentationBase, ImageSequential)) and _param is not None:
            input = module(input, params=_param)
            self._params.append(ParamItem(module_name, _param))
        else:
            assert (
                _param == {} or _param is None
            ), f"Non-augmentaion operation {module_name} require empty parameters. Got {module}."
            # In case of return_transform = True
            if isinstance(input, (tuple, list)):
                input = (module(input[0]), input[1])
            else:
                input = module(input)
            self._params.append(ParamItem(module_name, {}))
        return input

    def forward(
        self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], params: Optional[List[ParamItem]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        self._params = []
        named_modules = self.get_forward_sequence(params)
        params = [] if params is None else params
        for (name, module), param in zip_longest(named_modules, params):
            input = self.apply_to_input(input, name, module, param=param)  # type: ignore
        return input
