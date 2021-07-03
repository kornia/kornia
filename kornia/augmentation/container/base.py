from collections import OrderedDict
from typing import Any, Iterator, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn

from kornia.augmentation.base import _AugmentationBase, MixAugmentationBase

__all__ = ["SequentialBase", "ParamItem"]


class ParamItem(NamedTuple):
    name: str
    data: Optional[Union[dict, list]]


class SequentialBase(nn.Sequential):
    def __init__(
        self,
        *args: nn.Module,
        same_on_batch: Optional[bool] = None,
        return_transform: Optional[bool] = None,
        keepdim: Optional[bool] = None,
    ) -> None:
        # To name the modules properly
        _args = OrderedDict()
        for idx, mod in enumerate(args):
            if not isinstance(mod, nn.Module):
                raise NotImplementedError(f"Only nn.Module are supported at this moment. Got {mod}.")
            _args.update({f"{mod.__class__.__name__}_{idx}": mod})
        super().__init__(_args)
        self._same_on_batch = same_on_batch
        self._return_transform = return_transform
        self._keepdim = keepdim
        self._params: List[Any] = []
        self.update_attribute(same_on_batch, return_transform, keepdim)

    def update_attribute(
        self,
        same_on_batch: Optional[bool] = None,
        return_transform: Optional[bool] = None,
        keepdim: Optional[bool] = None,
    ) -> None:
        for mod in self.children():
            # MixAugmentation does not have return transform
            if isinstance(mod, _AugmentationBase) or isinstance(mod, MixAugmentationBase):
                if same_on_batch is not None:
                    mod.same_on_batch = same_on_batch
                if keepdim is not None:
                    mod.keepdim = keepdim
            if isinstance(mod, _AugmentationBase):
                if return_transform is not None:
                    mod.return_transform = return_transform

    @property
    def same_on_batch(self) -> Optional[bool]:
        return self._same_on_batch

    @same_on_batch.setter
    def same_on_batch(self, same_on_batch: Optional[bool]) -> None:
        self._same_on_batch = same_on_batch
        self.update_attribute(same_on_batch=same_on_batch)

    @property
    def return_transform(self) -> Optional[bool]:
        return self._return_transform

    @return_transform.setter
    def return_transform(self, return_transform: Optional[bool]) -> None:
        self._return_transform = return_transform
        self.update_attribute(return_transform=return_transform)

    @property
    def keepdim(self) -> Optional[bool]:
        return self._keepdim

    @keepdim.setter
    def keepdim(self, keepdim: Optional[bool]) -> None:
        self._keepdim = keepdim
        self.update_attribute(keepdim=keepdim)

    def clear_state(self) -> None:
        self._params = []

    def get_children_by_indices(self, indices: torch.Tensor) -> Iterator[Tuple[str, nn.Module]]:
        modules = list(self.named_children())
        for idx in indices:
            yield modules[idx]

    def get_children_by_params(self, params: List[ParamItem]) -> Iterator[Tuple[str, nn.Module]]:
        modules = list(self.named_children())
        for param in params:
            yield modules[list(dict(self.named_children()).keys()).index(param.name)]

    def get_params_by_module(self, named_modules: Iterator[Tuple[str, nn.Module]]) -> Iterator[ParamItem]:
        # This will not take module._params
        for name, _ in named_modules:
            yield ParamItem(name, None)
