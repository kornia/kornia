from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterator, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn

from kornia.augmentation import MixAugmentationBase
from kornia.augmentation._2d.mix.base import MixAugmentationBaseV2
from kornia.augmentation.base import _AugmentationBase

__all__ = ["SequentialBase", "ParamItem"]


class ParamItem(NamedTuple):
    name: str
    data: dict | list | None


class SequentialBase(nn.Sequential):
    r"""SequentialBase for creating kornia modulized processing pipeline.

    Args:
        *args : a list of kornia augmentation and image operation modules.
        same_on_batch: apply the same transformation across the batch.
            If None, it will not overwrite the function-wise settings.
        return_transform: if ``True`` return the matrix describing the transformation
            applied to each. If None, it will not overwrite the function-wise settings.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
            to the batch form (False). If None, it will not overwrite the function-wise settings.
    """

    def __init__(
        self,
        *args: nn.Module,
        same_on_batch: bool | None = None,
        return_transform: bool | None = None,
        keepdim: bool | None = None,
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
        self._params: list[ParamItem] | None = None
        self.update_attribute(same_on_batch, return_transform, keepdim)

    def update_attribute(
        self, same_on_batch: bool | None = None, return_transform: bool | None = None, keepdim: bool | None = None
    ) -> None:
        for mod in self.children():
            # MixAugmentation does not have return transform
            if isinstance(mod, (_AugmentationBase, MixAugmentationBase, MixAugmentationBaseV2)):
                if same_on_batch is not None:
                    mod.same_on_batch = same_on_batch
                if keepdim is not None:
                    mod.keepdim = keepdim
            if isinstance(mod, _AugmentationBase):
                if return_transform is not None:
                    mod.return_transform = return_transform
            if isinstance(mod, SequentialBase):
                mod.update_attribute(same_on_batch, return_transform, keepdim)

    def get_submodule(self, target: str) -> nn.Module:
        """Get submodule.

        This code is taken from torch 1.9.0 since it is not introduced
        back to torch 1.7.1. We included this for maintaining more
        backward torch versions.

        Args:
            target: The fully-qualified string name of the submodule
                to look for. (See above example for how to specify a
                fully-qualified string.)

        Returns:
            torch.nn.Module: The submodule referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``nn.Module``
        """
        if target == "":
            return self

        atoms: list[str] = target.split(".")
        mod: torch.nn.Module = self

        for item in atoms:

            if not hasattr(mod, item):
                raise AttributeError(mod._get_name() + " has no " "attribute `" + item + "`")

            mod = getattr(mod, item)

            if not isinstance(mod, torch.nn.Module):
                raise AttributeError("`" + item + "` is not " "an nn.Module")

        return mod

    @property
    def same_on_batch(self) -> bool | None:
        return self._same_on_batch

    @same_on_batch.setter
    def same_on_batch(self, same_on_batch: bool | None) -> None:
        self._same_on_batch = same_on_batch
        self.update_attribute(same_on_batch=same_on_batch)

    @property
    def return_transform(self) -> bool | None:
        return self._return_transform

    @return_transform.setter
    def return_transform(self, return_transform: bool | None) -> None:
        self._return_transform = return_transform
        self.update_attribute(return_transform=return_transform)

    @property
    def keepdim(self) -> bool | None:
        return self._keepdim

    @keepdim.setter
    def keepdim(self, keepdim: bool | None) -> None:
        self._keepdim = keepdim
        self.update_attribute(keepdim=keepdim)

    def clear_state(self) -> None:
        """Reset self._params state to None."""
        self._params = None

    def update_params(self, param: Any) -> None:
        """Update self._params state."""
        if self._params is None:
            self._params = [param]
        else:
            self._params.append(param)

    # TODO: Implement this for all submodules.
    def forward_parameters(self, batch_shape: torch.Size) -> list[ParamItem]:
        raise NotImplementedError

    def get_children_by_indices(self, indices: torch.Tensor) -> Iterator[tuple[str, nn.Module]]:
        modules = list(self.named_children())
        for idx in indices:
            yield modules[idx]

    def get_children_by_params(self, params: list[ParamItem]) -> Iterator[tuple[str, nn.Module]]:
        modules = list(self.named_children())
        # TODO: Wrong params passed here when nested ImageSequential
        for param in params:
            yield modules[list(dict(self.named_children()).keys()).index(param.name)]

    def get_params_by_module(self, named_modules: Iterator[tuple[str, nn.Module]]) -> Iterator[ParamItem]:
        # This will not take module._params
        for name, _ in named_modules:
            yield ParamItem(name, None)

    def contains_label_operations(self, params: list) -> bool:
        raise NotImplementedError

    def autofill_dim(self, input: torch.Tensor, dim_range: tuple[int, int] = (2, 4)) -> tuple[torch.Size, torch.Size]:
        """Fill tensor dim to the upper bound of dim_range.

        If input tensor dim is smaller than the lower bound of dim_range, an error will be thrown out.
        """
        ori_shape = input.shape
        if len(ori_shape) < dim_range[0] or len(ori_shape) > dim_range[1]:
            raise RuntimeError(f"input shape expected to be in {dim_range} while got {ori_shape}.")
        while len(input.shape) < dim_range[1]:
            input = input[None]
        return ori_shape, input.shape
