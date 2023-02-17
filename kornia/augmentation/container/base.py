from collections import OrderedDict
from itertools import zip_longest
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

import kornia.augmentation as K
from kornia.augmentation.base import _AugmentationBase
from kornia.core import Module, Tensor
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints

from .ops import BoxSequentialOps, InputSequentialOps, KeypointSequentialOps, MaskSequentialOps
from .params import ParamItem

__all__ = ["BasicSequentialBase", "ImageSequentialBase", "SequentialBase"]


class BasicSequentialBase(nn.Sequential):
    r"""BasicSequential for creating kornia modulized processing pipeline.

    Args:
        *args : a list of kornia augmentation and image operation modules.
    """

    def __init__(self, *args: Module) -> None:
        # To name the modules properly
        _args = OrderedDict()
        for idx, mod in enumerate(args):
            if not isinstance(mod, Module):
                raise NotImplementedError(f"Only Module are supported at this moment. Got {mod}.")
            _args.update({f"{mod.__class__.__name__}_{idx}": mod})
        super().__init__(_args)
        self._params: Optional[List[ParamItem]] = None

    def get_submodule(self, target: str) -> Module:
        """Get submodule.

        This code is taken from torch 1.9.0 since it is not introduced
        back to torch 1.7.1. We included this for maintaining more
        backward torch versions.

        Args:
            target: The fully-qualified string name of the submodule
                to look for. (See above example for how to specify a
                fully-qualified string.)

        Returns:
            Module: The submodule referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``Module``
        """
        if target == "":
            return self

        atoms: List[str] = target.split(".")
        mod = self

        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(mod._get_name() + " has no " "attribute `" + item + "`")

            mod = getattr(mod, item)

            if not isinstance(mod, Module):
                raise AttributeError("`" + item + "` is not " "an Module")

        return mod

    def clear_state(self) -> None:
        """Reset self._params state to None."""
        self._params = None

    # TODO: Implement this for all submodules.
    def forward_parameters(self, batch_shape: torch.Size) -> List[ParamItem]:
        raise NotImplementedError

    def get_children_by_indices(self, indices: Tensor) -> Iterator[Tuple[str, Module]]:
        modules = list(self.named_children())
        for idx in indices:
            yield modules[idx]

    def get_children_by_params(self, params: List[ParamItem]) -> Iterator[Tuple[str, Module]]:
        modules = list(self.named_children())
        # TODO: Wrong params passed here when nested ImageSequential
        for param in params:
            yield modules[list(dict(self.named_children()).keys()).index(param.name)]

    def get_params_by_module(self, named_modules: Iterator[Tuple[str, Module]]) -> Iterator[ParamItem]:
        # This will not take module._params
        for name, _ in named_modules:
            yield ParamItem(name, None)


class SequentialBase(BasicSequentialBase):
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

    def __init__(self, *args: Module, same_on_batch: Optional[bool] = None, keepdim: Optional[bool] = None) -> None:
        # To name the modules properly
        super().__init__(*args)
        self._same_on_batch = same_on_batch
        self._keepdim = keepdim
        self.update_attribute(same_on_batch, keepdim)

    def update_attribute(
        self,
        same_on_batch: Optional[bool] = None,
        return_transform: Optional[bool] = None,
        keepdim: Optional[bool] = None,
    ) -> None:
        for mod in self.children():
            # MixAugmentation does not have return transform
            if isinstance(mod, (_AugmentationBase, K.MixAugmentationBaseV2)):
                if same_on_batch is not None:
                    mod.same_on_batch = same_on_batch
                if keepdim is not None:
                    mod.keepdim = keepdim
            if isinstance(mod, SequentialBase):
                mod.update_attribute(same_on_batch, return_transform, keepdim)

    @property
    def same_on_batch(self) -> Optional[bool]:
        return self._same_on_batch

    @same_on_batch.setter
    def same_on_batch(self, same_on_batch: Optional[bool]) -> None:
        self._same_on_batch = same_on_batch
        self.update_attribute(same_on_batch=same_on_batch)

    @property
    def keepdim(self) -> Optional[bool]:
        return self._keepdim

    @keepdim.setter
    def keepdim(self, keepdim: Optional[bool]) -> None:
        self._keepdim = keepdim
        self.update_attribute(keepdim=keepdim)

    def autofill_dim(self, input: Tensor, dim_range: Tuple[int, int] = (2, 4)) -> Tuple[torch.Size, torch.Size]:
        """Fill tensor dim to the upper bound of dim_range.

        If input tensor dim is smaller than the lower bound of dim_range, an error will be thrown out.
        """
        ori_shape = input.shape
        if len(ori_shape) < dim_range[0] or len(ori_shape) > dim_range[1]:
            raise RuntimeError(f"input shape expected to be in {dim_range} while got {ori_shape}.")
        while len(input.shape) < dim_range[1]:
            input = input[None]
        return ori_shape, input.shape


class ImageSequentialBase(SequentialBase):
    def identity_matrix(self, input) -> Tensor:
        """Return identity matrix."""
        raise NotImplementedError

    def get_transformation_matrix(
        self,
        input: Tensor,
        params: Optional[List[ParamItem]] = None,
        recompute: bool = False,
        extra_args: Dict[str, Any] = {},
    ) -> Optional[Tensor]:
        """Compute the transformation matrix according to the provided parameters.

        Args:
            input: the input tensor.
            params: params for the sequence.
            recompute: if to recompute the transformation matrix according to the params.
                default: False.
        """
        raise NotImplementedError

    def forward_parameters(self, batch_shape: torch.Size) -> List[ParamItem]:
        raise NotImplementedError

    def get_forward_sequence(self, params: Optional[List[ParamItem]] = None) -> Iterator[Tuple[str, Module]]:
        """Get module sequence by input params."""
        raise NotImplementedError

    def transform_inputs(self, input: Tensor, params: List[ParamItem], extra_args: Dict[str, Any] = {}) -> Tensor:
        for param in params:
            module = self.get_submodule(param.name)
            input = InputSequentialOps.transform(input, module=module, param=param, extra_args=extra_args)
        return input

    def inverse_inputs(self, input: Tensor, params: List[ParamItem], extra_args: Dict[str, Any] = {}) -> Tensor:
        for (name, module), param in zip_longest(list(self.get_forward_sequence(params))[::-1], params[::-1]):
            input = InputSequentialOps.inverse(input, module=module, param=param, extra_args=extra_args)
        return input

    def transform_masks(self, input: Tensor, params: List[ParamItem], extra_args: Dict[str, Any] = {}) -> Tensor:
        for param in params:
            module = self.get_submodule(param.name)
            input = MaskSequentialOps.transform(input, module=module, param=param, extra_args=extra_args)
        return input

    def inverse_masks(self, input: Tensor, params: List[ParamItem], extra_args: Dict[str, Any] = {}) -> Tensor:
        for (name, module), param in zip_longest(list(self.get_forward_sequence(params))[::-1], params[::-1]):
            input = MaskSequentialOps.inverse(input, module=module, param=param, extra_args=extra_args)
        return input

    def transform_boxes(self, input: Boxes, params: List[ParamItem], extra_args: Dict[str, Any] = {}) -> Boxes:
        for param in params:
            module = self.get_submodule(param.name)
            input = BoxSequentialOps.transform(input, module=module, param=param, extra_args=extra_args)
        return input

    def inverse_boxes(self, input: Boxes, params: List[ParamItem], extra_args: Dict[str, Any] = {}) -> Boxes:
        for (name, module), param in zip_longest(list(self.get_forward_sequence(params))[::-1], params[::-1]):
            input = BoxSequentialOps.inverse(input, module=module, param=param, extra_args=extra_args)
        return input

    def transform_keypoints(
        self, input: Keypoints, params: List[ParamItem], extra_args: Dict[str, Any] = {}
    ) -> Keypoints:
        for param in params:
            module = self.get_submodule(param.name)
            input = KeypointSequentialOps.transform(input, module=module, param=param, extra_args=extra_args)
        return input

    def inverse_keypoints(
        self, input: Keypoints, params: List[ParamItem], extra_args: Dict[str, Any] = {}
    ) -> Keypoints:
        for (name, module), param in zip_longest(list(self.get_forward_sequence(params))[::-1], params[::-1]):
            input = KeypointSequentialOps.inverse(input, module=module, param=param, extra_args=extra_args)
        return input

    def inverse(
        self, input: Tensor, params: Optional[List[ParamItem]] = None, extra_args: Dict[str, Any] = {}
    ) -> Tensor:
        """Inverse transformation.

        Used to inverse a tensor according to the performed transformation by a forward pass, or with respect to
        provided parameters.
        """
        if params is None:
            if self._params is None:
                raise ValueError(
                    "No parameters available for inversing, please run a forward pass first "
                    "or passing valid params into this function."
                )
            params = self._params

        input = self.inverse_inputs(input, params, extra_args=extra_args)

        return input

    def forward(
        self, input: Tensor, params: Optional[List[ParamItem]] = None, extra_args: Dict[str, Any] = {}
    ) -> Tensor:
        self.clear_state()

        if params is None:
            inp = input
            _, out_shape = self.autofill_dim(inp, dim_range=(2, 4))
            params = self.forward_parameters(out_shape)

        input = self.transform_inputs(input, params=params, extra_args=extra_args)

        self._params = params
        return input
