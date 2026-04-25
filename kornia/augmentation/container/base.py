# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from collections import OrderedDict
from itertools import zip_longest
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch import nn

import kornia.augmentation as K
from kornia.augmentation.base import _AugmentationBase
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

    def __init__(self, *args: nn.Module) -> None:
        # To name the modules properly
        _args = OrderedDict()
        for idx, mod in enumerate(args):
            if not isinstance(mod, nn.Module):
                raise NotImplementedError(f"Only nn.Module are supported at this moment. Got {mod}.")
            _args.update({f"{mod.__class__.__name__}_{idx}": mod})
        super().__init__(_args)
        self._params: Optional[List[ParamItem]] = None

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
            nn.Module: The submodule referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``nn.Module``

        """
        if len(target) == 0:
            return self

        atoms: List[str] = target.split(".")
        mod = self

        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(mod._get_name() + " has no attribute `" + item + "`")

            mod = getattr(mod, item)

            if not isinstance(mod, nn.Module):
                raise AttributeError("`" + item + "` is not an nn.Module")

        return mod

    def clear_state(self) -> None:
        """Reset self._params state to None."""
        self._params = None

    # TODO: Implement this for all submodules.
    def forward_parameters(self, batch_shape: torch.Size) -> List[ParamItem]:
        """Generate parameters for a forward pass.

        Args:
            batch_shape: Input batch shape.

        Returns:
            Per-module parameters in execution order.
        """
        raise NotImplementedError

    def get_children_by_indices(self, indices: torch.Tensor) -> Iterator[Tuple[str, nn.Module]]:
        """Yield child modules selected by index.

        Args:
            indices: Indices of children to fetch.

        Yields:
            ``(name, module)`` pairs matching ``indices`` order.
        """
        modules = list(self.named_children())
        for idx in indices:
            yield modules[idx]

    def get_children_by_params(self, params: List[ParamItem]) -> Iterator[Tuple[str, nn.Module]]:
        """Yield child modules referenced by a list of params.

        Args:
            params: Parameter items whose ``name`` fields identify the children.

        Yields:
            ``(name, module)`` pairs matching ``params`` order.
        """
        modules = list(self.named_children())
        # TODO: Wrong params passed here when nested ImageSequential
        for param in params:
            yield modules[list(dict(self.named_children()).keys()).index(param.name)]

    def get_params_by_module(self, named_modules: Iterator[Tuple[str, nn.Module]]) -> Iterator[ParamItem]:
        """Create placeholder params for a module iterator.

        Args:
            named_modules: Modules that will be executed.

        Yields:
            :class:`ParamItem` entries with ``data=None``.
        """
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

    def __init__(self, *args: nn.Module, same_on_batch: Optional[bool] = None, keepdim: Optional[bool] = None) -> None:
        # To name the modules properly
        super().__init__(*args)
        self._same_on_batch = same_on_batch
        self._keepdim = keepdim
        self.update_attribute(same_on_batch, keepdim=keepdim)

    def update_attribute(
        self,
        same_on_batch: Optional[bool] = None,
        return_transform: Optional[bool] = None,
        keepdim: Optional[bool] = None,
    ) -> None:
        """Propagate sequence-level flags to child augmentation modules.

        Args:
            same_on_batch: Override for ``same_on_batch``.
            return_transform: Reserved for compatibility with older interfaces.
            keepdim: Override for ``keepdim``.
        """
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
        """Return the sequence-level ``same_on_batch`` setting."""
        return self._same_on_batch

    @same_on_batch.setter
    def same_on_batch(self, same_on_batch: Optional[bool]) -> None:
        self._same_on_batch = same_on_batch
        self.update_attribute(same_on_batch=same_on_batch)

    @property
    def keepdim(self) -> Optional[bool]:
        """Return the sequence-level ``keepdim`` setting."""
        return self._keepdim

    @keepdim.setter
    def keepdim(self, keepdim: Optional[bool]) -> None:
        self._keepdim = keepdim
        self.update_attribute(keepdim=keepdim)

    def autofill_dim(self, input: torch.Tensor, dim_range: Tuple[int, int] = (2, 4)) -> Tuple[torch.Size, torch.Size]:
        """Fill torch.Tensor dim to the upper bound of dim_range.

        If input torch.Tensor dim is smaller than the lower bound of dim_range, an error will be thrown out.
        """
        ori_shape = input.shape
        if len(ori_shape) < dim_range[0] or len(ori_shape) > dim_range[1]:
            raise RuntimeError(f"input shape expected to be in {dim_range} while got {ori_shape}.")
        while len(input.shape) < dim_range[1]:
            input = input[None]
        return ori_shape, input.shape


class ImageSequentialBase(SequentialBase):
    """Provide a base class for sequential image-only augmentations.

    This class handles the logic for applying a series of transformations
    sequentially to input tensors while managing the transformation matrices.
    """

    def identity_matrix(self, input: torch.Tensor) -> torch.Tensor:
        """Return identity matrix."""
        raise NotImplementedError

    def get_transformation_matrix(
        self,
        input: torch.Tensor,
        params: Optional[List[ParamItem]] = None,
        recompute: bool = False,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Optional[torch.Tensor]:
        """Compute the transformation matrix according to the provided parameters.

        Args:
            input: the input torch.Tensor.
            params: params for the sequence.
            recompute: if to recompute the transformation matrix according to the params.
                default: False.
            extra_args: Optional dictionary of extra arguments with specific options for different input types.
        """
        raise NotImplementedError

    def forward_parameters(self, batch_shape: torch.Size) -> List[ParamItem]:
        """Generate parameters for child modules.

        Args:
            batch_shape: Input batch shape.

        Returns:
            Per-module parameter list.
        """
        raise NotImplementedError

    def get_forward_sequence(self, params: Optional[List[ParamItem]] = None) -> Iterator[Tuple[str, nn.Module]]:
        """Get module sequence by input params."""
        raise NotImplementedError

    def transform_inputs(
        self, input: torch.Tensor, params: List[ParamItem], extra_args: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Apply the sequence to an input tensor.

        Args:
            input: Input tensor.
            params: Parameters aligned with the execution sequence.
            extra_args: Optional per-input-type overrides.

        Returns:
            Transformed tensor.
        """
        for param in params:
            module = self.get_submodule(param.name)
            input = InputSequentialOps.transform(input, module=module, param=param, extra_args=extra_args)
        return input

    def inverse_inputs(
        self, input: torch.Tensor, params: List[ParamItem], extra_args: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Apply inverse transforms for an input tensor.

        Args:
            input: Tensor produced by :meth:`transform_inputs`.
            params: Parameters used during forward execution.
            extra_args: Optional per-input-type overrides.

        Returns:
            Tensor mapped back through inverse operations.
        """
        for (_, module), param in zip_longest(list(self.get_forward_sequence(params))[::-1], params[::-1]):
            input = InputSequentialOps.inverse(input, module=module, param=param, extra_args=extra_args)
        return input

    def transform_masks(
        self, input: torch.Tensor, params: List[ParamItem], extra_args: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Apply the sequence to mask tensors.

        Args:
            input: Mask tensor.
            params: Parameters aligned with the execution sequence.
            extra_args: Optional per-input-type overrides.

        Returns:
            Transformed mask tensor.
        """
        for param in params:
            module = self.get_submodule(param.name)
            input = MaskSequentialOps.transform(input, module=module, param=param, extra_args=extra_args)
        return input

    def inverse_masks(
        self, input: torch.Tensor, params: List[ParamItem], extra_args: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Apply inverse transforms for mask tensors.

        Args:
            input: Mask tensor produced by :meth:`transform_masks`.
            params: Parameters used during forward execution.
            extra_args: Optional per-input-type overrides.

        Returns:
            Inverse-transformed mask tensor.
        """
        for (_, module), param in zip_longest(list(self.get_forward_sequence(params))[::-1], params[::-1]):
            input = MaskSequentialOps.inverse(input, module=module, param=param, extra_args=extra_args)
        return input

    def transform_boxes(
        self, input: Boxes, params: List[ParamItem], extra_args: Optional[Dict[str, Any]] = None
    ) -> Boxes:
        """Apply the sequence to bounding boxes.

        Args:
            input: Bounding boxes.
            params: Parameters aligned with the execution sequence.
            extra_args: Optional per-input-type overrides.

        Returns:
            Transformed boxes.
        """
        for param in params:
            module = self.get_submodule(param.name)
            input = BoxSequentialOps.transform(input, module=module, param=param, extra_args=extra_args)
        return input

    def inverse_boxes(
        self, input: Boxes, params: List[ParamItem], extra_args: Optional[Dict[str, Any]] = None
    ) -> Boxes:
        """Apply inverse transforms for bounding boxes.

        Args:
            input: Boxes produced by :meth:`transform_boxes`.
            params: Parameters used during forward execution.
            extra_args: Optional per-input-type overrides.

        Returns:
            Inverse-transformed boxes.
        """
        for (_, module), param in zip_longest(list(self.get_forward_sequence(params))[::-1], params[::-1]):
            input = BoxSequentialOps.inverse(input, module=module, param=param, extra_args=extra_args)
        return input

    def transform_keypoints(
        self, input: Keypoints, params: List[ParamItem], extra_args: Optional[Dict[str, Any]] = None
    ) -> Keypoints:
        """Apply the sequence to keypoints.

        Args:
            input: Keypoints.
            params: Parameters aligned with the execution sequence.
            extra_args: Optional per-input-type overrides.

        Returns:
            Transformed keypoints.
        """
        for param in params:
            module = self.get_submodule(param.name)
            input = KeypointSequentialOps.transform(input, module=module, param=param, extra_args=extra_args)
        return input

    def inverse_keypoints(
        self, input: Keypoints, params: List[ParamItem], extra_args: Optional[Dict[str, Any]] = None
    ) -> Keypoints:
        """Apply inverse transforms for keypoints.

        Args:
            input: Keypoints produced by :meth:`transform_keypoints` (conceptually
                ``(B, N, 2)`` coordinates, where the last dimension stores
                ``(x, y)``.
            params: Parameters used during forward execution.
            extra_args: Optional per-input-type overrides.

        Returns:
            Inverse-transformed keypoints.
        """
        for (_, module), param in zip_longest(list(self.get_forward_sequence(params))[::-1], params[::-1]):
            input = KeypointSequentialOps.inverse(input, module=module, param=param, extra_args=extra_args)
        return input

    def inverse(
        self, input: torch.Tensor, params: Optional[List[ParamItem]] = None, extra_args: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Inverse transformation.

        Used to inverse a torch.Tensor according to the performed transformation by a forward pass, or with respect to
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
        self, input: torch.Tensor, params: Optional[List[ParamItem]] = None, extra_args: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Run the sequential augmentation pipeline.

        Args:
            input: Input tensor.
            params: Optional precomputed parameters. If omitted, parameters are
                sampled from ``input``.
            extra_args: Optional per-input-type overrides.

        Returns:
            Augmented tensor.
        """
        self.clear_state()

        if params is None:
            inp = input
            _, out_shape = self.autofill_dim(inp, dim_range=(2, 4))
            params = self.forward_parameters(out_shape)

        input = self.transform_inputs(input, params=params, extra_args=extra_args)

        self._params = params
        return input


class TransformMatrixMinIn:
    """Enables computation matrix computation."""

    _valid_ops_for_transform_computation: Tuple[Any, ...] = ()
    _transformation_matrix_arg: str = "silent"

    def __init__(self, *args, **kwargs) -> None:  # type:ignore
        super().__init__(*args, **kwargs)
        self._transform_matrix: Optional[torch.Tensor] = None
        self._transform_matrices: List[Optional[torch.Tensor]] = []

    def _parse_transformation_matrix_mode(self, transformation_matrix_mode: str) -> None:
        _valid_transformation_matrix_args = {"silence", "silent", "rigid", "skip"}
        if transformation_matrix_mode not in _valid_transformation_matrix_args:
            raise ValueError(
                f"`transformation_matrix` has to be one of {_valid_transformation_matrix_args}. "
                f"Got {transformation_matrix_mode}."
            )
        self._transformation_matrix_arg = transformation_matrix_mode

    @property
    def transform_matrix(self) -> Optional[torch.Tensor]:
        # In AugmentationSequential, the parent class is accessed first.
        # So that it was None in the beginning. We hereby use lazy computation here.
        if self._transform_matrix is None and len(self._transform_matrices) != 0:
            self._transform_matrix = self._transform_matrices[0]
            for mat in self._transform_matrices[1:]:
                self._update_transform_matrix(mat)
        return self._transform_matrix

    def _update_transform_matrix_for_valid_op(self, module: nn.Module) -> None:
        raise NotImplementedError(module)

    def _update_transform_matrix_by_module(self, module: nn.Module) -> None:
        if self._transformation_matrix_arg == "skip":
            return
        if isinstance(module, self._valid_ops_for_transform_computation):
            self._update_transform_matrix_for_valid_op(module)
        elif self._transformation_matrix_arg == "rigid":
            raise RuntimeError(
                f"Non-rigid module `{module}` is not supported under `rigid` computation mode. "
                "Please either update the module or change the `transformation_matrix` argument."
            )

    def _update_transform_matrix(self, transform_matrix: Optional[torch.Tensor]) -> None:
        if self._transform_matrix is None:
            self._transform_matrix = transform_matrix
        else:
            self._transform_matrix = transform_matrix @ self._transform_matrix

    def _reset_transform_matrix_state(self) -> None:
        self._transform_matrix = None
        self._transform_matrices = []
