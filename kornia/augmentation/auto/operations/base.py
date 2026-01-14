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

from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import nn

from kornia.augmentation.base import AugmentationBase


class OperationBase(nn.Module):
    """Base class for augmentation operations (simplified, non-differentiable).

    Args:
        operation: Kornia augmentation module.
        initial_magnitude: targeted magnitude parameter name and its initial magnitude value.
            The magnitude parameter name shall align with the attribute inside the random_generator
            in each augmentation. If None, the augmentation will be randomly applied according to
            the augmentation sampling range.
        is_batch_operation: determine if to obtain the probability from `p` or `p_batch`.
            Set to True for most non-shape-persistent operations (e.g. cropping).
        magnitude_fn: optional function to transform magnitude values.
        symmetric_megnitude: if to randomly assign the magnitude as negative or not.

    """

    def __init__(
        self,
        operation: AugmentationBase,
        initial_magnitude: Optional[List[Tuple[str, Optional[float]]]] = None,
        is_batch_operation: bool = False,
        magnitude_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        symmetric_megnitude: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(operation, AugmentationBase):
            raise ValueError(f"Only Kornia augmentations supported. Got {operation}.")

        self.op = operation

        self._init_magnitude(initial_magnitude)

        self._is_batch_operation = is_batch_operation
        # Store probability as a simple float (non-learnable)
        if is_batch_operation:
            self._probability = self.op.p_batch
        else:
            self._probability = self.op.p

        self.symmetric_megnitude = symmetric_megnitude
        self._magnitude_fn = self._init_magnitude_fn(magnitude_fn)

    def _init_magnitude_fn(
        self, magnitude_fn: Optional[Callable[[torch.Tensor], torch.Tensor]]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        def _identity(x: torch.Tensor) -> torch.Tensor:
            return x

        def _random_flip(fn: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
            def f(x: torch.Tensor) -> torch.Tensor:
                flip = (torch.rand((x.shape[0],), device=x.device) > 0.5).float() * 2 - 1
                return fn(x) * flip

            return f

        if magnitude_fn is None:
            magnitude_fn = _identity

        if self.symmetric_megnitude:
            return _random_flip(magnitude_fn)

        return magnitude_fn

    def _get_magnitude_range(self, factor_name: str) -> Optional[torch.Tensor]:
        """Get the magnitude range from the augmentation operation.

        Tries multiple naming conventions to find the bound tensor.
        """
        # Try direct name first (e.g., "brightness")
        base_name = factor_name.replace("_factor", "")

        # List of possible attribute names to check
        possible_names = [
            base_name,  # e.g., "brightness", "contrast"
            f"{base_name}_bound",  # e.g., "degrees_bound", "thresholds_bound"
            factor_name,  # original name as fallback
        ]

        for name in possible_names:
            if hasattr(self.op, name):
                attr = getattr(self.op, name)
                if isinstance(attr, torch.Tensor):
                    return attr

        # Also check _param_generator for backward compatibility
        if hasattr(self.op, "_param_generator") and self.op._param_generator is not None:
            if hasattr(self.op._param_generator, factor_name):
                return getattr(self.op._param_generator, factor_name)

        return None

    def _init_magnitude(self, initial_magnitude: Optional[List[Tuple[str, Optional[float]]]]) -> None:
        if isinstance(initial_magnitude, (list, tuple)):
            if not all(isinstance(ini_mag, (list, tuple)) and len(ini_mag) == 2 for ini_mag in initial_magnitude):
                raise ValueError(f"`initial_magnitude` shall be a list of 2-element tuples. Got {initial_magnitude}")
            if len(initial_magnitude) != 1:
                raise NotImplementedError("Multi magnitudes operations are not yet supported.")

        if initial_magnitude is None:
            self._factor_name = None
            self._magnitude = None
            self.magnitude_range = None
        else:
            self._factor_name = initial_magnitude[0][0]
            self.magnitude_range = self._get_magnitude_range(self._factor_name)
            if self.magnitude_range is None:
                raise ValueError(f"No valid magnitude `{self._factor_name}` found in augmentation.")

            self._magnitude = None
            if initial_magnitude[0][1] is not None:
                # Store magnitude as simple tensor (non-learnable)
                self.register_buffer("_magnitude_value", torch.tensor([initial_magnitude[0][1]]))
                self._magnitude = self._magnitude_value

    def forward_parameters(
        self, batch_shape: torch.Size, mag: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if mag is None:
            mag = self.magnitude
        params = self.op.forward_parameters(batch_shape)

        if mag is not None:
            if self._factor_name is None:
                raise RuntimeError("No factor found in the params while `mag` is provided.")
            # For single factor operations, this is equivalent to `same_on_batch=True`
            params[self._factor_name] = params[self._factor_name].zero_() + mag

        if self._factor_name is not None:
            params[self._factor_name] = self._magnitude_fn(params[self._factor_name])

        return params

    def forward(self, input: torch.Tensor, params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        if params is None:
            params = self.forward_parameters(input.shape)

        batch_prob = params["batch_prob"][(...,) + ((None,) * (len(input.shape) - 1))].to(device=input.device)

        return batch_prob * self.op(input, params=params) + (1 - batch_prob) * input

    @property
    def transform_matrix(self) -> Optional[torch.Tensor]:
        if hasattr(self.op, "transform_matrix"):
            return self.op.transform_matrix
        return None

    @property
    def magnitude(self) -> Optional[torch.Tensor]:
        if self._magnitude is None:
            return None
        mag = self._magnitude
        if self.magnitude_range is not None:
            return mag.clamp(*self.magnitude_range)
        return mag

    @property
    def probability(self) -> float:
        return self._probability
