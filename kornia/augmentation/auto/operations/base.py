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

from typing import Callable, Dict, List, Optional, Tuple, TypeVar

import torch
from torch import nn

from kornia.augmentation.base import _AugmentationBase

T = TypeVar("T", bound="OperationBase")


class OperationBase(nn.Module):
    """Base class of differentiable augmentation operations.

    Args:
        operation: Kornia augmentation module.
        initial_magnitude: targeted magnitude parameter name and its initial magnitude value.
            The magnitude parameter name shall align with the attribute inside the random_generator
            in each augmentation. If None, the augmentation will be randomly applied according to
            the augmentation sampling range.
        temperature: retained for API compatibility.
        is_batch_operation: determine if to obtain the probability from `p` or `p_batch`.
            Set to True for most non-shape-persistent operations (e.g. cropping).

    Convention:
        - ``probability`` is initialized from the wrapped augmentation's ``p`` (``p_batch`` for a batch operation),
          clamped to ``[1e-7, 1 - 1e-7]`` and kept in ``state_dict()`` for checkpoint compatibility, but it takes
          no part in sampling and receives no gradient: the gate is drawn from the wrapped augmentation's own
          ``p`` and ``p_batch`` as a hard ``0`` or ``1``. ``magnitude``, where the operation has one, is clamped to
          the wrapped generator's range and receives a gradient where the wrapped augmentation is differentiable
          in it; ``forward_parameters`` substitutes it into the wrapped augmentation's draw.
        - ``forward`` linearly blends the wrapped output with the input using ``batch_prob``. Unless the wrapped
          ``p`` and ``p_batch`` are both ``1``, the wrapped augmentation first keeps rows whose gate is at most
          ``0.5`` unchanged, so a supplied fractional gate at or below ``0.5`` leaves its row untouched.
        - a symmetric magnitude applies the magnitude mapping first and then a random sign per row, so a mapping
          that quantizes to zero stays zero (``Posterize`` maps ``0.5`` to ``0`` bits with ``magnitude_range=(0, 8)``).
        - the concrete classes in ``kornia.augmentation.auto.operations.ops`` wrap public 2D augmentations and
          inherit their input, dtype, RNG and replay contracts. A wrapper pickles only with a named magnitude
          mapping and ``symmetric_megnitude=False`` (with default arguments, only ``Posterize``); otherwise its
          local closure blocks pickling (`#4469 <https://github.com/kornia/kornia/issues/4469>`_).

    """

    def __init__(
        self,
        operation: _AugmentationBase,
        initial_magnitude: Optional[List[Tuple[str, Optional[float]]]] = None,
        temperature: float = 0.1,
        is_batch_operation: bool = False,
        magnitude_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        symmetric_megnitude: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(operation, _AugmentationBase):
            raise ValueError(f"Only Kornia augmentations supported. Got {operation}.")

        self.op = operation

        self._init_magnitude(initial_magnitude)

        # Keep the legacy probability state for API and checkpoint compatibility.
        self.probability_range = (1e-7, 1 - 1e-7)
        self._is_batch_operation = is_batch_operation
        if is_batch_operation:
            self._probability = nn.Parameter(torch.empty(1).fill_(self.op.p_batch))
        else:
            self._probability = nn.Parameter(torch.empty(1).fill_(self.op.p))

        if temperature < 0:
            raise ValueError(f"Expect temperature value greater than 0. Got {temperature}.")
        self.register_buffer("temperature", torch.empty(1).fill_(temperature))

        self.symmetric_megnitude = symmetric_megnitude
        self._magnitude_fn = self._init_magnitude_fn(magnitude_fn)

    def _init_magnitude_fn(
        self, magnitude_fn: Optional[Callable[[torch.Tensor], torch.Tensor]]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        def _identity(x: torch.Tensor) -> torch.Tensor:
            return x

        def _random_flip(fn: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
            def f(x: torch.Tensor) -> torch.Tensor:
                # a sign, not a mask: multiplying by the bool would zero half the
                # magnitudes instead of negating them
                sign = torch.where(torch.rand((x.shape[0],), device=x.device) > 0.5, 1.0, -1.0)
                return fn(x) * sign.to(x.dtype)

            return f

        if magnitude_fn is None:
            magnitude_fn = _identity

        if self.symmetric_megnitude:
            return _random_flip(magnitude_fn)

        return magnitude_fn

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
            if self.op._param_generator is not None:
                self.magnitude_range = getattr(self.op._param_generator, self._factor_name)
            else:
                raise ValueError(f"No valid magnitude `{self._factor_name}` found in `{self.op._param_generator}`.")

            self._magnitude = None
            if initial_magnitude[0][1] is not None:
                self._magnitude = nn.Parameter(torch.empty(1).fill_(initial_magnitude[0][1]))

    def forward_parameters(
        self, batch_shape: torch.Size, mag: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Sample parameters for the wrapped augmentation op.

        Args:
            batch_shape: Input batch shape used for sampling.
            mag: Optional magnitude override. When set, it replaces the sampled
                factor before the magnitude mapping function is applied.

        Returns:
            Parameter dictionary consumed by :meth:`forward`.
        """
        if mag is None:
            mag = self.magnitude

        params = self.op.forward_parameters(batch_shape)

        if mag is not None:
            if self._factor_name is None:
                raise RuntimeError("No factor found in the params while `mag` is provided.")
            params[self._factor_name] = params[self._factor_name].zero_() + mag

        if self._factor_name is not None:
            params[self._factor_name] = self._magnitude_fn(params[self._factor_name])

        return params

    def forward(self, input: torch.Tensor, params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Apply the operation with probabilistic gating.

        Args:
            input: Input tensor.
            params: Optional precomputed parameters. If omitted, parameters are
                sampled from ``input.shape``.

        Returns:
            Tensor blended with the wrapped augmentation's output according to
            ``batch_prob``. The wrapped augmentation's own gate runs before this
            blend, as described in the class's Convention block.
        """
        if params is None:
            params = self.forward_parameters(input.shape)

        batch_prob = params["batch_prob"][(...,) + ((None,) * (len(input.shape) - 1))].to(
            device=input.device, dtype=input.dtype
        )

        return batch_prob * self.op(input, params=params) + (1 - batch_prob) * input

    @property
    def transform_matrix(self) -> Optional[torch.Tensor]:
        """Return the latest transform matrix from the wrapped op, if available.

        Returns:
            Transform matrix tensor or ``None`` for non-geometric operations.
        """
        if hasattr(self.op, "transform_matrix"):
            return self.op.transform_matrix
        return None

    @property
    def magnitude(self) -> Optional[torch.Tensor]:
        """Return the learned magnitude value for the operation.

        Returns:
            Magnitude tensor, clamped to ``magnitude_range`` when defined; otherwise
            the raw learnable value. Returns ``None`` when this operation has no
            magnitude parameter.
        """
        if self._magnitude is None:
            return None
        mag = self._magnitude
        if self.magnitude_range is not None:
            return mag.clamp(*self.magnitude_range)
        return mag

    @property
    def probability(self) -> torch.Tensor:
        """Return the operation probability after applying configured bounds.

        Returns:
            Probability tensor clamped to ``probability_range``.
        """
        return self._probability.clamp(*self.probability_range)
