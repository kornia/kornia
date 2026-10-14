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

"""Explicit augmentation random streams, without changing torch's global RNG state."""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator, ParamSpec, Self, TypeVar, cast

import torch

P = ParamSpec("P")
R = TypeVar("R")


class _GeneratorMixin(torch.nn.Module):
    _generator: torch.Generator | None = None

    def set_generator(self, generator: torch.Generator | None) -> Self:
        """Use ``generator`` for this module and its built-in augmentation children.

        ``None`` restores global sampling. The generator must match the sampling
        device, which is CPU by default and can differ from the image's device.
        Streams do not move with ``.to()`` and are not part of ``state_dict()``;
        save and restore their state explicitly with ``get_state``/``set_state``.
        Auto-augmentation policies, model presets and RandomDissolving are not supported here.
        Custom modules must explicitly thread ``self._generator`` into their draws.
        """
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError(f"Expected torch.Generator or None. Got {type(generator)}.")
        modules = [m for m in self.modules() if isinstance(m, _GeneratorMixin)]
        if generator is not None:
            for module in self.modules():
                name = type(module).__module__
                if name.startswith(("kornia.augmentation.auto.", "kornia.augmentation.presets.")) or (
                    type(module).__name__ == "RandomDissolving"
                ):
                    raise NotImplementedError(f"Explicit generators are not supported by {type(module).__name__}.")
        for module in modules:
            module._generator = generator
        return self

    @contextmanager
    def _generator_scope(self, generator: torch.Generator) -> Iterator[None]:
        previous = [(m, m._generator) for m in self.modules() if isinstance(m, _GeneratorMixin)]
        self.set_generator(generator)
        try:
            yield
        finally:
            for module, old in previous:
                module._generator = old


def _with_generator(function: Callable[P, R]) -> Callable[P, R]:
    """Accept a call-local stream, restoring each child's persistent stream afterwards."""

    @wraps(function)
    def wrapped(self: _GeneratorMixin, *args: Any, generator: torch.Generator | None = None, **kwargs: Any) -> R:
        # Leave the original default path intact, including under graph capture.
        if generator is None:
            return function(self, *args, **kwargs)
        with self._generator_scope(generator):
            return function(self, *args, **kwargs)

    return cast(Callable[P, R], wrapped)


class _GeneratorDirichlet(torch.autograd.Function):
    """torch's Dirichlet reparameterization with an explicit RNG (first-order only)."""

    @staticmethod
    def forward(ctx: Any, concentration: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        result = torch._sample_dirichlet(concentration, generator=generator)
        ctx.save_for_backward(result, concentration)
        return result

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        # Same implicit gradient as torch.distributions.Dirichlet; only the draw's RNG differs.
        result, concentration = ctx.saved_tensors
        total = concentration.sum(-1, True).expand_as(concentration)
        grad = torch._dirichlet_grad(result, concentration, total)
        return grad * (grad_output - (result * grad_output).sum(-1, True)), None


def _sample_distribution(
    distribution: torch.distributions.Distribution,
    shape: torch.Size,
    generator: torch.Generator | None,
    reparameterized: bool,
) -> torch.Tensor:
    """Sample the distributions used by built-in generators, preserving the default path."""
    if generator is None:
        return distribution.rsample(shape) if reparameterized else distribution.sample(shape)
    if isinstance(distribution, torch.distributions.Bernoulli):
        if reparameterized:
            raise NotImplementedError("Bernoulli does not support reparameterized sampling.")
        with torch.no_grad():
            return torch.bernoulli(distribution.probs.expand(shape + distribution.batch_shape), generator=generator)
    if not reparameterized:
        with torch.no_grad():
            return _sample_distribution(distribution, shape, generator, True)
    size = shape + distribution.batch_shape
    if isinstance(distribution, torch.distributions.Uniform):
        low, high = distribution.low, distribution.high
        return low + torch.rand(size, device=low.device, dtype=low.dtype, generator=generator) * (high - low)
    if isinstance(distribution, torch.distributions.Normal):
        loc, scale = distribution.loc, distribution.scale
        noise = torch.empty(size, device=loc.device, dtype=loc.dtype).normal_(generator=generator)
        return loc + noise * scale
    if isinstance(distribution, torch.distributions.Beta):
        concentration = torch.stack((distribution.concentration1, distribution.concentration0), -1)
        return _GeneratorDirichlet.apply(concentration.expand(size + (2,)), generator)[..., 0]
    raise NotImplementedError(f"Explicit generators are not supported for {type(distribution).__name__}.")
