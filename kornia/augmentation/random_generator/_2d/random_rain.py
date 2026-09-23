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

from __future__ import annotations

from typing import Union

import torch

from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _range_bound
from kornia.core.utils import _extract_device_dtype


def _closed_integer_range(
    value: Union[tuple[int, int], list[int], torch.Tensor], name: str, device: torch.device
) -> torch.Tensor:
    """Return the half-open sampler bounds ``[lo, hi + 1)`` for a closed integer range ``(lo, hi)``."""
    if isinstance(value, torch.Tensor):
        value = tuple(value.flatten().tolist())
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"`{name}` must be a (lower, upper) pair. Got {value}.")
    lower, upper = value
    # Whole numbers only.  This is also what disposes of ``nan`` and ``inf``, which are neither greater
    # nor smaller than the other bound and would sail through an ordering test into the sampler, and of
    # a fractional pair, which the clamp below draws worse than the truncation it replaces:
    # ``(0.5, 2.5)`` covers ``[0.5, 3.5)``, and folding the ``3`` back onto ``2.5`` leaves the cast to
    # ``torch.long`` to truncate it to ``2``, which then takes half the draws against a quarter before.
    if not (float(lower).is_integer() and float(upper).is_integer()):
        raise ValueError(f"`{name}` must be a pair of whole numbers. Got {value}.")
    if lower > upper:
        raise ValueError(f"`{name}`[0] should be smaller than or equal to `{name}`[1]. Got {value}.")
    # Built on ``device`` explicitly: a bare ``torch.tensor`` follows an ambient ``torch.set_default_device``
    # instead, which allocates the bounds on that device and, under a ``meta`` one, cannot move them back.
    return torch.tensor([float(lower), float(upper) + 1.0], device=device)


def _draw_closed_integer(
    batch_size: int, sampler: UniformDistribution, same_on_batch: bool, device: torch.device
) -> torch.Tensor:
    """Draw one integer per sample, uniformly over the closed range ``sampler`` was built from.

    ``floor``, not a truncating cast: truncation rounds toward zero, so a signed range such as
    ``drop_width=(-5, 5)`` would fold ``(-1, 0)`` and ``(0, 1)`` onto ``0`` and draw it twice as often as
    any other value.  The clamp removes the excluded end point: ``lo + u * (hi + 1 - lo)`` rounds up onto
    ``hi + 1`` in ``float32`` for the largest draw ``u = 1 - 2 ** -24`` (``5 + u * 16`` is exactly
    ``21.0``), which ``floor`` would keep, so about one draw in ``2 ** 24`` would leave the closed range.
    """
    drawn = _adapted_rsampling((batch_size,), sampler, same_on_batch).floor()
    return drawn.clamp(max=sampler.high - 1).to(device=device, dtype=torch.long)


class RainGenerator(RandomGeneratorBase):
    def __init__(
        self, number_of_drops: tuple[int, int], drop_height: tuple[int, int], drop_width: tuple[int, int]
    ) -> None:
        super().__init__()
        self.number_of_drops = number_of_drops
        self.drop_height = drop_height
        self.drop_width = drop_width

    def __repr__(self) -> str:
        return f"number_of_drops={self.number_of_drops}, drop_height={self.drop_height}, drop_width={self.drop_width}"

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        # Each range is a closed integer interval: the sampler covers ``[lo, hi + 1)`` and ``forward``
        # floors the draw, so every integer from ``lo`` to ``hi`` is drawn with the same probability.
        number_of_drops = _closed_integer_range(self.number_of_drops, "number_of_drops", device)
        drop_height = _closed_integer_range(self.drop_height, "drop_height", device)
        drop_width = _closed_integer_range(self.drop_width, "drop_width", device)

        drop_coordinates = _range_bound((0, 1), "drops_coordinate", center=0.5, bounds=(0, 1)).to(
            device=device, dtype=dtype
        )
        self.number_of_drops_sampler = UniformDistribution(number_of_drops[0], number_of_drops[1], validate_args=False)
        self.drop_height_sampler = UniformDistribution(drop_height[0], drop_height[1], validate_args=False)
        self.drop_width_sampler = UniformDistribution(drop_width[0], drop_width[1], validate_args=False)
        self.coordinates_sampler = UniformDistribution(drop_coordinates[0], drop_coordinates[1], validate_args=False)

    def forward(self, batch_shape: tuple[int, ...], same_on_batch: bool = False) -> dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.drop_width, self.drop_height, self.number_of_drops])
        number_of_drops_factor = _draw_closed_integer(batch_size, self.number_of_drops_sampler, same_on_batch, _device)
        drop_height_factor = _draw_closed_integer(batch_size, self.drop_height_sampler, same_on_batch, _device)
        drop_width_factor = _draw_closed_integer(batch_size, self.drop_width_sampler, same_on_batch, _device)
        coordinates_factor = _adapted_rsampling(
            (batch_size, int(number_of_drops_factor.max().item()) if number_of_drops_factor.numel() > 0 else 0, 2),
            self.coordinates_sampler,
            same_on_batch=same_on_batch,
        ).to(device=_device)
        return {
            "number_of_drops_factor": number_of_drops_factor,
            "coordinates_factor": coordinates_factor,
            "drop_height_factor": drop_height_factor,
            "drop_width_factor": drop_width_factor,
        }
