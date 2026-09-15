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

import torch

from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _range_bound
from kornia.core.utils import _extract_device_dtype


def _closed_integer_range(value: tuple[int, int], name: str) -> torch.Tensor:
    """Return ``[lo, hi + 1]`` as a float tensor for a closed integer range ``(lo, hi)``."""
    if isinstance(value, torch.Tensor):
        value = tuple(value.flatten().tolist())
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"`{name}` must be a (lower, upper) pair. Got {value}.")
    lower, upper = value
    if lower > upper:
        raise ValueError(f"`{name}` lower bound should be smaller than or equal to its upper bound. Got {value}.")
    return torch.tensor([float(lower), float(upper) + 1.0])


class RainGenerator(RandomGeneratorBase):
    def __init__(
        self, number_of_drops: tuple[int, int], drop_height: tuple[int, int], drop_width: tuple[int, int]
    ) -> None:
        super().__init__()
        self.number_of_drops = number_of_drops
        self.drop_height = drop_height
        self.drop_width = drop_width

    def __repr__(self) -> str:
        repr = f"number_of_drops={self.number_of_drops}, drop_height={self.drop_height}, drop_width={self.drop_width}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        # Each range is a closed integer interval: the sampler covers ``[lo, hi + 1)`` and the forward
        # floors the draw, so every integer from ``lo`` to ``hi`` is drawn with the same probability.
        number_of_drops = _closed_integer_range(self.number_of_drops, "number_of_drops").to(device)
        drop_height = _closed_integer_range(self.drop_height, "drop_height").to(device)
        drop_width = _closed_integer_range(self.drop_width, "drop_width").to(device)

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
        # self.ksize_factor.expand((batch_size, -1))
        # ``floor`` rather than a truncating integer cast: a signed range such as ``drop_width=(-5, 5)``
        # would otherwise fold ``(-1, 0)`` and ``(0, 1)`` onto ``0`` and draw it twice as often.
        number_of_drops_factor = (
            _adapted_rsampling((batch_size,), self.number_of_drops_sampler, same_on_batch)
            .floor()
            .to(device=_device, dtype=torch.long)
        )
        drop_height_factor = (
            _adapted_rsampling((batch_size,), self.drop_height_sampler, same_on_batch)
            .floor()
            .to(device=_device, dtype=torch.long)
        )
        drop_width_factor = (
            _adapted_rsampling((batch_size,), self.drop_width_sampler, same_on_batch)
            .floor()
            .to(device=_device, dtype=torch.long)
        )
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
