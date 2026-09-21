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

from typing import Dict, Tuple, Union

import torch

from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import (
    _adapted_rsampling,
    _check_positive_int_or_traced,
    _common_param_check,
    _joint_range_check,
)
from kornia.augmentation.utils.helpers import _constant_tensor
from kornia.core.utils import _extract_device_dtype

__all__ = ["RectangleEraseGenerator"]


class RectangleEraseGenerator(RandomGeneratorBase):
    r"""Get parameters for ```erasing``` transformation for erasing transform.

    Args:
        scale (torch.Tensor): range of size of the origin size cropped. Shape (2).
        ratio (torch.Tensor): range of aspect ratio of the origin aspect ratio cropped. Shape (2).
        value (float): value to be filled in the erased area.

    Returns:
        A dict of parameters to be passed for transformation.
            - widths (torch.Tensor): element-wise erasing widths with a shape of (B,).
            - heights (torch.Tensor): element-wise erasing heights with a shape of (B,).
            - xs (torch.Tensor): element-wise erasing x coordinates with a shape of (B,).
            - ys (torch.Tensor): element-wise erasing y coordinates with a shape of (B,).
            - values (torch.Tensor): element-wise filling values with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.

    """

    def __init__(
        self,
        scale: Union[torch.Tensor, Tuple[float, float]] = (0.02, 0.33),
        ratio: Union[torch.Tensor, Tuple[float, float]] = (0.3, 3.3),
        value: float = 0.0,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __repr__(self) -> str:
        return f"scale={self.scale}, resize_to={self.ratio}, value={self.value}"

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        scale = torch.as_tensor(self.scale, device=device, dtype=dtype)
        ratio = torch.as_tensor(self.ratio, device=device, dtype=dtype)

        if not (isinstance(self.value, (int, float)) and self.value >= 0 and self.value <= 1):
            raise AssertionError(f"'value' must be a number between 0 - 1. Got {self.value}.")
        _joint_range_check(scale, "scale", bounds=(0, float("inf")))
        _joint_range_check(ratio, "ratio", bounds=(0, float("inf")))

        self.scale_sampler = UniformDistribution(scale[0], scale[1], validate_args=False)

        if ratio[0] < 1.0 and ratio[1] > 1.0:
            self.ratio_sampler1 = UniformDistribution(ratio[0], 1, validate_args=False)
            self.ratio_sampler2 = UniformDistribution(1, ratio[1], validate_args=False)
            self.index_sampler = UniformDistribution(
                torch.tensor(0, device=device, dtype=dtype),
                torch.tensor(1, device=device, dtype=dtype),
                validate_args=False,
            )
        else:
            self.ratio_sampler = UniformDistribution(ratio[0], ratio[1], validate_args=False)
        position_sampler_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        self.uniform_sampler = UniformDistribution(
            torch.tensor(0, device=device, dtype=position_sampler_dtype),
            torch.tensor(1, device=device, dtype=position_sampler_dtype),
            validate_args=False,
        )

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        height = batch_shape[-2]
        width = batch_shape[-1]
        _check_positive_int_or_traced(height, "height")
        _check_positive_int_or_traced(width, "width")

        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.ratio, self.scale])
        images_area = height * width
        target_areas = (
            _adapted_rsampling((batch_size,), self.scale_sampler, same_on_batch).to(device=_device, dtype=_dtype)
            * images_area
        )

        if self.ratio[0] < 1.0 and self.ratio[1] > 1.0:
            aspect_ratios1 = _adapted_rsampling((batch_size,), self.ratio_sampler1, same_on_batch)
            aspect_ratios2 = _adapted_rsampling((batch_size,), self.ratio_sampler2, same_on_batch)
            if same_on_batch:
                rand_idxs = (
                    torch.round(_adapted_rsampling((1,), self.index_sampler, same_on_batch)).repeat(batch_size).bool()
                )
            else:
                rand_idxs = torch.round(_adapted_rsampling((batch_size,), self.index_sampler, same_on_batch)).bool()
            aspect_ratios = torch.where(rand_idxs, aspect_ratios1, aspect_ratios2)
        else:
            aspect_ratios = _adapted_rsampling((batch_size,), self.ratio_sampler, same_on_batch)

        aspect_ratios = aspect_ratios.to(device=_device, dtype=_dtype)

        # based on target areas and aspect ratios, rectangle params are computed
        heights = torch.round((target_areas * aspect_ratios) ** (1 / 2)).clamp(1.0, height)
        widths = torch.round((target_areas / aspect_ratios) ** (1 / 2)).clamp(1.0, width)

        position_dtype = torch.promote_types(self.uniform_sampler.low.dtype, _dtype)
        if position_dtype in (torch.float16, torch.bfloat16):
            position_dtype = torch.float32
        xs_ratio = _adapted_rsampling((batch_size,), self.uniform_sampler, same_on_batch).to(
            device=_device, dtype=position_dtype
        )
        ys_ratio = _adapted_rsampling((batch_size,), self.uniform_sampler, same_on_batch).to(
            device=_device, dtype=position_dtype
        )

        xs = xs_ratio * (width - widths.to(dtype=position_dtype) + 1)
        ys = ys_ratio * (height - heights.to(dtype=position_dtype) + 1)

        def _cast_position(position: torch.Tensor, size: torch.Tensor, limit: int) -> torch.Tensor:
            floored = position.floor()
            output = floored.to(device=_device, dtype=_dtype)
            if _dtype not in (torch.float16, torch.bfloat16):
                return output

            max_start = (
                _constant_tensor(limit, device=_device, dtype=position_dtype)
                - size.to(device=_device, dtype=position_dtype)
            ).floor()
            # Coordinates are nonnegative integers. Deliberately round the bound down
            # to the output format's local representable integer grid.
            max_start = torch.minimum(
                max_start,
                _constant_tensor(torch.finfo(_dtype).max, device=_device, dtype=position_dtype),
            )
            max_start = torch.clamp(max_start, min=0)
            mantissa_bits = 10 if _dtype == torch.float16 else 7
            clamped_start = torch.clamp(max_start, min=1)
            exponent = torch.floor(torch.log2(clamped_start))
            power = torch.pow(
                _constant_tensor(2.0, device=_device, dtype=position_dtype),
                exponent,
            )
            exponent = torch.where(power > clamped_start, exponent - 1, exponent)
            quantum = torch.pow(
                _constant_tensor(2.0, device=_device, dtype=position_dtype),
                torch.clamp(exponent - mantissa_bits, min=0),
            )
            max_start_output = (torch.floor(max_start / quantum) * quantum).to(device=_device, dtype=_dtype).detach()
            return torch.minimum(torch.maximum(output, torch.zeros_like(output)), max_start_output)

        return {
            "widths": widths.floor(),
            "heights": heights.floor(),
            "xs": _cast_position(xs, widths, width),
            "ys": _cast_position(ys, heights, height),
            "values": torch.full((batch_size,), self.value, device=_device, dtype=_dtype),
        }
