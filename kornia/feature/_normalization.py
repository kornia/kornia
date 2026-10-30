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
from collections.abc import Callable

import torch
from torch import nn


class _DescriptorSequential(nn.Sequential):
    """Keep a descriptor's final BatchNorm statistics and computation in float32 for half inputs."""

    def __init__(self, *args: nn.Module | OrderedDict[str, nn.Module]) -> None:
        super().__init__(*args)
        # Sequential slices share modules. Only the original descriptor's final norm is protected;
        # a slice ending at an earlier BatchNorm must not change that shared layer's dtype.
        if (
            len(self)
            and isinstance(self[-1], nn.BatchNorm2d)
            and not (len(args) == 1 and isinstance(args[0], OrderedDict))
        ):
            self[-1]._descriptor_float32_statistics = True
        for module in self:
            if getattr(module, "_descriptor_float32_statistics", False):
                for name in ("running_mean", "running_var"):
                    tensor = getattr(module, name)
                    if tensor is not None and tensor.dtype == torch.float16:
                        setattr(module, name, tensor.float())

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> "_DescriptorSequential":
        statistics = tuple(
            tensor
            for module in self
            if getattr(module, "_descriptor_float32_statistics", False)
            for tensor in (module.running_mean, module.running_var)
        )

        def convert(tensor: torch.Tensor) -> torch.Tensor:
            converted = fn(tensor)
            if converted.dtype == torch.float16 and any(tensor is statistic for statistic in statistics):
                # Recover from the original buffer: promoting an already-overflowed value cannot restore it.
                return tensor.to(device=converted.device, dtype=torch.float32)
            return converted

        return super()._apply(convert, recurse=recurse)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        for module in self:
            if getattr(module, "_descriptor_float32_statistics", False) and input.dtype == torch.float16:
                # MPS also requires the input and running statistics to have the same dtype.
                input = module(input.float()).to(input.dtype)
            else:
                input = module(input)
        return input
