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

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn


class Float32BatchNorm2d(nn.BatchNorm2d):
    """Keep float32 running stats and use float32 math for float16 inputs."""

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> "Float32BatchNorm2d":
        stats = {id(t) for t in (self.running_mean, self.running_var) if t is not None}

        def keep_stats_float32(t: torch.Tensor) -> torch.Tensor:
            converted = fn(t)
            if id(t) in stats and converted.dtype == torch.float16:
                return t.to(device=converted.device, dtype=torch.float32)
            return converted

        return super()._apply(keep_stats_float32, recurse)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        exponential_average_factor = 0.0 if self.momentum is None else self.momentum

        if self.training and self.track_running_stats and self.num_batches_tracked is not None:
            self.num_batches_tracked.add_(1)
            if self.momentum is None:
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)

        bn_training = self.training or (self.running_mean is None and self.running_var is None)

        output = F.batch_norm(
            input.float() if input.dtype == torch.float16 else input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
        return output.to(input.dtype)
