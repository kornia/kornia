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

from typing import Union

import torch
from torch import nn


class Rescale(nn.Module):
    r"""Initialize the Rescale operator.

    Args:
        factor: The scaling factor. Could be a float or a 0-d torch.tensor.

    """

    def __init__(self, factor: Union[float, torch.Tensor]) -> None:
        super().__init__()
        if isinstance(factor, float):
            self.factor = torch.tensor(factor)
        else:
            if not isinstance(factor, torch.Tensor) or factor.ndim != 0:
                raise TypeError(f"Expected factor to be a float or a 0-d torch.tensor, got {factor}.")
            self.factor = factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.factor
