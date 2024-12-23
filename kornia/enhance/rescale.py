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

from kornia.core import ImageModule as Module
from kornia.core import Tensor, tensor


class Rescale(Module):
    r"""Initialize the Rescale operator.

    Args:
        factor: The scaling factor. Could be a float or a 0-d tensor.

    """

    def __init__(self, factor: Union[float, Tensor]) -> None:
        super().__init__()
        if isinstance(factor, float):
            self.factor = tensor(factor)
        else:
            if not isinstance(factor, Tensor) or factor.ndim != 0:
                raise TypeError(f"Expected factor to be a float or a 0-d tensor, got {factor}.")
            self.factor = factor

    def forward(self, input: Tensor) -> Tensor:
        return input * self.factor
