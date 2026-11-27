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

from torch import Tensor, nn


class Metric(nn.Module):
    """Base class for representation comparison metrics.

    Subclasses must implement :meth:`forward` to compare two input
    representations and return a tensor containing the resulting score.

    Args:
        None
    """

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Compare two representations and return a tensor score.

        Args:
            x: First input representation.
            y: Second input representation.

        Returns:
            A tensor containing the comparison score.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError
