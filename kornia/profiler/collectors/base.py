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

from abc import ABC, abstractmethod
from typing import Dict

from torch import Tensor, nn


class Collector(ABC):
    """Abstract interface for collecting representations from a model.

    Subclasses implement model attachment, representation collection, and
    cleanup of any hooks registered during attachment.
    """

    @abstractmethod
    def attach(self, model: nn.Module) -> None:
        """Attach the collector to a model.

        Args:
            model: PyTorch model from which representations will be collected.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, **inputs: Tensor) -> Dict[str, Tensor]:
        """Run the attached model and return collected representations.

        Args:
            **inputs: Tensor-valued keyword arguments passed to the model.

        Returns:
            A dictionary mapping representation names to their tensors.

        Raises:
            RuntimeError: If the collector has not been attached to a model,
                when applicable to the implementation.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_hooks(self) -> None:
        """Remove hooks registered by the collector.

        Implementations should release hook registrations and associated
        references as appropriate.
        """
        raise NotImplementedError
