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

import pytest
from torch import Tensor, nn

from kornia.profiler.collectors.layer import LayerCollector
from kornia.profiler.profiler import Profiler


class SimpleModel(nn.Module):
    """Simple fully connected model for profiler failure tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the fully connected layer."""
        return self.fc(x)


def test_invalid_metric_does_not_attach_hooks() -> None:
    """Verify invalid metrics are rejected before hooks are registered."""
    model = SimpleModel()
    collector = LayerCollector(layers=["fc"])

    try:
        with pytest.raises(ValueError, match="Unknown metric"):
            Profiler(
                model=model,
                collector=collector,
                metrics=["invalid_metric"],
            )

        assert len(model.fc._forward_hooks) == 0
    finally:
        collector.remove_hooks()
