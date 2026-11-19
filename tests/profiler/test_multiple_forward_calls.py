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

import torch
from torch import Tensor, nn

from kornia.profiler.collectors.layer import LayerCollector
from kornia.profiler.profiler import Profiler


class SimpleModel(nn.Module):
    """Linear model with deterministic weights for feature capture tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2, bias=False)

        with torch.no_grad():
            self.fc.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                    ]
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the linear layer to the input tensor."""
        return self.fc(x)


def test_multiple_forward_calls_capture_separate_features() -> None:
    """Verify that repeated profiler calls preserve separate feature values."""
    model = SimpleModel()
    collector = LayerCollector(layers=["fc"])

    try:
        profiler = Profiler(
            model=model,
            collector=collector,
            metrics=["cosine"],
        )

        x1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        x2 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        x3 = torch.tensor([[0.0, 0.0, 1.0, 0.0]])

        profiler(group="repeated", x=x1)
        profiler(group="repeated", x=x2)
        profiler(group="repeated", x=x3)

        entries = profiler.storage["repeated"]

        assert len(entries) == 3

        features = [entry["features"]["fc"] for entry in entries]

        torch.testing.assert_close(
            features[0],
            torch.tensor([[1.0, 0.0]]),
        )
        torch.testing.assert_close(
            features[1],
            torch.tensor([[0.0, 1.0]]),
        )
        torch.testing.assert_close(
            features[2],
            torch.tensor([[0.0, 0.0]]),
        )

        # Each call should preserve its own captured feature values.
        assert not torch.equal(features[0], features[1])
        assert not torch.equal(features[1], features[2])
    finally:
        collector.remove_hooks()
