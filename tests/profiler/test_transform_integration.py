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

from kornia.profiler import LayerCollector, Profiler
from kornia.profiler.transforms import Flatten


class SimpleModel(nn.Module):
    """Simple convolutional model for transform integration tests."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the convolutional layer to the input tensor."""
        return self.conv(x)


def test_composed_transform() -> None:
    """Verify that composed transforms work with the profiler's metrics."""
    model = SimpleModel()
    collector = LayerCollector(layers=["conv"])

    try:
        profiler = Profiler(
            model,
            collector=collector,
            pre_metric_transform=Flatten() | "flatten",
            metrics=["cosine", "linear"],
        )

        profiler(group="a", x=torch.rand(2, 3, 8, 8))
        profiler(group="b", x=torch.rand(2, 3, 8, 8))
        profiler.compute(["a", "b"])

        results = profiler.df
        assert results is not None
        assert results.shape == (1, 2)
        assert "cosine_conv" in results.columns
        assert "linear_conv" in results.columns
    finally:
        collector.remove_hooks()


if __name__ == "__main__":
    test_composed_transform()
    print("PASS: composed transform works in Profiler")
