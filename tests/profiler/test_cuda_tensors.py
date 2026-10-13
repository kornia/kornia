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
import torch
from torch import Tensor, nn

from kornia.profiler.collectors.layer import LayerCollector
from kornia.profiler.profiler import Profiler

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)


class SimpleModel(nn.Module):
    """Simple fully connected model for CUDA profiler tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2).cuda()

    def forward(self, x: Tensor) -> Tensor:
        """Apply the fully connected layer."""
        return self.fc(x)


def test_profiler_handles_cuda_tensors() -> None:
    """Verify that profiling preserves CUDA-resident representations."""
    model = SimpleModel()
    collector = LayerCollector(layers=["fc"])

    try:
        profiler = Profiler(
            model=model,
            collector=collector,
            metrics=["cosine", "linear"],
        )

        x1 = torch.randn(2, 4, device="cuda")
        x2 = torch.randn(2, 4, device="cuda")

        profiler(group="original", x=x1)
        profiler(group="comparison", x=x2)
        profiler.compute(["original", "comparison"])

        results = profiler.df
        assert results is not None
        assert results.shape == (1, 2)
        assert "cosine_fc" in results.columns
        assert "linear_fc" in results.columns

        # Confirm that the collector captured CUDA-resident features.
        features = profiler.storage["original"][0]["features"]["fc"]
        assert features.device.type == "cuda"
    finally:
        collector.remove_hooks()
