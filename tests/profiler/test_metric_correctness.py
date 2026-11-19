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
from kornia.profiler.metrics import CosineSimilarity
from kornia.profiler.profiler import Profiler


class SimpleModel(nn.Module):
    """Simple linear model for profiler metric tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the linear layer to the input tensor."""
        return self.fc(x)


def test_cosine_similarity_identical_tensors() -> None:
    """Verify that identical tensors have cosine similarity equal to one."""
    metric = CosineSimilarity()
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    result = metric(x, x)

    assert torch.isfinite(result).all()
    torch.testing.assert_close(
        result,
        torch.tensor(1.0),
        atol=1e-6,
        rtol=1e-6,
    )


def test_cosine_similarity_orthogonal_tensors() -> None:
    """Verify that orthogonal tensors have cosine similarity equal to zero."""
    metric = CosineSimilarity()
    x = torch.tensor([[1.0, 0.0]])
    y = torch.tensor([[0.0, 1.0]])

    result = metric(x, y)

    assert torch.isfinite(result).all()
    torch.testing.assert_close(
        result,
        torch.tensor(0.0),
        atol=1e-6,
        rtol=1e-6,
    )


def test_profiler_reports_expected_cosine_similarity() -> None:
    """Verify that the profiler reports similarity for identical features."""
    model = SimpleModel()
    collector = LayerCollector(layers=["fc"])

    try:
        profiler = Profiler(
            model=model,
            collector=collector,
            metrics=["cosine"],
        )

        x = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

        profiler(group="original", x=x)
        profiler(group="comparison", x=x.clone())
        profiler.compute(["original", "comparison"])

        results = profiler.df
        assert results is not None
        assert results.shape == (1, 1)
        assert "cosine_fc" in results.columns

        result = results.loc[0, "cosine_fc"]
        assert result == pytest.approx(1.0, abs=1e-6)
    finally:
        collector.remove_hooks()
