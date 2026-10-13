import pytest
import torch
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
