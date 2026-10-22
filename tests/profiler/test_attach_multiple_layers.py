import torch
from torch import Tensor, nn

from kornia.profiler.collectors.layer import LayerCollector
def test_attach_to_multiple_models_with_layers_none() -> None:
    """Resolve all layer names independently for each attached model."""

    class FirstModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first_layer = nn.Linear(4, 3)

        def forward(self, x: Tensor) -> Tensor:
            return self.first_layer(x)

    class SecondModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.second_layer = nn.Linear(4, 2)

        def forward(self, x: Tensor) -> Tensor:
            return self.second_layer(x)

    collector = LayerCollector(layers=None)

    first_model = FirstModel()
    collector.attach(first_model)
    assert collector.layers is None
    assert "first_layer" in collector.active_layers

    collector.remove_hooks()

    second_model = SecondModel()
    collector.attach(second_model)
    try:
        assert collector.layers is None
        assert "second_layer" in collector.active_layers
        features = collector(x=torch.randn(2, 4))
        assert "second_layer" in features
        assert "first_layer" not in features
    finally:
        collector.remove_hooks()