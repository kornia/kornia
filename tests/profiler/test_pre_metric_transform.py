import torch
from torch import Tensor, nn

from kornia.profiler import LayerCollector, Profiler


class SimpleModel(nn.Module):
    """Simple convolutional model for metric transformation tests."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the convolutional layer to the input tensor."""
        return self.conv(x)


def run_profiler(pre_metric_transform: str) -> None:
    """Run the profiler with the specified pre-metric transformation."""
    model = SimpleModel()
    collector = LayerCollector(layers=["conv"])

    try:
        profiler = Profiler(
            model,
            collector=collector,
            pre_metric_transform=pre_metric_transform,
            metrics=["cosine", "linear"],
        )

        profiler(group="original", x=torch.rand(2, 3, 8, 8))
        profiler(group="changed", x=torch.rand(2, 3, 8, 8))
        profiler.compute(["original", "changed"])

        results = profiler.df
        assert results is not None
        assert results.shape == (1, 2)
        assert "cosine_conv" in results.columns
        assert "linear_conv" in results.columns
    finally:
        collector.remove_hooks()


def test_flatten_transform() -> None:
    """Verify that the flatten transformation supports the configured metrics."""
    run_profiler("flatten")


if __name__ == "__main__":
    test_flatten_transform()
    print("PASS: flatten transform")
