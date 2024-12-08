import pytest
import torch

from kornia.color import grayscale_to_rgb


@pytest.mark.parametrize("B", [1, 5])
@pytest.mark.parametrize("C", [1])
@pytest.mark.parametrize("H", [128, 237, 512])
@pytest.mark.parametrize("W", [128, 237, 512])
def test_grayscale_to_rgb(benchmark, device, dtype, torch_optimizer, B, C, H, W):
    data = torch.rand(B, C, H, W, device=device, dtype=dtype)
    op = torch_optimizer(grayscale_to_rgb)

    actual = benchmark(op, image=data)

    assert actual.shape == (B, 3, H, W)
