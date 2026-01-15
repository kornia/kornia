import pytest
import torch
from kornia.color import rgb_to_xyz, xyz_to_rgb


@pytest.mark.parametrize("B", [1, 8, 32])
@pytest.mark.parametrize("C", [3])
@pytest.mark.parametrize("H", [128, 256, 512])
@pytest.mark.parametrize("W", [128, 256, 512])
def test_rgb_to_xyz_benchmark(benchmark, device, dtype, torch_optimizer, B, C, H, W):
    """Benchmark the rgb_to_xyz function."""
    
    data = torch.rand(B, C, H, W, device=device, dtype=dtype)
    op = torch_optimizer(rgb_to_xyz)
    actual = benchmark(op, data)
    
    assert actual.shape == (B, 3, H, W)
    
    
@pytest.mark.parametrize("B", [1, 8, 32])
@pytest.mark.parametrize("C", [3])
@pytest.mark.parametrize("H", [128, 256, 512])
@pytest.mark.parametrize("W", [128, 256, 512])
def test_xyz_to_rgb_benchmark(benchmark, device, dtype, torch_optimizer, B, C, H, W):
    """Benchmark the xyz_to_rgb function."""
    
    data = torch.rand(B, C, H, W, device=device, dtype=dtype)
    op = torch_optimizer(xyz_to_rgb)
    actual = benchmark(op, image=data)
    
    assert actual.shape == (B, 3, H, W)