import torch
import pytest

from kornia.augmentation._2d.mix.patchmix import PatchMix

@pytest.mark.parametrize("batch_size,channels,height,width,patch_size", [
    (4, 3, 32, 32, 8),
    (2, 1, 16, 16, 4),
])
def test_patchmix_shape_and_type(batch_size, channels, height, width, patch_size):
    aug = PatchMix(alpha=1.0, patch_size=patch_size)
    x = torch.rand(batch_size, channels, height, width)
    params = aug.generate_parameters(x.shape)
    out, idx, lam = aug.apply_transform(x, params, {})
    assert out.shape == x.shape
    assert idx.shape[0] == batch_size
    assert lam.shape[0] == batch_size
    assert out.dtype == x.dtype

def test_patchmix_different_inputs():
    aug = PatchMix(alpha=1.0, patch_size=8)
    x = torch.rand(4, 3, 32, 32)
    y = torch.rand(4, 3, 32, 32)
    params = aug.generate_parameters(x.shape)
    out1, _, _ = aug.apply_transform(x, params, {})
    out2, _, _ = aug.apply_transform(y, params, {})
    assert not torch.allclose(out1, out2)

def test_patchmix_grad():
    aug = PatchMix(alpha=1.0, patch_size=8)
    x = torch.rand(2, 3, 16, 16, requires_grad=True)
    params = aug.generate_parameters(x.shape)
    out, _, _ = aug.apply_transform(x, params, {})
    out.sum().backward()
    assert x.grad is not None