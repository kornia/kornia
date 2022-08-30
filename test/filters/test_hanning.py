import pytest
import torch

import kornia
from kornia.testing import assert_close


@pytest.mark.parametrize("window_size", [5, 11])
def test_get_hanning_kernel(window_size, device, dtype):
    kernel = kornia.filters.get_hanning_kernel1d(window_size, dtype=dtype, device=device)
    assert kernel.shape == (window_size,)
    assert kernel.max().item() == pytest.approx(1.0)


@pytest.mark.parametrize("ksize_x", [5, 11])
@pytest.mark.parametrize("ksize_y", [3, 7])
def test_get_hanning_kernel2d(ksize_x, ksize_y, device, dtype):
    kernel = kornia.filters.get_hanning_kernel2d((ksize_x, ksize_y), dtype=dtype, device=device)
    assert kernel.shape == (ksize_x, ksize_y)
    assert kernel.max().item() == pytest.approx(1.0)


def test_get_hanning_kernel1d_5(device, dtype):
    kernel = kornia.filters.get_hanning_kernel1d(5, dtype=dtype, device=device)
    expected = torch.tensor([0, 0.5, 1.0, 0.5, 0], dtype=dtype, device=device)
    assert kernel.shape == (5,)
    assert_close(kernel, expected)


def test_get_hanning_kernel2d_3x4(device, dtype):
    kernel = kornia.filters.get_hanning_kernel2d((3, 4), dtype=dtype, device=device)
    expected = torch.tensor(
        [[0.0, 0.00, 0.00, 0.0], [0.0, 0.75, 0.75, 0.0], [0.0, 0.00, 0.00, 0.0]], dtype=dtype, device=device
    )
    assert kernel.shape == (3, 4)
    assert_close(kernel, expected)
