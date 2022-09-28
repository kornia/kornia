from __future__ import annotations

import pytest
import torch


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_smoke(batch_size):
    x = torch.rand(batch_size, 2, 3)
    assert x.shape == (batch_size, 2, 3), x.shape
