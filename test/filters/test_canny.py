import pytest

import kornia as kornia
import kornia.testing as utils  # test utils

import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck

import cv2

class TestCanny:
    def test_shape(self, device, dtype):
        inp = torch.zeros(1, 3, 4, 4, device=device, dtype=dtype)
        canny = kornia.filters.Canny()
        assert canny(inp).shape == (1, 3, 4, 4)

