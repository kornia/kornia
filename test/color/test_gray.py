import pytest
import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose
from common import device_type

import torchgeometry as K
import torchgeometry.color as color
import utils


class TestRgbToGrayscale:
    def test_rgb_to_grayscale(self):
        channels, height, width = 3, 4, 5
        img = torch.ones(channels, height, width)
        assert K.rgb_to_grayscale(img).shape == (1, height, width)
        #assert color.RgbToGrayscale()(img).shape == (1, height, width)

    def test_rgb_to_grayscale_batch(self):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width)
        assert color.RgbToGrayscale()(img).shape == \
            (batch_size, 1, height, width)

    def test_gradcheck(self):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(color.rgb_to_grayscale, (img,), raise_exception=True)

    def test_jit(self):
        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width)
        gray = color.RgbToGrayscale()
        gray_traced = torch.jit.trace(color.RgbToGrayscale(), img)
        assert_allclose(gray(img), gray_traced(img))
