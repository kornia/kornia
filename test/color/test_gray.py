import pytest

import kornia
import kornia.testing as utils  # test utils
from test.common import device_type

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestRgbToGrayscale:
    def test_rgb_to_grayscale(self):
        channels, height, width = 3, 4, 5
        img = torch.ones(channels, height, width)
        assert kornia.rgb_to_grayscale(img).shape == (1, height, width)

    def test_rgb_to_grayscale_batch(self):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width)
        assert kornia.rgb_to_grayscale(img).shape == \
            (batch_size, 1, height, width)

    def test_gradcheck(self):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.rgb_to_grayscale, (img,), raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width)
        gray = kornia.color.RgbToGrayscale()
        gray_traced = torch.jit.trace(kornia.color.RgbToGrayscale(), img)
        assert_allclose(gray(img), gray_traced(img))


class TestBgrToGrayscale:
    def test_bgr_to_grayscale(self):
        channels, height, width = 3, 4, 5
        img = torch.ones(channels, height, width)
        assert kornia.bgr_to_grayscale(img).shape == (1, height, width)

    def test_bgr_to_grayscale_batch(self):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width)
        assert kornia.bgr_to_grayscale(img).shape == \
            (batch_size, 1, height, width)

    def test_gradcheck(self):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.bgr_to_grayscale, (img,), raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width)
        gray = kornia.color.BgrToGrayscale()
        gray_traced = torch.jit.trace(kornia.color.BgrToGrayscale(), img)
        assert_allclose(gray(img), gray_traced(img))
