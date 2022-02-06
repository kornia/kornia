import numpy as np
import pytest
import torch

from kornia.color.gray import rgb_to_grayscale
from kornia.core import Image, ImageColor
from kornia.testing import assert_close


class TestImage:
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32, torch.float64])
    def test_from_tensor(self, device, dtype):
        data = torch.ones(3, 4, 5, device=device, dtype=dtype)
        img = Image.from_tensor(data, color=ImageColor.RGB)
        assert isinstance(img, Image)
        assert img.channels == 3
        assert img.height == 4
        assert img.width == 5
        assert img.color == ImageColor.RGB
        assert not img.is_batch
        assert img.shape == (3, 4, 5)
        assert img.dtype == dtype
        assert img.device == device

    def test_from_numpy(self, device, dtype):
        data = np.ones((4, 5, 3))
        img = Image.from_numpy(data, color=ImageColor.RGB)
        assert isinstance(img, Image)
        assert img.channels == 3
        assert img.height == 4
        assert img.width == 5
        assert img.color == ImageColor.RGB
        assert not img.is_batch
        # check clone
        img2 = img.clone()
        assert isinstance(img2, Image)
        img2 = img2.to(device, dtype)
        assert img2.dtype == dtype
        assert img2.device == device
        img3 = img2.to(torch.uint8)
        assert isinstance(img3, Image)
        assert img3.dtype == torch.uint8
        assert img3.device == device

    def test_from_list(self):
        data = [[[1.0, 2.0], [3.0, 4.0]]]
        img = Image.from_list(data, color=ImageColor.GRAY)
        assert isinstance(img, Image)
        assert img.channels == 1
        assert img.height == 2
        assert img.width == 2
        assert img.color == ImageColor.GRAY
        assert not img.is_batch

    def test_grayscale(self):
        data = torch.rand(3, 4, 5)
        img = Image.from_tensor(data, color=ImageColor.RGB)
        img_gray = img.grayscale()
        assert isinstance(img, Image)
        assert isinstance(img_gray, Image)
        assert img_gray.channels == 1
        assert img_gray.height == 4
        assert img_gray.width == 5
        assert img_gray.color == ImageColor.GRAY
        assert not img.is_batch
        assert_close(img_gray, rgb_to_grayscale(img))
        # inplace
        img = img.grayscale_()
        assert isinstance(img, Image)
        assert img.channels == 1
        assert img.height == 4
        assert img.width == 5
        assert img.color == ImageColor.GRAY
        assert not img.is_batch
        assert_close(img_gray, img)
