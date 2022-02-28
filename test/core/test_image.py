import numpy as np
import pytest
import torch

from kornia.color.gray import rgb_to_grayscale
from kornia.core import Image, ImageColor
from kornia.testing import assert_close


class TestImage:
    def test_smoke(self, device):
        data = torch.randint(0, 255, (3, 4, 5), dtype=torch.uint8)
        img = Image(data, ImageColor.BGR, is_normalized=False)
        img = img.to(device)
        assert isinstance(img, Image)
        assert img.channels == 3
        assert img.height == 4
        assert img.width == 5
        assert img.color == ImageColor.BGR
        assert not img.is_normalized
        assert not img.is_batch
        assert img.shape == (3, 4, 5)
        assert img.device == device

    def test_batch(self, device, dtype):
        data = torch.randint(0, 255, (2, 1, 4, 5), dtype=torch.uint8)
        img = Image(data, ImageColor.GRAY, is_normalized=False)
        assert img.dtype == torch.uint8
        img = img.to(device, dtype)
        assert isinstance(img, Image)
        assert img.channels == 1
        assert img.height == 4
        assert img.width == 5
        assert img.color == ImageColor.GRAY
        assert img.is_batch
        assert not img.is_normalized
        assert img.device == device
        assert img.dtype == dtype

        # slice
        x1 = img[1]
        assert isinstance(x1, Image)
        assert x1.channels == 1
        assert x1.height == 4
        assert x1.width == 5
        assert x1.color == ImageColor.GRAY
        assert not x1.is_batch
        assert not x1.is_normalized
        assert x1.device == device
        assert x1.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32, torch.float64])
    def test_from_tensor(self, device, dtype):
        data = torch.ones(3, 4, 5, device=device, dtype=dtype)
        img = Image.from_tensor(data, color=ImageColor.BGR, is_normalized=False)
        img = img.to(device)
        assert isinstance(img, Image)
        assert img.channels == 3
        assert img.height == 4
        assert img.width == 5
        assert img.color == ImageColor.BGR
        assert not img.is_batch
        assert img.shape == (3, 4, 5)
        assert img.dtype == dtype
        assert img.device == device
        assert not img.is_normalized

    def test_from_numpy(self, device, dtype):
        data = np.ones((4, 5, 3), dtype=np.uint8)
        img = Image.from_numpy(data, color=ImageColor.RGB, is_normalized=False)
        img = img.to(device, dtype)
        assert isinstance(img, Image)
        assert img.channels == 3
        assert img.height == 4
        assert img.width == 5
        assert img.color == ImageColor.RGB
        assert not img.is_batch
        assert not img.is_normalized
        assert img.device == device
        assert img.dtype == dtype
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
        assert not img3.is_normalized

    def test_from_list(self, device, dtype):
        data = [[[1.0, 2.0], [3.0, 4.0]]]
        img = Image.from_list(data, color=ImageColor.GRAY, is_normalized=True)
        img = img.to(device, dtype)
        assert isinstance(img, Image)
        assert img.channels == 1
        assert img.height == 2
        assert img.width == 2
        assert img.color == ImageColor.GRAY
        assert not img.is_batch
        assert img.is_normalized
        assert img.dtype == dtype
        assert img.device == device

    def test_normalize(self, device):
        data = torch.randint(0, 255, (3, 4, 5), device=device, dtype=torch.uint8)
        img = Image(data, color=ImageColor.RGB, is_normalized=False)
        assert not img.is_normalized

        img_norm = img.normalize()
        assert img_norm.mean().item() < 1.0
        assert img_norm.is_normalized

        img_denorm = img_norm.denormalize()
        assert img_denorm.mean().item() > 1.0
        assert not img_denorm.is_normalized
        assert_close(img_denorm, img)

    def test_grayscale(self, device):
        data = torch.randint(0, 255, (3, 4, 5), device=device, dtype=torch.uint8)
        img = Image(data, color=ImageColor.RGB, is_normalized=False)
        assert not img.is_normalized

        img_norm = img.normalize()
        img_gray = img_norm.apply(rgb_to_grayscale)
        assert isinstance(img, Image)
        assert isinstance(img_gray, Image)
        assert img_gray.channels == 1
        assert img_gray.height == 4
        assert img_gray.width == 5
        assert img_gray.color == ImageColor.GRAY
        assert not img.is_batch
        assert img_gray.is_normalized
