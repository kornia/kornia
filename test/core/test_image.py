import numpy as np
import pytest
import torch

from kornia.color.gray import rgb_to_grayscale
from kornia.core import Image, ImageColor
from kornia.geometry.transform import resize
from kornia.testing import assert_close


class TestImage:
    def test_smoke(self, device):
        data = torch.randint(0, 255, (3, 4, 5), device=device, dtype=torch.uint8)
        img = Image(data, ImageColor.RGB8)
        assert isinstance(img, Image)
        assert img.channels == 3
        assert img.height == 4
        assert img.width == 5
        assert img.color == ImageColor.RGB8
        assert not img.is_batch
        assert img.resolution == (4, 5)
        assert img.shape == (3, 4, 5)
        assert img._mean() is None
        assert img._std() is None
        assert img.device == device
        assert img.dtype == torch.uint8

    def test_batch(self, device, dtype):
        data = torch.randint(0, 255, (2, 1, 4, 5), device=device, dtype=dtype)
        img = Image(data, ImageColor.GRAY8)
        assert isinstance(img, Image)
        assert img.channels == 1
        assert img.height == 4
        assert img.width == 5
        assert img.color == ImageColor.GRAY8
        assert img.is_batch
        assert img.resolution == (4, 5)
        assert img.shape == (2, 1, 4, 5)
        assert img._mean() is None
        assert img._std() is None
        assert img.device == device
        assert img.dtype == dtype

        # slice
        x1 = img[1]
        assert isinstance(x1, Image)
        assert x1.channels == 1
        assert x1.height == 4
        assert x1.width == 5
        assert x1.color == ImageColor.GRAY8
        assert not x1.is_batch
        assert x1.resolution == (4, 5)
        assert x1.shape == (1, 4, 5)
        assert x1.device == device
        assert x1.dtype == dtype

    def test_numpy(self, device, dtype):
        # as it was from cv2.imread
        data = np.ones((4, 5, 3), dtype=np.uint8)
        img = Image.from_numpy(data, color=ImageColor.BGR8)
        img = img.to(device, dtype)
        assert isinstance(img, Image)
        assert img.channels == 3
        assert img.height == 4
        assert img.width == 5
        assert img.color == ImageColor.BGR8
        assert not img.is_batch
        assert img.resolution == (4, 5)
        assert img.shape == (3, 4, 5)
        assert img.device == device
        assert img.dtype == dtype
        assert_close(data, img.to_numpy())

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

    def test_dlpack(self, device, dtype):
        data = torch.rand((3, 4, 5), device=device, dtype=dtype)
        color = ImageColor.RGB8
        img = Image(data, color)
        assert_close(data, Image.from_dlpack(img.to_dlpack(), color).data)

    def test_denormalize(self, device, dtype):
        # opencv case
        data = np.ones((4, 5, 3), dtype=np.uint8)
        img = Image.from_numpy(data, ImageColor.BGR8)

        # type error: must be floating_point.
        # we assume that the user must convert to floating point
        with pytest.raises(TypeError):
            img.denormalize()

        # case where type it floating point but not mean/std provided
        img = img.to(device, dtype) / 255.0
        assert_close(img.denormalize(), img)

        # data in floating point and ranges [0, 1]
        data = torch.rand(3, 4, 5, device=device, dtype=dtype)
        img = Image(data, color=ImageColor.BGR8, mean=[0, 0, 0], std=[255, 255, 255])
        assert img.mean().item() <= 1.0
        assert img.denormalize().mean().item() > 1.0

    def test_apply(self, device):
        data = torch.randint(0, 255, (3, 4, 5), device=device, dtype=torch.uint8)
        img = Image(data, color=ImageColor.RGB8)
        assert isinstance(img, Image)
        # the user needs to normalize the image [0, 1] in floating point
        import pdb

        pdb.set_trace()
        img_norm = img.float() / 255.0
        img_gray = img_norm.apply(rgb_to_grayscale)
        assert isinstance(img_gray, Image)
        assert img_gray.channels == 1
        assert img_gray.height == 4
        assert img_gray.width == 5
        assert img_gray.resolution == (4, 5)

        # apply a resize function
        img_resize = img_gray.apply(resize, (2, 3))
        assert isinstance(img_resize, Image)
        assert img_resize.channels == 1
        assert img_resize.height == 2
        assert img_resize.width == 3
        assert img_resize.resolution == (2, 3)
