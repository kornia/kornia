from pathlib import Path

import numpy as np
import torch

from kornia.image.base import ChannelsOrder, ImageLayout, ImageSize, PixelFormat
from kornia.image.image import Image
from kornia.testing import assert_close


class TestImage:
    def test_smoke(self, device):
        data = torch.randint(0, 255, (3, 4, 5), device=device, dtype=torch.uint8)
        layout = ImageLayout(
            image_size=ImageSize(4, 5),
            channels=3,
            pixel_format=PixelFormat.RGB,
            channels_order=ChannelsOrder.CHANNELS_FIRST,
        )

        img = Image(data, layout)
        assert isinstance(img, Image)
        assert img.channels == 3
        assert img.height == 4
        assert img.width == 5
        assert img.shape == (3, 4, 5)
        assert img.device == device
        assert img.dtype == torch.uint8
        assert img.layout == layout
        assert img.pixel_format == PixelFormat.RGB
        assert img.channels_order == ChannelsOrder.CHANNELS_FIRST

    def test_numpy(self, device):
        # as it was from cv2.imread
        data = np.ones((4, 5, 3), dtype=np.uint8)
        img = Image.from_numpy(data, pixel_format=PixelFormat.RGB)
        img = img.to(device)
        assert isinstance(img, Image)
        assert img.channels == 3
        assert img.height == 4
        assert img.width == 5
        assert img.pixel_format == PixelFormat.RGB
        assert img.shape == (3, 4, 5)
        assert img.device == device
        assert img.dtype == torch.uint8
        assert_close(data, img.to_numpy())

        # check clone
        img2 = img.clone()
        assert isinstance(img2, Image)
        img2 = img2.to(device)
        assert img2.dtype == torch.uint8
        assert img2.device == device
        img3 = img2.to(torch.uint8)
        assert isinstance(img3, Image)
        assert img3.dtype == torch.uint8
        assert img3.device == device

    def test_dlpack(self, device, dtype):
        data = torch.rand((3, 4, 5), device=device, dtype=dtype)
        layout = ImageLayout(
            image_size=ImageSize(4, 5),
            channels=3,
            pixel_format=PixelFormat.RGB,
            channels_order=ChannelsOrder.CHANNELS_FIRST,
        )
        img = Image(data, layout=layout)
        assert_close(data, Image.from_dlpack(img.to_dlpack()).data)

    def test_load_write(self, tmp_path: Path) -> None:
        data = torch.randint(0, 255, (3, 4, 5), dtype=torch.uint8)
        img = Image.from_numpy(data.numpy(), channels_order=ChannelsOrder.CHANNELS_FIRST)

        file_name = tmp_path / "image.jpg"

        img.write(str(file_name))
        img2 = Image.from_file(str(file_name))

        # NOTE: the tolerance is high due to the jpeg compression
        assert (img.float().data - img2.float().data).pow(2).mean() > 0.5
