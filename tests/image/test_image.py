# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from pathlib import Path

import numpy as np
import pytest
import torch

from kornia.image.base import ChannelsOrder, ColorSpace, ImageLayout, ImageSize, PixelFormat
from kornia.image.image import Image
from kornia.utils._compat import torch_version_le

from testing.base import assert_close


class TestImage:
    def test_smoke(self, device):
        data = torch.randint(0, 255, (3, 4, 5), device=device, dtype=torch.uint8)
        pixel_format = PixelFormat(color_space=ColorSpace.RGB, bit_depth=8)
        layout = ImageLayout(image_size=ImageSize(4, 5), channels=3, channels_order=ChannelsOrder.CHANNELS_FIRST)

        img = Image(data, pixel_format, layout)
        assert isinstance(img, Image)
        assert img.channels == 3
        assert img.height == 4
        assert img.width == 5
        assert img.shape == (3, 4, 5)
        assert img.device == device
        assert img.dtype == torch.uint8
        assert img.layout == layout
        assert img.pixel_format.color_space == ColorSpace.RGB
        assert img.pixel_format.bit_depth == 8
        assert img.channels_order == ChannelsOrder.CHANNELS_FIRST

    def test_numpy(self, device):
        # as it was from cv2.imread
        data = np.ones((4, 5, 3), dtype=np.uint8)
        img = Image.from_numpy(data, color_space=ColorSpace.RGB)
        img = img.to(device)
        assert isinstance(img, Image)
        assert img.channels == 3
        assert img.height == 4
        assert img.width == 5
        assert img.pixel_format.color_space == ColorSpace.RGB
        assert img.shape == (4, 5, 3)
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
        pixel_format = PixelFormat(color_space=ColorSpace.RGB, bit_depth=data.element_size() * 8)
        layout = ImageLayout(image_size=ImageSize(4, 5), channels=3, channels_order=ChannelsOrder.CHANNELS_FIRST)
        img = Image(data, pixel_format=pixel_format, layout=layout)
        assert_close(data, Image.from_dlpack(img.to_dlpack()).data)

    @pytest.mark.skipif(torch_version_le(1, 9, 1), reason="dlpack is broken in torch<=1.9.1")
    @pytest.mark.xfail(reason="This may fail some time due to jpeg compression assertion")
    def test_load_write(self, tmp_path: Path) -> None:
        data = torch.randint(0, 255, (3, 4, 5), dtype=torch.uint8)
        img = Image.from_numpy(data.numpy(), channels_order=ChannelsOrder.CHANNELS_FIRST)

        file_name = tmp_path / "image.jpg"

        img.write(file_name)
        img2 = Image.from_file(file_name)

        # NOTE: the tolerance is high due to the jpeg compression
        assert (img.float().data - img2.float().data).pow(2).mean() <= 0.75

    def test_write_first_channel(self, tmp_path: Path) -> None:
        data = np.ones((4, 5, 3), dtype=np.uint8)
        img = Image.from_numpy(data, color_space=ColorSpace.RGB, channels_order=ChannelsOrder.CHANNELS_LAST)
        img.write(tmp_path / "image.jpg")

    def make_image(self, data: torch.Tensor, cs: ColorSpace, order: ChannelsOrder) -> Image:
        """Wrap a tensor into Kornia's Image with given color space & layout"""
        if order == ChannelsOrder.CHANNELS_FIRST:
            channels, H, W = data.shape
        else:
            H, W, channels = data.shape
        pf = PixelFormat(color_space=cs, bit_depth=data.element_size() * 8)
        layout = ImageLayout(image_size=ImageSize(height=H, width=W), channels=channels, channels_order=order)
        return Image(data.clone(), pf, layout)

    @pytest.mark.parametrize("order", [ChannelsOrder.CHANNELS_FIRST, ChannelsOrder.CHANNELS_LAST])
    def test_identity_to_color_space(self, order, device):
        # RGB → RGB no-op
        if order == ChannelsOrder.CHANNELS_FIRST:
            data = torch.arange(12, dtype=torch.uint8, device=device).view(3, 2, 2)
        else:
            data = torch.arange(12, dtype=torch.uint8, device=device).view(2, 2, 3)
        img = self.make_image(data, ColorSpace.RGB, order)
        out = img.to_color_space(ColorSpace.RGB)
        assert out is img

    @pytest.mark.parametrize(
        "src, dst",
        [
            (ColorSpace.RGB, ColorSpace.BGR),
            (ColorSpace.BGR, ColorSpace.RGB),
        ],
    )
    def test_flip_channels(self, src, dst, device):
        # verify RGB↔BGR channel reversal
        data = torch.stack(
            [torch.full((2, 2), fill_value=i, dtype=torch.uint8, device=device) for i in (10, 20, 30)]
        )  # C*H*W
        img = self.make_image(data, src, ChannelsOrder.CHANNELS_FIRST)
        out = img.to_color_space(dst)
        assert torch.equal(out.data, data[[2, 1, 0]])
        assert out.pixel_format.color_space == dst

    def test_gray_from_rgb(self, device):
        # single pixel luminosity formula
        data = torch.tensor([[[8]], [[16]], [[32]]], dtype=torch.uint8, device=device)
        img = self.make_image(data, ColorSpace.RGB, ChannelsOrder.CHANNELS_FIRST)
        out = img.to_color_space(ColorSpace.GRAY)
        expected = int(0.2989 * 8 + 0.5870 * 16 + 0.1140 * 32)
        assert out.pixel_format.color_space == ColorSpace.GRAY
        assert out.data.shape == (1, 1, 1)
        assert out.data.item() == expected

    def test_gray_to_rgb_and_bgr(self, device):
        # Gray→RGB/BGR replication
        gray = torch.tensor([[[42, 84]]], dtype=torch.uint8, device=device)  # 1*1*2
        img = self.make_image(gray, ColorSpace.GRAY, ChannelsOrder.CHANNELS_FIRST)
        rgb = img.to_color_space(ColorSpace.RGB)
        assert rgb.data.shape == (3, 1, 2)
        for c in range(3):
            assert torch.equal(rgb.data[c], gray[0])
        bgr = img.to_color_space(ColorSpace.BGR)
        assert torch.equal(bgr.data, rgb.data)
