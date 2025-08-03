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

    @pytest.mark.parametrize("src_cs,dst_cs", [
        (ColorSpace.RGB, ColorSpace.BGR),
        (ColorSpace.RGB, ColorSpace.GRAY),
        (ColorSpace.BGR, ColorSpace.RGB),
        (ColorSpace.BGR, ColorSpace.GRAY),
        (ColorSpace.GRAY, ColorSpace.RGB),
        (ColorSpace.GRAY, ColorSpace.BGR),
    ])
        
    @pytest.mark.parametrize("channels_order", [ChannelsOrder.CHANNELS_FIRST, ChannelsOrder.CHANNELS_LAST])
    def test_to_color_space_conversion(self, src_cs, dst_cs, channels_order):
        torch.manual_seed(42)
        height, width = 4, 5

        if src_cs in [ColorSpace.RGB, ColorSpace.BGR]:
            data = torch.randint(0, 255, (3, height, width), dtype=torch.uint8)
            if channels_order == ChannelsOrder.CHANNELS_LAST:
                data = data.permute(1, 2, 0)
            channels = 3
        else:  # GRAY
            data = torch.randint(0, 255, (1, height, width), dtype=torch.uint8)
            if channels_order == ChannelsOrder.CHANNELS_LAST:
                data = data.permute(1, 2, 0)
            channels = 1

        pixel_format = PixelFormat(color_space=src_cs, bit_depth=8)
        layout = ImageLayout(ImageSize(height, width), channels=channels, channels_order=channels_order)
        img = Image(data, pixel_format, layout)

        out = img.to_color_space(dst_cs)
        assert isinstance(out, Image)
        assert out.pixel_format.color_space == dst_cs

        if dst_cs in [ColorSpace.RGB, ColorSpace.BGR]:
            assert out.channels == 3
        elif dst_cs == ColorSpace.GRAY:
            assert out.channels == 1

        # Check that layout is preserved
        assert out.layout.channels_order == channels_order
        assert out.height == height
        assert out.width == width

    def test_to_color_space_noop(self):
        data = torch.randint(0, 255, (3, 4, 5), dtype=torch.uint8)
        pixel_format = PixelFormat(color_space=ColorSpace.RGB, bit_depth=8)
        layout = ImageLayout(ImageSize(4, 5), channels=3, channels_order=ChannelsOrder.CHANNELS_FIRST)
        img = Image(data, pixel_format, layout)
        out = img.to_color_space(ColorSpace.RGB)
        assert out is img  # should be same instance

    def test_to_color_space_invalid_conversion(self):
        data = torch.randint(0, 255, (3, 4, 5), dtype=torch.uint8)
        pixel_format = PixelFormat(color_space=ColorSpace.RGB, bit_depth=8)
        layout = ImageLayout(ImageSize(4, 5), channels=3, channels_order=ChannelsOrder.CHANNELS_FIRST)
        img = Image(data, pixel_format, layout)

        with pytest.raises(ValueError, match="Can't convert RGB →"):
            img.to_color_space(ColorSpace.UNKNOWN)

    def test_to_color_space_unsupported_source(self):
        # Simulate an invalid color space by creating a dummy PixelFormat
        class DummyColorSpace:
            pass

        dummy_cs = DummyColorSpace()
        data = torch.randint(0, 255, (3, 4, 5), dtype=torch.uint8)
        pixel_format = PixelFormat(color_space=dummy_cs, bit_depth=8)
        layout = ImageLayout(ImageSize(4, 5), channels=3, channels_order=ChannelsOrder.CHANNELS_FIRST)
        img = Image(data, pixel_format, layout)

        with pytest.raises(ValueError, match="Unsupported source color space"):
            img.to_color_space(ColorSpace.RGB)

