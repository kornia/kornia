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

from kornia.core._compat import torch_version_le
from kornia.core.exceptions import ShapeError
from kornia.image.base import ChannelsOrder, ColorSpace, ImageLayout, ImageSize, KORNIA_CHECK_IMAGE_LAYOUT, PixelFormat
from kornia.image.image import Image

from testing.base import BaseTester


class TestImage(BaseTester):
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
        np_img = np.asarray(img.to_numpy())
        np.testing.assert_array_equal(data, np_img)

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
        self.assert_close(data, Image.from_dlpack(img.to_dlpack()).data)

    # Channel first
    def test_rgb_gray_rgb_channels_first(self):
        rgb_val = torch.tensor([0.5, 0.2, 0.1], dtype=torch.float32)
        rgb_data = rgb_val.view(3, 1, 1)

        img_rgb = make_image(rgb_data, ColorSpace.RGB, ChannelsOrder.CHANNELS_FIRST)

        gray = img_rgb.to_gray()
        rgb_back = gray.to_rgb()

        expected_gray = img_rgb.to_gray().data.squeeze()
        self.assert_close(gray.data.squeeze(), expected_gray)

        # RGB reconstructed from gray should repeat luminance across channels
        expected_rgb = expected_gray.repeat(3)
        self.assert_close(rgb_back.data.squeeze(), expected_rgb)

    def test_bgr_gray_bgr_channels_first(self):
        rgb_val = torch.tensor([0.5, 0.2, 0.1], dtype=torch.float32)
        bgr_val = rgb_val.flip(0)
        bgr_data = bgr_val.view(3, 1, 1)

        img_bgr = make_image(bgr_data, ColorSpace.BGR, ChannelsOrder.CHANNELS_FIRST)

        gray = img_bgr.to_gray()
        bgr_back = gray.to_bgr()

        expected_gray = img_bgr.to_gray().data.squeeze()
        self.assert_close(gray.data.squeeze(), expected_gray)

        expected_bgr = expected_gray.repeat(3).flip(0)
        self.assert_close(bgr_back.data.squeeze(), expected_bgr)

    def test_rgb_bgr_swap_channels_first(self):
        rgb_val = torch.tensor([0.5, 0.2, 0.1], dtype=torch.float32)
        bgr_val = rgb_val.flip(0)

        rgb_data = rgb_val.view(3, 1, 1)
        bgr_data = bgr_val.view(3, 1, 1)

        img_rgb = make_image(rgb_data, ColorSpace.RGB, ChannelsOrder.CHANNELS_FIRST)
        img_bgr = make_image(bgr_data, ColorSpace.BGR, ChannelsOrder.CHANNELS_FIRST)

        self.assert_close(img_rgb.to_bgr().data.squeeze(), bgr_val)
        self.assert_close(img_bgr.to_rgb().data.squeeze(), rgb_val)

    # Channel last
    def test_rgb_gray_rgb_channels_last(self):
        rgb_val = torch.tensor([0.5, 0.2, 0.1], dtype=torch.float32)
        rgb_data = rgb_val.view(1, 1, 3)

        img_rgb = make_image(rgb_data, ColorSpace.RGB, ChannelsOrder.CHANNELS_LAST)

        gray = img_rgb.to_gray()
        rgb_back = gray.to_rgb()

        expected_gray = img_rgb.to_gray().data.squeeze()
        self.assert_close(gray.data.squeeze(), expected_gray)

        # RGB reconstructed from gray should repeat luminance across channels
        expected_rgb = expected_gray.repeat(3)
        self.assert_close(rgb_back.data.squeeze(), expected_rgb)

    def test_bgr_gray_bgr_channels_last(self):
        rgb_val = torch.tensor([0.5, 0.2, 0.1], dtype=torch.float32)
        bgr_val = rgb_val.flip(0)
        bgr_data = bgr_val.view(1, 1, 3)

        img_bgr = make_image(bgr_data, ColorSpace.BGR, ChannelsOrder.CHANNELS_LAST)

        gray = img_bgr.to_gray()
        bgr_back = gray.to_bgr()

        expected_gray = img_bgr.to_gray().data.squeeze()
        self.assert_close(gray.data.squeeze(), expected_gray)

        expected_bgr = expected_gray.repeat(3).flip(0)
        self.assert_close(bgr_back.data.squeeze(), expected_bgr)

    def test_rgb_bgr_swap_channels_last(self):
        rgb_val = torch.tensor([0.5, 0.2, 0.1], dtype=torch.float32)
        bgr_val = rgb_val.flip(0)

        rgb_data = rgb_val.view(1, 1, 3)
        bgr_data = bgr_val.view(1, 1, 3)

        img_rgb = make_image(rgb_data, ColorSpace.RGB, ChannelsOrder.CHANNELS_LAST)
        img_bgr = make_image(bgr_data, ColorSpace.BGR, ChannelsOrder.CHANNELS_LAST)

        self.assert_close(img_rgb.to_bgr().data.squeeze(), bgr_val)
        self.assert_close(img_bgr.to_rgb().data.squeeze(), rgb_val)

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


def make_image(data: torch.Tensor, cs: ColorSpace, order: ChannelsOrder) -> Image:
    if order not in [ChannelsOrder.CHANNELS_FIRST, ChannelsOrder.CHANNELS_LAST]:
        pytest.skip(f"Skipping unsupported channels_order: {order}")
    if order == ChannelsOrder.CHANNELS_FIRST:
        C, H, W = data.shape
    else:
        H, W, C = data.shape
    pf = PixelFormat(color_space=cs, bit_depth=data.element_size() * 8)
    layout = ImageLayout(image_size=ImageSize(H, W), channels=C, channels_order=order)
    return Image(data.clone(), pf, layout)


class TestCheckImageLayout(BaseTester):
    def test_invalid_shape_raises(self, device):
        data = torch.rand(3, 4, 5, device=device)
        layout = ImageLayout(ImageSize(10, 10), 3, ChannelsOrder.CHANNELS_FIRST)
        with pytest.raises(ShapeError):
            KORNIA_CHECK_IMAGE_LAYOUT(data, layout)

    def test_invalid_shape_no_raise(self, device):
        data = torch.rand(3, 4, 5, device=device)
        layout = ImageLayout(ImageSize(10, 10), 3, ChannelsOrder.CHANNELS_FIRST)
        assert not KORNIA_CHECK_IMAGE_LAYOUT(data, layout, raises=False)
