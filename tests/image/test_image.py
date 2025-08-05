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


@pytest.mark.parametrize("order", [ChannelsOrder.CHANNELS_FIRST, ChannelsOrder.CHANNELS_LAST])
def test_color_space_conversions(order):
    rgb_val = torch.tensor([0.5, 0.2, 0.1], dtype=torch.float32)
    bgr_val = rgb_val.flip(0)

    if order == ChannelsOrder.CHANNELS_FIRST:
        rgb_data = rgb_val.view(3, 1, 1)
        bgr_data = bgr_val.view(3, 1, 1)
    else:
        rgb_data = rgb_val.view(1, 1, 3)
        bgr_data = bgr_val.view(1, 1, 3)

    # Create images
    img_rgb = make_image(rgb_data, ColorSpace.RGB, order)
    img_bgr = make_image(bgr_data, ColorSpace.BGR, order)

    # 1) RGB -> Gray -> RGB
    gray = img_rgb.to_gray()
    lum = 0.299 * rgb_val[0] + 0.587 * rgb_val[1] + 0.114 * rgb_val[2]
    assert pytest.approx(gray.data.squeeze().item(), rel=1e-3) == lum
    rgb_back = gray.to_rgb().data.squeeze()
    expected = torch.tensor([lum, lum, lum])
    assert torch.allclose(rgb_back, expected)

    # 2) BGR -> Gray -> BGR
    gray2 = img_bgr.to_gray()
    assert pytest.approx(gray2.data.squeeze().item(), rel=1e-3) == lum
    bgr_back = gray2.to_bgr().data.squeeze()
    expected_bgr = expected.flip(0)
    assert torch.allclose(bgr_back, expected_bgr)

    # 3) RGB <-> BGR swap
    assert torch.allclose(img_rgb.to_bgr().data.squeeze(), bgr_val)
    assert torch.allclose(img_bgr.to_rgb().data.squeeze(), rgb_val)
