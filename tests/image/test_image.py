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

from kornia.core.exceptions import ShapeError, ValueCheckError
from kornia.image import ColorSpace
from kornia.image.base import KORNIA_CHECK_IMAGE_LAYOUT, ChannelsOrder, ImageLayout, ImageSize, PixelFormat
from kornia.image.image import Image
from kornia.image.image_print import image_to_string
from kornia.io import ImageLoadType, load_image

from testing.base import BaseTester


class TestImage(BaseTester):
    @pytest.mark.parametrize("channels,color_space", [(1, ColorSpace.GRAY), (3, ColorSpace.RGB)])
    @pytest.mark.parametrize("order", [ChannelsOrder.CHANNELS_FIRST, ChannelsOrder.CHANNELS_LAST])
    def test_to_numpy_preserves_layout_and_values(self, device, dtype, channels, color_space, order):
        """Preserve layout and metadata while bfloat16 exports use independent float32 storage."""
        data = torch.arange(channels * 6, device=device, dtype=dtype).reshape(channels, 2, 3)
        if order == ChannelsOrder.CHANNELS_LAST:
            data = data.permute(1, 2, 0)
        data.requires_grad_()
        image = make_image(data, color_space, order)
        stored_data = image.data

        actual = image.to_numpy()

        expected = data.detach().cpu()
        if dtype == torch.bfloat16:
            expected = expected.float()
        self.assert_close(torch.from_numpy(actual), expected)
        assert image.data is stored_data
        assert image.dtype == dtype
        assert image.channels_order == order
        assert image.pixel_format.color_space == color_space
        if dtype == torch.bfloat16:
            actual.fill(-1)
            assert (image.data >= 0).all()

    @pytest.mark.parametrize("array_dtype", [np.uint8, np.float32, np.float64])
    @pytest.mark.parametrize("order", [ChannelsOrder.CHANNELS_FIRST, ChannelsOrder.CHANNELS_LAST])
    def test_to_numpy_keeps_supported_cpu_dtypes_shared(self, array_dtype, order):
        """Keep supported CPU dtypes and their shared-memory NumPy view."""
        data = np.arange(18, dtype=array_dtype).reshape(3, 2, 3)
        if order == ChannelsOrder.CHANNELS_LAST:
            data = data.transpose(1, 2, 0)
        image = Image.from_numpy(data, channels_order=order)

        actual = image.to_numpy()

        assert actual.dtype == data.dtype
        assert np.shares_memory(actual, data)
        actual[0, 0, 0] = 42
        assert data[0, 0, 0] == 42

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

    @staticmethod
    def _pixel(values, order):
        """Build a one-pixel image tensor in the given channels order."""
        data = torch.tensor(values, dtype=torch.float32)
        return data.view(3, 1, 1) if order == ChannelsOrder.CHANNELS_FIRST else data.view(1, 1, 3)

    # BT.601 luma of RGB (0.5, 0.2, 0.1): 0.299 * 0.5 + 0.587 * 0.2 + 0.114 * 0.1. Reading the BGR pixel
    # (0.1, 0.2, 0.5) as RGB would give 0.2043 instead.
    _GRAY = 0.2783

    @pytest.mark.parametrize("order", [ChannelsOrder.CHANNELS_FIRST, ChannelsOrder.CHANNELS_LAST])
    def test_rgb_gray_rgb(self, order):
        img_rgb = make_image(self._pixel([0.5, 0.2, 0.1], order), ColorSpace.RGB, order)

        gray = img_rgb.to_gray()
        rgb_back = gray.to_rgb()

        self.assert_close(gray.data.squeeze(), torch.tensor(self._GRAY))
        # RGB reconstructed from gray repeats the luminance across channels
        self.assert_close(rgb_back.data.squeeze(), torch.full((3,), self._GRAY))

    @pytest.mark.parametrize("order", [ChannelsOrder.CHANNELS_FIRST, ChannelsOrder.CHANNELS_LAST])
    def test_bgr_gray_bgr(self, order):
        img_bgr = make_image(self._pixel([0.1, 0.2, 0.5], order), ColorSpace.BGR, order)

        gray = img_bgr.to_gray()
        bgr_back = gray.to_bgr()

        self.assert_close(gray.data.squeeze(), torch.tensor(self._GRAY))
        self.assert_close(bgr_back.data.squeeze(), torch.full((3,), self._GRAY))

    @pytest.mark.parametrize("order", [ChannelsOrder.CHANNELS_FIRST, ChannelsOrder.CHANNELS_LAST])
    def test_rgb_bgr_swap(self, order):
        rgb_val = torch.tensor([0.5, 0.2, 0.1], dtype=torch.float32)
        bgr_val = rgb_val.flip(0)

        img_rgb = make_image(self._pixel(rgb_val.tolist(), order), ColorSpace.RGB, order)
        img_bgr = make_image(self._pixel(bgr_val.tolist(), order), ColorSpace.BGR, order)

        self.assert_close(img_rgb.to_bgr().data.squeeze(), bgr_val)
        self.assert_close(img_bgr.to_rgb().data.squeeze(), rgb_val)

    def test_load_write(self, tmp_path: Path) -> None:
        # PNG is lossless, so the channels-first image comes back bit for bit.
        data = torch.randint(0, 255, (3, 4, 5), dtype=torch.uint8)
        img = Image.from_numpy(data.numpy(), channels_order=ChannelsOrder.CHANNELS_FIRST)

        file_name = tmp_path / "image.png"

        img.write(file_name)
        img2 = Image.from_file(file_name)

        assert img2.channels_order == ChannelsOrder.CHANNELS_FIRST
        assert torch.equal(img2.data, data)

    def test_to_returns_a_new_image_and_leaves_the_original(self, device):
        img = Image.from_numpy(np.zeros((4, 5, 3), dtype=np.uint8))
        out = img.to(device, torch.float32)
        assert out is not img
        assert img.dtype == torch.uint8
        assert img.pixel_format.bit_depth == 8
        assert out.dtype == torch.float32
        assert out.device == device
        assert out.pixel_format.bit_depth == 32
        assert out.clone().pixel_format.bit_depth == 32  # bit_depth follows the dtype, so clone() passes the check

    def test_float_leaves_the_original(self):
        img = Image.from_numpy(np.zeros((4, 5, 3), dtype=np.uint8))
        out = img.float()
        assert img.dtype == torch.uint8
        assert out.dtype == torch.float32
        assert out.pixel_format.bit_depth == 32

    @pytest.mark.parametrize("order", [ChannelsOrder.CHANNELS_LAST, ChannelsOrder.CHANNELS_FIRST])
    def test_from_numpy_accepts_a_2d_gray_array(self, order):
        data = np.arange(20, dtype=np.uint8).reshape(4, 5)  # H = 4, W = 5
        img = Image.from_numpy(data, color_space=ColorSpace.GRAY, channels_order=order)
        assert (img.channels, img.height, img.width) == (1, 4, 5)
        data[0, 1] = 99
        assert int(img.data.flatten()[1]) == 99  # shares memory, like a 3-D array

    def test_write_stores_true_colours_for_a_bgr_image(self, tmp_path):
        bgr = np.zeros((4, 5, 3), dtype=np.uint8)
        bgr[..., 0], bgr[..., 1], bgr[..., 2] = 2, 1, 0  # B, G, R -> true RGB (0, 1, 2)
        Image.from_numpy(bgr, color_space=ColorSpace.BGR).write(str(tmp_path / "bgr.png"))
        assert load_image(tmp_path / "bgr.png", ImageLoadType.RGB8)[:, 0, 0].tolist() == [0, 1, 2]

    def test_print_shows_true_colours_for_a_bgr_image(self, capsys):
        rgb = torch.zeros(3, 2, 3, dtype=torch.uint8)
        rgb[0] = 255  # red
        make_image(rgb.flip(0).permute(1, 2, 0).contiguous(), ColorSpace.BGR, ChannelsOrder.CHANNELS_LAST).print()
        assert capsys.readouterr().out == image_to_string(rgb) + "\n"

    @pytest.mark.parametrize(
        ("cs", "channels"), [(ColorSpace.GRAY, 3), (ColorSpace.RGB, 1), (ColorSpace.BGR, 4), (ColorSpace.RGB, 4)]
    )
    def test_color_space_must_match_the_channel_count(self, cs, channels):
        with pytest.raises(ValueCheckError, match="color space"):
            make_image(torch.zeros(4, 5, channels, dtype=torch.uint8), cs, ChannelsOrder.CHANNELS_LAST)

    def test_unknown_color_space_accepts_any_channel_count(self):
        img = make_image(torch.zeros(4, 5, 4, dtype=torch.uint8), ColorSpace.UNKNOWN, ChannelsOrder.CHANNELS_LAST)
        assert img.channels == 4

    def test_from_numpy_infers_the_color_space(self):
        assert Image.from_numpy(np.zeros((4, 5, 1), dtype=np.uint8)).pixel_format.color_space == ColorSpace.GRAY
        assert Image.from_numpy(np.zeros((4, 5), dtype=np.uint8)).pixel_format.color_space == ColorSpace.GRAY
        assert Image.from_numpy(np.zeros((4, 5, 3), dtype=np.uint8)).pixel_format.color_space == ColorSpace.RGB
        assert Image.from_numpy(np.zeros((4, 5, 4), dtype=np.uint8)).pixel_format.color_space == ColorSpace.UNKNOWN

    def test_from_numpy_rejects_a_mismatching_color_space(self):
        with pytest.raises(ValueCheckError, match="color space"):
            Image.from_numpy(np.zeros((4, 5, 1), dtype=np.uint8), color_space=ColorSpace.RGB)

    def test_from_dlpack_infers_the_color_space(self):
        gray = Image.from_dlpack(np.zeros((1, 4, 5), dtype=np.uint8).__dlpack__())
        assert gray.pixel_format.color_space == ColorSpace.GRAY
        multi = Image.from_dlpack(np.zeros((4, 4, 5), dtype=np.uint8).__dlpack__())
        assert multi.pixel_format.color_space == ColorSpace.UNKNOWN

    @pytest.mark.parametrize(
        ("channels", "color_space"),
        [
            (1, ColorSpace.GRAY),
            (3, ColorSpace.RGB),
            (3, ColorSpace.BGR),
            (3, ColorSpace.UNKNOWN),
            (4, ColorSpace.UNKNOWN),
        ],
    )
    @pytest.mark.parametrize("channels_order", [ChannelsOrder.CHANNELS_LAST, ChannelsOrder.CHANNELS_FIRST])
    def test_dlpack_round_trip_keeps_the_layout_and_color_space_6039(
        self, channels, color_space, channels_order, device, dtype
    ):
        # to_dlpack keeps the image's own layout and carries no color space, so a channels-last image used to come
        # back from from_dlpack with its height, width and channels swapped, and a BGR image came back as RGB.
        chw = torch.arange(channels * 6 * 8, device=device).reshape(channels, 6, 8).to(dtype)
        data = chw.permute(1, 2, 0).contiguous() if channels_order == ChannelsOrder.CHANNELS_LAST else chw
        img = make_image(data, color_space, channels_order)

        back = Image.from_dlpack(img.to_dlpack(), img.pixel_format.color_space, img.channels_order)

        assert (back.height, back.width, back.channels) == (6, 8, channels)
        assert back.layout == img.layout
        assert back.pixel_format == img.pixel_format
        assert back.data.data_ptr() == img.data.data_ptr()
        self.assert_close(back.data, data, rtol=0, atol=0)

    def test_from_dlpack_reads_a_channels_last_capsule_6039(self):
        hwc = np.zeros((6, 8, 3), dtype=np.uint8)
        img = Image.from_dlpack(hwc.__dlpack__(), channels_order=ChannelsOrder.CHANNELS_LAST)
        assert (img.height, img.width, img.channels) == (6, 8, 3)
        assert img.pixel_format.color_space == ColorSpace.RGB
        # The default is unchanged: a (C, H, W) capsule.
        chw = Image.from_dlpack(np.zeros((3, 6, 8), dtype=np.uint8).__dlpack__())
        assert (chw.channels_order, chw.height, chw.width, chw.channels) == (ChannelsOrder.CHANNELS_FIRST, 6, 8, 3)

    def test_from_dlpack_takes_an_explicit_color_space_6039(self):
        bgr = Image.from_dlpack(np.zeros((3, 6, 8), dtype=np.uint8).__dlpack__(), ColorSpace.BGR)
        assert bgr.pixel_format.color_space == ColorSpace.BGR
        with pytest.raises(ValueCheckError, match="color space"):
            Image.from_dlpack(
                np.zeros((6, 8, 4), dtype=np.uint8).__dlpack__(), ColorSpace.RGB, ChannelsOrder.CHANNELS_LAST
            )

    def test_from_dlpack_rejects_an_unknown_channels_order_6039(self):
        capsule = np.zeros((3, 6, 8), dtype=np.uint8).__dlpack__()
        with pytest.raises(ValueError, match="channels_order must be either"):
            Image.from_dlpack(capsule, channels_order="HWC")
        # the order is checked before the capsule, which can be read only once, is consumed
        assert Image.from_dlpack(capsule).channels == 3

    @pytest.mark.parametrize(
        ("dtype", "bits", "ok"),
        [(torch.uint8, 8, True), (torch.uint8, 12, False), (torch.uint16, 16, True), (torch.uint16, 12, False)],
    )
    def test_bit_depth_is_the_storage_bits(self, dtype, bits, ok):
        data = torch.zeros(3, 4, 5, dtype=dtype)
        pf = PixelFormat(color_space=ColorSpace.RGB, bit_depth=bits)
        layout = ImageLayout(ImageSize(4, 5), 3, ChannelsOrder.CHANNELS_FIRST)
        if ok:
            assert Image(data, pf, layout).pixel_format.bit_depth == bits
        else:
            with pytest.raises(ValueCheckError, match="bit depth"):
                Image(data, pf, layout)

    def test_pixel_format_checks_follow_disable_checks(self):
        from kornia.core.check import disable_checks, enable_checks

        layout = ImageLayout(ImageSize(4, 5), 1, ChannelsOrder.CHANNELS_FIRST)
        disable_checks()
        try:
            img = Image(torch.zeros(1, 4, 5, dtype=torch.uint8), PixelFormat(ColorSpace.RGB, 12), layout)
        finally:
            enable_checks()
        assert img.pixel_format.bit_depth == 12

    def test_color_space_is_exported(self):
        import kornia.image

        assert kornia.image.ColorSpace is ColorSpace
        assert "ColorSpace" in kornia.image.__all__

    @pytest.mark.parametrize("order", [ChannelsOrder.CHANNELS_LAST, ChannelsOrder.CHANNELS_FIRST])
    @pytest.mark.parametrize("height", [4, 3])
    def test_print_renders_the_channels_first_image(self, device, order, height, capsys) -> None:
        chw = (torch.arange(3 * height * 5, device=device).reshape(3, height, 5) * 4).to(torch.uint8)
        data = chw.permute(1, 2, 0).contiguous() if order == ChannelsOrder.CHANNELS_LAST else chw
        make_image(data, ColorSpace.RGB, order).print()
        assert capsys.readouterr().out == image_to_string(chw.cpu()) + "\n"  # the CPU result


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
    def test_channels_first_valid(self, device):
        data = torch.rand(3, 4, 5, device=device)
        layout = ImageLayout(ImageSize(4, 5), 3, ChannelsOrder.CHANNELS_FIRST)
        assert KORNIA_CHECK_IMAGE_LAYOUT(data, layout)

    def test_channels_last_valid(self, device):
        data = torch.rand(4, 5, 3, device=device)
        layout = ImageLayout(ImageSize(4, 5), 3, ChannelsOrder.CHANNELS_LAST)
        assert KORNIA_CHECK_IMAGE_LAYOUT(data, layout)

    def test_invalid_shape_raises(self, device):
        data = torch.rand(3, 4, 5, device=device)
        layout = ImageLayout(ImageSize(10, 10), 3, ChannelsOrder.CHANNELS_FIRST)
        with pytest.raises(ShapeError):
            KORNIA_CHECK_IMAGE_LAYOUT(data, layout)

    def test_invalid_shape_no_raise(self, device):
        data = torch.rand(3, 4, 5, device=device)
        layout = ImageLayout(ImageSize(10, 10), 3, ChannelsOrder.CHANNELS_FIRST)
        assert not KORNIA_CHECK_IMAGE_LAYOUT(data, layout, raises=False)

    # The sizes are compared as numbers. A tensor or float size whose str() is not all digits used to be skipped
    # like a dimension name, so a 4 x 5 image was accepted against any such height and width (#6244).
    @pytest.mark.parametrize(
        "size",
        [ImageSize(torch.tensor(9), torch.tensor(9)), ImageSize(9.0, 9.0), ImageSize(4.5, 5.0)],
        ids=["tensor", "float", "fractional"],
    )
    @pytest.mark.parametrize("order", [ChannelsOrder.CHANNELS_FIRST, ChannelsOrder.CHANNELS_LAST])
    def test_mismatched_tensor_or_float_size_raises_6244(self, size, order, device):
        data = torch.rand((3, 4, 5) if order == ChannelsOrder.CHANNELS_FIRST else (4, 5, 3), device=device)
        layout = ImageLayout(size, 3, order)
        with pytest.raises(ShapeError):
            Image(data, PixelFormat(ColorSpace.RGB, 32), layout)
        assert not KORNIA_CHECK_IMAGE_LAYOUT(data, layout, raises=False)

    @pytest.mark.parametrize(
        "size",
        [
            ImageSize(4, 5),
            ImageSize(4.0, 5.0),
            ImageSize(torch.tensor(4), torch.tensor(5)),
            ImageSize(torch.tensor(4.0), torch.tensor(5.0)),
        ],
        ids=["int", "float", "int_tensor", "float_tensor"],
    )
    def test_matching_size_of_any_type_is_accepted_6244(self, size, device):
        data = torch.rand(3, 4, 5, device=device)
        image = Image(data, PixelFormat(ColorSpace.RGB, 32), ImageLayout(size, 3, ChannelsOrder.CHANNELS_FIRST))
        assert (image.height, image.width) == (4, 5)

    def test_dynamo_layout_check_does_not_stringify_sizes_6244(self, device, torch_optimizer):
        # Turning a symbolic size into a string cannot be traced: under fullgraph=True with dynamic shapes, building
        # an Image raised torch._dynamo.exc.Unsupported (tp_repr_impl) instead of compiling one graph.
        from torch._dynamo.testing import CompileCounter

        def op(x):
            layout = ImageLayout(ImageSize(x.shape[1], x.shape[2]), 3, ChannelsOrder.CHANNELS_FIRST)
            return Image(x, PixelFormat(ColorSpace.RGB, 32), layout).to_gray().data

        counter = CompileCounter()
        compiled = torch_optimizer(op, backend=counter, fullgraph=True, dynamic=True)
        for height, width in ((4, 5), (6, 7)):
            data = torch.rand(3, height, width, device=device)
            self.assert_close(compiled(data), op(data))
        assert counter.frame_count == 1


class TestConventionsImage(BaseTester):
    """Pins for the unbatched layout, same-space conversions and NumPy aliasing of ``Image``."""

    def test_convention_image_is_unbatched(self):
        # Image holds one (H, W, C) or (C, H, W) image; a batched 4-D tensor is rejected by its dimension count.
        # Image.from_numpy of a 4-D array raises ShapeError too.
        batched = torch.zeros(2, 4, 5, 3, dtype=torch.uint8)
        layout = ImageLayout(ImageSize(4, 5), 3, ChannelsOrder.CHANNELS_LAST)
        with pytest.raises(ShapeError, match=r"expected \(4, 5, 3\), got \[2, 4, 5, 3\]"):
            Image(batched, PixelFormat(ColorSpace.RGB, 8), layout)
        with pytest.raises(ShapeError, match=r"got \[2, 4, 5, 3\]"):
            Image.from_numpy(np.zeros((2, 4, 5, 3), np.uint8))

    @pytest.mark.parametrize(
        ("color_space", "channels", "convert"),
        [
            (ColorSpace.GRAY, 1, "to_gray"),
            (ColorSpace.RGB, 3, "to_rgb"),
            (ColorSpace.BGR, 3, "to_bgr"),
        ],
    )
    @pytest.mark.parametrize("array_dtype", [np.uint8, np.uint16])
    def test_convention_color_conversion_to_the_same_space_returns_self(
        self, color_space, channels, convert, array_dtype
    ):
        image = Image.from_numpy(np.zeros((4, 5, channels), dtype=array_dtype), color_space=color_space)
        assert getattr(image, convert)() is image

    def test_convention_uint16_rgb_to_gray_is_unsupported(self):
        image = Image.from_numpy(np.ones((2, 3, 3), dtype=np.uint16))
        with pytest.raises(TypeError, match=r"Unknown data type: torch\.uint16"):
            image.to_gray()

    def test_convention_from_numpy_shares_memory(self):
        array = np.zeros((4, 5, 3), dtype=np.uint8)  # H = 4, W = 5
        image = Image.from_numpy(array)
        assert (image.height, image.width, image.channels) == (4, 5, 3)
        array[3, 4, 2] = 9
        assert image.data[3, 4, 2].item() == 9
        image.data[0, 1, 0] = 5
        assert array[0, 1, 0] == 5
