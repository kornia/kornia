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

import io
import re
from pathlib import Path

import kornia_rs
import numpy as np
import pytest
import torch

from kornia.io import ImageLoadType, load_image, write_image


def create_random_img8(height: int, width: int, channels: int) -> np.ndarray:
    return (np.random.rand(height, width, channels) * 255).astype(np.uint8)  # noqa: NPY002


def create_random_img8_torch(height: int, width: int, channels: int, device=None) -> torch.Tensor:
    return (torch.rand(channels, height, width, device=device) * 255).to(torch.uint8)


def write_random_png_u16(path: Path, mode: str) -> None:
    """Write a random 16-bit PNG, ``"mono"`` (grayscale) or ``"rgba"``; ``write_image`` cannot write RGBA."""
    channels = {"mono": 1, "rgba": 4}[mode]
    img_np = np.random.randint(0, 65535, (4, 5, channels)).astype(np.uint16)  # noqa: NPY002
    kornia_rs.io.write_image_png_u16(str(path), img_np, mode=mode)


@pytest.fixture(scope="session")
def png_image(tmp_path_factory):
    """RGB PNG written locally so load tests do not depend on the network."""
    filename = tmp_path_factory.mktemp("data") / "image.png"
    img_rgb = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)  # noqa: NPY002
    kornia_rs.io.write_image_png_u8(str(filename), img_rgb, mode="rgb")
    return filename


@pytest.fixture(scope="session")
def rgba_png_image(tmp_path_factory):
    """Create an RGBA PNG image for testing."""
    filename = tmp_path_factory.mktemp("data") / "rgba_image.png"
    img_rgba = np.random.randint(0, 255, (32, 32, 4), dtype=np.uint8)  # noqa: NPY002
    kornia_rs.io.write_image_png_u8(str(filename), img_rgba, mode="rgba")
    return filename


@pytest.fixture(scope="session")
def jpg_image(tmp_path_factory):
    """RGB JPEG written locally so load tests do not depend on the network."""
    filename = tmp_path_factory.mktemp("data") / "image.jpg"
    write_image(str(filename), create_random_img8_torch(32, 32, 3))
    return filename


@pytest.fixture(scope="session")
def images_fn(png_image, jpg_image):
    return {"png": png_image, "jpg": jpg_image}


class TestIoImage:
    @pytest.mark.parametrize(
        ("suffix", "channels", "dtype"),
        [
            (".png", 1, torch.uint8),  # read_image_png_u8
            (".png", 1, torch.uint16),  # read_image_png_u16
            (".png", 3, torch.uint8),  # read_image (RGB PNG)
            (".jpg", 3, torch.uint8),  # read_image_jpegturbo
            (".tiff", 3, torch.uint8),  # read_image (any other extension)
        ],
        ids=["png-gray8", "png-gray16", "png-rgb8", "jpg-rgb8", "tiff-rgb8"],
    )
    def test_truncated_image_decode_error_includes_path(self, suffix, channels, dtype, tmp_path: Path) -> None:
        pixels = (torch.arange(channels * 64 * 64, dtype=torch.int32) * 37 % 251).reshape(channels, 64, 64).to(dtype)
        path = tmp_path / f"truncated_{channels}c_{str(dtype).removeprefix('torch.')}{suffix}"
        write_image(path, pixels)
        data = path.read_bytes()
        path.write_bytes(data[: len(data) // 2])

        with pytest.raises(ValueError, match=re.escape(path.name)) as exc_info:
            load_image(path, ImageLoadType.UNCHANGED)

        assert exc_info.value.__cause__ is not None
        assert str(exc_info.value.__cause__) in str(exc_info.value)

    @pytest.mark.parametrize(("suffix", "error"), [(".png", FileNotFoundError), (".jpg", OSError), (".tiff", OSError)])
    def test_missing_file_keeps_its_os_error(self, suffix, error, tmp_path: Path) -> None:
        # a missing file is not a decode failure: callers catching FileNotFoundError or OSError keep working
        with pytest.raises(error):
            load_image(tmp_path / f"missing{suffix}")

    def test_smoke(self, tmp_path: Path) -> None:
        height, width = 4, 5
        img_th: torch.Tensor = create_random_img8_torch(height, width, 3)

        file_path = tmp_path / "image.jpg"
        write_image(str(file_path), img_th)

        assert file_path.is_file()

        img_load: torch.Tensor = load_image(str(file_path), ImageLoadType.UNCHANGED)

        assert img_th.shape == img_load.shape
        assert img_th.shape[1:] == (height, width)
        assert str(img_th.device) == "cpu"

    def test_device(self, device, png_image: Path) -> None:
        file_path = Path(png_image)

        assert file_path.is_file()

        img_th: torch.Tensor = load_image(file_path, ImageLoadType.UNCHANGED, str(device))
        assert str(img_th.device) == str(device)

    @pytest.mark.parametrize("ext", ["png", "jpg"])
    @pytest.mark.parametrize(
        "channels,load_type,expected_type,expected_channels",
        [
            # NOTE: these tests which should write and load images with channel size != 3, didn't do it
            # (1, ImageLoadType.GRAY8, torch.uint8, 1),
            (3, ImageLoadType.GRAY8, torch.uint8, 1),
            # (4, ImageLoadType.GRAY8, torch.uint8, 1),
            # (1, ImageLoadType.GRAY32, torch.float32, 1),
            (3, ImageLoadType.GRAY32, torch.float32, 1),
            # (4, ImageLoadType.GRAY32, torch.float32, 1),
            (3, ImageLoadType.RGB8, torch.uint8, 3),
            # (1, ImageLoadType.RGB8, torch.uint8, 3),
            (3, ImageLoadType.RGBA8, torch.uint8, 4),
            # (1, ImageLoadType.RGB32, torch.float32, 3),
            (3, ImageLoadType.RGB32, torch.float32, 3),
        ],
    )
    def test_load_image(self, images_fn, ext, channels, load_type, expected_type, expected_channels):
        file_path = images_fn[ext]

        assert file_path.is_file()

        img = load_image(file_path, load_type)
        assert img.shape[0] == expected_channels
        assert img.dtype == expected_type

    @pytest.mark.parametrize(
        "load_type,expected_type,expected_channels",
        [
            (ImageLoadType.UNCHANGED, torch.uint8, 4),
            (ImageLoadType.GRAY8, torch.uint8, 1),
            (ImageLoadType.GRAY32, torch.float32, 1),
            (ImageLoadType.RGB8, torch.uint8, 3),
            (ImageLoadType.RGBA8, torch.uint8, 4),
            (ImageLoadType.RGB32, torch.float32, 3),
        ],
    )
    def test_load_rgba_png(self, rgba_png_image, load_type, expected_type, expected_channels):
        img = load_image(rgba_png_image, load_type)
        assert img.shape[0] == expected_channels
        assert img.dtype == expected_type

    @pytest.mark.parametrize("channels,mode", [(1, "mono"), (3, "rgb")])
    def test_load_opaque_image_as_rgba(self, tmp_path: Path, channels: int, mode: str) -> None:
        pixels = np.full((2, 2, channels), 64, dtype=np.uint8)
        path = tmp_path / "image.png"
        kornia_rs.io.write_image_png_u8(str(path), pixels, mode=mode)

        rgba = load_image(path, ImageLoadType.RGBA8)

        assert rgba.shape == (4, 2, 2)
        assert torch.equal(rgba[:3], torch.full((3, 2, 2), 64, dtype=torch.uint8))
        assert torch.equal(rgba[3], torch.full((2, 2), 255, dtype=torch.uint8))

    def test_unchanged_loads_an_8_bit_grayscale_png_as_one_uint8_channel(self, tmp_path: Path) -> None:
        img = create_random_img8_torch(4, 5, 1)
        path = tmp_path / "image.png"
        write_image(path, img)
        loaded = load_image(path, ImageLoadType.UNCHANGED)
        assert loaded.dtype == torch.uint8
        assert loaded.shape == (1, 4, 5)
        assert torch.equal(loaded, img)

    @pytest.mark.parametrize("ext", ["jpg"])
    @pytest.mark.parametrize("channels", [3])
    def test_write_image(self, device, tmp_path, ext, channels):
        height, width = 4, 5
        img_th: torch.Tensor = create_random_img8_torch(height, width, channels, device)

        file_path = tmp_path / f"image.{ext}"
        write_image(file_path, img_th)

        assert file_path.is_file()


class TestDownloadImage:
    """Offline pins for ``kornia.io.sample.download_image``; ``urlopen`` is mocked, no network is touched."""

    URL = "https://raw.githubusercontent.com/kornia/data/main/panda.jpg"

    def test_saves_the_fetched_bytes(self, tmp_path):
        from unittest import mock

        from PIL import Image as PILImage

        from kornia.io.sample import download_image

        payload = io.BytesIO()
        PILImage.new("RGB", (4, 3), (10, 20, 30)).save(payload, format="PNG")
        with mock.patch("kornia.io.sample.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value.read.return_value = payload.getvalue()
            dst = tmp_path / "panda.png"
            download_image(self.URL, str(dst))

        mock_urlopen.assert_called_once_with(self.URL, timeout=30)
        with PILImage.open(dst) as saved:
            assert saved.size == (4, 3)
            assert saved.getpixel((0, 0)) == (10, 20, 30)

    def test_http_error_propagates(self, tmp_path):
        import urllib.error
        from unittest import mock

        from kornia.io.sample import download_image

        dst = tmp_path / "panda.png"
        with (
            mock.patch(
                "kornia.io.sample.urlopen",
                side_effect=urllib.error.HTTPError(self.URL, 404, "Not Found", {}, None),
            ),
            pytest.raises(urllib.error.HTTPError, match="404"),
        ):
            download_image(self.URL, str(dst))
        assert not dst.exists()


class TestKorniaRsImageIo:
    """``kornia.io`` calls ``kornia_rs.io``, where kornia_rs keeps its image readers and writers since 0.1.11.

    kornia_rs 0.1.11 moved every ``read_image_*``/``write_image_*`` out of the package root; kornia kept
    calling the root, so ``pip install kornia`` (which resolves the newest kornia_rs) lost JPEG loading and
    most writes until the floor moved to 0.1.14 and the calls to ``kornia_rs.io`` (kornia#4325).
    """

    @staticmethod
    def _used_function_names() -> set[str]:
        """Every ``_rs_io.<name>`` attribute in ``kornia/io/io.py``."""
        import ast
        import inspect

        import kornia.io.io as io_module

        return {
            node.attr
            for node in ast.walk(ast.parse(inspect.getsource(io_module)))
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "_rs_io"
        }

    def test_used_function_names_are_extracted_from_the_source(self):
        assert {"read_image_jpegturbo", "read_image", "write_image_tiff_f32"} <= self._used_function_names()

    def test_installed_kornia_rs_provides_every_used_function(self):
        missing = sorted(n for n in self._used_function_names() if not callable(getattr(kornia_rs.io, n, None)))
        assert missing == [], f"kornia_rs {kornia_rs.__version__}: {missing}"

    @pytest.mark.parametrize("ext", ["jpg", "png", "tiff"])
    def test_uint8_round_trip_every_extension(self, tmp_path, ext):
        img = create_random_img8_torch(5, 6, 3)
        path = tmp_path / f"image.{ext}"
        write_image(path, img)
        loaded = load_image(path, ImageLoadType.UNCHANGED)
        assert loaded.shape == img.shape
        if ext != "jpg":  # lossless containers come back bit-exact
            assert torch.equal(loaded, img)

    def test_lookup_happens_at_call_time(self, tmp_path, monkeypatch):
        """Patching the kornia_rs function is seen by the next call."""
        calls = []
        real = kornia_rs.io.write_image_png_u8

        def spy(*args, **kwargs):
            calls.append(args[0])
            return real(*args, **kwargs)

        monkeypatch.setattr(kornia_rs.io, "write_image_png_u8", spy)
        write_image(tmp_path / "image.png", create_random_img8_torch(2, 3, 3))
        assert calls == [str(tmp_path / "image.png")]


class TestWiderThanUint8Decodes:
    """kornia_rs decodes 16-bit PNG/TIFF and float TIFF; the 8-bit load types are not defined for them."""

    @pytest.mark.parametrize(
        "ext,dtype",
        [("png", torch.uint16), ("tiff", torch.uint16), ("tiff", torch.float32)],
    )
    def test_unchanged_returns_the_decoded_dtype(self, tmp_path, ext, dtype):
        if dtype == torch.uint16:
            img = torch.randint(0, 65535, (3, 4, 5), dtype=torch.int32).to(torch.uint16)
        else:
            img = torch.rand(3, 4, 5)
        path = tmp_path / f"image.{ext}"
        write_image(path, img)
        loaded = load_image(path, ImageLoadType.UNCHANGED)
        assert loaded.dtype == dtype
        assert torch.equal(loaded, img)

    def test_unchanged_loads_a_16_bit_grayscale_png(self, tmp_path):
        img = torch.randint(0, 65535, (1, 4, 5), dtype=torch.int32).to(torch.uint16)
        path = tmp_path / "image.png"
        write_image(path, img)
        loaded = load_image(path, ImageLoadType.UNCHANGED)
        assert loaded.dtype == torch.uint16
        assert torch.equal(loaded, img)

    def test_unchanged_loads_a_16_bit_rgba_png(self, tmp_path):
        img_np = np.random.randint(0, 65535, (4, 5, 4)).astype(np.uint16)  # noqa: NPY002
        path = tmp_path / "image.png"
        kornia_rs.io.write_image_png_u16(str(path), img_np, mode="rgba")
        loaded = load_image(path, ImageLoadType.UNCHANGED)
        assert loaded.dtype == torch.uint16
        assert torch.equal(loaded, torch.from_numpy(img_np).permute(2, 0, 1))

    @pytest.mark.parametrize("mode", ["mono", "rgba"])
    def test_16_bit_png_goes_to_the_16_bit_reader(self, tmp_path, monkeypatch, mode):
        """kornia_rs's ``read_image`` also decodes these files, so the dtype alone cannot show which reader ran."""
        path = tmp_path / "image.png"
        write_random_png_u16(path, mode)
        calls = []
        real = kornia_rs.io.read_image_png_u16

        def spy(*args, **kwargs):
            calls.append(args[1])
            return real(*args, **kwargs)

        monkeypatch.setattr(kornia_rs.io, "read_image_png_u16", spy)
        assert load_image(path, ImageLoadType.UNCHANGED).dtype == torch.uint16
        assert calls == [mode]

    @pytest.mark.parametrize("mode", ["mono", "rgba"])
    @pytest.mark.parametrize("load_type", [t for t in ImageLoadType if t != ImageLoadType.UNCHANGED])
    def test_eight_bit_load_types_reject_a_16_bit_png(self, tmp_path, mode, load_type):
        path = tmp_path / "image.png"
        write_random_png_u16(path, mode)
        expected = rf"decoded to torch\.uint16, and ImageLoadType\.{load_type.name} .*ImageLoadType\.UNCHANGED"
        with pytest.raises(NotImplementedError, match=expected):
            load_image(path, load_type)

    @pytest.mark.parametrize("mode", ["mono", "rgba"])
    def test_default_load_type_rejects_a_16_bit_png(self, tmp_path, mode):
        path = tmp_path / "image.png"
        write_random_png_u16(path, mode)
        with pytest.raises(NotImplementedError, match=r"decoded to torch\.uint16, and ImageLoadType\.RGB32"):
            load_image(path)

    @pytest.mark.parametrize("load_type", [ImageLoadType.RGB8, ImageLoadType.GRAY8, ImageLoadType.RGB32])
    def test_eight_bit_load_types_reject_a_float_decode(self, tmp_path, load_type):
        path = tmp_path / "image.tiff"
        write_image(path, torch.rand(3, 4, 5))
        expected = rf"decoded to torch\.float32, and ImageLoadType\.{load_type.name}"
        with pytest.raises(NotImplementedError, match=expected):
            load_image(path, load_type)
