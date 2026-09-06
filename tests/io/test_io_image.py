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
import sys
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pytest
import torch

from kornia.core._compat import torch_version_ge
from kornia.io import ImageLoadType, load_image, write_image

try:
    import kornia_rs
except ImportError:
    kornia_rs = None


def available_package() -> bool:
    return sys.version_info >= (3, 7, 0) and torch_version_ge(1, 10, 0) and kornia_rs is not None


def create_random_img8(height: int, width: int, channels: int) -> np.ndarray:
    return (np.random.rand(height, width, channels) * 255).astype(np.uint8)  # noqa: NPY002


def create_random_img8_torch(height: int, width: int, channels: int, device=None) -> torch.Tensor:
    return (torch.rand(channels, height, width, device=device) * 255).to(torch.uint8)


def _download_image(url: str, filename: str = "") -> Path:
    # TODO: move this to testing

    filename = url.rsplit("/", maxsplit=1)[-1] if len(filename) == 0 else filename
    # Download
    # url is a fixed https:// literal defined by each fixture above.
    with urlopen(url, timeout=60) as resp:  # noqa: S310
        bytesio = io.BytesIO(resp.read())
    # Save file
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())

    return Path(filename)


@pytest.fixture(scope="session")
def png_image(tmp_path_factory):
    url = "https://github.com/kornia/data/raw/main/simba.png"
    filename = tmp_path_factory.mktemp("data") / "image.png"
    filename = _download_image(url, str(filename))
    return filename


@pytest.fixture(scope="session")
def rgba_png_image(tmp_path_factory):
    """Create an RGBA PNG image for testing."""
    filename = tmp_path_factory.mktemp("data") / "rgba_image.png"
    from kornia.io.io import _rs_io  # the kornia_rs namespace that holds the writers on this kornia_rs version

    img_rgba = np.random.randint(0, 255, (32, 32, 4), dtype=np.uint8)  # noqa: NPY002
    _rs_io.write_image_png_u8(str(filename), img_rgba, mode="rgba")
    return filename


@pytest.fixture(scope="session")
def jpg_image(tmp_path_factory):
    url = "https://github.com/kornia/data/raw/main/crowd.jpg"
    filename = tmp_path_factory.mktemp("data") / "image.jpg"
    filename = _download_image(url, str(filename))
    return filename


@pytest.fixture(scope="session")
def images_fn(png_image, jpg_image):
    return {"png": png_image, "jpg": jpg_image}


@pytest.mark.skipif(not available_package(), reason="kornia_rs only supports python >=3.7 and pt >= 1.10.0")
class TestIoImage:
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


class TestImageIoBackend:
    """``kornia.io`` finds kornia_rs's image I/O wherever the installed kornia_rs keeps it (kornia#4325).

    Five released layouts exist: 0.1.9/0.1.10 export everything from the package root; 0.1.11 moved the
    readers and writers into ``kornia_rs.io`` (root keeps a deprecated ``read_image_any`` and the uint8
    JPEG/PNG writers); 0.1.12 and 0.1.13 have no ``read_image_jpegturbo`` at all; 0.1.14 has it in ``io``.
    """

    @staticmethod
    def _backend(module):
        from kornia.io.io import _KorniaRsImageIO

        return _KorniaRsImageIO(module)

    @staticmethod
    def _used_function_names() -> set[str]:
        """Every ``_rs_io.<name>`` attribute and ``_rs_io.find("<name>")`` argument in ``kornia/io/io.py``."""
        import ast
        import inspect

        import kornia.io.io as io_module

        names: set[str] = set()
        for node in ast.walk(ast.parse(inspect.getsource(io_module))):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "_rs_io":
                if node.attr != "find":
                    names.add(node.attr)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "find"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "_rs_io"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                names.add(node.args[0].value)
        return names

    def test_used_function_names_are_extracted_from_the_source(self):
        names = self._used_function_names()
        assert {"read_image_jpegturbo", "read_image", "read_image_any", "write_image_tiff_f32"} <= names

    def test_root_layout(self):
        from types import SimpleNamespace

        root = SimpleNamespace(read_image_any=lambda p: "any", write_image_png_u8=lambda *a: None)
        backend = self._backend(root)
        assert backend.find("read_image_any") is root.read_image_any
        assert backend.find("read_image") is None
        assert backend.write_image_png_u8 is root.write_image_png_u8

    def test_io_submodule_wins_over_the_root(self):
        from types import SimpleNamespace

        io_ns = SimpleNamespace(read_image=lambda p: "io", write_image_png_u8=lambda *a: "io")
        root = SimpleNamespace(io=io_ns, read_image_any=lambda p: "root", write_image_png_u8=lambda *a: "root")
        backend = self._backend(root)
        assert backend.write_image_png_u8 is io_ns.write_image_png_u8
        assert backend.find("read_image") is io_ns.read_image
        assert backend.find("read_image_any") is root.read_image_any  # falls back to the root

    def test_non_callable_attribute_is_not_a_function(self):
        from types import SimpleNamespace

        backend = self._backend(SimpleNamespace(io=SimpleNamespace(read_image=1), read_image=lambda p: "root"))
        assert backend.find("read_image")("x") == "root"

    def test_missing_function_raises_import_error_naming_the_version(self):
        from types import SimpleNamespace

        backend = self._backend(SimpleNamespace(__version__="0.1.13", io=SimpleNamespace()))
        with pytest.raises(ImportError, match=r"kornia_rs 0\.1\.13 has no image I/O function 'read_image_jpegturbo'"):
            backend.read_image_jpegturbo("x.jpg")

    def test_dunder_lookup_raises_attribute_error(self):
        """``inspect``/doctest collection probe ``__wrapped__`` and friends; those must not become ImportErrors."""
        from types import SimpleNamespace

        backend = self._backend(SimpleNamespace(read_image_any=lambda p: "any"))
        with pytest.raises(AttributeError):
            backend.__wrapped__  # noqa: B018
        assert not hasattr(backend, "__wrapped__")

    def test_installed_kornia_rs_provides_every_used_function(self):
        from kornia.io.io import _rs_io

        optional = {"read_image_jpegturbo", "read_image", "read_image_any"}  # each has a fallback in io.py
        missing = sorted(n for n in self._used_function_names() - optional if _rs_io.find(n) is None)
        assert missing == [], f"kornia_rs {kornia_rs.__version__}: {missing}"
        assert _rs_io.find("read_image") is not None or _rs_io.find("read_image_any") is not None

    @pytest.mark.parametrize("ext", ["jpg", "png", "tiff"])
    def test_uint8_round_trip_every_extension(self, tmp_path, ext):
        img = create_random_img8_torch(5, 6, 3)
        path = tmp_path / f"image.{ext}"
        write_image(path, img)
        loaded = load_image(path, ImageLoadType.UNCHANGED)
        assert loaded.shape == img.shape
        if ext != "jpg":  # lossless containers come back bit-exact
            assert torch.equal(loaded, img)

    def test_jpeg_loads_without_a_libjpeg_turbo_reader(self, tmp_path, monkeypatch):
        """The 0.1.12/0.1.13 layout: ``kornia_rs.io`` exists but has no ``read_image_jpegturbo``."""
        import kornia.io.io as io_module
        from kornia.io.io import _KorniaRsImageIO

        class _WithoutJpegTurbo:
            def __init__(self, namespace):
                self._namespace = namespace

            def __getattr__(self, name):
                if name == "read_image_jpegturbo":
                    raise AttributeError(name)
                attribute = getattr(self._namespace, name)
                return _WithoutJpegTurbo(attribute) if name == "io" else attribute

        pruned = _WithoutJpegTurbo(kornia_rs)
        if io_module._rs_io.find("read_image") is None:
            pytest.skip("kornia_rs < 0.1.11 has no generic reader to fall back to")
        monkeypatch.setattr(io_module, "_rs_io", _KorniaRsImageIO(pruned))
        assert io_module._rs_io.find("read_image_jpegturbo") is None

        img = create_random_img8_torch(5, 6, 3)
        path = tmp_path / "image.jpg"
        write_image(path, img)
        assert load_image(path, ImageLoadType.UNCHANGED).shape == img.shape

    def test_lookup_happens_at_call_time(self, tmp_path, monkeypatch):
        """Patching the kornia_rs function is seen by the next call, as it was before the resolver existed."""
        import kornia.io.io as io_module

        calls = []
        namespace = io_module._rs_io._namespaces[0]
        real = namespace.write_image_png_u8

        def spy(*args, **kwargs):
            calls.append(args[0])
            return real(*args, **kwargs)

        monkeypatch.setattr(namespace, "write_image_png_u8", spy)
        write_image(tmp_path / "image.png", create_random_img8_torch(2, 3, 3))
        assert calls == [str(tmp_path / "image.png")]


class TestWiderThanUint8Decodes:
    """kornia_rs 0.1.11+ decodes 16-bit PNG/TIFF and float TIFF, which 0.1.10 rejected at decode time."""

    @pytest.fixture(autouse=True)
    def _needs_generic_reader(self):
        from kornia.io.io import _rs_io

        if _rs_io.find("read_image") is None:
            pytest.skip("kornia_rs < 0.1.11 cannot decode 16-bit or float images")

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

    @pytest.mark.parametrize("load_type", [ImageLoadType.RGB8, ImageLoadType.GRAY8, ImageLoadType.RGB32])
    def test_eight_bit_load_types_reject_a_float_decode(self, tmp_path, load_type):
        path = tmp_path / "image.tiff"
        write_image(path, torch.rand(3, 4, 5))
        with pytest.raises(
            NotImplementedError, match=rf"decoded to torch\.float32, and ImageLoadType\.{load_type.name}"
        ):
            load_image(path, load_type)
