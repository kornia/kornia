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

import numpy as np
import pytest
import requests
import torch

from kornia.core import Tensor
from kornia.io import ImageLoadType, load_image, write_image
from kornia.utils._compat import torch_version_ge

try:
    import kornia_rs
except ImportError:
    kornia_rs = None


def available_package() -> bool:
    return sys.version_info >= (3, 7, 0) and torch_version_ge(1, 10, 0) and kornia_rs is not None


def create_random_img8(height: int, width: int, channels: int) -> np.ndarray:
    return (np.random.rand(height, width, channels) * 255).astype(np.uint8)  # noqa: NPY002


def create_random_img8_torch(height: int, width: int, channels: int, device=None) -> Tensor:
    return (torch.rand(channels, height, width, device=device) * 255).to(torch.uint8)


def create_random_img16_torch(height: int, width: int, channels: int, device=None) -> Tensor:
    return (torch.rand(channels, height, width, device=device) * 65535).to(torch.uint16)


def create_random_img32_torch(height: int, width: int, channels: int, device=None) -> Tensor:
    return torch.rand(channels, height, width, device=device, dtype=torch.float32)


def _download_image(url: str, filename: str = "") -> Path:
    # TODO: move this to testing

    filename = url.split("/")[-1] if len(filename) == 0 else filename
    # Download
    bytesio = io.BytesIO(requests.get(url, timeout=60).content)
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
        img_th: Tensor = create_random_img8_torch(height, width, 3)

        file_path = tmp_path / "image.jpg"
        write_image(str(file_path), img_th)

        assert file_path.is_file()

        img_load: Tensor = load_image(str(file_path), ImageLoadType.UNCHANGED)

        assert img_th.shape == img_load.shape
        assert img_th.shape[1:] == (height, width)
        assert str(img_th.device) == "cpu"

    def test_device(self, device, png_image: Path) -> None:
        file_path = Path(png_image)

        assert file_path.is_file()

        img_th: Tensor = load_image(file_path, ImageLoadType.UNCHANGED, str(device))
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

    @pytest.mark.parametrize("ext", ["jpg"])
    @pytest.mark.parametrize("channels", [3])
    def test_write_image(self, device, tmp_path, ext, channels):
        height, width = 4, 5
        img_th: Tensor = create_random_img8_torch(height, width, channels, device)

        file_path = tmp_path / f"image.{ext}"
        write_image(file_path, img_th)

        assert file_path.is_file()

    @pytest.mark.parametrize("ext", ["jpg", "png", "tiff"])
    @pytest.mark.parametrize("channels", [1, 3])
    def test_write_image_uint8_formats_channels(self, device, tmp_path, ext, channels):
        """Test writing uint8 images in different formats with different channel counts.

        Note: 4-channel (RGBA) images are not fully supported by the backend for writing.
        """
        height, width = 4, 5
        img_th: Tensor = create_random_img8_torch(height, width, channels, device)

        file_path = tmp_path / f"image.{ext}"
        write_image(file_path, img_th)

        assert file_path.is_file()
        img_load = load_image(file_path, ImageLoadType.UNCHANGED)
        # JPEG always loads as 3 channels, PNG/TIFF preserve channel count
        if ext == "jpg":
            assert img_load.shape[0] in [1, 3]  # JPEG may convert to RGB
        else:
            assert img_load.shape[0] == channels
        assert img_load.dtype == torch.uint8

    def test_write_image_uint8_2d(self, device, tmp_path):
        """Test writing 2D grayscale image (H, W)."""
        height, width = 4, 5
        img_th: Tensor = create_random_img8_torch(height, width, 1, device).squeeze(0)

        file_path = tmp_path / "image.jpg"
        write_image(file_path, img_th)

        assert file_path.is_file()
        img_load = load_image(file_path, ImageLoadType.UNCHANGED)
        # 2D images are converted to 3-channel RGB when saved as JPEG
        assert img_load.shape[0] == 3
        assert img_load.dtype == torch.uint8

    @pytest.mark.parametrize("ext", ["png", "tiff"])
    def test_write_image_uint16(self, device, tmp_path, ext):
        """Test writing uint16 images (PNG and TIFF only)."""
        height, width = 4, 5
        img_th: Tensor = create_random_img16_torch(height, width, 3, device)

        file_path = tmp_path / f"image.{ext}"
        write_image(file_path, img_th)

        assert file_path.is_file()
        # TODO: Add ticket to kornia-rs to fix loading back uint16 images
        # Need to fix kornia-rs - test will fail until backend supports uint16 loading
        img_load = load_image(file_path, ImageLoadType.UNCHANGED)
        assert img_load.shape == img_th.shape
        assert img_load.dtype == torch.uint16

    @pytest.mark.parametrize("ext", ["tiff"])
    def test_write_image_float32(self, device, tmp_path, ext):
        """Test writing float32 images (TIFF only)."""
        height, width = 4, 5
        img_th: Tensor = create_random_img32_torch(height, width, 3, device)

        file_path = tmp_path / f"image.{ext}"
        write_image(file_path, img_th)

        assert file_path.is_file()
        # TODO: Add ticket to kornia-rs to fix loading back float32 images
        # Need to fix kornia-rs - test will fail until backend supports float32 loading
        img_load = load_image(file_path, ImageLoadType.UNCHANGED)
        assert img_load.shape == img_th.shape
        assert img_load.dtype == torch.float32

    @pytest.mark.parametrize("quality", [None, 50, 80, 95])
    def test_write_image_jpeg_quality(self, device, tmp_path, quality):
        """Test that quality parameter works for JPEG files."""
        height, width = 4, 5
        img_th: Tensor = create_random_img8_torch(height, width, 3, device)

        file_path = tmp_path / "image.jpg"
        write_image(file_path, img_th, quality=quality)

        assert file_path.is_file()
        # Verify image can be loaded back
        img_load = load_image(file_path, ImageLoadType.UNCHANGED)
        assert img_load.shape == img_th.shape

    @pytest.mark.parametrize("ext", ["png", "tiff"])
    @pytest.mark.parametrize("quality", [None, 50, 95])
    def test_write_image_non_jpeg_quality_ignored(self, device, tmp_path, ext, quality):
        """Test that quality parameter is ignored for non-JPEG formats."""
        height, width = 4, 5
        img_th: Tensor = create_random_img8_torch(height, width, 3, device)

        file_path = tmp_path / f"image.{ext}"
        # Should work without errors even though quality is ignored
        write_image(file_path, img_th, quality=quality)

        assert file_path.is_file()
        img_load = load_image(file_path, ImageLoadType.UNCHANGED)
        assert img_load.shape == img_th.shape

    def test_write_image_invalid_extension(self, device, tmp_path):
        """Test that invalid file extensions raise an error."""
        height, width = 4, 5
        img_th: Tensor = create_random_img8_torch(height, width, 3, device)

        file_path = tmp_path / "image.bmp"
        with pytest.raises(Exception, match="Invalid file extension.*only .jpg, .jpeg, .png and .tiff are supported"):
            write_image(file_path, img_th)

    def test_write_image_invalid_shape(self, device, tmp_path):
        """Test that invalid image shapes raise an error."""
        # 1D tensor (invalid) - create using randint for uint8
        img_th: Tensor = torch.randint(0, 256, (10,), device=device, dtype=torch.uint8)

        file_path = tmp_path / "image.jpg"
        with pytest.raises(Exception, match="Invalid image shape.*Must be at least 2D"):
            write_image(file_path, img_th)

    def test_write_image_invalid_dtype(self, device, tmp_path):
        """Test that unsupported dtypes raise an error."""
        height, width = 4, 5
        # float64 is not supported
        img_th: Tensor = torch.rand(3, height, width, device=device, dtype=torch.float64)

        file_path = tmp_path / "image.jpg"
        with pytest.raises(NotImplementedError, match=r"Unsupported image dtype: torch\.float64"):
            write_image(file_path, img_th)

    def test_write_image_uint16_jpeg_unsupported(self, device, tmp_path):
        """Test that uint16 images cannot be written as JPEG."""
        height, width = 4, 5
        img_th: Tensor = create_random_img16_torch(height, width, 3, device)

        file_path = tmp_path / "image.jpg"
        with pytest.raises(NotImplementedError, match=r"Unsupported file extension: \.jpg for uint16 image"):
            write_image(file_path, img_th)

    def test_write_image_float32_jpeg_unsupported(self, device, tmp_path):
        """Test that float32 images cannot be written as JPEG."""
        height, width = 4, 5
        img_th: Tensor = create_random_img32_torch(height, width, 3, device)

        file_path = tmp_path / "image.jpg"
        with pytest.raises(NotImplementedError, match=r"Unsupported file extension: \.jpg for float32 image"):
            write_image(file_path, img_th)

    def test_write_image_float32_png_unsupported(self, device, tmp_path):
        """Test that float32 images cannot be written as PNG."""
        height, width = 4, 5
        img_th: Tensor = create_random_img32_torch(height, width, 3, device)

        file_path = tmp_path / "image.png"
        with pytest.raises(NotImplementedError, match=r"Unsupported file extension: \.png for float32 image"):
            write_image(file_path, img_th)
