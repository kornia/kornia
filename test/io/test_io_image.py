import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
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

    def test_device(self, device, tmp_path: Path) -> None:
        height, width = 4, 5
        img_np: np.ndarray = create_random_img8(height, width, 3)

        file_path = tmp_path / "image.png"
        cv2.imwrite(str(file_path), img_np)

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
    def test_load_image(self, tmp_path: Path, ext, channels, load_type, expected_type, expected_channels):
        height, width = 4, 5
        img_np: np.ndarray = create_random_img8(height, width, channels)

        file_path = tmp_path / f"image.{ext}"
        cv2.imwrite(str(file_path), img_np)

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
