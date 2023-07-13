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


def create_random_img8_torch(height: int, width: int, channels: int) -> Tensor:
    return (torch.rand(channels, height, width) * 255).to(torch.uint8)


@pytest.mark.skipif(not available_package(), reason="kornia_rs only supports python >=3.7 and pt >= 1.10.0")
class TestIoImage:
    def test_smoke(self, tmp_path: Path) -> None:
        height, width = 4, 5
        img_th: Tensor = create_random_img8_torch(height, width, 3)

        file_path = tmp_path / "image.jpg"
        write_image(file_path, img_th)

        assert file_path.is_file()

        img_load: Tensor = load_image(file_path, ImageLoadType.UNCHANGED)

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
    def test_types_color(self, tmp_path: Path, ext) -> None:
        height, width = 4, 5
        img_np: np.ndarray = create_random_img8(height, width, 3)

        file_path = tmp_path / f"image.{ext}"
        cv2.imwrite(str(file_path), img_np)

        assert file_path.is_file()

        img = load_image(file_path, ImageLoadType.GRAY8)
        assert img.shape[0] == 1
        assert img.dtype == torch.uint8

        img = load_image(file_path, ImageLoadType.GRAY32)
        assert img.shape[0] == 1
        assert img.dtype == torch.float32

        img = load_image(file_path, ImageLoadType.RGB8)
        assert img.shape[0] == 3
        assert img.dtype == torch.uint8

        img = load_image(file_path, ImageLoadType.RGB32)
        assert img.shape[0] == 3
        assert img.dtype == torch.float32

        img = load_image(file_path, ImageLoadType.RGBA8)
        assert img.shape[0] == 4
        assert img.dtype == torch.uint8

    @pytest.mark.parametrize("ext", ["png", "jpg"])
    def test_types_gray(self, tmp_path: Path, ext) -> None:
        height, width = 4, 5
        img_np: np.ndarray = create_random_img8(height, width, 1)

        file_path = tmp_path / f"image.{ext}"
        cv2.imwrite(str(file_path), img_np)

        assert file_path.is_file()

        img = load_image(file_path, ImageLoadType.GRAY8)
        assert img.shape[0] == 1
        assert img.dtype == torch.uint8

        img = load_image(file_path, ImageLoadType.GRAY32)
        assert img.shape[0] == 1
        assert img.dtype == torch.float32

        img = load_image(file_path, ImageLoadType.RGB8)
        assert img.shape[0] == 3
        assert img.dtype == torch.uint8

        img = load_image(file_path, ImageLoadType.RGB32)
        assert img.shape[0] == 3
        assert img.dtype == torch.float32

        img = load_image(file_path, ImageLoadType.RGBA8)
        assert img.shape[0] == 4
        assert img.dtype == torch.uint8
