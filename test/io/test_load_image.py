import os
import sys
import tempfile

import cv2
import numpy as np
import pytest
import torch

from kornia.core import Tensor
from kornia.io import ImageType, load_image


def create_random_img8(height: int, width: int, channels: int) -> np.ndarray:
    return (np.random.rand(height, width, channels) * 255).astype(np.uint8)


@pytest.mark.skipif(sys.version_info < (3, 7, 0), reason="kornia_rs only supports python >=3.7")
class TestLoadImage:
    def test_smoke(self):
        height, width = 4, 5
        img_np: np.ndarray = create_random_img8(height, width, 3)
        with tempfile.NamedTemporaryFile() as tmp:

            file_path: str = tmp.name + ".png"
            cv2.imwrite(file_path, img_np)
            assert os.path.isfile(file_path)

            img_cv: np.ndarray = cv2.imread(file_path)
            img_th: Tensor = load_image(file_path, ImageType.UNCHANGED)

            assert img_cv.shape[:2] == img_th.shape[1:]
            assert img_th.shape[1:] == (height, width)
            assert str(img_th.device) == "cpu"

    def test_device(self, device):
        height, width = 4, 5
        img_np: np.ndarray = create_random_img8(height, width, 3)
        with tempfile.NamedTemporaryFile() as tmp:

            file_path: str = tmp.name + ".png"
            cv2.imwrite(file_path, img_np)
            assert os.path.isfile(file_path)

            img_th: Tensor = load_image(file_path, ImageType.UNCHANGED, str(device))
            assert str(img_th.device) == str(device)

    @pytest.mark.parametrize("ext", ["png", "jpg"])
    def test_types_color(self, ext):
        height, width = 4, 5
        img_np: np.ndarray = create_random_img8(height, width, 3)
        with tempfile.NamedTemporaryFile() as tmp:

            file_path: str = tmp.name + f".{ext}"
            cv2.imwrite(file_path, img_np)
            assert os.path.isfile(file_path)

            img = load_image(file_path, ImageType.GRAY8)
            assert img.shape[0] == 1 and img.dtype == torch.uint8

            img = load_image(file_path, ImageType.GRAY32)
            assert img.shape[0] == 1 and img.dtype == torch.float32

            img = load_image(file_path, ImageType.RGB8)
            assert img.shape[0] == 3 and img.dtype == torch.uint8

            img = load_image(file_path, ImageType.RGB32)
            assert img.shape[0] == 3 and img.dtype == torch.float32

            img = load_image(file_path, ImageType.RGBA8)
            assert img.shape[0] == 4 and img.dtype == torch.uint8

    @pytest.mark.parametrize("ext", ["png", "jpg"])
    def test_types_gray(self, ext):
        height, width = 4, 5
        img_np: np.ndarray = create_random_img8(height, width, 1)
        with tempfile.NamedTemporaryFile() as tmp:

            file_path: str = tmp.name + f".{ext}"
            cv2.imwrite(file_path, img_np)
            assert os.path.isfile(file_path)

            img = load_image(file_path, ImageType.GRAY8)
            assert img.shape[0] == 1 and img.dtype == torch.uint8

            img = load_image(file_path, ImageType.GRAY32)
            assert img.shape[0] == 1 and img.dtype == torch.float32

            img = load_image(file_path, ImageType.RGB8)
            assert img.shape[0] == 3 and img.dtype == torch.uint8

            img = load_image(file_path, ImageType.RGB32)
            assert img.shape[0] == 3 and img.dtype == torch.float32

            img = load_image(file_path, ImageType.RGBA8)
            assert img.shape[0] == 4 and img.dtype == torch.uint8
