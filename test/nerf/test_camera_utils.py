from pathlib import Path

import pytest

from kornia.nerf.camera_utils import create_spiral_path
from kornia.nerf.colmap_parser import parse_colmap_output


@pytest.fixture
def colmap_cameras_path():
    return Path(__file__).parent / './cameras.txt'


@pytest.fixture
def colmap_images_path():
    return Path(__file__).parent / './images.txt'


def test_create_spiral_path(device, dtype, colmap_cameras_path, colmap_images_path) -> None:
    _, cameras = parse_colmap_output(colmap_cameras_path, colmap_images_path, device, dtype)
    try:
        create_spiral_path(cameras, 1, 30, 3)
    except Exception as err:
        assert False, err
