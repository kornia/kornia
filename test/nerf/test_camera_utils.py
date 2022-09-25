from pathlib import Path

import pytest
import torch

from kornia.geometry.conversions import QuaternionCoeffOrder, quaternion_to_rotation_matrix
from kornia.nerf.camera_utils import parse_colmap_output
from kornia.testing import assert_close


@pytest.fixture
def colmap_cameras_path():
    return Path(__file__).parent / './cameras.txt'


@pytest.fixture
def colmap_images_path():
    return Path(__file__).parent / './images.txt'


def test_parse_colmap_output(device, dtype, colmap_cameras_path, colmap_images_path) -> None:
    img_names, cameras = parse_colmap_output(colmap_cameras_path, colmap_images_path, device)
    assert_close(cameras.fx[0], 845.5654)
    assert_close(cameras.fy[0], 845.5654)
    assert_close(cameras.cx[0], 504)
    assert_close(cameras.cy[0], 378)

    qw = 0.99967420027533338
    qx = -0.016637661896892787
    qy = 0.012113517278577941
    qz = 0.015097821353390602
    tx = 3.2407154690493751
    ty = 2.3447342819165637
    tz = -1.0631749488011808

    q = torch.tensor([qw, qx, qy, qz], device=device)
    R = quaternion_to_rotation_matrix(q, order=QuaternionCoeffOrder.WXYZ)
    t = torch.tensor([tx, ty, tz], device=device)

    assert_close(R, cameras.rotation_matrix[2])
    assert_close(cameras.translation_vector[2], t.unsqueeze(-1))

    assert img_names[2] == 'image002.png'
