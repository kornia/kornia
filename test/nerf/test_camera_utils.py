from pathlib import Path

import pytest
import torch

from kornia.geometry.conversions import QuaternionCoeffOrder, quaternion_to_rotation_matrix
from kornia.nerf.camera_utils import create_spiral_path, parse_colmap_cameras, parse_colmap_output
from kornia.testing import assert_close


@pytest.fixture
def colmap_cameras_path():
    return Path(__file__).parent / './cameras.txt'


@pytest.fixture
def colmap_images_path():
    return Path(__file__).parent / './images.txt'


def test_parse_colmap_output(device, dtype, colmap_cameras_path, colmap_images_path) -> None:
    img_names, cameras = parse_colmap_output(colmap_cameras_path, colmap_images_path, device, dtype)
    assert_close(cameras.fx[0], torch.tensor(845.5654, device=device, dtype=dtype))
    assert_close(cameras.fy[0], torch.tensor(845.5654, device=device, dtype=dtype))
    assert_close(cameras.cx[0], torch.tensor(504, device=device, dtype=dtype))
    assert_close(cameras.cy[0], torch.tensor(378, device=device, dtype=dtype))

    qw = 0.99967420027533338
    qx = -0.016637661896892787
    qy = 0.012113517278577941
    qz = 0.015097821353390602
    tx = 3.2407154690493751
    ty = 2.3447342819165637
    tz = -1.0631749488011808

    q = torch.tensor([qw, qx, qy, qz], device=device, dtype=dtype)
    R = quaternion_to_rotation_matrix(q, order=QuaternionCoeffOrder.WXYZ)
    t = torch.tensor([tx, ty, tz], device=device, dtype=dtype)

    assert_close(R, cameras.rotation_matrix[2])
    assert_close(cameras.translation_vector[2], t.unsqueeze(-1))

    assert img_names[2] == 'image002.png'


@pytest.fixture
def colmap_simple_radial_cameras_path():
    return Path(__file__).parent / './cameras_simple_radial.txt'


def test_parse_simple_radial_camera_params(device, dtype, colmap_simple_radial_cameras_path) -> None:
    heights, widths, intrinsics = parse_colmap_cameras(colmap_simple_radial_cameras_path, device, dtype)
    assert heights[0] == 2731
    assert widths[0] == 1536
    intrinsic = intrinsics[0].to('cpu')
    assert_close(intrinsic[0, 0].item(), 2136.7647911326526)
    assert_close(intrinsic[1, 1].item(), 2136.7647911326526)
    assert_close(intrinsic[0, 2].item(), 768)
    assert_close(intrinsic[1, 2].item(), 1365.5)


def test_create_spiral_path(device, dtype, colmap_cameras_path, colmap_images_path) -> None:
    _, cameras = parse_colmap_output(colmap_cameras_path, colmap_images_path, device, dtype)
    try:
        create_spiral_path(cameras, 1, 30, 3)
    except Exception as err:
        assert False, err
