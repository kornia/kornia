import hashlib
import urllib.request

import pytest
import torch

from kornia.geometry.conversions import quaternion_to_rotation_matrix
from kornia.nerf.camera_utils import create_spiral_path, parse_colmap_output
from kornia.testing import assert_close

_ref = {
    'cameras': (
        'https://raw.githubusercontent.com/kornia/data/main/nerf/cameras.txt',
        'aee6dbd448be900a0d4de85d08914e47a84f025e16f8498cbf9f7bfc7eff09a6',
    ),
    'images': (
        'https://raw.githubusercontent.com/kornia/data/main/nerf/images.txt',
        'e6b66a9b76d92e498697edece33be76ae69aa28a960bc895a59495763de61286',
    ),
}


def _get_data(url: str, sha256: str) -> str:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:  # noqa: S310
        data = response.read()

    assert hashlib.sha256(data).hexdigest() == sha256

    return data.decode('utf-8')


@pytest.fixture()
def colmap_cameras_path(tmp_path):
    data = _get_data(*_ref['cameras'])

    p = tmp_path / "camera.txt"
    p.write_text(data)

    return p


@pytest.fixture()
def colmap_images_path(tmp_path):
    data = _get_data(*_ref['images'])

    p = tmp_path / "images.txt"
    p.write_text(data)

    return p


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
    R = quaternion_to_rotation_matrix(q)
    t = torch.tensor([tx, ty, tz], device=device, dtype=dtype)

    assert_close(R, cameras.rotation_matrix[2])
    assert_close(cameras.translation_vector[2], t.unsqueeze(-1))

    assert img_names[2] == 'image002.png'


def test_create_spiral_path(device, dtype, colmap_cameras_path, colmap_images_path) -> None:
    _, cameras = parse_colmap_output(colmap_cameras_path, colmap_images_path, device, dtype)
    try:
        create_spiral_path(cameras, 1, 30, 3)
    except Exception as err:
        assert False, err
