from pathlib import Path

import pytest

from kornia.geometry.nerf.camera_utils import CameraParser
from kornia.testing import assert_close


@pytest.fixture
def xml_path():
    return Path(__file__).parent / './cameras.xml'


def test_parse_camera_matrices(device, dtype, xml_path) -> None:
    camera_parser = CameraParser(xml_path, device)
    cameras = camera_parser.create_cameras()
    assert cameras.height[0] == 756
    assert cameras.width[0] == 1008
    assert_close(cameras.fx[0], 875.95911716912599)
    assert_close(cameras.fy[0], 875.95911716912599)
    assert_close(cameras.cx[0], -2.7834739868339686 + (cameras.width[0] - 1.0) / 2.0)
    assert_close(cameras.cy[0], 31.748254839395045 + (cameras.height[0] - 1.0) / 2.0)

    assert_close(cameras.rotation_matrix[3, 0, 0], 9.9814188389430214e-01)
    assert_close(cameras.rotation_matrix[3, 0, 1], 2.1967447258124558e-03)
    assert_close(cameras.rotation_matrix[3, 0, 2], -6.0892971093077258e-02)
    assert_close(cameras.rotation_matrix[3, 1, 0], 3.0174972532337795e-03)
    assert_close(cameras.rotation_matrix[3, 1, 1], -9.9990579780875144e-01)
    assert_close(cameras.rotation_matrix[3, 1, 2], 1.3389929752265439e-02)
    assert_close(cameras.rotation_matrix[3, 2, 0], -6.0857820584206385e-02)
    assert_close(cameras.rotation_matrix[3, 2, 1], -1.3548794081152171e-02)
    assert_close(cameras.rotation_matrix[3, 2, 2], -9.9805448541283903e-01)

    assert_close(cameras.translation_vector[5, 0, 0], -5.6508471844511410e-01)
    assert_close(cameras.translation_vector[5, 1, 0], 1.7676268597137265e-01)
    assert_close(cameras.translation_vector[5, 2, 0], 1.7946317899388795e-01)
