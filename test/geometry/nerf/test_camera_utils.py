from pathlib import Path

import pytest

import kornia.geometry.nerf.camera_utils as camera_utils


@pytest.fixture
def xml_path():
    return Path(__file__).parent / './cameras.xml'


def test_parse_camera_extrinsic_matrices(xml_path) -> None:
    root = camera_utils.read_camera_xml(xml_path)
    camera_utils.parse_camera_extrinsic_matrices(root)
