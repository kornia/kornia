# import xml.etree.ElementTree as ET

import torch
from defusedxml import lxml as ET


def read_camera_xml(xml_path: str):  # -> ET.Element:
    tree = ET.parse(xml_path)

    root = tree.getroot()
    return root


def __text_to_tensor(a: str) -> torch.tensor:
    pass


def parse_camera_extrinsic_matrices(root) -> torch.Tensor:
    chunk = root.find('chunk')
    camera_head = chunk.find('cameras')
    cameras = camera_head.findall('camera')
    for camera in cameras:
        transform = camera.find('transform')
        __text_to_tensor(transform.text)
    pass
