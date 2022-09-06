from typing import List

import torch
from defusedxml import lxml as ET

from kornia.geometry.camera import PinholeCamera
from kornia.geometry.nerf.types import Device


class CameraParser:
    def __init__(self, xml_path: str, device: Device) -> None:
        self._xml_path = xml_path
        self._device = device

    def __init_xml(self):  # -> ET.Element:
        tree = ET.parse(self._xml_path)
        root = tree.getroot()
        return root

    def __parse_camera_extrinsic_matrices(self, root) -> torch.Tensor:
        chunk = root.find('chunk')
        camera_head = chunk.find('cameras')
        cameras = camera_head.findall('camera')
        extrinsics: List[torch.Tensor] = []
        for camera in cameras:
            transform = camera.find('transform')
            extrinsics.append(
                torch.tensor(list(map(float, transform.text.split(' '))), device=self._device).reshape(4, 4)
            )
        return torch.stack(extrinsics)

    def __parse_camera_intrinsic_matrix(self, root) -> torch.Tensor:
        chunk = root.find('chunk')
        sensor_head = chunk.find('sensors')
        sensors = sensor_head.findall('sensor')
        for sensor in sensors:
            resolution = sensor.find('resolution')
            width = int(resolution.get('width'))
            height = int(resolution.get('height'))
            calibration = sensor.find('calibration')
            f = float(calibration.find('f').text)
            cx = float(calibration.find('cx').text)
            cy = float(calibration.find('cy').text)
            k1 = float(calibration.find('k1').text)
            k2 = float(calibration.find('k2').text)
            k3 = float(calibration.find('k3').text)
            p1 = float(calibration.find('p1').text)
            p2 = float(calibration.find('p2').text)
        return (
            self.__camera_intrinsic_matrix_from_parameters(
                f, (width - 1.0) / 2.0 + cx, (height - 1.0) / 2.0 + cy, k1, k2, k3, p1, p2
            ),
            width,
            height,
        )

    def __camera_intrinsic_matrix_from_parameters(self, f, cx, cy, k1, k2, k3, p1, p2) -> torch.Tensor:
        intrinsics = torch.eye(4, device=self._device, dtype=torch.float32)
        # FIXME: 1) Assuming one focal length in both directions; 2) ignoring distorion parameters for now
        intrinsics[0, 0] = f
        intrinsics[1, 1] = f
        intrinsics[0, 2] = cx
        intrinsics[1, 2] = cy
        return intrinsics

    def create_cameras(self) -> PinholeCamera:
        root = self.__init_xml()
        extrinsics = self.__parse_camera_extrinsic_matrices(root)
        intrinsic, width, height = self.__parse_camera_intrinsic_matrix(root)
        num_cams = extrinsics.shape[0]
        intrinsics = intrinsic.repeat(num_cams, 1, 1)
        return PinholeCamera(
            intrinsics,
            extrinsics,
            torch.tensor([height] * num_cams, device=self._device),
            torch.tensor([width] * num_cams, device=self._device),
        )


def cameras_for_ids(cameras: PinholeCamera, camera_ids: List[int]) -> PinholeCamera:
    intrinsics = cameras.intrinsics[camera_ids]
    extrinsics = cameras.extrinsics[camera_ids]
    height = cameras.height[camera_ids]
    width = cameras.width[camera_ids]
    return PinholeCamera(intrinsics, extrinsics, height, width)


def create_spiral_path(cameras: PinholeCamera, rad: float, num_views: int, num_circles: int) -> PinholeCamera:

    # Average locations over all cameras
    mean_center = torch.squeeze(torch.mean(cameras.translation_vector, dim=0))
    t = torch.linspace(0, 2 * torch.pi * num_circles, num_views)
    cos_t = torch.cos(t)
    sin_t = -torch.sin(t)
    sin_05t = -torch.sin(0.5 * t)
    translation_vector = torch.unsqueeze(mean_center, dim=0) + torch.stack((cos_t, sin_t, sin_05t)).permute((1, 0))
    mean_intrinsics = torch.mean(cameras.intrinsics, dim=0, keepdims=True).repeat((num_views, 1, 1))
    mean_extrinsics = torch.mean(cameras.extrinsics, dim=0, keepdims=True).repeat((num_views, 1, 1))
    extrinsics = mean_extrinsics
    extrinsics[:, :3, 3] = translation_vector
    height = torch.tensor([cameras.height[0]] * num_views)
    width = torch.tensor([cameras.width[0]] * num_views)
    return PinholeCamera(mean_intrinsics, extrinsics, height, width)
