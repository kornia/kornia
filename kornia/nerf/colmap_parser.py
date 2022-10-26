from typing import List, Tuple

import torch

from kornia.core import Device, Tensor
from kornia.geometry.camera import PinholeCamera
from kornia.geometry.conversions import QuaternionCoeffOrder, quaternion_to_rotation_matrix


def parse_colmap_cameras(
    cameras_path: str, device: Device, dtype: torch.dtype
) -> Tuple[List[int], List[int], List[Tensor]]:

    # Parse camera intrinsics
    with open(cameras_path) as f:
        lines = f.readlines()

    class CameraParams:
        def __init__(self, line: str) -> None:
            split_line = line.split(' ')
            model = split_line[1]
            self._width = int(split_line[2])
            self._height = int(split_line[3])
            if model == 'SIMPLE_PINHOLE':
                self._fx = float(split_line[4])
                self._fy = self._fx
                self._cx = float(split_line[5])
                self._cy = float(split_line[6])
            elif model == 'PINHOLE':
                self._fx = float(split_line[4])
                self._fy = float(split_line[5])
                self._cx = float(split_line[6])
                self._cy = float(split_line[7])
            elif model == 'SIMPLE_RADIAL':
                self._fx = float(split_line[4])
                self._fy = self._fx
                self._cx = float(split_line[5])
                self._cy = float(split_line[6])
                self._k = float(
                    split_line[7]
                )  # FIXME: Skewness is assigned here but ignored later since PinholeCamera does not support distortion

    heights: List[int] = []
    widths: List[int] = []
    intrinsics: List[Tensor] = []
    for line in lines:
        if line.startswith('#'):
            continue
        camera_params = CameraParams(line)
        heights.append(camera_params._height)
        widths.append(camera_params._width)
        intrinsic = torch.eye(4, device=device, dtype=dtype)
        intrinsic[0, 0] = camera_params._fx
        intrinsic[1, 1] = camera_params._fy
        intrinsic[0, 2] = camera_params._cx
        intrinsic[1, 2] = camera_params._cy
        intrinsics.append(intrinsic)
    return heights, widths, intrinsics


def parse_colmap_output(
    cameras_path: str, images_path: str, device: Device, dtype: torch.dtype, sort_by_image_names: bool = False
) -> Tuple[List[str], PinholeCamera]:
    r"""Parses colmap output to create an PinholeCamera for aligned scene cameras.

    Args:
        cameras_path: Path to camera.txt Colmap file with camera intrinsics: str
        images_path: Path to images.txt Colmap file with camera extrinsics for each image: str
        device: device for created camera object: Union[str, torch.device]
        dtype: type for created camera object: torch.dtype
        sort_by_image_names: sort camers by image names. Useful for the case where image shooting order is important.
        For example, to create a camera path for rendering that follows the path of original images: bool

    Returns:
        image names: List[str]
        scene camera object: PinholeCamera
    """

    # Parse camera intrinsics
    camera_heights, camera_widths, camera_intrinsics = parse_colmap_cameras(cameras_path, device, dtype)

    # Parse camera quaternions and translation vectors
    with open(images_path) as f:
        lines = f.readlines()
    heights: List[int] = []
    widths: List[int] = []
    intrinsics: List[Tensor] = []
    extrinsics: List[Tensor] = []
    img_names: List[str] = []
    for line in lines:
        if line.startswith('#'):
            continue

        # Read line with camera quaternion
        line = line.strip()
        if line.endswith('png') or line.endswith('jpg') or line.endswith('jpeg'):
            split_line = line.split(' ')
            qw = float(split_line[1])
            qx = float(split_line[2])
            qy = float(split_line[3])
            qz = float(split_line[4])
            tx = float(split_line[5])
            ty = float(split_line[6])
            tz = float(split_line[7])
            camera_ind = int(split_line[8]) - 1
            img_name = split_line[9:]  # Assuming all last fields in the line compose the image filename
            img_name = ' '.join(img_name)
            img_names.append(img_name)

            # Intrinsic
            intrinsics.append(camera_intrinsics[camera_ind])
            heights.append(camera_heights[camera_ind])
            widths.append(camera_widths[camera_ind])

            # Extrinsic
            q = torch.tensor([qw, qx, qy, qz], device=device)
            R = quaternion_to_rotation_matrix(q, order=QuaternionCoeffOrder.WXYZ)
            t = torch.tensor([tx, ty, tz], device=device)
            extrinsic = torch.eye(4, device=device, dtype=dtype)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = t
            extrinsics.append(extrinsic)

    if sort_by_image_names:
        sorted_by_img_names = sorted(zip(img_names, intrinsics, extrinsics, heights, widths))
        tuples = zip(*sorted_by_img_names)
        img_names, intrinsics, extrinsics, heights, widths = (list(tuple) for tuple in tuples)

    cameras = PinholeCamera(
        torch.stack(intrinsics),
        torch.stack(extrinsics),
        torch.tensor(heights, device=device),
        torch.tensor(widths, device=device),
    )
    return img_names, cameras


def parse_colmap_points_3d(points_3d_path: str, device: Device, dtype: torch.dtype) -> Tensor:
    r"""Parses colmap point3D file for point cloud coordinates.

    Args:
        points_3d_path: Path to points3D.txt colmap file with point cloud coordinates
        device: device for created camera object: Union[str, torch.device]
        dtype: type for created camera object: torch.dtype

    Returns:
        points_3d: Point cloud coordinates :math:`(*, 1, 3)`
    """
    with open(points_3d_path) as f:
        lines = f.readlines()
    x: List[float] = []
    y: List[float] = []
    z: List[float] = []
    points_3d: List[Tensor] = []
    for line in lines:
        if line.startswith('#'):
            continue

        # Read line for a point in 3D
        line = line.strip()
        split_line = line.split(' ')
        x.append(float(split_line[1]))
        y.append(float(split_line[2]))
        z.append(float(split_line[3]))
        points_3d.append(torch.tensor([float(split_line[1]), float(split_line[2]), float(split_line[3])]))
        # TODO: Parse here more fields as necessary for future usages
    return torch.stack(points_3d).unsqueeze(1)
