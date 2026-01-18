# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np
import pycolmap
from PIL import Image

from kornia.models.depth_anything_3.specs import Prediction
from kornia.models.depth_anything_3.utils.logger import logger

from .glb import _depths_to_world_points_with_colors


def export_to_colmap(
    prediction: Prediction,
    export_dir: str,
    image_paths: list[str],
    conf_thresh_percentile: float = 40.0,
    process_res_method: str = "upper_bound_resize",
) -> None:
    # 1. Data preparation
    conf_thresh = np.percentile(prediction.conf, conf_thresh_percentile)
    points, colors = _depths_to_world_points_with_colors(
        prediction.depth,
        prediction.intrinsics,
        prediction.extrinsics,  # w2c
        prediction.processed_images,
        prediction.conf,
        conf_thresh,
    )
    num_points = len(points)
    logger.info(f"Exporting to COLMAP with {num_points} points")
    num_frames = len(prediction.processed_images)
    h, w = prediction.processed_images.shape[1:3]
    points_xyf = _create_xyf(num_frames, h, w)
    points_xyf = points_xyf[prediction.conf >= conf_thresh]

    # 2. Set Reconstruction
    reconstruction = pycolmap.Reconstruction()

    point3d_ids = []
    for vidx in range(num_points):
        point3d_id = reconstruction.add_point3D(points[vidx], pycolmap.Track(), colors[vidx])
        point3d_ids.append(point3d_id)

    for fidx in range(num_frames):
        orig_w, orig_h = Image.open(image_paths[fidx]).size

        intrinsic = prediction.intrinsics[fidx]
        if process_res_method.endswith("resize"):
            intrinsic[:1] *= orig_w / w
            intrinsic[1:2] *= orig_h / h
        elif process_res_method == "crop":
            raise NotImplementedError("COLMAP export for crop method is not implemented")
        else:
            raise ValueError(f"Unknown process_res_method: {process_res_method}")

        pycolmap_intri = np.array([intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]])

        extrinsic = prediction.extrinsics[fidx]
        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(extrinsic[:3, :3]), extrinsic[:3, 3])

        # set and add camera
        camera = pycolmap.Camera()
        camera.camera_id = fidx + 1
        camera.model = pycolmap.CameraModelId.PINHOLE
        camera.width = orig_w
        camera.height = orig_h
        camera.params = pycolmap_intri
        reconstruction.add_camera(camera)

        # set and add rig (from camera)
        rig = pycolmap.Rig()
        rig.rig_id = camera.camera_id
        rig.add_ref_sensor(camera.sensor_id)
        reconstruction.add_rig(rig)

        # set image
        image = pycolmap.Image()
        image.image_id = fidx + 1
        image.camera_id = camera.camera_id

        # set and add frame (from image)
        frame = pycolmap.Frame()
        frame.frame_id = image.image_id
        frame.rig_id = camera.camera_id
        frame.add_data_id(image.data_id)
        frame.rig_from_world = cam_from_world
        reconstruction.add_frame(frame)

        # set point2d and update track
        point2d_list = []
        points_in_frame = points_xyf[:, 2].astype(np.int32) == fidx
        for vidx in np.where(points_in_frame)[0]:
            point2d = points_xyf[vidx][:2]
            point2d[0] *= orig_w / w
            point2d[1] *= orig_h / h
            point3d_id = point3d_ids[vidx]
            point2d_list.append(pycolmap.Point2D(point2d, point3d_id))
            reconstruction.point3D(point3d_id).track.add_element(image.image_id, len(point2d_list) - 1)

        # set and add image
        image.frame_id = image.image_id
        image.name = os.path.basename(image_paths[fidx])
        image.points2D = pycolmap.Point2DList(point2d_list)
        reconstruction.add_image(image)

    # 3. Export
    reconstruction.write(export_dir)


def _create_xyf(num_frames, height, width):
    """Creates a grid of pixel coordinates and frame indices (fidx) for all frames."""
    # Create coordinate grids for a single frame
    y_grid, x_grid = np.indices((height, width), dtype=np.int32)
    x_grid = x_grid[np.newaxis, :, :]
    y_grid = y_grid[np.newaxis, :, :]

    # Broadcast to all frames
    x_coords = np.broadcast_to(x_grid, (num_frames, height, width))
    y_coords = np.broadcast_to(y_grid, (num_frames, height, width))

    # Create frame indices and broadcast
    f_idx = np.arange(num_frames, dtype=np.int32)[:, np.newaxis, np.newaxis]
    f_coords = np.broadcast_to(f_idx, (num_frames, height, width))

    # Stack coordinates and frame indices
    points_xyf = np.stack((x_coords, y_coords, f_coords), axis=-1)

    return points_xyf
