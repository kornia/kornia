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

from __future__ import annotations

import os
import numpy as np
import trimesh

from kornia.models.depth_anything_3.specs import Prediction
from kornia.models.depth_anything_3.utils.logger import logger

from .depth_vis import export_to_depth_vis


def set_sky_depth(prediction: Prediction, sky_mask: np.ndarray, sky_depth_def: float = 98.0):
    non_sky_mask = ~sky_mask
    valid_depth = prediction.depth[non_sky_mask]
    if valid_depth.size > 0:
        max_depth = np.percentile(valid_depth, sky_depth_def)
        prediction.depth[sky_mask] = max_depth


def get_conf_thresh(
    prediction: Prediction,
    sky_mask: np.ndarray,
    conf_thresh: float,
    conf_thresh_percentile: float = 10.0,
    ensure_thresh_percentile: float = 90.0,
):
    if sky_mask is not None and (~sky_mask).sum() > 10:
        conf_pixels = prediction.conf[~sky_mask]
    else:
        conf_pixels = prediction.conf
    lower = np.percentile(conf_pixels, conf_thresh_percentile)
    upper = np.percentile(conf_pixels, ensure_thresh_percentile)
    conf_thresh = min(max(conf_thresh, lower), upper)
    return conf_thresh


def export_to_glb(
    prediction: Prediction,
    export_dir: str,
    num_max_points: int = 1_000_000,
    conf_thresh: float = 1.05,
    filter_black_bg: bool = False,
    filter_white_bg: bool = False,
    conf_thresh_percentile: float = 40.0,
    ensure_thresh_percentile: float = 90.0,
    sky_depth_def: float = 98.0,
    show_cameras: bool = True,
    camera_size: float = 0.03,
    export_depth_vis: bool = True,
) -> str:
    """Generate a 3D point cloud and camera wireframes and export them as a ``.glb`` file.

    The function builds a point cloud from the predicted depth maps, aligns it to the
    first camera in glTF coordinates (X-right, Y-up, Z-backward), optionally draws
    camera wireframes, and writes the result to ``scene.glb``. Auxiliary assets such as
    depth visualizations can also be generated alongside the main export.

    Args:
        prediction: Model prediction containing depth, confidence, intrinsics, extrinsics,
            and pre-processed images.
        export_dir: Output directory where the glTF assets will be written.
        num_max_points: Maximum number of points retained after downsampling.
        conf_thresh: Base confidence threshold used before percentile adjustments.
        filter_black_bg: Mark near-black background pixels for removal during confidence filtering.
        filter_white_bg: Mark near-white background pixels for removal during confidence filtering.
        conf_thresh_percentile: Lower percentile used when adapting the confidence threshold.
        ensure_thresh_percentile: Upper percentile clamp for the adaptive threshold.
        sky_depth_def: Percentile used to fill sky pixels with plausible depth values.
        show_cameras: Whether to render camera wireframes in the exported scene.
        camera_size: Relative camera wireframe scale as a fraction of the scene diagonal.
        export_depth_vis: Whether to export raster depth visualisations alongside the glTF.

    Returns:
        Path to the exported ``scene.glb`` file.
    """
    # 1) Use prediction.processed_images, which is already processed image data
    assert (
        prediction.processed_images is not None
    ), "Export to GLB: prediction.processed_images is required but not available"
    assert (
        prediction.depth is not None
    ), "Export to GLB: prediction.depth is required but not available"
    assert (
        prediction.intrinsics is not None
    ), "Export to GLB: prediction.intrinsics is required but not available"
    assert (
        prediction.extrinsics is not None
    ), "Export to GLB: prediction.extrinsics is required but not available"
    assert (
        prediction.conf is not None
    ), "Export to GLB: prediction.conf is required but not available"
    logger.info(f"conf_thresh_percentile: {conf_thresh_percentile}")
    logger.info(f"num max points: {num_max_points}")
    logger.info(f"Exporting to GLB with num_max_points: {num_max_points}")
    if prediction.processed_images is None:
        raise ValueError("prediction.processed_images is required but not available")

    images_u8 = prediction.processed_images  # (N,H,W,3) uint8

    # 2) Sky processing (if sky_mask is provided)
    if getattr(prediction, "sky_mask", None) is not None:
        set_sky_depth(prediction, prediction.sky_mask, sky_depth_def)

    # 3) Confidence threshold (if no conf, then no filtering)
    if filter_black_bg:
        prediction.conf[(prediction.processed_images < 16).all(axis=-1)] = 1.0
    if filter_white_bg:
        prediction.conf[(prediction.processed_images >= 240).all(axis=-1)] = 1.0
    conf_thr = get_conf_thresh(
        prediction,
        getattr(prediction, "sky_mask", None),
        conf_thresh,
        conf_thresh_percentile,
        ensure_thresh_percentile,
    )

    # 4) Back-project to world coordinates and get colors (world frame)
    points, colors = _depths_to_world_points_with_colors(
        prediction.depth,
        prediction.intrinsics,
        prediction.extrinsics,  # w2c
        images_u8,
        prediction.conf,
        conf_thr,
    )

    # 5) Based on first camera orientation + glTF axis system, center by point cloud,
    # construct alignment transform, and apply to point cloud
    A = _compute_alignment_transform_first_cam_glTF_center_by_points(
        prediction.extrinsics[0], points
    )  # (4,4)

    if points.shape[0] > 0:
        points = trimesh.transform_points(points, A)

    # 6) Clean + downsample
    points, colors = _filter_and_downsample(points, colors, num_max_points)

    # 7) Assemble scene (add point cloud first)
    scene = trimesh.Scene()
    if scene.metadata is None:
        scene.metadata = {}
    scene.metadata["hf_alignment"] = A  # For camera wireframes and external reuse

    if points.shape[0] > 0:
        pc = trimesh.points.PointCloud(vertices=points, colors=colors)
        scene.add_geometry(pc)

    # 8) Draw cameras (wireframe pyramids), using the same transform A
    if show_cameras and prediction.intrinsics is not None and prediction.extrinsics is not None:
        scene_scale = _estimate_scene_scale(points, fallback=1.0)
        H, W = prediction.depth.shape[1:]
        _add_cameras_to_scene(
            scene=scene,
            K=prediction.intrinsics,
            ext_w2c=prediction.extrinsics,
            image_sizes=[(H, W)] * prediction.depth.shape[0],
            scale=scene_scale * camera_size,
        )

    # 9) Export
    os.makedirs(export_dir, exist_ok=True)
    out_path = os.path.join(export_dir, "scene.glb")
    scene.export(out_path)

    if export_depth_vis:
        export_to_depth_vis(prediction, export_dir)
        os.system(f"cp -r {export_dir}/depth_vis/0000.jpg {export_dir}/scene.jpg")
    return out_path


# =========================
# utilities
# =========================


def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    """
    Accept (4,4) or (3,4) extrinsic parameters, return (4,4) homogeneous matrix.
    """
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        H = np.eye(4, dtype=ext.dtype)
        H[:3, :4] = ext
        return H
    raise ValueError(f"extrinsic must be (4,4) or (3,4), got {ext.shape}")


def _depths_to_world_points_with_colors(
    depth: np.ndarray,
    K: np.ndarray,
    ext_w2c: np.ndarray,
    images_u8: np.ndarray,
    conf: np.ndarray | None,
    conf_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each frame, transform (u,v,1) through K^{-1} to get rays,
    multiply by depth to camera frame, then use (w2c)^{-1} to transform to world frame.
    Simultaneously extract colors.
    """
    N, H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)  # (H*W,3)

    pts_all, col_all = [], []

    for i in range(N):
        d = depth[i]  # (H,W)
        valid = np.isfinite(d) & (d > 0)
        if conf is not None:
            valid &= conf[i] >= conf_thr
        if not np.any(valid):
            continue

        d_flat = d.reshape(-1)
        vidx = np.flatnonzero(valid.reshape(-1))

        K_inv = np.linalg.inv(K[i])  # (3,3)
        c2w = np.linalg.inv(_as_homogeneous44(ext_w2c[i]))  # (4,4)

        rays = K_inv @ pix[vidx].T  # (3,M)
        Xc = rays * d_flat[vidx][None, :]  # (3,M)
        Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
        Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)  # (M,3)

        cols = images_u8[i].reshape(-1, 3)[vidx].astype(np.uint8)  # (M,3)

        pts_all.append(Xw)
        col_all.append(cols)

    if len(pts_all) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    return np.concatenate(pts_all, 0), np.concatenate(col_all, 0)


def _filter_and_downsample(points: np.ndarray, colors: np.ndarray, num_max: int):
    if points.shape[0] == 0:
        return points, colors
    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]
    if points.shape[0] > num_max:
        idx = np.random.choice(points.shape[0], num_max, replace=False)
        points, colors = points[idx], colors[idx]
    return points, colors


def _estimate_scene_scale(points: np.ndarray, fallback: float = 1.0) -> float:
    if points.shape[0] < 2:
        return fallback
    lo = np.percentile(points, 5, axis=0)
    hi = np.percentile(points, 95, axis=0)
    diag = np.linalg.norm(hi - lo)
    return float(diag if np.isfinite(diag) and diag > 0 else fallback)


def _compute_alignment_transform_first_cam_glTF_center_by_points(
    ext_w2c0: np.ndarray,
    points_world: np.ndarray,
) -> np.ndarray:
    """Computes the transformation matrix to align the scene with glTF standards.

    This function calculates a 4x4 homogeneous matrix that centers the scene's
    point cloud and transforms its coordinate system from the computer vision (CV)
    standard to the glTF standard.

    The transformation process involves three main steps:
    1.  **Initial Alignment**: Orients the world coordinate system to match the
        first camera's view (x-right, y-down, z-forward).
    2.  **Coordinate System Conversion**: Converts the CV camera frame to the
        glTF frame (x-right, y-up, z-backward) by flipping the Y and Z axes.
    3.  **Centering**: Translates the entire scene so that the median of the
        point cloud becomes the new origin (0,0,0).

    Returns:
        A 4x4 homogeneous transformation matrix (torch.Tensor or np.ndarray)
        that applies these transformations.  A: X' = A @ [X;1]
    """

    w2c0 = _as_homogeneous44(ext_w2c0).astype(np.float64)

    # CV -> glTF axis transformation
    M = np.eye(4, dtype=np.float64)
    M[1, 1] = -1.0  # flip Y
    M[2, 2] = -1.0  # flip Z

    # Don't center first
    A_no_center = M @ w2c0

    # Calculate point cloud center in new coordinate system (use median to resist outliers)
    if points_world.shape[0] > 0:
        pts_tmp = trimesh.transform_points(points_world, A_no_center)
        center = np.median(pts_tmp, axis=0)
    else:
        center = np.zeros(3, dtype=np.float64)

    T_center = np.eye(4, dtype=np.float64)
    T_center[:3, 3] = -center

    A = T_center @ A_no_center
    return A


def _add_cameras_to_scene(
    scene: trimesh.Scene,
    K: np.ndarray,
    ext_w2c: np.ndarray,
    image_sizes: list[tuple[int, int]],
    scale: float,
) -> None:
    """Draws camera frustums to visualize their position and orientation.

    This function renders each camera as a wireframe pyramid, originating from
    the camera's center and extending to the corners of its imaging plane.

    It reads the 'hf_alignment' metadata from the scene to ensure the
    wireframes are correctly aligned with the 3D point cloud.
    """
    N = K.shape[0]
    if N == 0:
        return

    # Alignment matrix consistent with point cloud (use identity matrix if missing)
    A = None
    try:
        A = scene.metadata.get("hf_alignment", None) if scene.metadata else None
    except Exception:
        A = None
    if A is None:
        A = np.eye(4, dtype=np.float64)

    for i in range(N):
        H, W = image_sizes[i]
        segs = _camera_frustum_lines(K[i], ext_w2c[i], W, H, scale)  # (8,2,3) world frame
        # Apply unified transformation
        segs = trimesh.transform_points(segs.reshape(-1, 3), A).reshape(-1, 2, 3)
        path = trimesh.load_path(segs)
        color = _index_color_rgb(i, N)
        if hasattr(path, "colors"):
            path.colors = np.tile(color, (len(path.entities), 1))
        scene.add_geometry(path)


def _camera_frustum_lines(
    K: np.ndarray, ext_w2c: np.ndarray, W: int, H: int, scale: float
) -> np.ndarray:
    corners = np.array(
        [
            [0, 0, 1.0],
            [W - 1, 0, 1.0],
            [W - 1, H - 1, 1.0],
            [0, H - 1, 1.0],
        ],
        dtype=float,
    )  # (4,3)

    K_inv = np.linalg.inv(K)
    c2w = np.linalg.inv(_as_homogeneous44(ext_w2c))

    # camera center in world
    Cw = (c2w @ np.array([0, 0, 0, 1.0]))[:3]

    # rays -> z=1 plane points (camera frame)
    rays = (K_inv @ corners.T).T
    z = rays[:, 2:3]
    z[z == 0] = 1.0
    plane_cam = (rays / z) * scale  # (4,3)

    # to world
    plane_w = []
    for p in plane_cam:
        pw = (c2w @ np.array([p[0], p[1], p[2], 1.0]))[:3]
        plane_w.append(pw)
    plane_w = np.stack(plane_w, 0)  # (4,3)

    segs = []
    # center to corners
    for k in range(4):
        segs.append(np.stack([Cw, plane_w[k]], 0))
    # rectangle edges
    order = [0, 1, 2, 3, 0]
    for a, b in zip(order[:-1], order[1:]):
        segs.append(np.stack([plane_w[a], plane_w[b]], 0))

    return np.stack(segs, 0)  # (8,2,3)


def _index_color_rgb(i: int, n: int) -> np.ndarray:
    h = (i + 0.5) / max(n, 1)
    s, v = 0.85, 0.95
    r, g, b = _hsv_to_rgb(h, s, v)
    return (np.array([r, g, b]) * 255).astype(np.uint8)


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return r, g, b
