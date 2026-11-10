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

"""Camera models: project 3D points to pixels and back, through lenses and stereo rigs.

Points are in the camera frame, +X right, +Y down and +Z forward, unless a function takes extrinsics. Pixels are
``(u, v)`` = (column, row) with integer pixel centres, so a centred principal point is ``cx = (W - 1) / 2``.
Normalized coordinates ``(x / z, y / z)`` are where a camera ray meets the ``z = 1`` plane. Intrinsics come as a
``(*, 3, 3)`` ``K``, as a flat ``[fx, fy, cx, cy, ...]`` vector, or, in :class:`PinholeCamera`, as ``(B, 4, 4)``
matrices next to world-to-camera extrinsics. Depth is the camera-frame ``z`` unless a ``normalize`` flag reads it as
the Euclidean ray length. The camera-conventions page of the documentation compares these with other libraries.

Which function to use:

- Camera-frame points to pixels with a ``3x3`` ``K``: :func:`project_points`; pixels at given depths back to
  camera-frame points: :func:`unproject_points`.
- World points to pixels and back through cameras with extrinsics: :class:`PinholeCamera`. A depth map to
  camera-frame points, and those to another camera's pixels, with its ``(B, 4, 4)`` matrices: :func:`pixel2cam`
  and :func:`cam2pixel`.
- Camera-frame points to the image plane and back, without intrinsics, to compose a camera model:
  :func:`project_points_z1` and :func:`unproject_points_z1` (perspective, onto the normalized ``z = 1`` plane),
  :func:`project_points_orthographic` and :func:`unproject_points_orthographic` (dropping ``z``); their derivatives
  :func:`dx_project_points_z1` (the ``(..., 2, 3)`` Jacobian) and :func:`dx_project_points_orthographic`
  (``du/dx`` alone).
- Normalized points to pixels and back with a flat parameter vector: :func:`distort_points_affine` and
  :func:`undistort_points_affine` (``[fx, fy, cx, cy]``, no lens distortion), :func:`distort_points_kannala_brandt`
  and :func:`undistort_points_kannala_brandt` (fisheye, eight parameters); their Jacobians
  :func:`dx_distort_points_affine` and :func:`dx_distort_points_kannala_brandt`. For OpenCV's Brown-Conrady
  coefficients on pixel coordinates, use :mod:`kornia.geometry.calibration`.
- 3D points from a rectified stereo disparity map: :class:`StereoCamera`, from the two projection matrices, or
  :func:`reproject_disparity_to_3D`, from a ``Q`` matrix.
"""

from .distortion_affine import distort_points_affine, dx_distort_points_affine, undistort_points_affine
from .distortion_kannala_brandt import (
    distort_points_kannala_brandt,
    dx_distort_points_kannala_brandt,
    undistort_points_kannala_brandt,
)
from .perspective import project_points, unproject_points
from .pinhole import PinholeCamera, cam2pixel, pixel2cam
from .projection_orthographic import (
    dx_project_points_orthographic,
    project_points_orthographic,
    unproject_points_orthographic,
)
from .projection_z1 import dx_project_points_z1, project_points_z1, unproject_points_z1
from .stereo import StereoCamera, reproject_disparity_to_3D

__all__ = [
    "PinholeCamera",
    "StereoCamera",
    "cam2pixel",
    "distort_points_affine",
    "distort_points_kannala_brandt",
    "dx_distort_points_affine",
    "dx_distort_points_kannala_brandt",
    "dx_project_points_orthographic",
    "dx_project_points_z1",
    "pixel2cam",
    "project_points",
    "project_points_orthographic",
    "project_points_z1",
    "reproject_disparity_to_3D",
    "undistort_points_affine",
    "undistort_points_kannala_brandt",
    "unproject_points",
    "unproject_points_orthographic",
    "unproject_points_z1",
]
