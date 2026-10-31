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

r"""Lens distortion of pixel coordinates with OpenCV's Brown-Conrady model, and camera pose from 3D-2D points.

Image points are pixel coordinates ``(u, v)`` = (column, row) with integer pixel centres, and ``K`` is a ``(*, 3, 3)``
pinhole matrix. ``dist`` is OpenCV's coefficient vector
:math:`(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])` of 4, 5, 8, 12 or 14
elements: radial, tangential, thin-prism and sensor-tilt terms, as ``cv2.projectPoints`` and
``cv2.undistortPoints`` read it.

Which function to use:

- Add the distortion to ideal pixels: :func:`distort_points`. Remove it from measured pixels:
  :func:`undistort_points`; from a whole image: :func:`undistort_image`.
- The sensor-tilt matrix of the last two coefficients, or its inverse: :func:`tilt_projection`.
- A world-to-camera pose from ``N >= 6`` world points and their pixels: :func:`solve_pnp_dlt`.
- The Kannala-Brandt fisheye model, on normalized points and a flat parameter vector:
  :func:`~kornia.geometry.camera.distort_points_kannala_brandt` and
  :func:`~kornia.geometry.camera.undistort_points_kannala_brandt`.
"""

from .distort import distort_points, tilt_projection
from .pnp import solve_pnp_dlt
from .undistort import undistort_image, undistort_points

__all__ = ["distort_points", "solve_pnp_dlt", "tilt_projection", "undistort_image", "undistort_points"]
