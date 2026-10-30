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

r"""Two-view epipolar geometry: estimate F and E, recover the relative pose, triangulate.

The two-view functions take the first image's points first (``points1``, ``x1`` or ``pts1``), then the second
image's; the estimators take ``(B, N, 2)`` tensors. A fundamental matrix ``F`` relates pixel coordinates and an
essential matrix ``E`` normalized camera coordinates :math:`K^{-1} [u, v, 1]^\top`, both with
:math:`x_2^\top F x_1 = 0`: the second image's point on the left, as in OpenCV. Camera poses are world-to-camera
extrinsics, :math:`X_{cam} = R X + t`, and the relative motion ``(R, t)`` of camera 2 to camera 1 takes camera-1
coordinates to camera-2 coordinates, :math:`X_2 = R X_1 + t`; an essential matrix is :math:`[t]_\times R` of that
motion. :ref:`Two-view geometry <two-view-conventions>` compares these conventions with OpenCV's.

Which function to use:

- A model from correspondences known to be inliers: :func:`find_fundamental` from pixels (8 or 7 points),
  :func:`find_essential` from normalized camera coordinates (Nister's 5 points, up to ten candidates). For a plane
  or a pure rotation, :func:`~kornia.geometry.homography.find_homography_dlt`; with outliers,
  :class:`~kornia.geometry.ransac.RANSAC`.
- One model from another: :func:`essential_from_fundamental` and :func:`fundamental_from_essential` with the two
  cameras' intrinsics, :func:`essential_from_Rt` from two camera poses, :func:`fundamental_from_projections` from
  two projection matrices, and :func:`project_to_essential` onto the nearest essential matrix.
- The relative pose from an essential matrix: :func:`motion_from_essential_choose_solution` picks it by cheirality
  from pixel correspondences, and :func:`motion_from_essential` lists the four candidates;
  :func:`decompose_essential_matrix` (SVD) and :func:`decompose_essential_matrix_no_svd` (closed form) return the
  two rotations and the translation that they combine. The relative motion of two known camera poses:
  :func:`relative_camera_motion`.
- How well correspondences fit a fundamental matrix: :func:`sampson_epipolar_distance` (first-order geometric
  error), :func:`symmetrical_epipolar_distance` (both points to their epipolar lines),
  :func:`left_to_right_epipolar_distance` (in image 2) and :func:`right_to_left_epipolar_distance` (in image 1).
- Epipolar lines: :func:`compute_correspond_epilines`; the point on a line nearest to a correspondence,
  :func:`get_closest_point_on_epipolar_line`, built on :func:`get_perpendicular`.
- 3D points from two views: :func:`triangulate_points`.
- Cameras: :func:`projection_from_KRt` composes ``P = K [R | t]`` and :func:`KRt_from_projection` decomposes it;
  :func:`projections_from_fundamental` gives a canonical pair for ``F``; :func:`depth_from_point` gives the depth of
  points in a camera. Intrinsics: :func:`intrinsics_like` (from a focal length and an image size),
  :func:`random_intrinsics`, :func:`scale_intrinsics` (for a resized image).
- Helpers of the estimators: :func:`normalize_points` (Hartley normalization), :func:`normalize_transformation`
  (scale to a unit bottom-right entry), :func:`cross_product_matrix` (the matrix :math:`[x]_\times`).
- A random synthetic scene to test estimators on: :func:`generate_scene`.
"""

from ._metrics import (
    left_to_right_epipolar_distance,
    right_to_left_epipolar_distance,
    sampson_epipolar_distance,
    symmetrical_epipolar_distance,
)
from .essential import (
    decompose_essential_matrix,
    decompose_essential_matrix_no_svd,
    essential_from_fundamental,
    essential_from_Rt,
    find_essential,
    motion_from_essential,
    motion_from_essential_choose_solution,
    project_to_essential,
    relative_camera_motion,
)
from .fundamental import (
    compute_correspond_epilines,
    find_fundamental,
    fundamental_from_essential,
    fundamental_from_projections,
    get_closest_point_on_epipolar_line,
    get_perpendicular,
    normalize_points,
    normalize_transformation,
)
from .numeric import cross_product_matrix
from .projection import (
    KRt_from_projection,
    depth_from_point,
    intrinsics_like,
    projection_from_KRt,
    projections_from_fundamental,
    random_intrinsics,
    scale_intrinsics,
)
from .scene import generate_scene
from .triangulation import triangulate_points

__all__ = [
    "KRt_from_projection",
    "compute_correspond_epilines",
    "cross_product_matrix",
    "decompose_essential_matrix",
    "decompose_essential_matrix_no_svd",
    "depth_from_point",
    "essential_from_Rt",
    "essential_from_fundamental",
    "find_essential",
    "find_fundamental",
    "fundamental_from_essential",
    "fundamental_from_projections",
    "generate_scene",
    "get_closest_point_on_epipolar_line",
    "get_perpendicular",
    "intrinsics_like",
    "left_to_right_epipolar_distance",
    "motion_from_essential",
    "motion_from_essential_choose_solution",
    "normalize_points",
    "normalize_transformation",
    "project_to_essential",
    "projection_from_KRt",
    "projections_from_fundamental",
    "random_intrinsics",
    "relative_camera_motion",
    "right_to_left_epipolar_distance",
    "sampson_epipolar_distance",
    "scale_intrinsics",
    "symmetrical_epipolar_distance",
    "triangulate_points",
]
