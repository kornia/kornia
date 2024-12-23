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
