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

"""ONNX export survey — area geomB: conversions, linalg, homography, epipolar, camera, calibration, depth,
liegroup, quaternion, ransac, solvers, pointcloud."""

import sys

import torch
from harness import case, run_cases

import kornia as K
from kornia.geometry import conversions as C
from kornia.geometry import epipolar as E
from kornia.geometry import linalg as L
from kornia.geometry.camera import PinholeCamera, StereoCamera
from kornia.geometry.liegroup import Se2, Se3, So2, So3
from kornia.geometry.quaternion import Quaternion

torch.manual_seed(0)

# ----------------------------------------------------------------------------- synthetic geometry
B, N = 1, 12
IMG = torch.rand(1, 3, 32, 40)
Kmat = E.intrinsics_like(50.0, IMG)  # (1,3,3): fx=fy=50, cx=20, cy=16
Kmat2 = E.intrinsics_like(60.0, IMG)


def _rot(axis_angle):
    return C.axis_angle_to_rotation_matrix(torch.tensor([axis_angle]))  # (1,3,3)


R1 = _rot([0.0, 0.0, 0.0])
t1 = torch.zeros(1, 3, 1)
R2 = _rot([0.05, -0.15, 0.08])
t2 = torch.tensor([[[0.6], [0.1], [0.2]]])
P1 = E.projection_from_KRt(Kmat, R1, t1)  # (1,3,4)
P2 = E.projection_from_KRt(Kmat2, R2, t2)
X3d = torch.rand(1, N, 3) * torch.tensor([2.0, 1.5, 1.0]) + torch.tensor([-1.0, -0.75, 3.0])  # in front of cam
pts1 = L.transform_points(P1, X3d)  # (1,N,2)
pts2 = L.transform_points(P2, X3d)
E_gt = E.essential_from_Rt(R1, t1, R2, t2)
F_gt = E.fundamental_from_essential(E_gt, Kmat, Kmat2)
F_gt = F_gt / F_gt[..., 2:3, 2:3]
T01 = C.Rt_to_matrix4x4(R2, t2)  # (1,4,4)
T02 = C.Rt_to_matrix4x4(_rot([0.1, 0.02, -0.03]), torch.tensor([[[0.1], [-0.2], [0.3]]]))

# planar homography scene
H_gt = torch.tensor([[[1.05, 0.02, 2.0], [-0.03, 0.98, -1.0], [0.0005, -0.0002, 1.0]]])
hp1 = torch.rand(1, N, 2) * torch.tensor([40.0, 32.0])
hp2 = L.transform_points(H_gt, hp1)
hp1_4 = hp1[:, :4]
hp2_4 = hp2[:, :4]
ls1 = torch.rand(1, N, 2, 2) * torch.tensor([40.0, 32.0])
ls2 = L.transform_points(H_gt, ls1.reshape(1, -1, 2)).reshape(1, N, 2, 2)
w_ones = torch.ones(1, N)

# rotations / quaternions
aa = torch.tensor([[0.1, -0.2, 0.3], [1.0, 0.5, -0.4]])  # (2,3) axis-angle
Rm = C.axis_angle_to_rotation_matrix(aa)  # (2,3,3)
q = C.normalize_quaternion(torch.tensor([[0.9, 0.1, -0.2, 0.3], [0.2, 0.7, 0.1, -0.5]]))  # (2,4) wxyz
qb = C.normalize_quaternion(torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.8, -0.1, 0.4, 0.2]]))
euler = (torch.tensor([0.1, -0.4]), torch.tensor([0.2, 0.3]), torch.tensor([-0.3, 1.0]))


# pinhole cameras
def _pinhole(k_in, r_in, t_in):
    intr = torch.eye(4)[None].clone()
    intr[:, :3, :3] = k_in
    extr = C.Rt_to_matrix4x4(r_in, t_in)
    return PinholeCamera(intr, extr, torch.tensor([32]), torch.tensor([40]))


cam1 = _pinhole(Kmat, R1, t1)
cam2 = _pinhole(Kmat2, R2, t2)
depth_img = torch.rand(1, 1, 32, 40) * 2.0 + 2.0

# stereo
left34 = torch.zeros(1, 3, 4)
left34[:, :3, :3] = Kmat
right34 = left34.clone()
right34[:, 0, 3] = -50.0 * 0.1  # fx * baseline (negative)
disparity = torch.rand(1, 32, 40, 1) * 5.0 + 1.0

# distortion (OpenCV model)
dist5 = torch.tensor([[0.1, -0.02, 0.001, 0.002, 0.005]])
dist14 = torch.tensor([[0.1, -0.02, 0.001, 0.002, 0.005, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.01, 0.02]])
img_pts = torch.rand(1, N, 2) * torch.tensor([40.0, 32.0])
kb_params = torch.tensor([50.0, 50.0, 20.0, 16.0, 0.1, 0.01, 0.001, 0.0001])
aff_params = torch.tensor([50.0, 50.0, 20.0, 16.0])
z1_pts = torch.rand(N, 2) * 0.6 - 0.3
cam_pts3 = torch.rand(N, 3) + torch.tensor([-0.5, -0.5, 2.0])

# liegroup parameter vectors
so3_v = torch.tensor([[0.1, -0.2, 0.3], [0.5, 0.4, -0.1]])
se3_v = torch.cat([torch.tensor([[0.3, -0.1, 0.2], [1.0, 0.5, -0.5]]), so3_v], dim=-1)  # (2,6) [t, omega]
so2_th = torch.tensor([0.3, -1.2])
se2_v = torch.tensor([[0.5, -0.2, 0.3], [1.0, 0.7, -1.2]])  # (2,3) [x, y, theta]
So3M = So3.exp(so3_v).matrix()
Se3M = Se3.exp(se3_v).matrix()
So2M = So2.exp(so2_th).matrix()
Se2M = Se2.exp(se2_v).matrix()


def _cc(name, fn, inputs, kwargs=None, **kw):
    return case("geometry.conversions." + name, "geometry.conversions", fn, inputs, kwargs, **kw)


def _cl(name, fn, inputs, kwargs=None, **kw):
    return case("geometry.linalg." + name, "geometry.linalg", fn, inputs, kwargs, **kw)


def _ch(name, fn, inputs, kwargs=None, **kw):
    return case("geometry.homography." + name, "geometry.homography", fn, inputs, kwargs, **kw)


def _ce(name, fn, inputs, kwargs=None, **kw):
    return case("geometry.epipolar." + name, "geometry.epipolar", fn, inputs, kwargs, **kw)


def _cam(name, fn, inputs, kwargs=None, **kw):
    return case("geometry.camera." + name, "geometry.camera", fn, inputs, kwargs, **kw)


def _cal(name, fn, inputs, kwargs=None, **kw):
    return case("geometry.calibration." + name, "geometry.calibration", fn, inputs, kwargs, **kw)


def _cd(name, fn, inputs, kwargs=None, **kw):
    return case("geometry.depth." + name, "geometry.depth", fn, inputs, kwargs, **kw)


def _clg(name, fn, inputs, kwargs=None, **kw):
    return case("geometry.liegroup." + name, "geometry.liegroup", fn, inputs, kwargs, **kw)


def _cq(name, fn, inputs, kwargs=None, **kw):
    return case("geometry.quaternion." + name, "geometry.quaternion", fn, inputs, kwargs, **kw)


def _cs(name, fn, inputs, kwargs=None, **kw):
    return case("geometry.solvers." + name, "geometry.solvers", fn, inputs, kwargs, **kw)


CASES = [
    # ------------------------------------------------------------------ conversions
    _cc("rad2deg", C.rad2deg, [torch.rand(2, 3) * 6.0]),
    _cc("deg2rad", C.deg2rad, [torch.rand(2, 3) * 360.0]),
    _cc("pol2cart", C.pol2cart, [torch.rand(2, 5) + 0.5, torch.rand(2, 5) * 6.0]),
    _cc("cart2pol", C.cart2pol, [torch.randn(2, 5), torch.randn(2, 5)]),
    _cc("angle_to_rotation_matrix", C.angle_to_rotation_matrix, [torch.rand(2, 3) * 360.0], note="angle in degrees"),
    _cc("convert_points_from_homogeneous", C.convert_points_from_homogeneous, [torch.rand(1, N, 3) + 0.5]),
    _cc("convert_points_to_homogeneous", C.convert_points_to_homogeneous, [torch.rand(1, N, 2)]),
    _cc("convert_affinematrix_to_homography", C.convert_affinematrix_to_homography, [torch.rand(2, 2, 3)]),
    _cc(
        "convert_affinematrix_to_homography3d",
        C.convert_affinematrix_to_homography3d,
        [torch.rand(2, 3, 4)],
        tags=("3d",),
    ),
    _cc(
        "normalize_pixel_coordinates",
        C.normalize_pixel_coordinates,
        [img_pts],
        {"height": 32, "width": 40},
        note="H,W baked",
    ),
    _cc(
        "denormalize_pixel_coordinates",
        C.denormalize_pixel_coordinates,
        [torch.rand(1, N, 2) * 2 - 1],
        {"height": 32, "width": 40},
        note="H,W baked",
    ),
    _cc(
        "normalize_pixel_coordinates3d",
        C.normalize_pixel_coordinates3d,
        [torch.rand(1, N, 3) * torch.tensor([40.0, 32.0, 8.0])],
        {"depth": 8, "height": 32, "width": 40},
        note="D,H,W baked",
        tags=("3d",),
    ),
    _cc(
        "denormalize_pixel_coordinates3d",
        C.denormalize_pixel_coordinates3d,
        [torch.rand(1, N, 3) * 2 - 1],
        {"depth": 8, "height": 32, "width": 40},
        note="D,H,W baked",
        tags=("3d",),
    ),
    _cc("normalize_points_with_intrinsics", C.normalize_points_with_intrinsics, [img_pts, Kmat]),
    _cc("denormalize_points_with_intrinsics", C.denormalize_points_with_intrinsics, [torch.rand(1, N, 2) - 0.5, Kmat]),
    _cc(
        "normalize_homography",
        C.normalize_homography,
        [H_gt],
        {"dsize_src": (32, 40), "dsize_dst": (32, 40)},
        note="dsize baked",
    ),
    _cc(
        "denormalize_homography",
        C.denormalize_homography,
        [H_gt],
        {"dsize_src": (32, 40), "dsize_dst": (32, 40)},
        note="dsize baked",
    ),
    _cc(
        "normalize_homography3d",
        C.normalize_homography3d,
        [torch.eye(4)[None] + torch.rand(1, 4, 4) * 0.01],
        {"dsize_src": (8, 32, 40), "dsize_dst": (8, 32, 40)},
        note="dsize baked",
        tags=("3d",),
    ),
    _cc(
        "normal_transform_pixel",
        lambda d: C.normal_transform_pixel(32, 40, device=d.device, dtype=d.dtype),
        [torch.zeros(1)],
        note="H,W baked; graph fully constant",
    ),
    _cc(
        "normal_transform_pixel3d",
        lambda d: C.normal_transform_pixel3d(8, 32, 40, device=d.device, dtype=d.dtype),
        [torch.zeros(1)],
        note="D,H,W baked; graph fully constant",
        tags=("3d",),
    ),
    _cc("quaternion_to_axis_angle", C.quaternion_to_axis_angle, [q]),
    _cc("quaternion_to_angle_axis", None, [], skip="deprecated alias of quaternion_to_axis_angle"),
    _cc("quaternion_to_rotation_matrix", C.quaternion_to_rotation_matrix, [q]),
    _cc("quaternion_log_to_exp", C.quaternion_log_to_exp, [aa]),
    _cc("quaternion_exp_to_log", C.quaternion_exp_to_log, [q]),
    _cc("normalize_quaternion", C.normalize_quaternion, [torch.rand(2, 4) + 0.1]),
    _cc("vector_to_skew_symmetric_matrix", C.vector_to_skew_symmetric_matrix, [torch.rand(2, 3)]),
    _cc("rotation_matrix_to_axis_angle", C.rotation_matrix_to_axis_angle, [Rm]),
    _cc("rotation_matrix_to_angle_axis", None, [], skip="deprecated alias of rotation_matrix_to_axis_angle"),
    _cc("rotation_matrix_to_quaternion", C.rotation_matrix_to_quaternion, [Rm]),
    _cc("axis_angle_to_quaternion", C.axis_angle_to_quaternion, [aa]),
    _cc("angle_axis_to_quaternion", None, [], skip="deprecated alias of axis_angle_to_quaternion"),
    _cc("axis_angle_to_rotation_matrix", C.axis_angle_to_rotation_matrix, [aa]),
    _cc("angle_axis_to_rotation_matrix", None, [], skip="deprecated alias of axis_angle_to_rotation_matrix"),
    _cc("quaternion_from_euler", C.quaternion_from_euler, list(euler)),
    _cc(
        "euler_from_quaternion", lambda qq: C.euler_from_quaternion(qq[..., 0], qq[..., 1], qq[..., 2], qq[..., 3]), [q]
    ),
    _cc("Rt_to_matrix4x4", C.Rt_to_matrix4x4, [R2, t2]),
    _cc("matrix4x4_to_Rt", C.matrix4x4_to_Rt, [T01]),
    _cc("worldtocam_to_camtoworld_Rt", C.worldtocam_to_camtoworld_Rt, [R2, t2]),
    _cc("camtoworld_to_worldtocam_Rt", C.camtoworld_to_worldtocam_Rt, [R2, t2]),
    _cc("camtoworld_graphics_to_vision_4x4", C.camtoworld_graphics_to_vision_4x4, [T01]),
    _cc("camtoworld_vision_to_graphics_4x4", C.camtoworld_vision_to_graphics_4x4, [T01]),
    _cc("camtoworld_graphics_to_vision_Rt", C.camtoworld_graphics_to_vision_Rt, [R2, t2]),
    _cc("camtoworld_vision_to_graphics_Rt", C.camtoworld_vision_to_graphics_Rt, [R2, t2]),
    _cc("ARKitQTVecs_to_ColmapQTVecs", C.ARKitQTVecs_to_ColmapQTVecs, [q[:1], torch.tensor([[[0.1], [0.2], [0.3]]])]),
    # ------------------------------------------------------------------ linalg
    _cl("relative_transformation", L.relative_transformation, [T01, T02]),
    _cl("compose_transformations", L.compose_transformations, [T01, T02]),
    _cl("inverse_transformation", L.inverse_transformation, [T01]),
    _cl("transform_points", L.transform_points, [T01, X3d]),
    _cl("transform_points[2d-homography]", L.transform_points, [H_gt, hp1]),
    _cl("point_line_distance", L.point_line_distance, [hp1, torch.rand(1, N, 3) + 0.1]),
    _cl("squared_norm", L.squared_norm, [X3d]),
    _cl("batched_dot_product", L.batched_dot_product, [X3d, X3d + 0.1]),
    _cl("batched_squared_norm", L.batched_squared_norm, [X3d]),
    _cl("euclidean_distance", L.euclidean_distance, [X3d, X3d + 0.1]),
    # ------------------------------------------------------------------ homography
    _ch(
        "find_homography_dlt",
        K.geometry.homography.find_homography_dlt,
        [hp1, hp2],
        note="solver=lu (default)",
        atol=1e-3,
    ),
    _ch(
        "find_homography_dlt[svd]", K.geometry.homography.find_homography_dlt, [hp1, hp2], {"solver": "svd"}, atol=1e-3
    ),
    _ch("find_homography_dlt[weights]", K.geometry.homography.find_homography_dlt, [hp1, hp2, w_ones], atol=1e-3),
    _ch(
        "find_homography_dlt_iterated",
        K.geometry.homography.find_homography_dlt_iterated,
        [hp1, hp2, w_ones],
        {"n_iter": 3},
        note="n_iter baked",
        atol=1e-3,
    ),
    _ch("find_homography_lines_dlt", K.geometry.homography.find_homography_lines_dlt, [ls1, ls2], atol=1e-3),
    _ch(
        "find_homography_lines_dlt_iterated",
        K.geometry.homography.find_homography_lines_dlt_iterated,
        [ls1, ls2, w_ones],
        {"n_iter": 3},
        note="n_iter baked",
        atol=1e-3,
    ),
    _ch(
        "line_segment_transfer_error_one_way",
        K.geometry.homography.line_segment_transfer_error_one_way,
        [ls1, ls2, H_gt],
    ),
    _ch("oneway_transfer_error", K.geometry.homography.oneway_transfer_error, [hp1, hp2, H_gt]),
    _ch("symmetric_transfer_error", K.geometry.homography.symmetric_transfer_error, [hp1, hp2, H_gt]),
    _ch(
        "sample_is_valid_for_homography",
        K.geometry.homography.sample_is_valid_for_homography,
        [hp1_4, hp2_4],
        note="returns bool (B,)",
    ),
    _ch("normalize_points", K.geometry.homography.normalize_points, [hp1]),
    _ch("safe_inverse_with_mask", K.geometry.homography.safe_inverse_with_mask, [H_gt]),
    _ch("safe_solve_with_mask", K.geometry.homography.safe_solve_with_mask, [torch.rand(1, 3, 1), H_gt]),
    # ------------------------------------------------------------------ epipolar
    _ce("find_fundamental", E.find_fundamental, [pts1, pts2], note="8POINT", atol=1e-3),
    _ce("find_fundamental[weights]", E.find_fundamental, [pts1, pts2, w_ones], atol=1e-3),
    _ce(
        "find_fundamental[7POINT]",
        E.find_fundamental,
        [pts1[:, :7], pts2[:, :7]],
        {"method": "7POINT"},
        note="returns up to 3 F (B,3*3,3)",
        atol=1e-3,
    ),
    _ce(
        "find_essential", E.find_essential, [pts1[:, :5], pts2[:, :5]], note="5pt Nister, returns (B,10,3,3)", atol=1e-3
    ),
    _ce("find_essential[weights]", E.find_essential, [pts1[:, :5], pts2[:, :5], w_ones[:, :5]], atol=1e-3),
    _ce("project_to_essential", E.project_to_essential, [E_gt + torch.rand(1, 3, 3) * 0.01]),
    _ce("essential_from_fundamental", E.essential_from_fundamental, [F_gt, Kmat, Kmat2]),
    _ce("essential_from_Rt", E.essential_from_Rt, [R1, t1, R2, t2]),
    _ce(
        "decompose_essential_matrix", E.decompose_essential_matrix, [E_gt], note="SVD-based; sign ambiguity", atol=1e-3
    ),
    _ce("decompose_essential_matrix_no_svd", E.decompose_essential_matrix_no_svd, [E_gt], atol=1e-3),
    _ce("motion_from_essential", E.motion_from_essential, [E_gt], atol=1e-3),
    _ce(
        "motion_from_essential_choose_solution",
        E.motion_from_essential_choose_solution,
        [E_gt, Kmat, Kmat2, pts1, pts2],
        atol=1e-3,
    ),
    _ce(
        "motion_from_essential_choose_solution[mask]",
        E.motion_from_essential_choose_solution,
        [E_gt, Kmat, Kmat2, pts1, pts2, torch.ones(1, N, dtype=torch.bool)],
        atol=1e-3,
    ),
    _ce("relative_camera_motion", E.relative_camera_motion, [R1, t1, R2, t2]),
    _ce("fundamental_from_essential", E.fundamental_from_essential, [E_gt, Kmat, Kmat2]),
    _ce("fundamental_from_projections", E.fundamental_from_projections, [P1, P2]),
    _ce("compute_correspond_epilines", E.compute_correspond_epilines, [pts1, F_gt]),
    _ce("normalize_points", E.normalize_points, [pts1]),
    _ce("normalize_transformation", E.normalize_transformation, [F_gt * 3.0]),
    _ce("get_perpendicular", E.get_perpendicular, [torch.rand(1, N, 3) + 0.1, pts1]),
    _ce("get_closest_point_on_epipolar_line", E.get_closest_point_on_epipolar_line, [pts1, pts2, F_gt]),
    _ce("sampson_epipolar_distance", E.sampson_epipolar_distance, [pts1, pts2, F_gt]),
    _ce(
        "sampson_epipolar_distance[no-matmul]",
        E.sampson_epipolar_distance,
        [pts1, pts2, F_gt],
        {"use_matmul_at_less_than_points": 0},
        note="einsum path",
    ),
    _ce("symmetrical_epipolar_distance", E.symmetrical_epipolar_distance, [pts1, pts2, F_gt]),
    _ce("left_to_right_epipolar_distance", E.left_to_right_epipolar_distance, [pts1, pts2, F_gt]),
    _ce("right_to_left_epipolar_distance", E.right_to_left_epipolar_distance, [pts1, pts2, F_gt]),
    _ce("projection_from_KRt", E.projection_from_KRt, [Kmat, R2, t2]),
    _ce("projections_from_fundamental", E.projections_from_fundamental, [F_gt], atol=1e-3),
    _ce("KRt_from_projection", E.KRt_from_projection, [P2], atol=1e-3),
    _ce("depth_from_point", E.depth_from_point, [R2, t2, X3d]),
    _ce("intrinsics_like", lambda im: E.intrinsics_like(50.0, im), [IMG], note="focal baked; only shape of image used"),
    _ce("scale_intrinsics", E.scale_intrinsics, [Kmat, torch.tensor([0.5])]),
    _ce("scale_intrinsics[float]", E.scale_intrinsics, [Kmat], {"scale_factor": 0.5}, note="scale baked"),
    _ce(
        "random_intrinsics",
        lambda d: E.random_intrinsics(d, d + 100.0),
        [torch.tensor(1.0)],
        check=False,
        note="uses torch.rand inside",
    ),
    _ce("cross_product_matrix", E.cross_product_matrix, [torch.rand(2, 3)]),
    _ce("triangulate_points", E.triangulate_points, [P1, P2, pts1, pts2], note="solver=eigh (default)", atol=1e-3),
    _ce("triangulate_points[svd]", E.triangulate_points, [P1, P2, pts1, pts2], {"solver": "svd"}, atol=1e-3),
    _ce("generate_scene", None, [], skip="random scene generator with int-only args (returns dict of random tensors)"),
    # ------------------------------------------------------------------ camera
    _cam("project_points", K.geometry.camera.project_points, [X3d, Kmat]),
    _cam("unproject_points", K.geometry.camera.unproject_points, [pts1, X3d[..., 2:3], Kmat]),
    _cam(
        "unproject_points[normalize]",
        K.geometry.camera.unproject_points,
        [pts1, X3d[..., 2:3], Kmat],
        {"normalize": True},
    ),
    _cam(
        "cam2pixel",
        K.geometry.camera.cam2pixel,
        [torch.rand(1, 32, 40, 3) + torch.tensor([0.0, 0.0, 2.0]), cam1.intrinsics],
    ),
    _cam(
        "pixel2cam",
        K.geometry.camera.pixel2cam,
        [
            depth_img,
            cam1.intrinsics_inverse(),
            torch.cat([K.geometry.camera.stereo.create_meshgrid(32, 40, False), torch.ones(1, 32, 40, 1)], -1),
        ],
    ),
    _cam(
        "PinholeCamera.project",
        lambda intr, extr, p: PinholeCamera(intr, extr, torch.tensor([32]), torch.tensor([40])).project(p),
        [cam2.intrinsics, cam2.extrinsics, X3d],
    ),
    _cam(
        "PinholeCamera.unproject",
        lambda intr, extr, p, d: PinholeCamera(intr, extr, torch.tensor([32]), torch.tensor([40])).unproject(p, d),
        [cam2.intrinsics, cam2.extrinsics, pts2, X3d[..., 2:3]],
        atol=1e-3,
    ),
    _cam(
        "PinholeCamera.intrinsics_inverse",
        lambda intr, extr: PinholeCamera(intr, extr, torch.tensor([32]), torch.tensor([40])).intrinsics_inverse(),
        [cam2.intrinsics, cam2.extrinsics],
    ),
    _cam(
        "PinholeCamera.scale",
        lambda intr, extr, s: PinholeCamera(intr, extr, torch.tensor([32]), torch.tensor([40])).scale(s).intrinsics,
        [cam2.intrinsics, cam2.extrinsics, torch.tensor([0.5])],
    ),
    _cam(
        "PinholeCamera.properties",
        lambda intr, extr: (
            lambda c: (
                c.fx,
                c.fy,
                c.cx,
                c.cy,
                c.tx,
                c.ty,
                c.tz,
                c.rt_matrix,
                c.camera_matrix,
                c.rotation_matrix,
                c.translation_vector,
            )
        )(PinholeCamera(intr, extr, torch.tensor([32]), torch.tensor([40]))),
        [cam2.intrinsics, cam2.extrinsics],
    ),
    _cam(
        "pinhole.pinhole_matrix",
        K.geometry.camera.pinhole.pinhole_matrix,
        [torch.tensor([[50.0, 50.0, 20.0, 16.0, 32.0, 40.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0]])],
        note="legacy 12-vector pinhole",
    ),
    _cam(
        "pinhole.inverse_pinhole_matrix",
        K.geometry.camera.pinhole.inverse_pinhole_matrix,
        [torch.tensor([[50.0, 50.0, 20.0, 16.0, 32.0, 40.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0]])],
    ),
    _cam(
        "pinhole.scale_pinhole",
        K.geometry.camera.pinhole.scale_pinhole,
        [torch.tensor([[50.0, 50.0, 20.0, 16.0, 32.0, 40.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0]]), torch.tensor([0.5])],
    ),
    _cam(
        "pinhole.get_optical_pose_base",
        K.geometry.camera.pinhole.get_optical_pose_base,
        [torch.tensor([[50.0, 50.0, 20.0, 16.0, 32.0, 40.0, 0.1, 0.2, 0.3, 0.1, -0.1, 0.05]])],
        note="KORNIA BUG: function body is `raise NotImplementedError` (pinhole.py:628)",
    ),
    _cam(
        "pinhole.homography_i_H_ref",
        K.geometry.camera.pinhole.homography_i_H_ref,
        [
            torch.tensor([[50.0, 50.0, 20.0, 16.0, 32.0, 40.0, 0.1, 0.2, 0.3, 0.1, -0.1, 0.05]]),
            torch.tensor([[50.0, 50.0, 20.0, 16.0, 32.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        ],
        note="KORNIA BUG: calls get_optical_pose_base which is a NotImplementedError stub",
    ),
    _cam(
        "StereoCamera.reproject_disparity_to_3D",
        lambda l, r, d: StereoCamera(l, r).reproject_disparity_to_3D(d),
        [left34, right34, disparity],
    ),
    _cam("StereoCamera.Q", lambda l, r: StereoCamera(l, r).Q, [left34, right34]),
    _cam(
        "stereo.reproject_disparity_to_3D",
        K.geometry.camera.stereo.reproject_disparity_to_3D,
        [disparity, StereoCamera(left34, right34).Q],
    ),
    _cam("project_points_z1", K.geometry.camera.project_points_z1, [cam_pts3]),
    _cam("unproject_points_z1", K.geometry.camera.unproject_points_z1, [z1_pts]),
    _cam("unproject_points_z1[extension]", K.geometry.camera.unproject_points_z1, [z1_pts, torch.rand(N) + 1.0]),
    _cam("dx_project_points_z1", K.geometry.camera.dx_project_points_z1, [cam_pts3]),
    _cam("project_points_orthographic", K.geometry.camera.project_points_orthographic, [cam_pts3]),
    _cam(
        "unproject_points_orthographic", K.geometry.camera.unproject_points_orthographic, [z1_pts, torch.rand(N) + 1.0]
    ),
    _cam("dx_project_points_orthographic", K.geometry.camera.dx_project_points_orthographic, [cam_pts3]),
    _cam("distort_points_affine", K.geometry.camera.distort_points_affine, [z1_pts, aff_params]),
    _cam("undistort_points_affine", K.geometry.camera.undistort_points_affine, [img_pts[0], aff_params]),
    _cam("dx_distort_points_affine", K.geometry.camera.dx_distort_points_affine, [z1_pts, aff_params]),
    _cam("distort_points_kannala_brandt", K.geometry.camera.distort_points_kannala_brandt, [z1_pts, kb_params]),
    _cam(
        "undistort_points_kannala_brandt",
        K.geometry.camera.undistort_points_kannala_brandt,
        [img_pts[0], kb_params],
        note="10 unrolled Gauss-Newton iters",
    ),
    _cam("dx_distort_points_kannala_brandt", K.geometry.camera.dx_distort_points_kannala_brandt, [z1_pts, kb_params]),
    # ------------------------------------------------------------------ calibration
    _cal("distort_points", K.geometry.calibration.distort_points, [img_pts, Kmat, dist5]),
    _cal(
        "distort_points[14]",
        K.geometry.calibration.distort_points,
        [img_pts, Kmat, dist14],
        note="14-coeff model incl. tilt",
    ),
    _cal("distort_points[new_K]", K.geometry.calibration.distort_points, [img_pts, Kmat, dist5, Kmat2]),
    _cal(
        "undistort_points", K.geometry.calibration.undistort_points, [img_pts, Kmat, dist5], note="num_iters=5 default"
    ),
    _cal("undistort_points[14]", K.geometry.calibration.undistort_points, [img_pts, Kmat, dist14]),
    _cal("undistort_image", K.geometry.calibration.undistort_image, [IMG, Kmat[0], dist5[0]]),
    _cal(
        "undistort_image[batch>1]",
        K.geometry.calibration.undistort_image,
        [torch.rand(2, 3, 32, 40), Kmat.expand(2, 3, 3).contiguous(), dist5.expand(2, 5).contiguous()],
        tags=("batch>1",),
    ),
    _cal("tilt_projection", K.geometry.calibration.tilt_projection, [torch.tensor([[0.01]]), torch.tensor([[0.02]])]),
    _cal(
        "tilt_projection[inverse]",
        K.geometry.calibration.tilt_projection,
        [torch.tensor([[0.01]]), torch.tensor([[0.02]])],
        {"return_inverse": True},
    ),
    _cal("solve_pnp_dlt", K.geometry.calibration.solve_pnp_dlt, [X3d, pts2, Kmat2], atol=1e-3),
    _cal("solve_pnp_dlt[weights]", K.geometry.calibration.solve_pnp_dlt, [X3d, pts2, Kmat2, w_ones], atol=1e-3),
    # ------------------------------------------------------------------ depth
    _cd("depth_to_3d", K.geometry.depth.depth_to_3d, [depth_img, Kmat]),
    _cd("depth_to_3d[normalize]", K.geometry.depth.depth_to_3d, [depth_img, Kmat], {"normalize_points": True}),
    _cd("depth_to_3d_v2", K.geometry.depth.depth_to_3d_v2, [depth_img[:, 0], Kmat]),
    _cd(
        "depth_to_3d_v2[xyz_grid]",
        lambda d, k, g: K.geometry.depth.depth_to_3d_v2(d, k, xyz_grid=g),
        [depth_img[:, 0], Kmat, K.geometry.depth.unproject_meshgrid(32, 40, Kmat)],
    ),
    _cd("unproject_meshgrid", lambda k: K.geometry.depth.unproject_meshgrid(32, 40, k), [Kmat], note="H,W baked"),
    _cd("depth_to_normals", K.geometry.depth.depth_to_normals, [depth_img, Kmat]),
    _cd(
        "depth_from_disparity",
        K.geometry.depth.depth_from_disparity,
        [disparity[..., 0], torch.tensor([0.1]), torch.tensor([50.0])],
    ),
    _cd(
        "depth_from_disparity[float]",
        K.geometry.depth.depth_from_disparity,
        [disparity[..., 0]],
        {"baseline": 0.1, "focal": 50.0},
        note="baseline/focal baked",
    ),
    _cd(
        "depth_from_plane_equation",
        K.geometry.depth.depth_from_plane_equation,
        [torch.tensor([[0.1, 0.2, 0.97]]), torch.tensor([[3.0]]), img_pts, Kmat],
    ),
    _cd("warp_frame_depth", K.geometry.depth.warp_frame_depth, [IMG, depth_img, T01, Kmat]),
    _cd(
        "DepthWarper",
        lambda d, p: (lambda w: (w.compute_projection_matrix(cam1), w)[1])(K.geometry.depth.DepthWarper(cam2, 32, 40))(  # noqa: PLW0108
            d, p
        ),
        [depth_img, IMG],
        note="cameras baked in module; H,W baked",
    ),
    _cd(
        "DepthWarper.warp_grid",
        lambda d: (lambda w: (w.compute_projection_matrix(cam1), w)[1])(
            K.geometry.depth.DepthWarper(cam2, 32, 40)
        ).warp_grid(d),
        [depth_img],
        note="cameras baked",
    ),
    _cd(
        "DepthWarper.compute_subpixel_step",
        lambda d: (
            (lambda w: (w.compute_projection_matrix(cam1), w)[1])(
                K.geometry.depth.DepthWarper(cam2, 32, 40)
            ).compute_subpixel_step()
            + 0 * d.sum()
        ),
        [torch.zeros(1)],
        note="fully constant graph",
    ),
    _cd(
        "depth_warp",
        lambda d, p: K.geometry.depth.depth_warp(cam2, cam1, d, p, 32, 40),
        [depth_img, IMG],
        note="cameras + H,W baked",
    ),
    # ------------------------------------------------------------------ liegroup
    _clg("So3.exp.matrix", lambda v: So3.exp(v).matrix(), [so3_v]),
    _clg("So3.exp.q", lambda v: So3.exp(v).q.data, [so3_v]),
    _clg("So3.log", lambda v: So3.exp(v).log(), [so3_v]),
    _clg("So3.from_matrix", lambda m: So3.from_matrix(m).q.data, [So3M]),
    _clg("So3.from_wxyz", lambda w: So3.from_wxyz(w).matrix(), [q]),
    _clg("So3.inverse", lambda v: So3.exp(v).inverse().matrix(), [so3_v]),
    _clg("So3.mul", lambda a, b: (So3.exp(a) * So3.exp(b)).matrix(), [so3_v, so3_v.flip(0)]),
    _clg("So3.mul[point]", lambda a, p: So3.exp(a) * p, [so3_v, torch.rand(2, 3)], note="rotate a (B,3) point"),
    _clg("So3.adjoint", lambda v: So3.exp(v).adjoint(), [so3_v]),
    _clg("So3.hat", So3.hat, [so3_v]),
    _clg("So3.vee", lambda v: So3.vee(So3.hat(v)), [so3_v]),
    _clg("So3.Jl", So3.Jl, [so3_v]),
    _clg("So3.Jr", So3.Jr, [so3_v]),
    _clg("So3.left_jacobian", So3.left_jacobian, [so3_v]),
    _clg("So3.right_jacobian", So3.right_jacobian, [so3_v]),
    _clg("So3.rot_x", lambda x: So3.rot_x(x).matrix(), [so2_th]),
    _clg("So3.rot_y", lambda x: So3.rot_y(x).matrix(), [so2_th]),
    _clg("So3.rot_z", lambda x: So3.rot_z(x).matrix(), [so2_th]),
    _clg(
        "So3.random",
        lambda d: So3.random(2, device=d.device, dtype=d.dtype).matrix() + 0 * d,
        [torch.zeros(1)],
        check=False,
        note="random sampling in graph",
    ),
    _clg(
        "So3.identity",
        lambda d: So3.identity(2, device=d.device, dtype=d.dtype).matrix() + 0 * d,
        [torch.zeros(1)],
        note="fully constant graph",
    ),
    _clg("Se3.exp.matrix", lambda v: Se3.exp(v).matrix(), [se3_v]),
    _clg("Se3.log", lambda v: Se3.exp(v).log(), [se3_v]),
    _clg("Se3.from_matrix", lambda m: Se3.from_matrix(m).matrix(), [Se3M]),
    _clg("Se3.from_qxyz", lambda w: Se3.from_qxyz(w).matrix(), [torch.cat([q, torch.rand(2, 3)], -1)]),
    _clg("Se3.inverse", lambda v: Se3.exp(v).inverse().matrix(), [se3_v]),
    _clg("Se3.mul", lambda a, b: (Se3.exp(a) * Se3.exp(b)).matrix(), [se3_v, se3_v.flip(0)]),
    _clg("Se3.mul[point]", lambda a, p: Se3.exp(a) * p, [se3_v, torch.rand(2, 3)], note="transform a (B,3) point"),
    _clg("Se3.adjoint", lambda v: Se3.exp(v).adjoint(), [se3_v]),
    _clg("Se3.hat", Se3.hat, [se3_v]),
    _clg("Se3.vee", lambda v: Se3.vee(Se3.hat(v)), [se3_v]),
    _clg("Se3.trans", lambda x, y, z: Se3.trans(x, y, z).matrix(), [so2_th, so2_th + 1, so2_th - 1]),
    _clg("Se3.rot_x", lambda x: Se3.rot_x(x).matrix(), [so2_th]),
    _clg(
        "Se3.random",
        lambda d: Se3.random(2, device=d.device, dtype=d.dtype).matrix() + 0 * d,
        [torch.zeros(1)],
        check=False,
        note="random sampling in graph",
    ),
    _clg("So2.exp.matrix", lambda th: So2.exp(th).matrix(), [so2_th]),
    _clg("So2.exp.z", lambda th: So2.exp(th).z, [so2_th], note="complex output"),
    _clg("So2.log", lambda th: So2.exp(th).log(), [so2_th]),
    _clg("So2.from_matrix", lambda m: So2.from_matrix(m).log(), [So2M]),
    _clg("So2.inverse", lambda th: So2.exp(th).inverse().matrix(), [so2_th]),
    _clg("So2.mul", lambda a, b: (So2.exp(a) * So2.exp(b)).matrix(), [so2_th, so2_th.flip(0)]),
    _clg("So2.mul[point]", lambda a, p: So2.exp(a) * p, [so2_th, torch.rand(2, 2)]),
    _clg("So2.adjoint", lambda th: So2.exp(th).adjoint(), [so2_th]),
    _clg("So2.hat", So2.hat, [so2_th]),
    _clg("So2.vee", lambda th: So2.vee(So2.hat(th)), [so2_th]),
    _clg(
        "So2.random",
        lambda d: So2.random(2, device=d.device, dtype=d.dtype).matrix() + 0 * d,
        [torch.zeros(1)],
        check=False,
        note="random sampling in graph",
    ),
    _clg("Se2.exp.matrix", lambda v: Se2.exp(v).matrix(), [se2_v]),
    _clg("Se2.log", lambda v: Se2.exp(v).log(), [se2_v]),
    _clg("Se2.from_matrix", lambda m: Se2.from_matrix(m).matrix(), [Se2M]),
    _clg("Se2.inverse", lambda v: Se2.exp(v).inverse().matrix(), [se2_v]),
    _clg("Se2.mul", lambda a, b: (Se2.exp(a) * Se2.exp(b)).matrix(), [se2_v, se2_v.flip(0)]),
    _clg("Se2.mul[point]", lambda a, p: Se2.exp(a) * p, [se2_v, torch.rand(2, 2)]),
    _clg("Se2.adjoint", lambda v: Se2.exp(v).adjoint(), [se2_v]),
    _clg("Se2.hat", Se2.hat, [se2_v]),
    _clg("Se2.vee", lambda v: Se2.vee(Se2.hat(v)), [se2_v]),
    _clg("Se2.trans", lambda x, y: Se2.trans(x, y).matrix(), [so2_th, so2_th + 1]),
    _clg(
        "Se2.random",
        lambda d: Se2.random(2, device=d.device, dtype=d.dtype).matrix() + 0 * d,
        [torch.zeros(1)],
        check=False,
        note="random sampling in graph",
    ),
    # ------------------------------------------------------------------ quaternion
    _cq("Quaternion.matrix", lambda d: Quaternion(d).matrix(), [q]),
    _cq("Quaternion.from_matrix", lambda m: Quaternion.from_matrix(m).data, [Rm]),
    _cq("Quaternion.from_axis_angle", lambda v: Quaternion.from_axis_angle(v).data, [aa]),
    _cq("Quaternion.to_axis_angle", lambda d: Quaternion(d).to_axis_angle(), [q]),
    _cq("Quaternion.from_euler", lambda r, p, y: Quaternion.from_euler(r, p, y).data, list(euler)),
    _cq("Quaternion.to_euler", lambda d: Quaternion(d).to_euler(), [q]),
    _cq("Quaternion.norm", lambda d: Quaternion(d).norm(), [q * 2.0]),
    _cq("Quaternion.squared_norm", lambda d: Quaternion(d).squared_norm(), [q * 2.0]),
    _cq("Quaternion.normalize", lambda d: Quaternion(d).normalize().data, [q * 2.0]),
    _cq("Quaternion.conj", lambda d: Quaternion(d).conj().data, [q]),
    _cq("Quaternion.inv", lambda d: Quaternion(d).inv().data, [q * 2.0]),
    _cq("Quaternion.mul", lambda a, b: (Quaternion(a) * Quaternion(b)).data, [q, qb]),
    _cq(
        "Quaternion.mul[scalar-tensor]",
        lambda a, p: (Quaternion(a) * p).data,
        [q, torch.rand(2)],
        note="scalar (B,) tensor -> scalar quaternion",
    ),
    _cq("Quaternion.add", lambda a, b: (Quaternion(a) + Quaternion(b)).data, [q, qb]),
    _cq("Quaternion.sub", lambda a, b: (Quaternion(a) - Quaternion(b)).data, [q, qb]),
    _cq("Quaternion.neg", lambda a: (-Quaternion(a)).data, [q]),
    _cq("Quaternion.truediv", lambda a, b: (Quaternion(a) / Quaternion(b)).data, [q, qb]),
    _cq("Quaternion.pow", lambda a: (Quaternion(a) ** 0.5).data, [q], note="t=0.5 baked"),
    _cq("Quaternion.slerp", lambda a, b: Quaternion(a).slerp(Quaternion(b), 0.3).data, [q, qb], note="t=0.3 baked"),
    _cq("Quaternion.polar_angle", lambda a: Quaternion(a).polar_angle, [q]),
    _cq(
        "Quaternion.properties",
        lambda a: (lambda Q: (Q.w, Q.x, Q.y, Q.z, Q.real, Q.vec, Q.scalar, Q.coeffs))(Quaternion(a)),
        [q],
    ),
    _cq(
        "Quaternion.random",
        lambda d: Quaternion.random(2, device=d.device, dtype=d.dtype).data + 0 * d,
        [torch.zeros(1)],
        check=False,
        note="random sampling in graph",
    ),
    _cq(
        "Quaternion.identity",
        lambda d: Quaternion.identity(2, device=d.device, dtype=d.dtype).data + 0 * d,
        [torch.zeros(1)],
        note="fully constant graph",
    ),
    _cq(
        "average_quaternions",
        lambda t: K.geometry.quaternion.average_quaternions(Quaternion(t)).data,
        [torch.stack([q[0], qb[0], q[1]])],
        note="eigen-decomposition of 4x4",
    ),
    # ------------------------------------------------------------------ ransac
    case(
        "geometry.ransac.RANSAC[homography,no-seed]",
        "geometry.ransac",
        K.geometry.ransac.RANSAC("homography", inl_th=2.0, max_iter=3, batch_size=64, seed=None),
        [hp1[0], hp2[0]],
        check=False,
        note="diagnostic: seed=None avoids torch.Generator; shows next blocker",
    ),
    _cam(
        "StereoCamera.reproject_disparity_to_3D[baked-cams]",
        lambda d: StereoCamera(left34, right34).reproject_disparity_to_3D(d),
        [disparity],
        note="diagnostic: cameras baked as constants",
    ),
    case(
        "geometry.ransac.RANSAC[homography]",
        "geometry.ransac",
        K.geometry.ransac.RANSAC("homography", inl_th=2.0, max_iter=3, batch_size=64, seed=0),
        [torch.cat([hp1[0], torch.rand(20, 2) * 40], 0), torch.cat([hp2[0], torch.rand(20, 2) * 40], 0)],
        check=False,
        note="sampling inside; max_iter=3; 12 inliers + 20 outliers",
    ),
    case(
        "geometry.ransac.RANSAC[fundamental]",
        "geometry.ransac",
        K.geometry.ransac.RANSAC("fundamental", inl_th=1.0, max_iter=3, batch_size=64, seed=0),
        [torch.cat([pts1[0], torch.rand(20, 2) * 40], 0), torch.cat([pts2[0], torch.rand(20, 2) * 40], 0)],
        check=False,
        note="sampling inside; 8pt inside",
    ),
    case(
        "geometry.ransac.RANSAC[fundamental_7pt]",
        "geometry.ransac",
        K.geometry.ransac.RANSAC("fundamental_7pt", inl_th=1.0, max_iter=3, batch_size=64, seed=0),
        [torch.cat([pts1[0], torch.rand(20, 2) * 40], 0), torch.cat([pts2[0], torch.rand(20, 2) * 40], 0)],
        check=False,
        note="sampling inside; 7pt inside",
    ),
    case(
        "geometry.ransac.RANSAC[homography_from_linesegments]",
        "geometry.ransac",
        K.geometry.ransac.RANSAC("homography_from_linesegments", inl_th=2.0, max_iter=3, batch_size=64, seed=0),
        [torch.cat([ls1[0], torch.rand(20, 2, 2) * 40], 0), torch.cat([ls2[0], torch.rand(20, 2, 2) * 40], 0)],
        check=False,
        note="sampling inside",
    ),
    case(
        "geometry.ransac.RANSAC.estimate_model_from_minsample",
        "geometry.ransac",
        K.geometry.ransac.RANSAC("homography", seed=0),
        [hp1_4, hp2_4],
        method="estimate_model_from_minsample",
        atol=1e-3,
    ),
    case(
        "geometry.ransac.RANSAC.verify",
        "geometry.ransac",
        lambda k1, k2, m: K.geometry.ransac.RANSAC("homography", seed=0).verify(k1, k2, m, 2.0)[:2],
        [hp1[0], hp2[0], H_gt],
        note="inl_th baked; returns (inliers, models, score float, ...) -> only tensors",
    ),
    # ------------------------------------------------------------------ solvers
    _cs("solve_quadratic", K.geometry.solvers.solve_quadratic, [torch.tensor([[1.0, -3.0, 2.0], [2.0, 0.5, -1.0]])]),
    _cs(
        "solve_cubic", K.geometry.solvers.solve_cubic, [torch.tensor([[1.0, -6.0, 11.0, -6.0], [1.0, 0.0, -1.0, 0.5]])]
    ),
    _cs(
        "solve_quartic",
        K.geometry.solvers.solve_quartic,
        [torch.tensor([[1.0, -10.0, 35.0, -50.0, 24.0], [1.0, 0.0, -5.0, 0.0, 4.0]])],
    ),
    _cs("multiply_deg_one_poly", K.geometry.solvers.multiply_deg_one_poly, [torch.rand(2, 4), torch.rand(2, 4)]),
    _cs(
        "multiply_deg_two_one_poly", K.geometry.solvers.multiply_deg_two_one_poly, [torch.rand(2, 10), torch.rand(2, 4)]
    ),
    _cs("determinant_to_polynomial", K.geometry.solvers.determinant_to_polynomial, [torch.rand(2, 3, 13)]),
    _cs("null_vector_3x4", K.geometry.solvers.null_vector_3x4, [P2], atol=1e-3),
    # ------------------------------------------------------------------ pointcloud
    case(
        "geometry.pointcloud.save_pointcloud_ply", "geometry.pointcloud", None, [], skip="file I/O (writes PLY to disk)"
    ),
    case(
        "geometry.pointcloud.load_pointcloud_ply",
        "geometry.pointcloud",
        None,
        [],
        skip="file I/O (reads PLY from disk)",
    ),
    case(
        "geometry.pointcloud.save_pointcloud_ply_binary",
        "geometry.pointcloud",
        None,
        [],
        skip="file I/O (writes PLY to disk)",
    ),
    case(
        "geometry.pointcloud.load_pointcloud_ply_binary",
        "geometry.pointcloud",
        None,
        [],
        skip="file I/O (reads PLY from disk)",
    ),
]

if __name__ == "__main__":
    run_cases(CASES, sys.argv[1], only=sys.argv[2:] or None)
