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

"""ONNX export survey — area geomA: kornia.geometry.transform / grid / bbox / boxes / keypoints / subpix /
line / plane / ray / vector / pose."""

import math
import sys

import torch
from harness import case, run_cases

import kornia as K
import kornia.geometry.transform as T
from kornia.geometry.bbox import (
    bbox_generator,
    bbox_generator3d,
    bbox_to_mask,
    bbox_to_mask3d,
    infer_bbox_shape,
    infer_bbox_shape3d,
    nms,
    transform_bbox,
    validate_bbox,
    validate_bbox3d,
)
from kornia.geometry.boxes import Boxes, Boxes3D, VideoBoxes
from kornia.geometry.grid import create_meshgrid, create_meshgrid3d
from kornia.geometry.keypoints import Keypoints, Keypoints3D
from kornia.geometry.line import ParametrizedLine, fit_line
from kornia.geometry.plane import Hyperplane, fit_plane
from kornia.geometry.pose import NamedPose
from kornia.geometry.ray import Ray
from kornia.geometry.subpix import (
    AdaptiveQuadInterp3d,
    ConvQuadInterp3d,
    ConvSoftArgmax2d,
    ConvSoftArgmax3d,
    IterativeQuadInterp3d,
    NonMaximaSuppression2d,
    NonMaximaSuppression3d,
    SpatialSoftArgmax2d,
    conv_quad_interp3d,
    conv_soft_argmax2d,
    conv_soft_argmax3d,
    iterative_quad_interp3d,
    nms2d,
    nms3d,
    nms3d_minmax,
    render_gaussian2d,
    spatial_expectation2d,
    spatial_soft_argmax2d,
    spatial_softmax2d,
)
from kornia.geometry.vector import Scalar, Vector2, Vector3

torch.manual_seed(0)

# ----------------------------------------------------------------------------- shared inputs
B, C, H, W = 1, 3, 32, 40
img = torch.rand(B, C, H, W)
img2 = torch.rand(2, 3, 48, 64)
vol = torch.rand(1, 1, 8, 16, 16)

# 2D transforms
center2d = torch.tensor([[W / 2.0, H / 2.0]])
angle = torch.tensor([15.0])
scale2 = torch.tensor([[1.1, 0.9]])
trans2 = torch.tensor([[3.0, -2.0]])
shear2 = torch.tensor([[0.1, 0.05]])
aff23 = T.get_affine_matrix2d(trans2, center2d, scale2, angle, sx=shear2[:, 0], sy=shear2[:, 1])[:, :2, :]  # (1,2,3)
homo33 = T.get_affine_matrix2d(trans2, center2d, scale2, angle)  # (1,3,3)
homo33_pert = homo33.clone()
homo33_pert[:, 2, 0] = 1e-3
homo33_pert[:, 2, 1] = -5e-4
# normalized homography (pixel -> normalized [-1,1] coordinates) for homography_warp with normalized_homography=True
homo_norm = K.geometry.conversions.normalize_homography(homo33_pert, (H, W), (H, W))

# 3D transforms
D3, H3, W3 = 8, 16, 16
center3d = torch.tensor([[W3 / 2.0, H3 / 2.0, D3 / 2.0]])
angles3 = torch.tensor([[10.0, -5.0, 8.0]])
scale3 = torch.tensor([[1.05, 0.95, 1.0]])
trans3 = torch.tensor([[0.5, -0.5, 0.3]])
aff44_3d = T.get_affine_matrix3d(trans3, center3d, scale3, angles3)  # (1,4,4)
aff34_3d = aff44_3d[:, :3, :]  # (1,3,4)
proj34 = T.get_projective_transform(center3d, angles3, scale3)  # (1,3,4)

# points / corners
src_pts = torch.tensor([[[2.0, 3.0], [36.0, 2.0], [37.0, 29.0], [3.0, 28.0]]])  # (1,4,2)
dst_pts = torch.tensor([[[0.0, 0.0], [W - 1.0, 0.0], [W - 1.0, H - 1.0], [0.0, H - 1.0]]])
src_pts3d = torch.tensor(
    [
        [
            [1.0, 1.0, 1.0],
            [14.0, 1.0, 1.0],
            [14.0, 14.0, 1.0],
            [1.0, 14.0, 1.0],
            [1.0, 1.0, 6.0],
            [14.0, 1.0, 6.0],
            [14.0, 14.0, 6.0],
            [1.0, 14.0, 6.0],
        ]
    ]
)  # (1,8,3)
dst_pts3d = torch.tensor(
    [
        [
            [0.0, 0.0, 0.0],
            [W3 - 1.0, 0.0, 0.0],
            [W3 - 1.0, H3 - 1.0, 0.0],
            [0.0, H3 - 1.0, 0.0],
            [0.0, 0.0, D3 - 1.0],
            [W3 - 1.0, 0.0, D3 - 1.0],
            [W3 - 1.0, H3 - 1.0, D3 - 1.0],
            [0.0, H3 - 1.0, D3 - 1.0],
        ]
    ]
)
# tps
tps_src = torch.tensor([[[-0.8, -0.8], [0.8, -0.8], [0.8, 0.8], [-0.8, 0.8], [0.0, 0.0]]])  # (1,5,2) normalized
tps_dst = tps_src + 0.05 * torch.randn(1, 5, 2)
tps_kernel_w, tps_affine_w = T.get_tps_transform(tps_src, tps_dst)

# remap maps (pixel coordinates)
grid_px = create_meshgrid(H, W, normalized_coordinates=False)  # (1,H,W,2)
map_x = grid_px[..., 0] + 1.5
map_y = grid_px[..., 1] - 0.7

# boxes
boxes_xyxy = torch.tensor([[[2.0, 3.0, 20.0, 25.0], [10.0, 5.0, 35.0, 30.0]]])  # (1,2,4)
boxes_quad = Boxes.from_tensor(boxes_xyxy, mode="xyxy").data  # (1,2,4,2)
boxes3d_xyzxyz = torch.tensor([[[1.0, 2.0, 1.0, 10.0, 12.0, 6.0], [3.0, 3.0, 2.0, 14.0, 14.0, 7.0]]])  # (1,2,6)
boxes3d_hexa = Boxes3D.from_tensor(boxes3d_xyzxyz, mode="xyzxyz").data  # (1,2,8,3)
nms_boxes = torch.tensor([[10.0, 10.0, 20.0, 20.0], [15.0, 5.0, 15.0, 25.0], [100.0, 100.0, 150.0, 150.0]])
nms_scores = torch.tensor([0.9, 0.8, 0.7])
kpts = torch.tensor([[[5.0, 6.0], [20.0, 10.0], [30.0, 25.0]]])  # (1,3,2)
kpts3d = torch.tensor([[[5.0, 6.0, 2.0], [10.0, 10.0, 4.0], [12.0, 3.0, 6.0]]])  # (1,3,3)

# subpix
heat = torch.rand(1, 2, 16, 24)
heat3d = torch.rand(1, 1, 5, 16, 24)  # scale-space (B, C, D=levels, H, W)

# lines / planes
line_pts = torch.stack([torch.linspace(0, 5, 10), 2 * torch.linspace(0, 5, 10) + 1], -1)[None] + 0.01 * torch.randn(
    1, 10, 2
)
plane_pts = torch.randn(20, 3)
plane_pts[:, 2] = 0.3 * plane_pts[:, 0] - 0.2 * plane_pts[:, 1] + 1.0 + 0.01 * torch.randn(20)
p3 = torch.tensor([1.0, 2.0, 3.0])
q3 = torch.tensor([4.0, 5.0, 7.0])
v3 = torch.randn(4, 3)

# pose
Rt = aff44_3d.clone()  # not orthonormal (has scale) -> build a proper SE3 from axis-angle
so3 = K.geometry.liegroup.So3.exp(torch.tensor([[0.1, 0.2, -0.1]]))
pose44 = torch.eye(4)[None].repeat(1, 1, 1)
pose44[:, :3, :3] = so3.matrix()
pose44[:, :3, 3] = torch.tensor([[0.5, -1.0, 2.0]])
pts3 = torch.randn(1, 6, 3)


# ----------------------------------------------------------------------------- cases
G = "geometry.transform"
CASES = [
    # ---- 2D warps
    case(
        "transform.warp_perspective", G, T.warp_perspective, [img, homo33_pert], {"dsize": (H, W)}, note="dsize baked"
    ),
    case(
        "transform.warp_perspective[nearest,border]",
        G,
        T.warp_perspective,
        [img, homo33_pert],
        {"dsize": (H, W), "mode": "nearest", "padding_mode": "border"},
        note="dsize baked",
    ),
    case(
        "transform.warp_perspective[fill]",
        G,
        lambda x, M, fv: T.warp_perspective(x, M, dsize=(H, W), padding_mode="fill", fill_value=fv),
        [img, homo33_pert, torch.tensor([0.2, 0.4, 0.6])],
        note="padding_mode=fill; fill_value live input; dsize baked",
    ),
    case(
        "transform.warp_perspective[batch2,resize]",
        G,
        T.warp_perspective,
        [img2, homo33_pert.repeat(2, 1, 1)],
        {"dsize": (24, 30)},
        tags=("batch>1",),
        note="dsize != input size",
    ),
    case("transform.warp_affine", G, T.warp_affine, [img, aff23], {"dsize": (H, W)}, note="dsize baked"),
    case(
        "transform.warp_affine[align_corners=False,reflection]",
        G,
        T.warp_affine,
        [img, aff23],
        {"dsize": (H, W), "align_corners": False, "padding_mode": "reflection"},
    ),
    case(
        "transform.warp_affine[fill]",
        G,
        lambda x, M, fv: T.warp_affine(x, M, dsize=(H, W), padding_mode="fill", fill_value=fv),
        [img, aff23, torch.tensor([0.5, 0.5, 0.5])],
        note="fill_value live input; dsize baked",
    ),
    case(
        "transform.warp_grid",
        G,
        T.warp_grid,
        [create_meshgrid(H, W, normalized_coordinates=True), homo_norm],
        note="grid is a live input (1,H,W,2)",
    ),
    case("transform.remap", G, T.remap, [img, map_x, map_y], {"align_corners": True}),
    case(
        "transform.remap[normalized,nearest]",
        G,
        T.remap,
        [img, (map_x / (W - 1)) * 2 - 1, (map_y / (H - 1)) * 2 - 1],
        {"align_corners": True, "normalized_coordinates": True, "mode": "nearest"},
    ),
    case(
        "transform.homography_warp",
        G,
        T.homography_warp,
        [img, homo_norm],
        {"dsize": (H, W)},
        note="normalized homography; dsize baked",
    ),
    case(
        "transform.homography_warp[pixel_homography]",
        G,
        T.homography_warp,
        [img, homo33_pert],
        {"dsize": (H, W), "normalized_coordinates": False, "normalized_homography": False},
    ),
    case("transform.get_tps_transform", G, T.get_tps_transform, [tps_src, tps_dst], tags=("points",)),
    case(
        "transform.warp_points_tps",
        G,
        T.warp_points_tps,
        [torch.rand(1, 7, 2) * 1.6 - 0.8, tps_src, tps_kernel_w, tps_affine_w],
        tags=("points",),
        note="query points distinct from kernel centers",
    ),
    case(
        "transform.warp_points_tps[at_centers]",
        G,
        T.warp_points_tps,
        [tps_src.clone(), tps_src, tps_kernel_w, tps_affine_w],
        tags=("points",),
        note="query points == kernel centers (zero distance) -> exercises the eps guard in _kernel_distance",
    ),
    case("transform.warp_image_tps", G, T.warp_image_tps, [img, tps_src, tps_kernel_w, tps_affine_w]),
    case(
        "transform.elastic_transform2d",
        G,
        T.elastic_transform2d,
        [img, torch.rand(1, 2, H, W) * 2 - 1],
        {"kernel_size": (31, 31), "sigma": (8.0, 8.0), "alpha": (1.0, 1.0)},
        note="noise is live input; kernel/sigma/alpha baked",
    ),
    case(
        "transform.elastic_transform2d[tensor_sigma_alpha]",
        G,
        lambda x, n, s, a: T.elastic_transform2d(x, n, kernel_size=(31, 31), sigma=s, alpha=a),
        [img, torch.rand(1, 2, H, W) * 2 - 1, torch.tensor([8.0, 8.0]), torch.tensor([1.0, 1.0])],
        note="sigma/alpha as live tensors; kernel_size baked",
    ),
    # ---- 2D affine family
    case("transform.affine", G, T.affine, [img, aff23]),
    case("transform.rotate", G, T.rotate, [img, angle]),
    case("transform.rotate[center]", G, T.rotate, [img, angle, center2d]),
    case("transform.translate", G, T.translate, [img, trans2]),
    case("transform.scale", G, T.scale, [img, scale2]),
    case("transform.scale[scalar]", G, T.scale, [img, torch.tensor([1.2])]),
    case("transform.shear", G, T.shear, [img, shear2]),
    case("transform.hflip", G, T.hflip, [img]),
    case("transform.vflip", G, T.vflip, [img]),
    case("transform.rot180", G, T.rot180, [img]),
    case("transform.resize", G, T.resize, [img], {"size": (24, 30)}, note="size baked"),
    case("transform.resize[int,side=short]", G, T.resize, [img], {"size": 16, "side": "short"}, note="size=int baked"),
    case("transform.resize[nearest]", G, T.resize, [img], {"size": (24, 30), "interpolation": "nearest"}),
    case(
        "transform.resize[bicubic,align_corners]",
        G,
        T.resize,
        [img],
        {"size": (24, 30), "interpolation": "bicubic", "align_corners": True},
    ),
    case("transform.resize[antialias,area]", G, T.resize, [img], {"size": (16, 20), "antialias": True}),
    case(
        "transform.resize_to_be_divisible",
        G,
        T.resize_to_be_divisible,
        [img],
        {"divisible_factor": 14},
        note="factor baked; output size fixed at trace",
    ),
    case("transform.rescale", G, T.rescale, [img], {"factor": 0.5}),
    case("transform.rescale[tuple]", G, T.rescale, [img], {"factor": (1.5, 0.75)}),
    # ---- pyramids
    case("transform.pyrdown", G, T.pyrdown, [img]),
    case("transform.pyrdown[factor=3]", G, T.pyrdown, [img], {"factor": 3.0}),
    case("transform.pyrup", G, T.pyrup, [img]),
    case(
        "transform.build_pyramid",
        G,
        T.build_pyramid,
        [img2],
        {"max_level": 3},
        tags=("batch>1",),
        note="returns list of 3 tensors",
    ),
    case(
        "transform.build_laplacian_pyramid",
        G,
        T.build_laplacian_pyramid,
        [img2],
        {"max_level": 3},
        tags=("batch>1",),
        note="returns list of 3 tensors",
    ),
    case("transform.PyrDown", G, T.PyrDown(), [img]),
    case("transform.PyrUp", G, T.PyrUp(), [img]),
    case(
        "transform.ScalePyramid",
        G,
        T.ScalePyramid(n_levels=3, min_size=15),
        [torch.rand(1, 1, 48, 64)],
        note="returns (pyramid list, sigmas list, pixel_dists list) flattened",
    ),
    case(
        "transform.ScalePyramid[double_image]",
        G,
        T.ScalePyramid(n_levels=3, min_size=15, double_image=True),
        [torch.rand(1, 1, 32, 40)],
    ),
    # ---- 3D warps
    case("transform.warp_affine3d", G, T.warp_affine3d, [vol, aff34_3d], {"dsize": (D3, H3, W3)}, tags=("3d",)),
    case(
        "transform.warp_affine3d[nearest]",
        G,
        T.warp_affine3d,
        [vol, aff34_3d],
        {"dsize": (D3, H3, W3), "flags": "nearest", "align_corners": False},
        tags=("3d",),
    ),
    case(
        "transform.warp_perspective3d", G, T.warp_perspective3d, [vol, aff44_3d], {"dsize": (D3, H3, W3)}, tags=("3d",)
    ),
    case(
        "transform.warp_grid3d",
        G,
        T.warp_grid3d,
        [
            create_meshgrid3d(D3, H3, W3, normalized_coordinates=True),
            K.geometry.conversions.normalize_homography3d(aff44_3d, (D3, H3, W3), (D3, H3, W3)),
        ],
        tags=("3d",),
    ),
    case(
        "transform.homography_warp3d",
        G,
        T.homography_warp3d,
        [vol, K.geometry.conversions.normalize_homography3d(aff44_3d, (D3, H3, W3), (D3, H3, W3))],
        {"dsize": (D3, H3, W3)},
        tags=("3d",),
    ),
    case("transform.affine3d", G, T.affine3d, [vol, aff34_3d], tags=("3d",)),
    case(
        "transform.rotate3d",
        G,
        T.rotate3d,
        [vol, torch.tensor([10.0]), torch.tensor([-5.0]), torch.tensor([8.0])],
        tags=("3d",),
    ),
    case(
        "transform.rotate3d[center]",
        G,
        T.rotate3d,
        [vol, torch.tensor([10.0]), torch.tensor([-5.0]), torch.tensor([8.0]), center3d],
        tags=("3d",),
    ),
    # ---- matrix helpers
    case("transform.get_perspective_transform", G, T.get_perspective_transform, [src_pts, dst_pts], tags=("points",)),
    case(
        "transform.get_perspective_transform3d",
        G,
        T.get_perspective_transform3d,
        [src_pts3d, dst_pts3d],
        tags=("points", "3d"),
    ),
    case(
        "transform.get_projective_transform", G, T.get_projective_transform, [center3d, angles3, scale3], tags=("3d",)
    ),
    case("transform.get_rotation_matrix2d", G, T.get_rotation_matrix2d, [center2d, angle, scale2]),
    case("transform.get_translation_matrix2d", G, T.get_translation_matrix2d, [trans2]),
    case("transform.get_shear_matrix2d", G, T.get_shear_matrix2d, [center2d, shear2[:, 0], shear2[:, 1]]),
    case("transform.get_shear_matrix2d[sx_only]", G, T.get_shear_matrix2d, [center2d, shear2[:, 0]]),
    case(
        "transform.get_shear_matrix3d",
        G,
        T.get_shear_matrix3d,
        [
            center3d,
            torch.tensor([0.1]),
            torch.tensor([0.05]),
            torch.tensor([-0.05]),
            torch.tensor([0.02]),
            torch.tensor([0.03]),
            torch.tensor([-0.02]),
        ],
        tags=("3d",),
    ),
    case("transform.get_affine_matrix2d", G, T.get_affine_matrix2d, [trans2, center2d, scale2, angle]),
    case(
        "transform.get_affine_matrix2d[shear]",
        G,
        T.get_affine_matrix2d,
        [trans2, center2d, scale2, angle, shear2[:, 0], shear2[:, 1]],
    ),
    case("transform.get_affine_matrix3d", G, T.get_affine_matrix3d, [trans3, center3d, scale3, angles3], tags=("3d",)),
    case(
        "transform.get_affine_matrix3d[shear]",
        G,
        T.get_affine_matrix3d,
        [
            trans3,
            center3d,
            scale3,
            angles3,
            torch.tensor([0.1]),
            torch.tensor([0.05]),
            torch.tensor([-0.05]),
            torch.tensor([0.02]),
            torch.tensor([0.03]),
            torch.tensor([-0.02]),
        ],
        tags=("3d",),
    ),
    case("transform.invert_affine_transform", G, T.invert_affine_transform, [aff23]),
    case(
        "transform.projection_from_Rt",
        G,
        T.projection_from_Rt,
        [so3.matrix(), torch.tensor([[[0.5], [-1.0], [2.0]]])],
        tags=("3d",),
    ),
    # ---- crops 2D
    case(
        "transform.crop_by_indices",
        G,
        T.crop_by_indices,
        [img, src_pts],
        note="size inferred from box (data-dependent)",
    ),
    case(
        "transform.crop_by_indices[size]", G, T.crop_by_indices, [img, src_pts], {"size": (20, 24)}, note="size baked"
    ),
    case("transform.crop_by_boxes", G, T.crop_by_boxes, [img, src_pts, dst_pts]),
    case("transform.crop_by_transform_mat", G, T.crop_by_transform_mat, [img, homo33], {"out_size": (H, W)}),
    case("transform.center_crop", G, T.center_crop, [img], {"size": (20, 24)}),
    case("transform.crop_and_resize", G, T.crop_and_resize, [img, src_pts], {"size": (20, 24)}),
    case("transform.CenterCrop2D", G, T.CenterCrop2D((20, 24)), [img]),
    case("transform.CenterCrop2D[resample]", G, T.CenterCrop2D((20, 24), cropping_mode="resample"), [img]),
    # ---- crops 3D
    case("transform.crop_by_boxes3d", G, T.crop_by_boxes3d, [vol, src_pts3d, dst_pts3d], tags=("3d",)),
    case(
        "transform.crop_by_transform_mat3d",
        G,
        T.crop_by_transform_mat3d,
        [vol, aff44_3d],
        {"out_size": (D3, H3, W3)},
        tags=("3d",),
    ),
    case("transform.center_crop3d", G, T.center_crop3d, [vol], {"size": (6, 12, 12)}, tags=("3d",)),
    case("transform.crop_and_resize3d", G, T.crop_and_resize3d, [vol, src_pts3d], {"size": (6, 12, 12)}, tags=("3d",)),
    # ---- modules
    case("transform.Rotate", G, T.Rotate(angle), [img], note="angle is a module constant"),
    case("transform.Translate", G, T.Translate(trans2), [img]),
    case("transform.Scale", G, T.Scale(scale2), [img]),
    case("transform.Shear", G, T.Shear(shear2), [img]),
    case("transform.Affine", G, T.Affine(angle=angle, translation=trans2, scale_factor=scale2, shear=shear2), [img]),
    case("transform.Hflip", G, T.Hflip(), [img]),
    case("transform.Vflip", G, T.Vflip(), [img]),
    case("transform.Rot180", G, T.Rot180(), [img]),
    case("transform.Resize", G, T.Resize((24, 30)), [img]),
    case("transform.Resize[int]", G, T.Resize(24, side="long"), [img]),
    case("transform.Rescale", G, T.Rescale(0.5), [img]),
    case("transform.HomographyWarper", G, T.HomographyWarper(H, W), [img, homo_norm], note="normalized homography"),
    case(
        "transform.HomographyWarper[pixel]",
        G,
        T.HomographyWarper(H, W, normalized_coordinates=False),
        [img, homo33_pert],
    ),
    case(
        "transform.HomographyWarper[precomputed]",
        G,
        None,
        [],
        skip="precompute_warp_grid path stores grid as state; "
        "same graph as HomographyWarper with the homography folded into a constant",
    ),
    case("transform.BaseWarper", G, None, [], skip="abstract base class"),
    case("transform.ImageRegistrator", G, None, [], skip="optimizer loop (Adam over num_iterations) — not a graph"),
    case(
        "transform.image_registrator.Homography",
        G,
        lambda d: T.Homography()(),
        [torch.zeros(1)],
        note="parametric model; forward() takes no input, graph is fully constant",
    ),
    case(
        "transform.image_registrator.Similarity",
        G,
        lambda d: T.Similarity()(),
        [torch.zeros(1)],
        note="parametric model; forward() takes no input, graph is fully constant",
    ),
    case(
        "transform.image_registrator.Homography.forward_inverse",
        G,
        lambda d: T.Homography().forward_inverse(),
        [torch.zeros(1)],
        note="torch.inverse on 3x3 parameter",
    ),
    case("transform.image_registrator.BaseModel", G, None, [], skip="abstract base class"),
]

# ----------------------------------------------------------------------------- grid
G = "geometry.grid"
CASES += [
    case(
        "grid.create_meshgrid",
        G,
        lambda d: create_meshgrid(16, 24, device=d.device, dtype=d.dtype),
        [torch.zeros(1)],
        note="H,W baked; graph is fully constant",
    ),
    case(
        "grid.create_meshgrid[pixel]",
        G,
        lambda d: create_meshgrid(16, 24, normalized_coordinates=False, device=d.device, dtype=d.dtype),
        [torch.zeros(1)],
        note="H,W baked",
    ),
    case(
        "grid.create_meshgrid3d",
        G,
        lambda d: create_meshgrid3d(4, 16, 24, device=d.device, dtype=d.dtype),
        [torch.zeros(1)],
        tags=("3d",),
        note="D,H,W baked",
    ),
    case(
        "grid.create_meshgrid[dynamic_from_input]",
        G,
        lambda x: (
            create_meshgrid(x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype) + 0 * x[:, :1].permute(0, 2, 3, 1)
        ),
        [img],
        note="H,W taken from input shape; still static under export",
    ),
]

# ----------------------------------------------------------------------------- bbox
G = "geometry.bbox"
CASES += [
    case(
        "bbox.bbox_generator",
        G,
        bbox_generator,
        [torch.tensor([2.0, 5.0]), torch.tensor([3.0, 4.0]), torch.tensor([10.0, 12.0]), torch.tensor([8.0, 9.0])],
    ),
    case(
        "bbox.bbox_generator3d",
        G,
        bbox_generator3d,
        [
            torch.tensor([2.0]),
            torch.tensor([3.0]),
            torch.tensor([1.0]),
            torch.tensor([10.0]),
            torch.tensor([8.0]),
            torch.tensor([4.0]),
        ],
        tags=("3d",),
    ),
    case(
        "bbox.bbox_to_mask",
        G,
        bbox_to_mask,
        [boxes_quad[0]],
        {"width": W, "height": H},
        note="H,W baked; input (N,4,2)",
    ),
    case(
        "bbox.bbox_to_mask3d",
        G,
        bbox_to_mask3d,
        [boxes3d_hexa[0]],
        {"size": (D3, H3, W3)},
        tags=("3d",),
        note="size baked; input (N,8,3)",
    ),
    case("bbox.infer_bbox_shape", G, infer_bbox_shape, [boxes_quad[0]]),
    case("bbox.infer_bbox_shape3d", G, infer_bbox_shape3d, [boxes3d_hexa[0]], tags=("3d",)),
    case("bbox.transform_bbox", G, transform_bbox, [homo33, boxes_xyxy], note="mode=xyxy default (B,N,4)"),
    case(
        "bbox.transform_bbox[xywh]",
        G,
        transform_bbox,
        [homo33, torch.tensor([[[2.0, 3.0, 18.0, 22.0], [10.0, 5.0, 25.0, 25.0]]])],
        {"mode": "xywh"},
    ),
    case("bbox.transform_bbox[polygon]", G, transform_bbox, [homo33, boxes_quad], note="(B,N,4,2) polygon input"),
    case(
        "bbox.nms",
        G,
        nms,
        [nms_boxes, nms_scores],
        {"iou_threshold": 0.5},
        note="data-dependent output size (keep indices); iou_threshold baked",
    ),
    case("bbox.validate_bbox", G, validate_bbox, [boxes_quad[0]], note="returns Python bool"),
    case("bbox.validate_bbox3d", G, validate_bbox3d, [boxes3d_hexa[0]], tags=("3d",), note="returns Python bool"),
]

# ----------------------------------------------------------------------------- boxes
G = "geometry.boxes"
CASES += [
    case("boxes.Boxes.from_tensor[xyxy]", G, lambda b: Boxes.from_tensor(b, mode="xyxy").data, [boxes_xyxy]),
    case(
        "boxes.Boxes.from_tensor[xywh]",
        G,
        lambda b: Boxes.from_tensor(b, mode="xywh").data,
        [torch.tensor([[[2.0, 3.0, 18.0, 22.0], [10.0, 5.0, 25.0, 25.0]]])],
    ),
    case("boxes.Boxes.from_tensor[xyxy_plus]", G, lambda b: Boxes.from_tensor(b, mode="xyxy_plus").data, [boxes_xyxy]),
    case("boxes.Boxes.from_tensor[vertices]", G, lambda b: Boxes.from_tensor(b, mode="vertices").data, [boxes_quad]),
    case(
        "boxes.Boxes.from_tensor[no_validate]",
        G,
        lambda b: Boxes.from_tensor(b, mode="xyxy", validate_boxes=False).data,
        [boxes_xyxy],
        note="validate_boxes=False skips the data-dependent assertion",
    ),
    case("boxes.Boxes.to_tensor[xyxy]", G, lambda b: Boxes(b, mode="vertices_plus").to_tensor("xyxy"), [boxes_quad]),
    case("boxes.Boxes.to_tensor[xywh]", G, lambda b: Boxes(b, mode="vertices_plus").to_tensor("xywh"), [boxes_quad]),
    case(
        "boxes.Boxes.to_tensor[vertices]",
        G,
        lambda b: Boxes(b, mode="vertices_plus").to_tensor("vertices"),
        [boxes_quad],
    ),
    case("boxes.Boxes.get_boxes_shape", G, lambda b: Boxes(b, mode="vertices_plus").get_boxes_shape(), [boxes_quad]),
    case("boxes.Boxes.compute_area", G, lambda b: Boxes(b, mode="vertices_plus").compute_area(), [boxes_quad]),
    case(
        "boxes.Boxes.to_mask", G, lambda b: Boxes(b, mode="vertices_plus").to_mask(H, W), [boxes_quad], note="H,W baked"
    ),
    case(
        "boxes.Boxes.transform_boxes",
        G,
        lambda b, M: Boxes(b, mode="vertices_plus").transform_boxes(M).data,
        [boxes_quad, homo33],
    ),
    case(
        "boxes.Boxes.transform_boxes_",
        G,
        lambda b, M: Boxes(b, mode="vertices_plus").transform_boxes_(M).data,
        [boxes_quad, homo33],
    ),
    case(
        "boxes.Boxes.translate",
        G,
        lambda b, s: Boxes(b, mode="vertices_plus").translate(s).data,
        [boxes_quad, torch.tensor([[3.0, -2.0]])],
    ),
    case(
        "boxes.Boxes.translate[fast]",
        G,
        None,
        [],
        skip="NotImplementedError stub (kornia/geometry/boxes.py:759, method='fast')",
    ),
    case(
        "boxes.Boxes.clamp[tuple]",
        G,
        None,
        [],
        skip="NotImplementedError stub for tuple bounds (kornia/geometry/boxes.py:391)",
    ),
    case(
        "boxes.Boxes.clamp[tensor]",
        G,
        lambda b, tl, br: Boxes(b, mode="vertices_plus").clamp(tl, br).data,
        [boxes_quad, torch.tensor([[5.0, 5.0]]), torch.tensor([[30.0, 20.0]])],
        note="bounds (B,2) tensors",
    ),
    case(
        "boxes.Boxes.pad",
        G,
        lambda b, p: Boxes(b, mode="vertices_plus").pad(p).data,
        [boxes_quad, torch.tensor([[2, 2, 3, 3]])],
        note="padding_size (B,4) int",
    ),
    case(
        "boxes.Boxes.unpad",
        G,
        lambda b, p: Boxes(b, mode="vertices_plus").unpad(p).data,
        [boxes_quad, torch.tensor([[2, 2, 3, 3]])],
    ),
    case("boxes.Boxes.trim", G, None, [], skip="NotImplementedError stub (kornia/geometry/boxes.py:442)"),
    case(
        "boxes.Boxes.filter_boxes_by_area",
        G,
        lambda b: Boxes(b, mode="vertices_plus").filter_boxes_by_area(min_area=100.0).data,
        [boxes_quad],
        note="zeros out boxes outside range",
    ),
    case(
        "boxes.Boxes.merge",
        G,
        lambda a, b: Boxes(a, mode="vertices_plus").merge(Boxes(b, mode="vertices_plus")).data,
        [boxes_quad, boxes_quad + 1.0],
    ),
    case(
        "boxes.Boxes.index_put",
        G,
        lambda b, v: Boxes(b, mode="vertices_plus").index_put((torch.tensor([0]), torch.tensor([1])), v).data,
        [boxes_quad, boxes_quad[0, :1]],
        note="indices baked as constants",
    ),
    case("boxes.Boxes.clone", G, lambda b: Boxes(b, mode="vertices_plus").clone().data, [boxes_quad]),
    case("boxes.Boxes.type", G, lambda b: Boxes(b, mode="vertices_plus").type(torch.float64).data, [boxes_quad]),
    case("boxes.Boxes.to_tensor[list_input]", G, None, [], skip="input is a Python list of tensors (ragged batch)"),
    case(
        "boxes.Boxes.to_tensor[as_padded_sequence]",
        G,
        None,
        [],
        skip="ragged list path; returns Python list of tensors",
    ),
    case(
        "boxes.Boxes3D.from_tensor[xyzxyz]",
        G,
        lambda b: Boxes3D.from_tensor(b, mode="xyzxyz").data,
        [boxes3d_xyzxyz],
        tags=("3d",),
    ),
    case(
        "boxes.Boxes3D.from_tensor[xyzwhd]",
        G,
        lambda b: Boxes3D.from_tensor(b, mode="xyzwhd").data,
        [torch.tensor([[[1.0, 2.0, 1.0, 9.0, 10.0, 5.0]]])],
        tags=("3d",),
    ),
    case(
        "boxes.Boxes3D.to_tensor[xyzxyz]",
        G,
        lambda b: Boxes3D(b, mode="xyzxyz_plus").to_tensor("xyzxyz"),
        [boxes3d_hexa],
        tags=("3d",),
    ),
    case(
        "boxes.Boxes3D.to_tensor[xyzwhd]",
        G,
        lambda b: Boxes3D(b, mode="xyzxyz_plus").to_tensor("xyzwhd"),
        [boxes3d_hexa],
        tags=("3d",),
    ),
    case(
        "boxes.Boxes3D.to_tensor[vertices]",
        G,
        lambda b: Boxes3D(b, mode="xyzxyz_plus").to_tensor("vertices"),
        [boxes3d_hexa],
        tags=("3d",),
    ),
    case(
        "boxes.Boxes3D.get_boxes_shape",
        G,
        lambda b: Boxes3D(b, mode="xyzxyz_plus").get_boxes_shape(),
        [boxes3d_hexa],
        tags=("3d",),
    ),
    case(
        "boxes.Boxes3D.to_mask",
        G,
        lambda b: Boxes3D(b, mode="xyzxyz_plus").to_mask(D3, H3, W3),
        [boxes3d_hexa],
        tags=("3d",),
        note="D,H,W baked",
    ),
    case(
        "boxes.Boxes3D.transform_boxes",
        G,
        lambda b, M: Boxes3D(b, mode="xyzxyz_plus").transform_boxes(M).data,
        [boxes3d_hexa, aff44_3d],
        tags=("3d",),
    ),
    case(
        "boxes.Boxes3D.transform_boxes_",
        G,
        lambda b, M: Boxes3D(b, mode="xyzxyz_plus").transform_boxes_(M).data,
        [boxes3d_hexa, aff44_3d],
        tags=("3d",),
    ),
    case(
        "boxes.VideoBoxes.from_tensor",
        G,
        lambda b: VideoBoxes.from_tensor(b).data,
        [boxes_quad[None].repeat(1, 2, 1, 1, 1)],
        note="(B,T,N,4,2) vertices",
    ),
    case(
        "boxes.VideoBoxes.to_tensor",
        G,
        lambda b: VideoBoxes.from_tensor(b).to_tensor("xyxy"),
        [boxes_quad[None].repeat(1, 2, 1, 1, 1)],
    ),
    case(
        "boxes.VideoBoxes.transform_boxes",
        G,
        lambda b, M: VideoBoxes.from_tensor(b).transform_boxes(M).data,
        [boxes_quad[None].repeat(1, 2, 1, 1, 1), homo33.repeat(2, 1, 1)],
        note="M must be (B*T,3,3): internal data is flattened to (B*T,N,4,2)",
    ),
]

# ----------------------------------------------------------------------------- keypoints
G = "geometry.keypoints"
CASES += [
    case("keypoints.Keypoints.from_tensor", G, lambda k: Keypoints.from_tensor(k).data, [kpts], tags=("points",)),
    case("keypoints.Keypoints.to_tensor", G, lambda k: Keypoints(k).to_tensor(), [kpts], tags=("points",)),
    case(
        "keypoints.Keypoints.transform_keypoints",
        G,
        lambda k, M: Keypoints(k).transform_keypoints(M).data,
        [kpts, homo33],
        tags=("points",),
    ),
    case(
        "keypoints.Keypoints.transform_keypoints_",
        G,
        lambda k, M: Keypoints(k).transform_keypoints_(M).data,
        [kpts, homo33],
        tags=("points",),
    ),
    case(
        "keypoints.Keypoints.pad",
        G,
        lambda k, p: Keypoints(k).pad(p).data,
        [kpts, torch.tensor([[2, 2, 3, 3]])],
        tags=("points",),
    ),
    case(
        "keypoints.Keypoints.unpad",
        G,
        lambda k, p: Keypoints(k).unpad(p).data,
        [kpts, torch.tensor([[2, 2, 3, 3]])],
        tags=("points",),
    ),
    case(
        "keypoints.Keypoints.index_put",
        G,
        lambda k, v: Keypoints(k).index_put((torch.tensor([0]), torch.tensor([1])), v).data,
        [kpts, kpts[0, :1]],
        tags=("points",),
        note="indices baked",
    ),
    case("keypoints.Keypoints.clone", G, lambda k: Keypoints(k).clone().data, [kpts], tags=("points",)),
    case("keypoints.Keypoints.type", G, lambda k: Keypoints(k).type(torch.float64).data, [kpts], tags=("points",)),
    case(
        "keypoints.Keypoints3D.from_tensor",
        G,
        lambda k: Keypoints3D.from_tensor(k).data,
        [kpts3d],
        tags=("points", "3d"),
    ),
    case("keypoints.Keypoints3D.to_tensor", G, lambda k: Keypoints3D(k).to_tensor(), [kpts3d], tags=("points", "3d")),
    case(
        "keypoints.Keypoints3D.transform_keypoints",
        G,
        None,
        [],
        skip="NotImplementedError stub (kornia/geometry/keypoints.py:377)",
    ),
    case(
        "keypoints.Keypoints3D.transform_keypoints_",
        G,
        None,
        [],
        skip="NotImplementedError stub (kornia/geometry/keypoints.py:377)",
    ),
    case("keypoints.Keypoints3D.pad", G, None, [], skip="NotImplementedError stub (kornia/geometry/keypoints.py:355)"),
    case(
        "keypoints.Keypoints3D.unpad", G, None, [], skip="NotImplementedError stub (kornia/geometry/keypoints.py:364)"
    ),
    case("keypoints.Keypoints3D.clone", G, lambda k: Keypoints3D(k).clone().data, [kpts3d], tags=("points", "3d")),
]

# ----------------------------------------------------------------------------- subpix
G = "geometry.subpix"
CASES += [
    case("subpix.conv_soft_argmax2d", G, conv_soft_argmax2d, [heat]),
    case(
        "subpix.conv_soft_argmax2d[output_value,pixel,stride2]",
        G,
        conv_soft_argmax2d,
        [heat],
        {"output_value": True, "normalized_coordinates": False, "stride": (2, 2)},
    ),
    case(
        "subpix.conv_soft_argmax2d[tensor_temperature]",
        G,
        lambda x, t: conv_soft_argmax2d(x, temperature=t),
        [heat, torch.tensor(0.5)],
        note="temperature as live 0-d tensor; `if temperature <= 0` is a data-dependent guard",
    ),
    case("subpix.conv_soft_argmax3d", G, conv_soft_argmax3d, [heat3d], tags=("3d",)),
    case(
        "subpix.conv_soft_argmax3d[normalized,bonus]",
        G,
        conv_soft_argmax3d,
        [heat3d],
        {"normalized_coordinates": True, "strict_maxima_bonus": 10.0},
        tags=("3d",),
    ),
    case("subpix.conv_quad_interp3d", G, conv_quad_interp3d, [heat3d], tags=("3d",)),
    case(
        "subpix.conv_quad_interp3d[no_scale_steps]",
        G,
        conv_quad_interp3d,
        [heat3d],
        {"allow_scale_steps": False},
        tags=("3d",),
    ),
    case(
        "subpix.conv_quad_interp3d[precomputed_mask]",
        G,
        lambda x: conv_quad_interp3d(x, precomputed_nms_mask=nms3d(x, (3, 3, 3), mask_only=True)),
        [heat3d],
        tags=("3d",),
    ),
    case(
        "subpix.iterative_quad_interp3d",
        G,
        iterative_quad_interp3d,
        [heat3d],
        tags=("3d",),
        note="data-dependent candidate set",
    ),
    case(
        "subpix.iterative_quad_interp3d[max_candidates]",
        G,
        iterative_quad_interp3d,
        [heat3d],
        {"max_candidates": 32},
        tags=("3d",),
    ),
    case("subpix.spatial_softmax2d", G, spatial_softmax2d, [heat]),
    case("subpix.spatial_softmax2d[temperature]", G, spatial_softmax2d, [heat, torch.tensor(0.5)]),
    case("subpix.spatial_expectation2d", G, spatial_expectation2d, [spatial_softmax2d(heat)]),
    case(
        "subpix.spatial_expectation2d[pixel]",
        G,
        spatial_expectation2d,
        [spatial_softmax2d(heat)],
        {"normalized_coordinates": False},
    ),
    case("subpix.spatial_soft_argmax2d", G, spatial_soft_argmax2d, [heat]),
    case("subpix.spatial_soft_argmax2d[temperature]", G, spatial_soft_argmax2d, [heat, torch.tensor(0.5)]),
    case(
        "subpix.render_gaussian2d",
        G,
        render_gaussian2d,
        [torch.tensor([[0.1, -0.2], [0.3, 0.4]]), torch.tensor([[0.2, 0.3], [0.1, 0.1]])],
        {"size": (16, 24)},
        note="size baked; mean/std (N,2) normalized",
    ),
    case(
        "subpix.render_gaussian2d[pixel]",
        G,
        render_gaussian2d,
        [torch.tensor([[5.0, 8.0], [12.0, 20.0]]), torch.tensor([[2.0, 3.0], [1.0, 1.0]])],
        {"size": (16, 24), "normalized_coordinates": False},
    ),
    case("subpix.nms2d", G, nms2d, [heat], {"kernel_size": (3, 3)}),
    case("subpix.nms2d[mask_only,5x5]", G, nms2d, [heat], {"kernel_size": (5, 5), "mask_only": True}),
    case("subpix.nms3d", G, nms3d, [heat3d], {"kernel_size": (3, 3, 3)}, tags=("3d",)),
    case("subpix.nms3d[mask_only]", G, nms3d, [heat3d], {"kernel_size": (3, 3, 3), "mask_only": True}, tags=("3d",)),
    case("subpix.nms3d_minmax", G, nms3d_minmax, [heat3d], tags=("3d",)),
    case("subpix.SpatialSoftArgmax2d", G, SpatialSoftArgmax2d(), [heat]),
    case("subpix.ConvSoftArgmax2d", G, ConvSoftArgmax2d(), [heat]),
    case("subpix.ConvSoftArgmax3d", G, ConvSoftArgmax3d(), [heat3d], tags=("3d",)),
    case("subpix.ConvQuadInterp3d", G, ConvQuadInterp3d(), [heat3d], tags=("3d",)),
    case("subpix.IterativeQuadInterp3d", G, IterativeQuadInterp3d(), [heat3d], tags=("3d",)),
    case("subpix.AdaptiveQuadInterp3d", G, AdaptiveQuadInterp3d(), [heat3d], tags=("3d",), note="mode=auto"),
    case("subpix.AdaptiveQuadInterp3d[conv]", G, AdaptiveQuadInterp3d(mode="conv"), [heat3d], tags=("3d",)),
    case("subpix.AdaptiveQuadInterp3d[patch]", G, AdaptiveQuadInterp3d(mode="patch"), [heat3d], tags=("3d",)),
    case("subpix.NonMaximaSuppression2d", G, NonMaximaSuppression2d((3, 3)), [heat]),
    case("subpix.NonMaximaSuppression3d", G, NonMaximaSuppression3d((3, 3, 3)), [heat3d], tags=("3d",)),
]

# ----------------------------------------------------------------------------- line / plane / ray / vector / pose
G = "geometry.line"
CASES += [
    case(
        "line.fit_line", G, lambda p: (lambda ln: (ln.origin, ln.direction))(fit_line(p)), [line_pts], tags=("points",)
    ),
    case(
        "line.fit_line[weights]",
        G,
        lambda p, w: (lambda ln: (ln.origin, ln.direction))(fit_line(p, w)),
        [line_pts, torch.rand(1, 10)],
        tags=("points",),
    ),
    case(
        "line.ParametrizedLine.through",
        G,
        lambda a, b: (lambda ln: (ln.origin, ln.direction))(ParametrizedLine.through(a, b)),
        [p3, q3],
    ),
    case(
        "line.ParametrizedLine.point_at",
        G,
        lambda a, b, t: ParametrizedLine.through(a, b).point_at(t),
        [p3, q3, torch.tensor(0.3)],
    ),
    case(
        "line.ParametrizedLine.point_at[float]", G, lambda a, b: ParametrizedLine.through(a, b).point_at(0.3), [p3, q3]
    ),
    case(
        "line.ParametrizedLine.projection",
        G,
        lambda a, b, p: ParametrizedLine.through(a, b).projection(p),
        [p3, q3, torch.tensor([0.0, 1.0, 0.0])],
    ),
    case(
        "line.ParametrizedLine.distance",
        G,
        lambda a, b, p: ParametrizedLine.through(a, b).distance(p),
        [p3, q3, torch.tensor([0.0, 1.0, 0.0])],
    ),
    case(
        "line.ParametrizedLine.squared_distance",
        G,
        lambda a, b, p: ParametrizedLine.through(a, b).squared_distance(p),
        [p3, q3, torch.tensor([0.0, 1.0, 0.0])],
    ),
    case(
        "line.ParametrizedLine.intersect",
        G,
        lambda a, b, n, e: ParametrizedLine.through(a, b).intersect(Hyperplane.from_vector(Vector3(n), Vector3(e))),
        [p3, q3, torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0, 0.0, 5.0])],
        note="returns (lambda, point)",
    ),
    case(
        "line.ParametrizedLine.dim",
        G,
        lambda a, b: ParametrizedLine.through(a, b).dim(),
        [p3, q3],
        note="returns Python int",
    ),
    case(
        "ray.Ray",
        G,
        lambda a, b: (lambda r: (r.origin, r.direction))(Ray.through(a, b)),
        [p3, q3],
        note="Ray is an alias of ParametrizedLine",
    ),
]
G = "geometry.plane"
CASES += [
    case(
        "plane.fit_plane",
        G,
        lambda p: (lambda pl: (pl.normal.data, pl.offset.data))(fit_plane(Vector3(p))),
        [plane_pts],
        tags=("points",),
        note="SVD",
    ),
    case(
        "plane.Hyperplane.through",
        G,
        lambda a, b, c: (lambda pl: (pl.normal.data, pl.offset.data))(Hyperplane.through(a, b, c)),
        [p3, q3, torch.tensor([0.0, 1.0, 5.0])],
    ),
    case(
        "plane.Hyperplane.through[2d]",
        G,
        lambda a, b: (lambda pl: (pl.normal.data, pl.offset.data))(Hyperplane.through(a, b)),
        [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 5.0]])],
        note="KORNIA BUG: 2-D branch wraps a 2-vector into Vector3 -> always fails validation (plane.py:185)",
    ),
    case(
        "plane.Hyperplane.from_vector",
        G,
        lambda n, e: (lambda pl: (pl.normal.data, pl.offset.data))(Hyperplane.from_vector(Vector3(n), Vector3(e))),
        [torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0, 0.0, 5.0])],
    ),
    case(
        "plane.Hyperplane.signed_distance",
        G,
        lambda n, e, p: Hyperplane.from_vector(Vector3(n), Vector3(e)).signed_distance(Vector3(p)).data,
        [torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0, 0.0, 5.0]), v3],
    ),
    case(
        "plane.Hyperplane.abs_distance",
        G,
        lambda n, e, p: Hyperplane.from_vector(Vector3(n), Vector3(e)).abs_distance(Vector3(p)).data,
        [torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0, 0.0, 5.0]), v3],
    ),
    case(
        "plane.Hyperplane.projection",
        G,
        lambda n, e, p: Hyperplane.from_vector(Vector3(n), Vector3(e)).projection(Vector3(p)).data,
        [torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0, 0.0, 5.0]), v3],
    ),
]
G = "geometry.vector"
CASES += [
    case("vector.Vector3.normalized", G, lambda v: Vector3(v).normalized().data, [v3]),
    case("vector.Vector3.dot", G, lambda a, b: Vector3(a).dot(Vector3(b)).data, [v3, torch.randn(4, 3)]),
    case("vector.Vector3.squared_norm", G, lambda v: Vector3(v).squared_norm().data, [v3]),
    case(
        "vector.Vector3.from_coords",
        G,
        lambda x, y, z: Vector3.from_coords(x, y, z).data,
        [torch.randn(4), torch.randn(4), torch.randn(4)],
    ),
    case("vector.Vector3.xyz", G, lambda v: (Vector3(v).x, Vector3(v).y, Vector3(v).z), [v3]),
    case(
        "vector.Vector3.arith",
        G,
        lambda a, b: (Vector3(a) + Vector3(b)).data * 2.0 - (Vector3(a) - Vector3(b)).data,
        [v3, torch.randn(4, 3)],
        note="TensorWrapper __add__/__sub__/__mul__",
    ),
    case("vector.Vector2.normalized", G, lambda v: Vector2(v).normalized().data, [torch.randn(4, 2)]),
    case("vector.Vector2.dot", G, lambda a, b: Vector2(a).dot(Vector2(b)).data, [torch.randn(4, 2), torch.randn(4, 2)]),
    case("vector.Vector2.squared_norm", G, lambda v: Vector2(v).squared_norm().data, [torch.randn(4, 2)]),
    case(
        "vector.Vector2.from_coords", G, lambda x, y: Vector2.from_coords(x, y).data, [torch.randn(4), torch.randn(4)]
    ),
    case("vector.Scalar", G, lambda s: (Scalar(s) * 2.0).data, [torch.randn(4)]),
    case("vector.Vector3.random", G, None, [], skip="random constructor; no tensor input, non-deterministic"),
]
G = "geometry.pose"
CASES += [
    case(
        "pose.NamedPose.from_matrix.transform_points",
        G,
        lambda M, p: NamedPose.from_matrix(M, "a", "b").transform_points(p),
        [pose44, pts3[0]],
        tags=("points", "3d"),
        note="points (N,3) with a batch-1 pose; (B,N,3) is rejected by So3.__mul__",
    ),
    case(
        "pose.NamedPose.from_matrix.pose",
        G,
        lambda M: NamedPose.from_matrix(M, "a", "b").pose.matrix(),
        [pose44],
        tags=("3d",),
    ),
    case(
        "pose.NamedPose.from_rt",
        G,
        lambda R, t: NamedPose.from_rt(R, t, "a", "b").pose.matrix(),
        [so3.matrix(), torch.tensor([[0.5, -1.0, 2.0]])],
        tags=("3d",),
        note="rotation as (B,3,3) matrix",
    ),
    case(
        "pose.NamedPose.inverse",
        G,
        lambda M: NamedPose.from_matrix(M, "a", "b").inverse().pose.matrix(),
        [pose44],
        tags=("3d",),
    ),
    case(
        "pose.NamedPose.mul",
        G,
        lambda M: (NamedPose.from_matrix(M, "b", "c") * NamedPose.from_matrix(M, "a", "b")).pose.matrix(),
        [pose44],
        tags=("3d",),
    ),
    case(
        "pose.NamedPose.rotation_translation",
        G,
        lambda M: (lambda np_: (np_.rotation.matrix(), np_.translation))(NamedPose.from_matrix(M, "a", "b")),
        [pose44],
        tags=("3d",),
    ),
    case(
        "pose.NamedPose.from_matrix[se2]",
        G,
        lambda M, p: NamedPose.from_matrix(M, "a", "b").transform_points(p),
        [
            torch.tensor(
                [[[math.cos(0.3), -math.sin(0.3), 1.0], [math.sin(0.3), math.cos(0.3), -2.0], [0.0, 0.0, 1.0]]]
            ),
            torch.randn(5, 2),
        ],
        tags=("points",),
        note="3x3 -> Se2 path; points (N,2)",
    ),
    case("pose.check_matrix_shape", G, None, [], skip="validation helper, returns None"),
    case(
        "pose.Quaternion/Se2/Se3/So2/So3",
        G,
        None,
        [],
        skip="re-exports of kornia.geometry.liegroup / quaternion — covered by the liegroup area",
    ),
]

if __name__ == "__main__":
    run_cases(CASES, sys.argv[1], only=sys.argv[2:] or None)
