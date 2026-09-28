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

"""ONNX export survey cases for kornia.feature (survey v2)."""

from __future__ import annotations

import sys
import traceback

import torch
import torch.nn.functional as F
from harness import case, run_cases

import kornia.feature as KF

torch.manual_seed(0)

# ----------------------------------------------------------------------------- shared inputs
IMG = torch.rand(1, 1, 32, 40)  # gray
IMG3 = torch.rand(1, 3, 32, 40)
IMG_BIG = torch.rand(1, 1, 64, 80)  # detectors need a pyramid
IMG3_BIG = torch.rand(1, 3, 64, 80)
IMG3_96 = torch.rand(1, 3, 64, 96)  # divisible by 16 and 32 (DISK / XFeat)
IMG3_DEDODE = torch.rand(1, 3, 56, 84)  # divisible by 14
PATCH32 = torch.rand(2, 1, 32, 32)
PATCH41 = torch.rand(2, 1, 41, 41)
PATCH19 = torch.rand(2, 1, 19, 19)


def make_lafs(B: int = 1, N: int = 5, H: int = 32, W: int = 40, scale: float = 4.0, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    xy = torch.stack([torch.rand(B, N, generator=g) * (W - 16) + 8, torch.rand(B, N, generator=g) * (H - 16) + 8], -1)
    sc = torch.full((B, N, 1, 1), scale) * (0.75 + 0.5 * torch.rand(B, N, 1, 1, generator=g))
    ori = torch.rand(B, N, 1, generator=g) * 360.0 - 180.0
    return KF.laf_from_center_scale_ori(xy, sc, ori)


LAF = make_lafs()  # (1, 5, 2, 3) inside 32x40
LAF_BIG = make_lafs(1, 8, 64, 80, 6.0, seed=1)
XY = LAF[..., :, 2].clone()  # (1, 5, 2)
SCALE = KF.get_laf_scale(LAF)  # (1, 5, 1, 1)
ORI = KF.get_laf_orientation(LAF)  # (1, 5, 1)
ANGLES = torch.rand(1, 5, 1) * 90.0
# ellipse (x, y, a, b, c) with [a b; b c] positive definite
_s = torch.tensor([1 / 16.0, 1 / 25.0, 1 / 9.0, 1 / 20.0, 1 / 12.0])
ELLS = torch.stack([XY[0, :, 0], XY[0, :, 1], _s, 0.1 * _s, 1.3 * _s], -1)[None]  # (1, 5, 5)
H_01 = torch.tensor([[[1.02, 0.05, 1.5], [-0.03, 0.98, -1.0], [1e-4, -2e-4, 1.0]]])  # (1, 3, 3)

# descriptors: desc2 = perturbed desc1 (+ 2 distractors) so mutual/ratio tests actually match
_g = torch.Generator().manual_seed(3)
DESC1 = F.normalize(torch.randn(10, 128, generator=_g), dim=1)
DESC2 = torch.cat([DESC1 + 0.05 * torch.randn(10, 128, generator=_g), torch.randn(2, 128, generator=_g)])
DESC2 = F.normalize(DESC2[torch.randperm(12, generator=_g)], dim=1)
LAFS1 = make_lafs(1, 10, 64, 80, 5.0, seed=5)
LAFS2 = make_lafs(1, 12, 64, 80, 5.0, seed=6)
KPTS0 = KF.get_laf_center(LAFS1)  # (1, 10, 2)
KPTS1 = KF.get_laf_center(LAFS2)  # (1, 12, 2)
IMG_SIZE = torch.tensor([[80.0, 64.0]])  # (1, 2) as (W, H)


class DictPair(torch.nn.Module):
    """Wrap a model taking {"image0", "image1"} so the model is a registered submodule (torch.export requirement)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, a, b):
        return self.m({"image0": a, "image1": b})


class LGWrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, k0, d0, k1, d1, s):
        return self.m(
            {
                "image0": {"keypoints": k0, "descriptors": d0, "image_size": s},
                "image1": {"keypoints": k1, "descriptors": d1, "image_size": s},
            }
        )


class DISKWrap(torch.nn.Module):
    def __init__(self, m, **kw):
        super().__init__()
        self.m = m
        self.kw = kw

    def forward(self, im):
        feats = self.m(im, **self.kw)
        return tuple(t for f in feats for t in (f.keypoints, f.descriptors, f.detection_scores))


class ALIKEDWrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, im):
        feats = self.m(im)
        return tuple(t for f in feats for t in (f.keypoints, f.descriptors, f.keypoint_scores))


def try_build(name: str, fn):
    """Construct a pretrained model; return (model, None) or (None, reason)."""
    try:
        return fn(), None
    except Exception as e:
        traceback.print_exc()
        return None, f"could not build/download {name}: {type(e).__name__}: {str(e)[:120]}"


def pcase(name, group, builder, inputs, kwargs=None, **kw):
    """Case whose target is a pretrained model built lazily; skip with the download error if it fails."""
    target, reason = try_build(name, builder)
    if target is None:
        return case(name, group, None, inputs, kwargs, skip=reason, **kw)
    return case(name, group, target, inputs, kwargs, **kw)


CASES: list = []

# ----------------------------------------------------------------------------- responses
R = "feature.responses"
CASES += [
    case("feature.harris_response", R, KF.harris_response, [IMG]),
    case("feature.harris_response[sigmas]", R, KF.harris_response, [IMG, torch.tensor([1.5])], note="sigmas (B,) live"),
    case("feature.harris_response[diff]", R, KF.harris_response, [IMG], {"grads_mode": "diff"}),
    case("feature.gftt_response", R, KF.gftt_response, [IMG]),
    case("feature.hessian_response", R, KF.hessian_response, [IMG]),
    case(
        "feature.dog_response",
        R,
        KF.dog_response,
        [torch.rand(1, 1, 4, 32, 40)],
        note="input is (B, C, L, H, W) scale levels",
        tags=("3d",),
    ),
    case("feature.dog_response_single", R, KF.dog_response_single, [IMG], {"sigma1": 1.0, "sigma2": 1.6}),
    case("feature.CornerHarris", R, KF.CornerHarris(0.04), [IMG]),
    case("feature.CornerGFTT", R, KF.CornerGFTT(), [IMG]),
    case("feature.BlobHessian", R, KF.BlobHessian(), [IMG]),
    case(
        "feature.BlobDoG",
        R,
        KF.BlobDoG(),
        [torch.rand(1, 1, 4, 32, 40)],
        tags=("3d",),
        note="docstring says (B,C,H,W) but dog_response requires 5-D",
    ),
    case("feature.BlobDoGSingle", R, KF.BlobDoGSingle(), [IMG]),
    case("feature.BlobHessian[batch2,rgb]", R, KF.BlobHessian(), [torch.rand(2, 3, 32, 40)], tags=("batch>1",)),
]

# ----------------------------------------------------------------------------- laf.py
L = "feature.laf"
CASES += [
    case("feature.laf_from_center_scale_ori", L, KF.laf_from_center_scale_ori, [XY, SCALE, ORI]),
    case("feature.laf_from_center_scale_ori[xy-only]", L, KF.laf_from_center_scale_ori, [XY], note="scale/ori default"),
    case("feature.get_laf_scale", L, KF.get_laf_scale, [LAF]),
    case("feature.get_laf_center", L, KF.get_laf_center, [LAF]),
    case("feature.get_laf_orientation", L, KF.get_laf_orientation, [LAF]),
    case("feature.set_laf_orientation", L, KF.set_laf_orientation, [LAF, ANGLES]),
    case("feature.rotate_laf", L, KF.rotate_laf, [LAF, ANGLES]),
    case("feature.scale_laf", L, KF.scale_laf, [LAF], {"scale_coef": 2.0}, note="scale_coef float baked"),
    case(
        "feature.scale_laf[tensor]",
        L,
        KF.scale_laf,
        [LAF, torch.rand(1, 5, 1, 1) + 0.5],
        note="scale_coef (B,N,1,1) live",
    ),
    case("feature.make_upright", L, KF.make_upright, [LAF]),
    case("feature.ellipse_to_laf", L, KF.ellipse_to_laf, [ELLS]),
    case("feature.laf_to_boundary_points", L, KF.laf_to_boundary_points, [LAF], {"n_pts": 12}, note="n_pts baked"),
    case(
        "feature.get_laf_pts_to_draw",
        L,
        KF.laf.get_laf_pts_to_draw,
        [LAF],
        {"img_idx": 0},
        note="returns two Python lists -> not exportable by design",
    ),
    case("feature.denormalize_laf", L, KF.denormalize_laf, [KF.normalize_laf(LAF, IMG), IMG]),
    case("feature.normalize_laf", L, KF.normalize_laf, [LAF, IMG]),
    case(
        "feature.laf.generate_patch_grid_from_normalized_LAF",
        L,
        KF.laf.generate_patch_grid_from_normalized_LAF,
        [IMG, KF.normalize_laf(LAF, IMG)],
        {"PS": 16},
        note="PS baked",
    ),
    case("feature.extract_patches_simple", L, KF.extract_patches_simple, [IMG, LAF], {"PS": 16}, note="PS baked"),
    case(
        "feature.extract_patches_simple[rgb,batch2]",
        L,
        KF.extract_patches_simple,
        [torch.rand(2, 3, 32, 40), make_lafs(2, 5)],
        {"PS": 16},
        tags=("batch>1",),
    ),
    case(
        "feature.extract_patches_from_pyramid",
        L,
        KF.extract_patches_from_pyramid,
        [IMG_BIG, LAF_BIG],
        {"PS": 16},
        note="PS baked",
    ),
    case("feature.laf_is_valid", L, KF.laf_is_valid, [LAF], note="bool (B,N) output"),
    case("feature.laf_is_inside_image", L, KF.laf_is_inside_image, [LAF, IMG], {"border": 2}, note="bool output"),
    case("feature.laf_to_three_points", L, KF.laf_to_three_points, [LAF]),
    case("feature.laf_from_three_points", L, KF.laf_from_three_points, [KF.laf_to_three_points(LAF)]),
    case("feature.perspective_transform_lafs", L, KF.perspective_transform_lafs, [H_01, LAF]),
    case("feature.KORNIA_CHECK_LAF", L, KF.KORNIA_CHECK_LAF, [LAF], note="validation helper; returns bool/None"),
]

# ----------------------------------------------------------------------------- orientation / affine shape
O = "feature.orientation_affine"
CASES += [
    case("feature.PassLAF", O, KF.PassLAF(), [LAF, IMG]),
    case("feature.PatchDominantGradientOrientation", O, KF.PatchDominantGradientOrientation(32), [PATCH32]),
    pcase(
        "feature.OriNet",
        O,
        lambda: KF.OriNet(pretrained=True),
        [PATCH32],
        tags=("model", "pretrained"),
        note="weights: OriNet.pth",
    ),
    case("feature.LAFOrienter", O, KF.LAFOrienter(19), [LAF_BIG, IMG_BIG], note="patch_size=19 (default PDGO)"),
    pcase(
        "feature.LAFOrienter[OriNet]",
        O,
        lambda: KF.LAFOrienter(32, angle_detector=KF.OriNet(pretrained=True)),
        [LAF_BIG, IMG_BIG],
        tags=("model", "pretrained"),
    ),
    case("feature.PatchAffineShapeEstimator", O, KF.PatchAffineShapeEstimator(19), [PATCH19]),
    case("feature.LAFAffineShapeEstimator", O, KF.LAFAffineShapeEstimator(19), [LAF_BIG, IMG_BIG]),
    pcase(
        "feature.LAFAffNetShapeEstimator",
        O,
        lambda: KF.LAFAffNetShapeEstimator(pretrained=True),
        [LAF_BIG, IMG_BIG],
        tags=("model", "pretrained"),
        note="weights: AffNet.pth",
    ),
]

# ----------------------------------------------------------------------------- descriptors
D = "feature.descriptors"
CASES += [
    case("feature.SIFTDescriptor", D, KF.SIFTDescriptor(41), [PATCH41], note="patch_size=41 default; rootsift"),
    case("feature.SIFTDescriptor[32,no-rootsift]", D, KF.SIFTDescriptor(32, rootsift=False), [PATCH32]),
    case("feature.siftdesc.sift_describe", D, KF.siftdesc.sift_describe, [PATCH32], {"patch_size": 32}),
    case(
        "feature.siftdesc.get_sift_pooling_kernel",
        D,
        lambda d: KF.siftdesc.get_sift_pooling_kernel(25).to(d.device),
        [torch.zeros(1)],
        note="ksize baked; fully constant graph",
    ),
    case("feature.DenseSIFTDescriptor", D, KF.DenseSIFTDescriptor(), [IMG]),
    pcase(
        "feature.MKDDescriptor",
        D,
        lambda: KF.MKDDescriptor(patch_size=32),
        [PATCH32],
        tags=("pretrained",),
        note="whitening weights mkd-concat-64.pth; patch_size=32",
    ),
    pcase("feature.MKDDescriptor[no-whitening]", D, lambda: KF.MKDDescriptor(patch_size=32, whitening=None), [PATCH32]),
    pcase(
        "feature.HardNet",
        D,
        lambda: KF.HardNet(pretrained=True),
        [PATCH32],
        tags=("model", "pretrained"),
        note="weights: checkpoint_liberty_with_aug.pth",
    ),
    pcase(
        "feature.HardNet8",
        D,
        lambda: KF.HardNet8(pretrained=True),
        [PATCH32],
        tags=("model", "pretrained"),
        note="weights: hardnet8v2.pt",
    ),
    pcase(
        "feature.HyNet",
        D,
        lambda: KF.HyNet(pretrained=True),
        [PATCH32],
        tags=("model", "pretrained"),
        note="weights: HyNet_LIB.pth",
    ),
    pcase(
        "feature.TFeat",
        D,
        lambda: KF.TFeat(pretrained=True),
        [PATCH32],
        tags=("model", "pretrained"),
        note="weights: tfeat-liberty.params",
    ),
    pcase(
        "feature.SOSNet",
        D,
        lambda: KF.SOSNet(pretrained=True),
        [PATCH32],
        tags=("model", "pretrained"),
        note="weights: sosnet_32x32_liberty.pth",
    ),
    case("feature.FilterResponseNorm2d", D, KF.FilterResponseNorm2d(8), [torch.rand(1, 8, 16, 20)]),
    case("feature.TLU", D, KF.TLU(8), [torch.rand(1, 8, 16, 20)]),
    pcase(
        "feature.LAFDescriptor",
        D,
        lambda: KF.LAFDescriptor(KF.HardNet(pretrained=True), patch_size=32),
        [IMG_BIG, LAF_BIG],
        tags=("model", "pretrained"),
    ),
    case(
        "feature.LAFDescriptor[SIFT,rgb]",
        D,
        KF.LAFDescriptor(KF.SIFTDescriptor(32), patch_size=32),
        [IMG3_BIG, LAF_BIG],
        note="grayscale_descriptor=True converts rgb",
    ),
    case(
        "feature.get_laf_descriptors",
        D,
        KF.get_laf_descriptors,
        [IMG_BIG, LAF_BIG],
        {"patch_descriptor": KF.SIFTDescriptor(32), "patch_size": 32},
    ),
    case(
        "feature.steerers.DiscreteSteerer",
        D,
        KF.steerers.DiscreteSteerer(torch.linalg.qr(torch.randn(128, 128))[0]),
        [DESC1],
    ),
    case(
        "feature.steerers.DiscreteSteerer.steer_descriptions",
        D,
        KF.steerers.DiscreteSteerer(torch.linalg.qr(torch.randn(128, 128))[0]),
        [DESC1],
        {"steerer_power": 3, "normalize": True},
        method="steer_descriptions",
    ),
]

# ----------------------------------------------------------------------------- detectors / pipelines (classical)
T = "feature.detectors"
CASES += [
    case(
        "feature.ScaleSpaceDetector",
        T,
        KF.ScaleSpaceDetector(num_features=32),
        [IMG_BIG],
        note="num_features=32 baked; default ScalePyramid(3,1.6,15) + BlobHessian + ConvQuadInterp3d",
    ),
    case(
        "feature.ScaleSpaceDetector[mask]",
        T,
        KF.ScaleSpaceDetector(num_features=32),
        [IMG_BIG, (torch.rand(1, 1, 64, 80) > 0.3).float()],
        note="mask live",
    ),
    case(
        "feature.MultiResolutionDetector",
        T,
        KF.MultiResolutionDetector(KF.BlobHessian(), num_features=32),
        [IMG_BIG],
        note="BlobHessian response; num_features=32",
    ),
    pcase(
        "feature.KeyNet",
        T,
        lambda: KF.KeyNet(pretrained=True),
        [IMG],
        tags=("model", "pretrained"),
        note="weights: keynet_pytorch.pth; dense response map",
    ),
    pcase(
        "feature.KeyNetDetector",
        T,
        lambda: KF.KeyNetDetector(pretrained=True, num_features=32),
        [IMG_BIG],
        tags=("model", "pretrained"),
    ),
    case("feature.SIFTFeature", T, KF.SIFTFeature(num_features=32), [IMG_BIG], note="num_features=32"),
    case(
        "feature.SIFTFeature[upright]",
        T,
        KF.SIFTFeature(num_features=32, upright=True),
        [IMG_BIG],
        note="RGB input is rejected eagerly ('model must return one response map'); gray only",
    ),
    case("feature.SIFTFeatureScaleSpace", T, KF.SIFTFeatureScaleSpace(num_features=32), [IMG_BIG]),
    pcase(
        "feature.GFTTAffNetHardNet",
        T,
        lambda: KF.GFTTAffNetHardNet(num_features=32),
        [IMG_BIG],
        tags=("model", "pretrained"),
    ),
    pcase(
        "feature.HesAffNetHardNet",
        T,
        lambda: KF.HesAffNetHardNet(num_features=32),
        [IMG_BIG],
        tags=("model", "pretrained"),
    ),
    pcase(
        "feature.KeyNetHardNet", T, lambda: KF.KeyNetHardNet(num_features=32), [IMG_BIG], tags=("model", "pretrained")
    ),
    pcase(
        "feature.KeyNetAffNetHardNet",
        T,
        lambda: KF.KeyNetAffNetHardNet(num_features=32),
        [IMG_BIG],
        tags=("model", "pretrained"),
    ),
    case(
        "feature.LocalFeature",
        T,
        KF.LocalFeature(KF.ScaleSpaceDetector(num_features=32), KF.LAFDescriptor(KF.SIFTDescriptor(32), 32)),
        [IMG_BIG],
        note="ScaleSpaceDetector + SIFT",
    ),
]

# ----------------------------------------------------------------------------- deep detectors / describers
M = "feature.models"
_disk, _disk_err = try_build("DISK", lambda: KF.DISK.from_pretrained("depth"))
CASES += [
    case(
        "feature.DISK",
        M,
        DISKWrap(_disk, n=64, window_size=5) if _disk else None,
        [IMG3_96],
        skip=_disk_err,
        tags=("model", "pretrained"),
        note="weights depth-save.pth; n=64 top-k; H,W %16",
    ),
    case(
        "feature.DISK[n=None]",
        M,
        DISKWrap(_disk, n=None, window_size=5) if _disk else None,
        [IMG3_96],
        skip=_disk_err,
        tags=("model", "pretrained"),
        note="all NMS maxima (data-dependent count)",
    ),
    case(
        "feature.DISK.heatmap_and_dense_descriptors",
        M,
        _disk,
        [IMG3_96],
        skip=_disk_err,
        method="heatmap_and_dense_descriptors",
        tags=("model", "pretrained"),
        note="dense head only",
    ),
]
_dedode, _dedode_err = try_build(
    "DeDoDe", lambda: KF.DeDoDe.from_pretrained("L-upright", "B-upright", amp_dtype=torch.float32)
)
CASES += [
    case(
        "feature.DeDoDe",
        M,
        _dedode,
        [IMG3_DEDODE],
        {"n": 64},
        skip=_dedode_err,
        tags=("model", "pretrained"),
        note="detector L-upright + descriptor B-upright (VGG19), float32; n=64",
    ),
    case(
        "feature.DeDoDe[no-imagenet-norm]",
        M,
        _dedode,
        [IMG3_DEDODE],
        {"n": 64, "apply_imagenet_normalization": False},
        skip=_dedode_err,
        tags=("model", "pretrained"),
        note="bypasses kornia.enhance.Normalize ONNX guard",
    ),
    case(
        "feature.DeDoDe.detect",
        M,
        _dedode,
        [IMG3_DEDODE],
        {"n": 64},
        skip=_dedode_err,
        method="detect",
        tags=("model", "pretrained"),
    ),
    case(
        "feature.DeDoDe.detect[no-imagenet-norm]",
        M,
        _dedode,
        [IMG3_DEDODE],
        {"n": 64, "apply_imagenet_normalization": False},
        skip=_dedode_err,
        method="detect",
        tags=("model", "pretrained"),
    ),
    case(
        "feature.DeDoDe.describe",
        M,
        _dedode,
        [IMG3_DEDODE, torch.rand(1, 64, 2) * 2 - 1],
        skip=_dedode_err,
        method="describe",
        tags=("model", "pretrained"),
        note="keypoints normalised [-1,1]",
    ),
    case(
        "feature.DeDoDe.describe[no-imagenet-norm]",
        M,
        _dedode,
        [IMG3_DEDODE, torch.rand(1, 64, 2) * 2 - 1],
        {"apply_imagenet_normalization": False},
        skip=_dedode_err,
        method="describe",
        tags=("model", "pretrained"),
    ),
]
_aliked, _aliked_err = try_build("ALIKED", lambda: KF.ALIKED.from_pretrained("aliked-t16", max_num_keypoints=64))
CASES += [
    case(
        "feature.ALIKED",
        M,
        ALIKEDWrap(_aliked) if _aliked else None,
        [IMG3_96],
        skip=_aliked_err,
        tags=("model", "pretrained"),
        note="aliked-t16; max_num_keypoints=64",
    ),
    case(
        "feature.ALIKED.forward_laf",
        M,
        _aliked,
        [IMG3_96],
        skip=_aliked_err,
        method="forward_laf",
        tags=("model", "pretrained"),
    ),
    case(
        "feature.ALIKED.extract_dense_map",
        M,
        _aliked,
        [IMG3_96],
        skip=_aliked_err,
        method="extract_dense_map",
        tags=("model", "pretrained"),
        note="dense backbone incl. deformable conv",
    ),
]
_xfeat, _xfeat_err = try_build("XFeat", lambda: KF.XFeat.from_pretrained(top_k=64))
CASES += [
    case(
        "feature.XFeatModel",
        M,
        _xfeat.net if _xfeat else None,
        [IMG3_96],
        skip=_xfeat_err,
        tags=("model", "pretrained"),
        note="dense backbone (feats, keypoints, heatmap); H,W %32",
    ),
    case(
        "feature.XFeat.detectAndCompute",
        M,
        _xfeat,
        [IMG3_96],
        skip=_xfeat_err,
        method="detectAndCompute",
        tags=("model", "pretrained"),
        note="top_k=64",
    ),
    case(
        "feature.XFeat.detectAndComputeDense",
        M,
        _xfeat,
        [IMG3_96],
        {"multiscale": False},
        skip=_xfeat_err,
        method="detectAndComputeDense",
        tags=("model", "pretrained"),
    ),
    case(
        "feature.XFeat.match_xfeat",
        M,
        _xfeat,
        [IMG3_96, torch.rand(1, 3, 64, 96)],
        skip=_xfeat_err,
        method="match_xfeat",
        tags=("model", "pretrained"),
    ),
    case(
        "feature.XFeat.match_xfeat_star",
        M,
        _xfeat,
        [IMG3_96, torch.rand(1, 3, 64, 96)],
        skip=_xfeat_err,
        method="match_xfeat_star",
        tags=("model", "pretrained"),
    ),
    case(
        "feature.InterpolateSparse2d",
        M,
        KF.InterpolateSparse2d("bicubic"),
        [torch.rand(1, 8, 16, 20), torch.rand(1, 6, 2) * torch.tensor([19.0, 15.0])],
        {"H": 16, "W": 20},
        note="H,W baked",
    ),
]
_sold2d, _sold2d_err = try_build("SOLD2_detector", lambda: KF.SOLD2_detector(pretrained=True))
_sold2, _sold2_err = try_build("SOLD2", lambda: KF.SOLD2(pretrained=True))
CASES += [
    case(
        "feature.SOLD2_detector",
        M,
        _sold2d,
        [IMG_BIG],
        skip=_sold2d_err,
        tags=("model", "pretrained"),
        note="weights sold2_wireframe.pth; returns list of line segments + heatmaps",
    ),
    case("feature.SOLD2", M, _sold2, [IMG_BIG], skip=_sold2_err, tags=("model", "pretrained")),
]
_defmo, _defmo_err = try_build("DeFMO", lambda: KF.DeFMO(pretrained=True))
CASES += [
    case(
        "feature.DeFMO",
        M,
        _defmo,
        [torch.rand(1, 6, 64, 96)],
        skip=_defmo_err,
        tags=("model", "pretrained"),
        note="weights encoder_best.pt + rendering_best.pt",
    ),
]

# ----------------------------------------------------------------------------- matchers
X = "feature.matching"
CASES += [
    case("feature.match_nn", X, KF.match_nn, [DESC1, DESC2]),
    case(
        "feature.match_nn[dm]",
        X,
        KF.match_nn,
        [DESC1, DESC2, torch.cdist(DESC1, DESC2)],
        note="dm passed live (bypasses torch.cdist)",
    ),
    case("feature.match_mnn", X, KF.match_mnn, [DESC1, DESC2]),
    case("feature.match_mnn[dm]", X, KF.match_mnn, [DESC1, DESC2, torch.cdist(DESC1, DESC2)], note="dm passed live"),
    case("feature.match_snn", X, KF.match_snn, [DESC1, DESC2], {"th": 0.8}),
    case("feature.match_smnn", X, KF.match_smnn, [DESC1, DESC2], {"th": 0.8}),
    case("feature.match_fginn", X, KF.match_fginn, [DESC1, DESC2, LAFS1, LAFS2], {"th": 0.8, "spatial_th": 10.0}),
    case(
        "feature.match_fginn[mutual]",
        X,
        KF.match_fginn,
        [DESC1, DESC2, LAFS1, LAFS2],
        {"th": 0.8, "spatial_th": 10.0, "mutual": True},
    ),
    case(
        "feature.match_adalam",
        X,
        KF.match_adalam,
        [DESC1, DESC2, LAFS1, LAFS2],
        {"hw1": (64, 80), "hw2": (64, 80)},
        note="hw baked",
    ),
    case("feature.DescriptorMatcher[nn]", X, KF.DescriptorMatcher("nn"), [DESC1, DESC2]),
    case("feature.DescriptorMatcher[snn]", X, KF.DescriptorMatcher("snn", 0.8), [DESC1, DESC2]),
    case("feature.DescriptorMatcher[smnn]", X, KF.DescriptorMatcher("smnn", 0.8), [DESC1, DESC2]),
    case(
        "feature.GeometryAwareDescriptorMatcher[fginn]",
        X,
        KF.GeometryAwareDescriptorMatcher("fginn"),
        [DESC1, DESC2, LAFS1, LAFS2],
    ),
    case(
        "feature.GeometryAwareDescriptorMatcher[adalam]",
        X,
        KF.GeometryAwareDescriptorMatcher("adalam"),
        [DESC1, DESC2, LAFS1, LAFS2],
    ),
    case(
        "feature.matching.DescriptorMatcherWithSteerer[global]",
        X,
        KF.matching.DescriptorMatcherWithSteerer(
            KF.steerers.DiscreteSteerer(torch.linalg.qr(torch.randn(128, 128))[0]), 4, "global", "smnn", 0.98
        ),
        [DESC1, DESC2],
        {"normalize": True},
    ),
    case(
        "feature.matching.DescriptorMatcherWithSteerer[local]",
        X,
        KF.matching.DescriptorMatcherWithSteerer(
            KF.steerers.DiscreteSteerer(torch.linalg.qr(torch.randn(128, 128))[0]), 4, "local", "smnn", 0.98
        ),
        [DESC1, DESC2],
        {"normalize": True},
    ),
]

_loftr, _loftr_err = try_build("LoFTR", lambda: KF.LoFTR(pretrained="outdoor"))
CASES += [
    case(
        "feature.LoFTR",
        X,
        DictPair(_loftr) if _loftr else None,
        [IMG_BIG, torch.rand(1, 1, 64, 80)],
        skip=_loftr_err,
        tags=("model", "pretrained"),
        note="weights loftr_outdoor.ckpt; gray 1x1x64x80 pair",
    ),
]
_lg, _lg_err = try_build("LightGlue", lambda: KF.LightGlue("disk").eval())
_lg_np, _lg_np_err = try_build(
    "LightGlue[no-prune]", lambda: KF.LightGlue("disk", depth_confidence=-1, width_confidence=-1, flash=False).eval()
)


def _lg_call(m):
    return lambda k0, d0, k1, d1, s: m(
        {
            "image0": {"keypoints": k0, "descriptors": d0, "image_size": s},
            "image1": {"keypoints": k1, "descriptors": d1, "image_size": s},
        }
    )


CASES += [
    case(
        "feature.LightGlue",
        X,
        LGWrap(_lg) if _lg else None,
        [KPTS0, DESC1[None], KPTS1, DESC2[None], IMG_SIZE],
        skip=_lg_err,
        tags=("model", "pretrained"),
        note="disk weights; default conf (depth/width confidence early-exit)",
    ),
    case(
        "feature.LightGlue[no-prune]",
        X,
        LGWrap(_lg_np) if _lg_np else None,
        [KPTS0, DESC1[None], KPTS1, DESC2[None], IMG_SIZE],
        skip=_lg_np_err,
        tags=("model", "pretrained"),
        note="depth_confidence=-1, width_confidence=-1, flash=False",
    ),
]
_lgm, _lgm_err = try_build("LightGlueMatcher", lambda: KF.LightGlueMatcher("disk").eval())
CASES += [
    case(
        "feature.LightGlueMatcher",
        X,
        _lgm,
        [DESC1, DESC2, LAFS1, LAFS2],
        {"hw1": (64, 80), "hw2": (64, 80)},
        skip=_lgm_err,
        tags=("model", "pretrained"),
        note="hw baked",
    ),
]
_lfm, _lfm_err = try_build(
    "LocalFeatureMatcher",
    lambda: KF.LocalFeatureMatcher(KF.SIFTFeature(num_features=32), KF.DescriptorMatcher("snn", 0.9)),
)
CASES += [
    case(
        "feature.LocalFeatureMatcher",
        X,
        DictPair(_lfm) if _lfm else None,
        [IMG_BIG, torch.rand(1, 1, 64, 80)],
        skip=_lfm_err,
        note="SIFTFeature(32) + snn matcher; dict in/out",
    ),
    case(
        "feature.OnnxLightGlue",
        X,
        None,
        [],
        skip="already an onnxruntime wrapper (loads fabio-sim LightGlue-ONNX *_fused.onnx graphs); not a torch module",
    ),
    case("feature.DISKFeatures", M, None, [], skip="dataclass output container (no forward)"),
    case("feature.ALIKEDFeatures", M, None, [], skip="dataclass output container (no forward)"),
    case("feature.scale_space_detector.get_default_detector_config", T, None, [], skip="returns a Python dict"),
]

if __name__ == "__main__":
    run_cases(CASES, sys.argv[1], only=sys.argv[2:] or None)
