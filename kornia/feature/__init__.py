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

r"""Detect, describe and match local features, from classical operators to pretrained models.

Images are floating-point tensors ``(B, C, H, W)`` in :math:`[0, 1]`. Keypoints use ``(x, y)`` pixels with integer
pixel centres, except :class:`DeDoDe` and :class:`LoFTR`, which place pixel ``i`` at ``i + 0.5``. Line endpoints use
``(row, col)``. See each model's conventions for resizing and coordinate limitations.

A local affine frame (LAF) ``[A | c]`` has shape ``(B, N, 2, 3)`` in pixels. It describes a patch with centre ``c``,
scale :math:`\sqrt{|\det A|}` and orientation ``atan2(a12, a11)`` in degrees. Detectors that return LAFs pad unused
slots with zero frames. Use :func:`laf_is_filled` to keep detected features and discard padded slots; apply the
same mask to LAFs, responses and descriptors.

Which function to use:

- Match images: :class:`LoFTR` without a detector; :class:`LocalFeatureMatcher` with a local feature and a
  descriptor-only matcher; :meth:`~kornia.feature.XFeat.match_xfeat` or :meth:`~kornia.feature.XFeat.match_xfeat_star`.
- Detect and describe learned keypoints: :class:`ALIKED` (subpixel), :class:`DISK` (integer pixels), :class:`XFeat`
  (64-D descriptors) or :class:`DeDoDe` (separate detector and descriptor networks); load with ``from_pretrained``.
- Store per-image learned features: :class:`ALIKEDFeatures` and :class:`DISKFeatures` hold keypoints and descriptors.
- Compose a detector and a LAF descriptor: :class:`LocalFeature`; the presets below return the same output tuple.
- Extract SIFT features: :class:`SIFTFeatureScaleSpace` refines scale-space extrema;
  :class:`SIFTFeature` detects independently at each pyramid level and takes one image per call.
- Extract HardNet features from blobs: :class:`DoGHardNet`, :class:`DoGAffNetHardNet` or :class:`HesAffNetHardNet`;
  from corners: :class:`GFTTAffNetHardNet`; the AffNet presets also adapt affine shape.
- Extract HardNet features with KeyNet: :class:`KeyNetHardNet` or :class:`KeyNetAffNetHardNet`, one image per call.
- Detect LAFs: :class:`ScaleSpaceDetector` refines scale-space extrema in a batch; :class:`MultiResolutionDetector`
  finds per-level 2D maxima in one image; :class:`KeyNetDetector` uses the learned :class:`KeyNet` response.
- Score corners: :func:`harris_response` or :func:`gftt_response`; module forms: :class:`CornerHarris`,
  :class:`CornerGFTT`.
- Score blobs: :func:`hessian_response` or :func:`dog_response_single`; module forms: :class:`BlobHessian`,
  :class:`BlobDoGSingle`; for Gaussian scale stacks: :func:`dog_response` and :class:`BlobDoG`.
- Describe grayscale patches: :class:`SIFTDescriptor` or :class:`MKDDescriptor`; learned 32 x 32 descriptors:
  :class:`HardNet`, :class:`HardNet8`, :class:`HyNet`, :class:`SOSNet`, :class:`TFeat`.
- Describe LAFs in an image: :class:`LAFDescriptor` reuses a patch descriptor, :func:`get_laf_descriptors` takes one
  per call, and :class:`SIFTDescriptorFromPyramid` computes SIFT without extracting patches.
- Describe every position: :class:`DenseSIFTDescriptor`; convert SIFT bin layouts:
  :func:`convert_sift_descriptor_layout`.
- Orient LAFs: :class:`LAFOrienter` with :class:`PatchDominantGradientOrientation` or the learned :class:`OriNet`.
- Adapt affine shape: :class:`LAFAffineShapeEstimator` with :class:`PatchAffineShapeEstimator`, or the learned
  :class:`LAFAffNetShapeEstimator`; pass LAFs through unchanged: :class:`PassLAF`.
- Match descriptors: :func:`match_nn` (nearest), :func:`match_mnn` (mutual), :func:`match_snn` (ratio test),
  :func:`match_smnn` (mutual and ratio), or :class:`DescriptorMatcher`; padded batches: :func:`match_smnn_batched`.
- Match descriptors with LAF geometry: :func:`match_fginn`, :func:`match_adalam` or
  :class:`GeometryAwareDescriptorMatcher`.
- Match with a learned model: :class:`LightGlueMatcher` takes descriptors and LAFs; :class:`LightGlue` takes feature
  dictionaries; :class:`OnnxLightGlue` runs an exported model with onnxruntime.
- Verify matches geometrically: :class:`~kornia.geometry.ransac.RANSAC` fits a model and selects its inliers.
- Build LAFs: :func:`laf_from_center_scale_ori`, :func:`ellipse_to_laf` or :func:`laf_from_three_points`.
- Read LAFs: :func:`get_laf_center`, :func:`get_laf_scale`, :func:`get_laf_orientation`, :func:`laf_to_three_points`
  or :func:`laf_to_boundary_points`.
- Transform LAFs: :func:`scale_laf`, :func:`rotate_laf`, :func:`set_laf_orientation`, :func:`make_upright`,
  :func:`perspective_transform_lafs`, :func:`normalize_laf` or :func:`denormalize_laf`.
- Check LAFs: :func:`laf_is_filled` selects detections, :func:`laf_is_valid` checks finite nonsingular frames,
  :func:`laf_is_inside_image` checks bounds, and :func:`KORNIA_CHECK_LAF` checks shape.
- Extract patches: :func:`extract_patches_from_pyramid` limits aliasing; :func:`extract_patches_simple` samples
  directly.
- Process line segments: :class:`SOLD2` detects, describes and matches; :class:`SOLD2_detector` detects only.
- Reuse network components: :class:`FilterResponseNorm2d` and :class:`TLU` from HyNet; :class:`XFeatModel` and
  :class:`InterpolateSparse2d` from XFeat.
- Deblur a fast-moving object: :class:`DeFMO` produces RGBA sub-frames.

:class:`LocalFeatureMatcher` accepts descriptor-only matchers. To use :class:`LightGlueMatcher`, extract features
first and pass their descriptors and LAFs to it directly. For learned keypoints, construct LAFs with
:func:`laf_from_center_scale_ori`; subtract 0.5 from DeDoDe coordinates first and match XFeat outputs with gradients
disabled. See the model and matcher docstrings for complete recipes.

Choosing an implementation by device. Kornia's local-feature pipelines run at their authors' speed or faster on
CUDA, x86 and Apple M1 CPUs and Apple MPS, and are as accurate on the cohorts measured. Choose differently here:

- :class:`ALIKED` on CUDA: compile it for repeated inference. Eager, it runs at 0.8x the authors' speed; compiled, at
  1.6x. On CPUs keep it eager, at 1.7-2.0x.
- SIFT on a CPU: use OpenCV's, 8-14x faster than kornia's on the Apple M1 CPU. Kornia's is for images on a GPU,
  gradients or batches; :class:`SIFTFeature` takes one image per call, :class:`SIFTFeatureScaleSpace` a batch.
- :func:`match_snn` and :func:`match_smnn` on CUDA or the Apple M1 CPU: faiss's exact k-nearest-neighbour search
  returns the same matches up to 2.2x faster. On the x86 CPU, kornia is the fastest.
- :meth:`~kornia.feature.XFeat.match_xfeat_star` departs from the authors' refinement on purpose: much more
  accurate on planar scenes, mixed on PhotoTourism (see :class:`XFeat`).

Figures per device and the protocol (640 px images, 2,048 features, PyTorch 2.14):
https://kornia.readthedocs.io/en/latest/applications/image_matching.html#feature-speed-by-device
"""

from .affine_shape import LAFAffineShapeEstimator, LAFAffNetShapeEstimator, PatchAffineShapeEstimator
from .aliked import ALIKED, ALIKEDFeatures
from .dedode import DeDoDe
from .defmo import DeFMO
from .disk import DISK, DISKFeatures
from .hardnet import HardNet, HardNet8
from .hynet import TLU, FilterResponseNorm2d, HyNet
from .integrated import (
    DoGAffNetHardNet,
    DoGHardNet,
    GFTTAffNetHardNet,
    HesAffNetHardNet,
    KeyNetAffNetHardNet,
    KeyNetHardNet,
    LAFDescriptor,
    LightGlueMatcher,
    LocalFeature,
    LocalFeatureMatcher,
    SIFTFeature,
    SIFTFeatureScaleSpace,
    get_laf_descriptors,
)
from .keynet import KeyNet, KeyNetDetector
from .laf import (
    KORNIA_CHECK_LAF,
    denormalize_laf,
    ellipse_to_laf,
    extract_patches_from_pyramid,
    extract_patches_simple,
    get_laf_center,
    get_laf_orientation,
    get_laf_scale,
    laf_from_center_scale_ori,
    laf_from_three_points,
    laf_is_filled,
    laf_is_inside_image,
    laf_is_valid,
    laf_to_boundary_points,
    laf_to_three_points,
    make_upright,
    normalize_laf,
    perspective_transform_lafs,
    rotate_laf,
    scale_laf,
    set_laf_orientation,
)
from .lightglue import LightGlue
from .lightglue_onnx import OnnxLightGlue
from .loftr import LoFTR
from .matching import (
    DescriptorMatcher,
    GeometryAwareDescriptorMatcher,
    match_adalam,
    match_fginn,
    match_mnn,
    match_nn,
    match_smnn,
    match_smnn_batched,
    match_snn,
)
from .mkd import MKDDescriptor
from .orientation import LAFOrienter, OriNet, PatchDominantGradientOrientation
from .responses import (
    BlobDoG,
    BlobDoGSingle,
    BlobHessian,
    CornerGFTT,
    CornerHarris,
    dog_response,
    dog_response_single,
    gftt_response,
    harris_response,
    hessian_response,
)
from .scale_space_detector import MultiResolutionDetector, PassLAF, ScaleSpaceDetector
from .sift import SIFTDescriptorFromPyramid
from .siftdesc import DenseSIFTDescriptor, SIFTDescriptor, convert_sift_descriptor_layout
from .sold2 import SOLD2, SOLD2_detector
from .sosnet import SOSNet
from .tfeat import TFeat
from .xfeat import InterpolateSparse2d, XFeat, XFeatModel

__all__ = [
    "ALIKED",
    "DISK",
    "KORNIA_CHECK_LAF",
    "SOLD2",
    "TLU",
    "ALIKEDFeatures",
    "BlobDoG",
    "BlobDoGSingle",
    "BlobHessian",
    "CornerGFTT",
    "CornerHarris",
    "DISKFeatures",
    "DeDoDe",
    "DeFMO",
    "DenseSIFTDescriptor",
    "DescriptorMatcher",
    "DoGAffNetHardNet",
    "DoGHardNet",
    "FilterResponseNorm2d",
    "GFTTAffNetHardNet",
    "GeometryAwareDescriptorMatcher",
    "HardNet",
    "HardNet8",
    "HesAffNetHardNet",
    "HyNet",
    "InterpolateSparse2d",
    "KeyNet",
    "KeyNetAffNetHardNet",
    "KeyNetDetector",
    "KeyNetHardNet",
    "LAFAffNetShapeEstimator",
    "LAFAffineShapeEstimator",
    "LAFDescriptor",
    "LAFOrienter",
    "LightGlue",
    "LightGlueMatcher",
    "LoFTR",
    "LocalFeature",
    "LocalFeatureMatcher",
    "MKDDescriptor",
    "MultiResolutionDetector",
    "OnnxLightGlue",
    "OriNet",
    "PassLAF",
    "PatchAffineShapeEstimator",
    "PatchDominantGradientOrientation",
    "SIFTDescriptor",
    "SIFTDescriptorFromPyramid",
    "SIFTFeature",
    "SIFTFeatureScaleSpace",
    "SOLD2_detector",
    "SOSNet",
    "ScaleSpaceDetector",
    "TFeat",
    "XFeat",
    "XFeatModel",
    "convert_sift_descriptor_layout",
    "denormalize_laf",
    "dog_response",
    "dog_response_single",
    "ellipse_to_laf",
    "extract_patches_from_pyramid",
    "extract_patches_simple",
    "get_laf_center",
    "get_laf_descriptors",
    "get_laf_orientation",
    "get_laf_scale",
    "gftt_response",
    "harris_response",
    "hessian_response",
    "laf_from_center_scale_ori",
    "laf_from_three_points",
    "laf_is_filled",
    "laf_is_inside_image",
    "laf_is_valid",
    "laf_to_boundary_points",
    "laf_to_three_points",
    "make_upright",
    "match_adalam",
    "match_fginn",
    "match_mnn",
    "match_nn",
    "match_smnn",
    "match_smnn_batched",
    "match_snn",
    "normalize_laf",
    "perspective_transform_lafs",
    "rotate_laf",
    "scale_laf",
    "set_laf_orientation",
]
