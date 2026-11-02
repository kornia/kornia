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

Images are float ``(B, C, H, W)`` tensors in :math:`[0, 1]`. Keypoints are ``(x, y)`` pixels with integer pixel centres,
the frame of :func:`laf_from_center_scale_ori`, except that :class:`DeDoDe` and :class:`LoFTR` put pixel ``i`` at
``i + 0.5`` and :class:`SOLD2` and :class:`SOLD2_detector` give line endpoints as ``(row, col)``. :class:`XFeat` and
:class:`MultiResolutionDetector`, with :class:`KeyNetDetector`, :class:`SIFTFeature`, :class:`KeyNetHardNet` and
:class:`KeyNetAffNetHardNet` built on it, report a pixel of a resized image or pyramid level at its index times the
resize factor, which is that pixel's centre only for a factor of 1 (see their Convention blocks). A local affine frame
(LAF) ``[A | c]`` is ``(B, N, 2, 3)`` in pixels, with centre ``c``, scale :math:`\sqrt{|\det A|}` and orientation
``atan2(a12, a11)`` in degrees. The detectors that return LAFs, the :class:`LocalFeature` presets and
:meth:`~kornia.feature.ALIKED.forward_laf` pad the slots no detection filled with zero LAFs, which :func:`laf_is_filled`
finds: drop those slots before matching.

Which function to use:

- Match two images in one call: :class:`LoFTR`, grayscale images without a detector; :class:`LocalFeatureMatcher`, a
  local feature on both and a descriptor-only matcher such as :class:`DescriptorMatcher` (call :class:`LightGlueMatcher`
  yourself); :meth:`~kornia.feature.XFeat.match_xfeat` and :meth:`~kornia.feature.XFeat.match_xfeat_star`.
- Detect and describe keypoints with a learned network, loaded with ``from_pretrained``: :class:`ALIKED`, sub-pixel
  keypoints of RGB or grayscale images of any size, one :class:`ALIKEDFeatures` per image; :class:`DISK`, integer-pixel
  keypoints of RGB images, one :class:`DISKFeatures` per image; :class:`XFeat`, any channel count; :class:`DeDoDe`, two
  networks, the ``n`` best keypoints of each RGB image. For :class:`LightGlueMatcher`, centre LAFs on the ``(N, 2)``
  keypoints of each image, ``laf_from_center_scale_ori(keypoints[None])``. XFeat's outputs are inference tensors: call
  the matcher on them under ``torch.no_grad()`` or ``torch.inference_mode()``, or ``clone()`` them first (#6533).
- Detect and describe LAFs in one call: :class:`LocalFeature`, a given detector and LAF descriptor. Its presets, on
  grayscale images: SIFT, :class:`SIFTFeatureScaleSpace` (refined DoG scale-space extrema; start new pipelines with
  ``descriptor_backend="pyramid"``) or :class:`SIFTFeature` (a DoG on each pyramid level); HardNet on DoG blobs
  (:class:`DoGHardNet`, :class:`DoGAffNetHardNet`), Hessian blobs (:class:`HesAffNetHardNet`), corners
  (:class:`GFTTAffNetHardNet`) or KeyNet keypoints (:class:`KeyNetHardNet`, :class:`KeyNetAffNetHardNet`), the AffNet
  ones with adapted affine shapes. SIFTFeature and the KeyNet presets take one image per call, the rest a batch.
- Detect LAFs only: :class:`ScaleSpaceDetector`, sub-pixel scale-space maxima, for a batch;
  :class:`MultiResolutionDetector`, 2D maxima on each pyramid level, one image per call; :class:`KeyNetDetector`, the
  latter on the learned :class:`KeyNet` response. Response maps of each pixel and channel, without non-maxima
  suppression: :func:`harris_response` and :func:`gftt_response` (corners), :func:`hessian_response` and
  :func:`dog_response_single` (blobs), :func:`dog_response` (a scale stack); the detectors take their module forms,
  :class:`CornerHarris`, :class:`CornerGFTT`, :class:`BlobHessian`, :class:`BlobDoGSingle` and :class:`BlobDoG`.
- Describe grayscale patches ``(B, 1, PS, PS)``: :class:`SIFTDescriptor` and :class:`MKDDescriptor`, handcrafted, of
  their ``patch_size``; :class:`HardNet`, :class:`HardNet8`, :class:`HyNet`, :class:`SOSNet` and :class:`TFeat`,
  learned, 32 x 32, all unit-norm but TFeat. LAFs on an image: :func:`get_laf_descriptors` or its module form
  :class:`LAFDescriptor` (HardNet by default), one sampled patch per LAF; :class:`SIFTDescriptorFromPyramid`, without
  patches. Every position: :class:`DenseSIFTDescriptor`. SIFT layouts: :func:`convert_sift_descriptor_layout`.
- Orient LAFs on a grayscale image: :class:`LAFOrienter`, with :class:`PatchDominantGradientOrientation` or the learned
  :class:`OriNet`. Adapt their affine shape: :class:`LAFAffineShapeEstimator`, with :class:`PatchAffineShapeEstimator`,
  or the learned :class:`LAFAffNetShapeEstimator`. Keep them: :class:`PassLAF`.
- Match descriptors ``(B1, D)`` and ``(B2, D)`` into ``(B3, 1)`` distances, ratios or inverse confidences, lower is
  better (:class:`LightGlueMatcher`: scores, higher is better), and index pairs ``(B3, 2)``: :func:`match_nn`, every
  nearest neighbour; :func:`match_mnn`, mutual ones; :func:`match_snn`, Lowe's ratio test; :func:`match_smnn`, both; the
  four as a module, :class:`DescriptorMatcher`. With the LAFs too: :func:`match_fginn` and :func:`match_adalam`, as a
  module :class:`GeometryAwareDescriptorMatcher`, and the learned :class:`LightGlueMatcher`. Padded batches of pairs:
  :func:`match_smnn_batched`. LightGlue on dicts of batched keypoints: :class:`LightGlue`; an exported ONNX model run by
  onnxruntime, one pair per call on CPU or CUDA: :class:`OnnxLightGlue`. Then :class:`~kornia.geometry.ransac.RANSAC`
  keeps the matches that one geometric model explains.
- LAFs: build, :func:`laf_from_center_scale_ori`, :func:`ellipse_to_laf`, :func:`laf_from_three_points`; read,
  :func:`get_laf_center`, :func:`get_laf_scale`, :func:`get_laf_orientation`, :func:`laf_to_three_points`,
  :func:`laf_to_boundary_points`; transform, :func:`scale_laf`, :func:`rotate_laf`, :func:`set_laf_orientation`,
  :func:`make_upright`, :func:`perspective_transform_lafs`, :func:`normalize_laf`, :func:`denormalize_laf`; check,
  :func:`laf_is_filled` (padding), :func:`laf_is_valid` (finite, nonsingular), :func:`laf_is_inside_image`,
  :func:`KORNIA_CHECK_LAF` (shape); patches, :func:`extract_patches_from_pyramid` or, without anti-aliasing,
  :func:`extract_patches_simple`.
- Line segments of grayscale images: :class:`SOLD2` detects, describes and matches them; :class:`SOLD2_detector` only
  detects them.
- Layers and other models: :class:`FilterResponseNorm2d` and :class:`TLU`, the normalisation and activation of
  :class:`HyNet`; :class:`XFeatModel`, the network of :class:`XFeat`, and :class:`InterpolateSparse2d`, XFeat's sampler
  of the dense maps at the keypoints; :class:`DeFMO`, which deblurs a fast-moving object into RGBA sub-frames.

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
