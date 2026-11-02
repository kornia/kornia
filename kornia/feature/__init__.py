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

"""Kornia Feature — Feature detection, description, and matching for Kornia.

This subpackage provides modules for keypoint detection, descriptors, and feature matching.

Choosing an implementation by device. Each figure below is the reference implementation's latency divided by
kornia's, so above 1 favours kornia. The reference is the authors' code, or OpenCV for SIFT. The figures were
measured on kornia's main branch between 30 October and 2 November 2026, with PyTorch 2.14, 640 px images and
2,048 features, on an Apple M1, an Intel i7-14700K and an NVIDIA RTX 4090. On every cohort, kornia's accuracy is
within 2.5 points of the reference, except :meth:`XFeat.match_xfeat_star`, whose refinement departs from the
authors' on purpose (see :class:`XFeat`).

- CUDA: most ports tie. :class:`LoFTR` (1.26x), :class:`DeDoDe` with the G descriptor (1.19x) and the polar or
  concatenated :class:`MKDDescriptor` (1.1-1.3x) are faster. Eager :class:`ALIKED` is slower (0.8x). With
  ``torch.compile`` it runs at 1.6x the authors' eager speed at a fixed image size, and 1.2x with ``dynamic=True``.
- x86 CPU: kornia is faster or tied everywhere except DeDoDe with the B descriptor (0.96x): :class:`ALIKED` 1.7x,
  :class:`XFeat` 1.6x, :class:`DISK` with :class:`LightGlue` 1.5x, :class:`HardNet`, :class:`SOSNet`,
  :class:`HyNet` and :class:`TFeat` 1.2-1.3x, :class:`MKDDescriptor` 3.9-8.5x. Keep ALIKED eager there: compiled,
  its forward is slower than eager.
- Apple M1 CPU: :class:`ALIKED` 2.0x, :class:`XFeat` 1.5x and :class:`MKDDescriptor` 1.5-2.5x; the other ports are
  within 10 % (DISK's noisy 1.2x aside). OpenCV's SIFT is 8-14x faster than :class:`SIFTFeature` and
  :class:`SIFTFeatureScaleSpace`.
- Apple MPS: :class:`MKDDescriptor` 2.9-6.3x and :class:`HyNet` 1.35x are faster; the other ports, ALIKED
  included, are within about 10 %.
- Matching: :func:`match_nn`, :func:`match_mnn`, :func:`match_snn` and :func:`match_smnn` return the same matches as
  OpenCV's ``BFMatcher`` and faiss's exact k-nearest-neighbour search. On the x86 CPU kornia is the fastest of the
  three. On CUDA, faiss runs the ratio tests (:func:`match_snn`, :func:`match_smnn`) up to 2.2x faster. On the
  Apple M1 CPU, kornia is faster for :func:`match_nn` and :func:`match_mnn` (1.9-2.4x) and faiss for the ratio
  tests (1.3-2.0x).

The tables and the protocol are in the image matching guide:
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
