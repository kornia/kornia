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

from .affine_shape import LAFAffineShapeEstimator, LAFAffNetShapeEstimator, PatchAffineShapeEstimator
from .dedode import DeDoDe
from .defmo import DeFMO
from .disk import DISK, DISKFeatures
from .hardnet import HardNet, HardNet8
from .hynet import TLU, FilterResponseNorm2d, HyNet
from .integrated import (
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
    laf_is_inside_image,
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
from .siftdesc import DenseSIFTDescriptor, SIFTDescriptor
from .sold2 import SOLD2, SOLD2_detector
from .sosnet import SOSNet
from .tfeat import TFeat

__all__ = [
    "DISK",
    "KORNIA_CHECK_LAF",
    "SOLD2",
    "TLU",
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
    "DescriptorMatcher",
    "FilterResponseNorm2d",
    "GFTTAffNetHardNet",
    "GFTTAffNetHardNet",
    "GeometryAwareDescriptorMatcher",
    "HardNet",
    "HardNet8",
    "HesAffNetHardNet",
    "HyNet",
    "KeyNet",
    "KeyNet",
    "KeyNetAffNetHardNet",
    "KeyNetDetector",
    "KeyNetHardNet",
    "LAFAffNetShapeEstimator",
    "LAFAffineShapeEstimator",
    "LAFDescriptor",
    "LAFDescriptor",
    "LAFOrienter",
    "LightGlue",
    "LightGlueMatcher",
    "LoFTR",
    "LocalFeature",
    "LocalFeature",
    "LocalFeatureMatcher",
    "LocalFeatureMatcher",
    "MKDDescriptor",
    "MultiResolutionDetector",
    "OnnxLightGlue",
    "OriNet",
    "PassLAF",
    "PatchAffineShapeEstimator",
    "PatchDominantGradientOrientation",
    "SIFTDescriptor",
    "SIFTFeature",
    "SIFTFeature",
    "SIFTFeatureScaleSpace",
    "SOLD2_detector",
    "SOSNet",
    "ScaleSpaceDetector",
    "TFeat",
    "denormalize_laf",
    "dog_response",
    "dog_response_single",
    "ellipse_to_laf",
    "extract_patches_from_pyramid",
    "extract_patches_simple",
    "get_laf_center",
    "get_laf_descriptors",
    "get_laf_descriptors",
    "get_laf_orientation",
    "get_laf_scale",
    "gftt_response",
    "harris_response",
    "hessian_response",
    "laf_from_center_scale_ori",
    "laf_from_three_points",
    "laf_is_inside_image",
    "laf_to_boundary_points",
    "laf_to_three_points",
    "make_upright",
    "match_adalam",
    "match_fginn",
    "match_mnn",
    "match_mnn",
    "match_nn",
    "match_nn",
    "match_smnn",
    "match_smnn",
    "match_snn",
    "match_snn",
    "normalize_laf",
    "perspective_transform_lafs",
    "rotate_laf",
    "scale_laf",
    "set_laf_orientation",
]
