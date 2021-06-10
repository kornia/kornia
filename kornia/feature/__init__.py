from .nms import nms2d, nms3d, NonMaximaSuppression2d, NonMaximaSuppression3d
from .responses import (
    BlobDoG,
    BlobHessian,
    CornerGFTT,
    CornerHarris,
    dog_response,
    gftt_response,
    harris_response,
    hessian_response,
)

# Backward compatibility
non_maxima_suppression2d = nms2d
non_maxima_suppression3d = nms3d

from .affine_shape import LAFAffineShapeEstimator, LAFAffNetShapeEstimator, PatchAffineShapeEstimator
from .hardnet import HardNet, HardNet8
from .laf import (
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
    raise_error_if_laf_is_not_valid,
    scale_laf,
    set_laf_orientation,
)
from .matching import match_mnn, match_nn, match_smnn, match_snn
from .mkd import MKDDescriptor
from .orientation import LAFOrienter, OriNet, PatchDominantGradientOrientation
from .scale_space_detector import PassLAF, ScaleSpaceDetector
from .siftdesc import SIFTDescriptor
from .sosnet import SOSNet
from .tfeat import TFeat

__all__ = [
    "nms2d",
    "nms3d",
    "non_maxima_suppression2d",
    "non_maxima_suppression3d",
    "harris_response",
    "gftt_response",
    "hessian_response",
    "dog_response",
    "NonMaximaSuppression2d",
    "NonMaximaSuppression3d",
    "CornerHarris",
    "CornerGFTT",
    "BlobHessian",
    "BlobDoG",
    "extract_patches_from_pyramid",
    "extract_patches_simple",
    "normalize_laf",
    "denormalize_laf",
    "laf_to_boundary_points",
    "ellipse_to_laf",
    "make_upright",
    "get_laf_scale",
    "get_laf_center",
    "get_laf_orientation",
    "set_laf_orientation",
    "scale_laf",
    "SIFTDescriptor",
    "MKDDescriptor",
    "HardNet",
    "HardNet8",
    "TFeat",
    "OriNet",
    "LAFAffNetShapeEstimator",
    "PassLAF",
    "ScaleSpaceDetector",
    "LAFAffineShapeEstimator",
    "PatchAffineShapeEstimator",
    "LAFOrienter",
    "PatchDominantGradientOrientation",
    "raise_error_if_laf_is_not_valid",
    "laf_is_inside_image",
    "laf_from_center_scale_ori",
    "laf_to_three_points",
    "laf_from_three_points",
    "match_nn",
    "match_mnn",
    "match_snn",
    "match_smnn",
]
