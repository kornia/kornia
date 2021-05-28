from .responses import (CornerHarris,
                        CornerGFTT,
                        BlobHessian,
                        BlobDoG,
                        harris_response,
                        gftt_response,
                        hessian_response,
                        dog_response)
from .nms import (NonMaximaSuppression2d,
                  nms2d,
                  NonMaximaSuppression3d,
                  nms3d)

# Backward compatibility
from .nms import nms2d as non_maxima_suppression2d
from .nms import nms3d as non_maxima_suppression3d

from .laf import (extract_patches_from_pyramid,
                  extract_patches_simple,
                  normalize_laf,
                  denormalize_laf,
                  laf_to_boundary_points,
                  ellipse_to_laf,
                  make_upright,
                  scale_laf,
                  get_laf_scale,
                  get_laf_center,
                  get_laf_orientation,
                  set_laf_orientation,
                  raise_error_if_laf_is_not_valid,
                  laf_from_center_scale_ori,
                  laf_is_inside_image,
                  laf_to_three_points,
                  laf_from_three_points)
from .siftdesc import SIFTDescriptor
from .mkd import MKDDescriptor
from .hardnet import HardNet, HardNet8
from .tfeat import TFeat
from .sosnet import SOSNet
from .scale_space_detector import ScaleSpaceDetector, PassLAF
from .affine_shape import LAFAffineShapeEstimator, PatchAffineShapeEstimator, LAFAffNetShapeEstimator
from .orientation import LAFOrienter, PatchDominantGradientOrientation, OriNet
from .matching import match_nn, match_mnn, match_snn, match_smnn


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
