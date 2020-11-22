from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia
from kornia.nn import (
    BlobHessian,
    ConvSoftArgmax3d,
    PassLAF,
    ScalePyramid
)


def _scale_index_to_scale(max_coords: torch.Tensor, sigmas: torch.Tensor, num_levels: int) -> torch.Tensor:
    """Auxilary function for ScaleSpaceDetector. Converts scale level index from ConvSoftArgmax3d
    to the actual scale, using the sigmas from the ScalePyramid output
    Args:
        max_coords: (torch.Tensor): tensor [BxNx3].
        sigmas: (torch.Tensor): tensor [BxNxD], D >= 1

    Returns:
        torch.Tensor:  tensor [BxNx3].
    """
    # depth (scale) in coord_max is represented as (float) index, not the scale yet.
    # we will interpolate the scale using pytorch.grid_sample function
    # Because grid_sample is for 4d input only, we will create fake 2nd dimension
    # ToDo: replace with 3d input, when grid_sample will start to support it

    # Reshape for grid shape
    B, N, _ = max_coords.shape
    L: int = sigmas.size(1)
    scale_coords = max_coords[:, :, 0].contiguous().view(-1, 1, 1, 1)
    # Replace the scale_x_y
    out = torch.cat([sigmas[0, 0] * torch.pow(2.0, scale_coords / float(num_levels)).view(B, N, 1),
                     max_coords[:, :, 1:]], dim=2)
    return out


def _create_octave_mask(mask: torch.Tensor, octave_shape: List[int]) -> torch.Tensor:
    """Downsamples a mask based on the given octave shape."""
    mask_shape = octave_shape[-2:]
    mask_octave = F.interpolate(mask, mask_shape, mode='bilinear', align_corners=False)  # type: ignore
    return mask_octave.unsqueeze(1)


class ScaleSpaceDetector(kornia.nn.ScaleSpaceDetector):
    """Module for differentiable local feature detection, as close as possible to classical
     local feature detectors like Harris, Hessian-Affine or SIFT (DoG).
     It has 5 modules inside: scale pyramid generator, response ("cornerness") function,
     soft nms function, affine shape estimator and patch orientation estimator.
     Each of those modules could be replaced with learned custom one, as long, as
     they respect output shape.

    Args:
        num_features: (int) Number of features to detect. default = 500. In order to keep everything batchable,
                      output would always have num_features outputed, even for completely homogeneous images.
        mr_size: (float), default 6.0. Multiplier for local feature scale compared to the detection scale.
                    6.0 is matching OpenCV 12.0 convention for SIFT.
        scale_pyr_module: (nn.Module), which generates scale pyramid.
                         See :class:`~kornia.geometry.ScalePyramid` for details. Default is ScalePyramid(3, 1.6, 10)
        resp_module: (nn.Module), which calculates 'cornerness' of the pixel. Default is BlobHessian().
        nms_module: (nn.Module), which outputs per-patch coordinates of the response maxima.
                    See :class:`~kornia.geometry.ConvSoftArgmax3d` for details.
        ori_module: (nn.Module) for local feature orientation estimation.  Default is :class:`~kornia.feature.PassLAF`,
                    which does nothing. See :class:`~kornia.feature.LAFOrienter` for details.
        aff_module:  (nn.Module) for local feature affine shape estimation. Default is :class:`~kornia.feature.PassLAF`,
                    which does nothing. See :class:`~kornia.feature.LAFAffineShapeEstimator` for details.
        minima_are_also_good:  (bool) if True, then both response function minima and maxima are detected
                                Useful for symmetric response functions like DoG or Hessian. Default is False
    """

    def __init__(self,
                 num_features: int = 500,
                 mr_size: float = 6.0,
                 scale_pyr_module: nn.Module = ScalePyramid(3, 1.6, 15),
                 resp_module: nn.Module = BlobHessian(),
                 nms_module: nn.Module = ConvSoftArgmax3d((3, 3, 3),
                                                          (1, 1, 1),
                                                          (1, 1, 1),
                                                          normalized_coordinates=False,
                                                          output_value=True),
                 ori_module: nn.Module = PassLAF(),
                 aff_module: nn.Module = PassLAF(),
                 minima_are_also_good: bool = False,
                 scale_space_response=False):
        super(ScaleSpaceDetector, self).__init__(
            num_features, mr_size, scale_pyr_module, resp_module, nms_module, ori_module, aff_module,
            minima_are_also_good, scale_space_response
        )
        kornia.deprecation_warning("kornia.feature.ScaleSpaceDetector", "kornia.nn.ScaleSpaceDetector")
