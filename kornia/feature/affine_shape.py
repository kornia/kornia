from typing import Tuple

import torch
import torch.nn as nn
import math
from kornia.filters import get_gaussian_kernel2d
from kornia.filters import SpatialGradient
from kornia.feature.laf import (ellipse_to_laf,
                                get_laf_scale,
                                raise_error_if_laf_is_not_valid,
                                scale_laf, make_upright)
from kornia.feature import extract_patches_from_pyramid


class PatchAffineShapeEstimator(nn.Module):
    def __init__(self, patch_size: int = 19, eps: float = 1e-10):
        """Module, which estimates the second moment matrix of the patch gradients in order to determine the
        affine shape of the local feature as in
        A. Baumberg (2000). "Reliable feature matching across widely separated views".
        In Proc.of CVPR. pp. I:1774--1781.
        Args:
            patch_size: int, default = 19
            eps: float, for safe division, default is 1e-10"""
        super(PatchAffineShapeEstimator, self).__init__()
        self.patch_size: int = patch_size
        self.gradient: nn.Module = SpatialGradient('sobel', 1)
        self.eps: float = eps
        sigma: float = float(self.patch_size) / math.sqrt(2.0)
        self.weighting: torch.Tensor = get_gaussian_kernel2d((self.patch_size, self.patch_size), (sigma, sigma), True)
        return

    def __repr__(self):
        return self.__class__.__name__ + '('\
            'patch_size=' + str(self.patch_size) + ', ' + \
            'eps=' + str(self.eps) + ')'

    def forward(self, patch: torch.Tensor) -> torch.Tensor:   # type: ignore
        """Args:
            patch: 4d tensor, shape [Bx1xHxW]
        Returns:
            ellipse_shape: 3d tensor, shape [Bx1x5] """
        if not torch.is_tensor(patch):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(patch)))
        if not len(patch.shape) == 4:
            raise ValueError("Invalid input shape, we expect Bx1xHxW. Got: {}"
                             .format(patch.shape))
        B, CH, W, H = patch.size()
        if (W != self.patch_size) or (H != self.patch_size) or (CH != 1):
            raise TypeError(
                "input shape should be must be [Bx1x{}x{}]. "
                "Got {}".format(self.patch_size, self.patch_size, patch.size()))
        self.weighting = self.weighting.to(patch.dtype).to(patch.device)
        grads: torch.Tensor = self.gradient(patch) * self.weighting
        # unpack the edges
        gx: torch.Tensor = grads[:, :, 0]
        gy: torch.Tensor = grads[:, :, 1]
        # abc == 1st axis, mixture, 2nd axis. Ellipse_shape is a 2nd moment matrix.
        ellipse_shape = torch.cat([gx.pow(2).mean(dim=2).mean(dim=2, keepdim=True),
                                   (gx * gy).mean(dim=2).mean(dim=2, keepdim=True),
                                   gy.pow(2).mean(dim=2).mean(dim=2, keepdim=True)], dim=2)

        # Now lets detect degenerate cases: when 2 or 3 elements are close to zero (e.g. if patch is completely black
        bad_mask = ((ellipse_shape < self.eps).float().sum(dim=2, keepdim=True) >= 2).to(ellipse_shape.dtype)
        # We will replace degenerate shape with circular shapes.
        circular_shape = torch.tensor([1.0, 0., 1.0]).to(ellipse_shape.device).to(ellipse_shape.dtype).view(1, 1, 3)
        ellipse_shape = ellipse_shape * (1.0 - bad_mask) + circular_shape * bad_mask  # type: ignore
        # normalization
        ellipse_shape = ellipse_shape / ellipse_shape.max(dim=2, keepdim=True)[0]
        return ellipse_shape


class LAFAffineShapeEstimator(nn.Module):
    """Module, which extracts patches using input images and local affine frames (LAFs),
    then runs PatchAffineShapeEstimator on patches to estimate LAFs shape.
    Then original LAF shape is replaced with estimated one. The original LAF orientation is not preserved,
     so it is recommended to first run LAFAffineShapeEstimator and then LAFOrienter
    Args:
            patch_size: int, default = 32"""
    def __init__(self,
                 patch_size: int = 32) -> None:
        super(LAFAffineShapeEstimator, self).__init__()
        self.patch_size = patch_size
        self.affine_shape_detector = PatchAffineShapeEstimator(self.patch_size)
        return

    def __repr__(self):
        return self.__class__.__name__ + '('\
            'patch_size=' + str(self.patch_size) + ')'

    def forward(self, laf: torch.Tensor, img: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Args:
                    laf: 4d tensor, shape [BxNx2x3]
                    img: 4d tensor, shape [Bx1xHxW]
                Returns:
                    laf_out: 4d tensor, shape [BxNx2x3] """
        raise_error_if_laf_is_not_valid(laf)
        img_message: str = "Invalid img shape, we expect BxCxHxW. Got: {}".format(img.shape)
        if not torch.is_tensor(img):
            raise TypeError("img type is not a torch.Tensor. Got {}"
                            .format(type(img)))
        if len(img.shape) != 4:
            raise ValueError(img_message)
        if laf.size(0) != img.size(0):
            raise ValueError("Batch size of laf and img should be the same. Got {}, {}"
                             .format(img.size(0), laf.size(0)))
        B, N = laf.shape[:2]
        PS: int = self.patch_size
        patches: torch.Tensor = extract_patches_from_pyramid(img,
                                                             make_upright(laf),
                                                             PS, True).view(-1, 1, PS, PS)
        ellipse_shape: torch.Tensor = self.affine_shape_detector(patches)
        ellipses = torch.cat([laf.view(-1, 2, 3)[..., 2].unsqueeze(1), ellipse_shape], dim=2).view(B, N, 5)
        scale_orig = get_laf_scale(laf)
        laf_out = ellipse_to_laf(ellipses)
        ellipse_scale = get_laf_scale(laf_out)
        laf_out = scale_laf(laf_out, scale_orig / ellipse_scale)
        return laf_out
