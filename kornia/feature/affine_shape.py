from typing import Tuple

import torch
import torch.nn as nn
import math
from kornia.filters import get_gaussian_kernel2d
from kornia.filters import SpatialGradient
from kornia.feature.laf import (ellipse_to_laf,
                                make_upright,
                                get_laf_scale,
                                raise_error_if_laf_is_not_valid,
                                scale_laf)
from kornia.feature import extract_patches_from_pyramid


class PatchAffineShapeEstimator(nn.Module):
    def __init__(self, patch_size: int = 19):
        super(PatchAffineShapeEstimator, self).__init__()
        self.patch_size: int = patch_size
        self.gradient: nn.Module = SpatialGradient('sobel', 1)
        sigma: float = float(self.patch_size) / math.sqrt(2.0)
        self.weighting: torch.Tensor = get_gaussian_kernel2d((self.patch_size, self.patch_size), (sigma, sigma), True)
        return

    def forward(self, patch: torch.Tensor) -> torch.Tensor:   # type: ignore
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
        ellipse_shape = torch.cat([gx.pow(2).mean(dim=2).mean(dim=2, keepdim=True),
                                   (gx * gy).mean(dim=2).mean(dim=2, keepdim=True),
                                   gy.pow(2).mean(dim=2).mean(dim=2, keepdim=True)], dim=2)
        return ellipse_shape


class LAFAffineShapeEstimator(nn.Module):
    def __init__(self,
                 patch_size: int = 32):
        super(LAFAffineShapeEstimator, self).__init__()
        self.patch_size = patch_size
        self.affine_shape_detector = PatchAffineShapeEstimator(self.patch_size)
        return

    def forward(self, laf: torch.Tensor, img: torch.Tensor) -> torch.Tensor:  # type: ignore
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
        patches: torch.Tensor = extract_patches_from_pyramid(img, laf, PS=self.patch_size).view(-1,
                                                                                                1,
                                                                                                self.patch_size,
                                                                                                self.patch_size)
        ellipse_shape: torch.Tensor = self.affine_shape_detector(patches)
        ellipses = torch.cat([laf.view(-1, 2, 3)[..., 2].unsqueeze(1), ellipse_shape], dim=2).view(B, N, 5)
        scale_orig = get_laf_scale(laf)
        laf_out = ellipse_to_laf(ellipses)
        ellipse_scale = get_laf_scale(laf_out)
        laf_out = scale_laf(laf_out, scale_orig / ellipse_scale)
        return laf_out
