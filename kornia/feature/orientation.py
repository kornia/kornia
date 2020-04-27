from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from kornia.filters import get_gaussian_kernel2d
from kornia.filters import SpatialGradient
from kornia.constants import pi
from kornia.feature import (extract_patches_from_pyramid, make_upright,
                            normalize_laf, raise_error_if_laf_is_not_valid)
from kornia.geometry import rad2deg, angle_to_rotation_matrix


class PassLAF(nn.Module):
    """Dummy module to use instead of local feature orientation or affine shape estimator"""

    def forward(self, laf: torch.Tensor, img: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        Args:
            laf: torch.Tensor: 4d tensor
            img (torch.Tensor): the input image tensor

        Return:
            torch.Tensor: unchanged laf from the input."""
        return laf


class PatchDominantGradientOrientation(nn.Module):
    """Module, which estimates the dominant gradient orientation of the given patches, in radians.
    Zero angle points towards right.

    Args:
            patch_size: int, default = 32
            num_angular_bins: int, default is 36
            eps: float, for safe division, and arctan, default is 1e-8"""

    def __init__(self,
                 patch_size: int = 32,
                 num_angular_bins: int = 36, eps: float = 1e-8):
        super(PatchDominantGradientOrientation, self).__init__()
        self.patch_size = patch_size
        self.num_ang_bins = num_angular_bins
        self.gradient = SpatialGradient('sobel', 1)
        self.eps = eps
        self.angular_smooth = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False, padding_mode="circular")
        with torch.no_grad():
            self.angular_smooth.weight[:] = torch.tensor([[[0.33, 0.34, 0.33]]])
        sigma: float = float(self.patch_size) / math.sqrt(2.0)
        self.weighting = get_gaussian_kernel2d((self.patch_size, self.patch_size), (sigma, sigma), True)
        return

    def __repr__(self):
        return self.__class__.__name__ + '('\
            'patch_size=' + str(self.patch_size) + ', ' + \
            'num_ang_bins=' + str(self.num_ang_bins) + ', ' + \
            'eps=' + str(self.eps) + ')'

    def forward(self, patch: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Args:
            patch: (torch.Tensor) shape [Bx1xHxW]
        Returns:
            patch: (torch.Tensor) shape [Bx1] """
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
        self.angular_smooth = self.angular_smooth.to(patch.dtype).to(patch.device)
        grads: torch.Tensor = self.gradient(patch)
        # unpack the edges
        gx: torch.Tensor = grads[:, :, 0]
        gy: torch.Tensor = grads[:, :, 1]

        mag: torch.Tensor = torch.sqrt(gx * gx + gy * gy + self.eps)
        ori: torch.Tensor = torch.atan2(gy, gx + self.eps) + 2.0 * pi

        o_big = float(self.num_ang_bins) * (ori + 1.0 * pi) / (2.0 * pi)
        bo0_big = torch.floor(o_big)
        wo1_big = o_big - bo0_big
        bo0_big = bo0_big % self.num_ang_bins
        bo1_big = (bo0_big + 1) % self.num_ang_bins
        wo0_big = (1.0 - wo1_big) * mag  # type: ignore
        wo1_big = wo1_big * mag
        ang_bins = []
        for i in range(0, self.num_ang_bins):
            ang_bins.append(F.adaptive_avg_pool2d((bo0_big == i).to(patch.dtype) * wo0_big +
                                                  (bo1_big == i).to(patch.dtype) * wo1_big, (1, 1)))
        ang_bins = torch.cat(ang_bins, 1).view(-1, 1, self.num_ang_bins)   # type: ignore
        ang_bins = self.angular_smooth(ang_bins)   # type: ignore
        values, indices = ang_bins.view(-1, self.num_ang_bins).max(1)  # type: ignore
        angle = -((2. * pi * indices.to(patch.dtype) / float(self.num_ang_bins)) - pi)  # type: ignore
        return angle


class LAFOrienter(nn.Module):
    """Module, which extracts patches using input images and local affine frames (LAFs),
    then runs :class:`~kornia.feature.PatchDominantGradientOrientation`
    on patches and then rotates the LAFs by the estimated angles

    Args:
            patch_size: int, default = 32
            num_angular_bins: int, default is 36"""

    def __init__(self,
                 patch_size: int = 32,
                 num_angular_bins: int = 36):
        super(LAFOrienter, self).__init__()
        self.patch_size = patch_size
        self.num_ang_bins = num_angular_bins
        self.angle_detector = PatchDominantGradientOrientation(self.patch_size, self.num_ang_bins)
        return

    def __repr__(self):
        return self.__class__.__name__ + '('\
            'patch_size=' + str(self.patch_size) + ', ' + \
            'num_ang_bins=' + str(self.num_ang_bins) + ')'

    def forward(self, laf: torch.Tensor, img: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        Args:
            laf: (torch.Tensor), shape [BxNx2x3]
            img: (torch.Tensor), shape [Bx1xHxW]

        Returns:
            laf_out: (torch.Tensor), shape [BxNx2x3] """
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
        patches: torch.Tensor = extract_patches_from_pyramid(img,
                                                             laf,
                                                             self.patch_size).view(-1,
                                                                                   1,
                                                                                   self.patch_size,
                                                                                   self.patch_size)
        angles_radians: torch.Tensor = self.angle_detector(patches).view(B, N)
        rotmat: torch.Tensor = angle_to_rotation_matrix(rad2deg(angles_radians)).view(B * N, 2, 2)

        laf_out: torch.Tensor = torch.cat([torch.bmm(make_upright(laf).view(B * N, 2, 3)[:, :2, :2], rotmat),
                                           laf.view(B * N, 2, 3)[:, :2, 2:]], dim=2).view(B, N, 2, 3)
        return laf_out
