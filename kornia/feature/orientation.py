import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.constants import pi
from kornia.feature import (
    extract_patches_from_pyramid,
    get_laf_orientation,
    raise_error_if_laf_is_not_valid,
    set_laf_orientation,
)
from kornia.filters import get_gaussian_kernel2d, SpatialGradient
from kornia.geometry import rad2deg

urls: Dict[str, str] = {}
urls["orinet"] = "https://github.com/ducha-aiki/affnet/raw/master/pretrained/OriNet.pth"


class PassLAF(nn.Module):
    """Dummy module to use instead of local feature orientation or affine shape estimator"""

    def forward(self, laf: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            laf: 4d tensor.
            img: the input image tensor.

        Return:
            torch.Tensor: unchanged laf from the input."""
        return laf


class PatchDominantGradientOrientation(nn.Module):
    """Module, which estimates the dominant gradient orientation of the given patches, in radians.

    Zero angle points towards right.

    Args:
        patch_size:
        num_angular_bins:
        eps: for safe division, and arctan.
    """

    def __init__(self, patch_size: int = 32, num_angular_bins: int = 36, eps: float = 1e-8):
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

    def __repr__(self):
        return (
            self.__class__.__name__ + '('
            'patch_size='
            + str(self.patch_size)
            + ', '
            + 'num_ang_bins='
            + str(self.num_ang_bins)
            + ', '
            + 'eps='
            + str(self.eps)
            + ')'
        )

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        """Args:
            patch: (torch.Tensor) shape [Bx1xHxW]
        Returns:
            torch.Tensor: angle shape [B]"""
        if not isinstance(patch, torch.Tensor):
            raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(patch)))
        if not len(patch.shape) == 4:
            raise ValueError("Invalid input shape, we expect Bx1xHxW. Got: {}".format(patch.shape))
        B, CH, W, H = patch.size()
        if (W != self.patch_size) or (H != self.patch_size) or (CH != 1):
            raise TypeError(
                "input shape should be must be [Bx1x{}x{}]. "
                "Got {}".format(self.patch_size, self.patch_size, patch.size())
            )
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
        wo0_big = (1.0 - wo1_big) * mag
        wo1_big = wo1_big * mag
        ang_bins_list = []
        for i in range(0, self.num_ang_bins):
            ang_bins_i = F.adaptive_avg_pool2d(
                (bo0_big == i).to(patch.dtype) * wo0_big + (bo1_big == i).to(patch.dtype) * wo1_big, (1, 1)
            )
            ang_bins_list.append(ang_bins_i)
        ang_bins = torch.cat(ang_bins_list, 1).view(-1, 1, self.num_ang_bins)
        ang_bins = self.angular_smooth(ang_bins)
        values, indices = ang_bins.view(-1, self.num_ang_bins).max(1)
        angle = -((2.0 * pi * indices.to(patch.dtype) / float(self.num_ang_bins)) - pi)
        return angle


class OriNet(nn.Module):
    """Network, which estimates the canonical orientation of the given 32x32 patches, in radians.

    Zero angle points towards right. This is based on the original code from paper
    "Repeatability Is Not Enough: Learning Discriminative Affine Regions via Discriminability"".
    See :cite:`AffNet2018` for more details.

    Args:
        pretrained: Download and set pretrained weights to the model.
        eps: to avoid division by zero in atan2.

    Returns:
        Angle in radians.

    Shape:
        - Input: (B, 1, 32, 32)
        - Output: (B)

    Examples:
        >>> input = torch.rand(16, 1, 32, 32)
        >>> orinet = OriNet()
        >>> angle = orinet(input) # 16
    """

    def __init__(self, pretrained: bool = False, eps: float = 1e-8):
        super(OriNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(64, 2, kernel_size=8, stride=1, padding=1, bias=True),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.eps = eps
        # use torch.hub to load pretrained model
        if pretrained:
            pretrained_dict = torch.hub.load_state_dict_from_url(
                urls['orinet'], map_location=lambda storage, loc: storage
            )
            self.load_state_dict(pretrained_dict['state_dict'], strict=False)

    @staticmethod
    def _normalize_input(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        "Utility function that normalizes the input by batch." ""
        sp, mp = torch.std_mean(x, dim=(-3, -2, -1), keepdim=True)
        # WARNING: we need to .detach() input, otherwise the gradients produced by
        # the patches extractor with F.grid_sample are very noisy, making the detector
        # training totally unstable.
        return (x - mp.detach()) / (sp.detach() + eps)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        """Args:
            patch: (torch.Tensor) shape [Bx1xHxW]
        Returns:
            patch: (torch.Tensor) shape [B]"""
        xy = self.features(self._normalize_input(patch)).view(-1, 2)
        angle = torch.atan2(xy[:, 0] + 1e-8, xy[:, 1] + self.eps)
        return angle


class LAFOrienter(nn.Module):
    """Module, which extracts patches using input images and local affine frames (LAFs).

    Then runs :class:`~kornia.feature.PatchDominantGradientOrientation` or
    :class:`~kornia.feature.OriNet` on patches and then rotates the LAFs by the estimated angles

    Args:
        patch_size:
        num_angular_bins:
        angle_detector: Patch orientation estimator, e.g. :class:`~kornia.feature.PatchDominantGradientOrientation`
          or OriNet.
    """  # noqa pylint: disable

    def __init__(self, patch_size: int = 32, num_angular_bins: int = 36, angle_detector: Optional[nn.Module] = None):
        super(LAFOrienter, self).__init__()
        self.patch_size = patch_size
        self.num_ang_bins = num_angular_bins
        self.angle_detector: nn.Module
        if angle_detector is None:
            self.angle_detector = PatchDominantGradientOrientation(self.patch_size, self.num_ang_bins)
        else:
            self.angle_detector = angle_detector

    def __repr__(self):
        return (
            self.__class__.__name__ + '('
            'patch_size=' + str(self.patch_size) + ', ' + 'angle_detector=' + str(self.angle_detector) + ')'
        )

    def forward(self, laf: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            laf: shape [BxNx2x3]
            img: shape [Bx1xHxW]

        Returns:
            laf_out, shape [BxNx2x3]
        """
        raise_error_if_laf_is_not_valid(laf)
        img_message: str = "Invalid img shape, we expect BxCxHxW. Got: {}".format(img.shape)
        if not isinstance(img, torch.Tensor):
            raise TypeError("img type is not a torch.Tensor. Got {}".format(type(img)))
        if len(img.shape) != 4:
            raise ValueError(img_message)
        if laf.size(0) != img.size(0):
            raise ValueError(
                "Batch size of laf and img should be the same. Got {}, {}".format(img.size(0), laf.size(0))
            )
        B, N = laf.shape[:2]
        patches: torch.Tensor = extract_patches_from_pyramid(img, laf, self.patch_size).view(
            -1, 1, self.patch_size, self.patch_size
        )
        angles_radians: torch.Tensor = self.angle_detector(patches).view(B, N)
        prev_angle = get_laf_orientation(laf).view_as(angles_radians)
        laf_out: torch.Tensor = set_laf_orientation(laf, rad2deg(angles_radians) + prev_angle)
        return laf_out
