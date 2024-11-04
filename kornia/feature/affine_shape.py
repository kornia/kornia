import math
import warnings
from typing import Dict, Optional

import torch
from torch import nn

from kornia.core.check import KORNIA_CHECK_LAF, KORNIA_CHECK_SHAPE
from kornia.filters.kernels import get_gaussian_kernel2d
from kornia.filters.sobel import SpatialGradient

from .laf import (
    ellipse_to_laf,
    extract_patches_from_pyramid,
    get_laf_orientation,
    get_laf_scale,
    make_upright,
    scale_laf,
    set_laf_orientation,
)

urls: Dict[str, str] = {}
urls["affnet"] = "https://github.com/ducha-aiki/affnet/raw/master/pretrained/AffNet.pth"


class PatchAffineShapeEstimator(nn.Module):
    r"""Module, which estimates the second moment matrix of the patch gradients.

    The method determines the affine shape of the local feature as in :cite:`baumberg2000`.

    Args:
        patch_size: the input image patch size.
        eps: for safe division.
    """

    def __init__(self, patch_size: int = 19, eps: float = 1e-10) -> None:
        super().__init__()
        self.patch_size: int = patch_size
        self.gradient: nn.Module = SpatialGradient("sobel", 1)
        self.eps: float = eps
        sigma: float = float(self.patch_size) / math.sqrt(2.0)
        self.weighting: torch.Tensor = get_gaussian_kernel2d((self.patch_size, self.patch_size), (sigma, sigma), True)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(patch_size={self.patch_size}, eps={self.eps})"

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch: :math:`(B, 1, H, W)`

        Returns:
            torch.Tensor: ellipse_shape :math:`(B, 1, 3)`

        """
        KORNIA_CHECK_SHAPE(patch, ["B", "1", "H", "W"])
        self.weighting = self.weighting.to(patch.dtype).to(patch.device)
        grads: torch.Tensor = self.gradient(patch) * self.weighting
        # unpack the edges
        gx: torch.Tensor = grads[:, :, 0]
        gy: torch.Tensor = grads[:, :, 1]
        # abc == 1st axis, mixture, 2nd axis. Ellipse_shape is a 2nd moment matrix.
        ellipse_shape = torch.cat(
            [
                gx.pow(2).mean(dim=2).mean(dim=2, keepdim=True),
                (gx * gy).mean(dim=2).mean(dim=2, keepdim=True),
                gy.pow(2).mean(dim=2).mean(dim=2, keepdim=True),
            ],
            dim=2,
        )

        # Now lets detect degenerate cases: when 2 or 3 elements are close to zero (e.g. if patch is completely black
        bad_mask = ((ellipse_shape < self.eps).float().sum(dim=2, keepdim=True) >= 2).to(ellipse_shape.dtype)
        # We will replace degenerate shape with circular shapes.
        circular_shape = torch.tensor([1.0, 0.0, 1.0]).to(ellipse_shape.device).to(ellipse_shape.dtype).view(1, 1, 3)
        ellipse_shape = ellipse_shape * (1.0 - bad_mask) + circular_shape * bad_mask
        # normalization
        ellipse_shape = ellipse_shape / ellipse_shape.max(dim=2, keepdim=True)[0]
        return ellipse_shape


class LAFAffineShapeEstimator(nn.Module):
    """Module, which extracts patches using input images and local affine frames (LAFs).

    Then runs :class:`~kornia.feature.PatchAffineShapeEstimator` on patches to estimate LAFs shape.

    Then original LAF shape is replaced with estimated one. The original LAF orientation is not preserved,
    so it is recommended to first run LAFAffineShapeEstimator and then LAFOrienter,


    Args:
        patch_size: the input image patch size.
        affine_shape_detector: Patch affine shape estimator, :class:`~kornia.feature.PatchAffineShapeEstimator`.
        preserve_orientation: if True, the original orientation is preserved.
    """  # pylint: disable

    def __init__(
        self, patch_size: int = 32, affine_shape_detector: Optional[nn.Module] = None, preserve_orientation: bool = True
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.affine_shape_detector = affine_shape_detector or PatchAffineShapeEstimator(self.patch_size)
        self.preserve_orientation = preserve_orientation
        if preserve_orientation:
            warnings.warn(
                "`LAFAffineShapeEstimator` default behaviour is changed "
                "and now it does preserve original LAF orientation. "
                "Make sure your code accounts for this.",
                DeprecationWarning,
                stacklevel=2,
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(patch_size={self.patch_size}, "
            f"affine_shape_detector={self.affine_shape_detector}, "
            f"preserve_orientation={self.preserve_orientation})"
        )

    def forward(self, laf: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            LAF: :math:`(B, N, 2, 3)`
            img: :math:`(B, 1, H, W)`

        Returns:
            LAF_out: :math:`(B, N, 2, 3)`
        """
        KORNIA_CHECK_LAF(laf)
        KORNIA_CHECK_SHAPE(img, ["B", "1", "H", "W"])
        B, N = laf.shape[:2]
        PS: int = self.patch_size
        patches: torch.Tensor = extract_patches_from_pyramid(img, make_upright(laf), PS, True).view(-1, 1, PS, PS)
        ellipse_shape: torch.Tensor = self.affine_shape_detector(patches)
        ellipses = torch.cat([laf.view(-1, 2, 3)[..., 2].unsqueeze(1), ellipse_shape], dim=2).view(B, N, 5)
        scale_orig = get_laf_scale(laf)
        if self.preserve_orientation:
            ori_orig = get_laf_orientation(laf)
        laf_out = ellipse_to_laf(ellipses)
        ellipse_scale = get_laf_scale(laf_out)
        laf_out = scale_laf(laf_out, scale_orig / ellipse_scale)
        if self.preserve_orientation:
            laf_out = set_laf_orientation(laf_out, ori_orig)
        return laf_out


class LAFAffNetShapeEstimator(nn.Module):
    """Module, which extracts patches using input images and local affine frames (LAFs).

    Then runs AffNet on patches to estimate LAFs shape. This is based on the original code from paper
    "Repeatability Is Not Enough: Learning Discriminative Affine Regions via Discriminability"".
    See :cite:`AffNet2018` for more details.

    Then original LAF shape is replaced with estimated one. The original LAF orientation is not preserved,
    so it is recommended to first run LAFAffineShapeEstimator and then LAFOrienter.

    Args:
        pretrained: Download and set pretrained weights to the model.
    """

    def __init__(self, pretrained: bool = False, preserve_orientation: bool = True) -> None:
        super().__init__()
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
            nn.Conv2d(64, 3, kernel_size=8, stride=1, padding=0, bias=True),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.patch_size = 32
        # use torch.hub to load pretrained model
        if pretrained:
            pretrained_dict = torch.hub.load_state_dict_from_url(urls["affnet"], map_location=torch.device("cpu"))
            self.load_state_dict(pretrained_dict["state_dict"], strict=False)
        self.preserve_orientation = preserve_orientation
        if preserve_orientation:
            warnings.warn(
                "`LAFAffNetShapeEstimator` default behaviour is changed "
                "and now it does preserve original LAF orientation. "
                "Make sure your code accounts for this.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.eval()

    @staticmethod
    def _normalize_input(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Utility function that normalizes the input by batch."""
        sp, mp = torch.std_mean(x, dim=(-3, -2, -1), keepdim=True)
        # WARNING: we need to .detach() input, otherwise the gradients produced by
        # the patches extractor with F.grid_sample are very noisy, making the detector
        # training totally unstable.
        return (x - mp.detach()) / (sp.detach() + eps)

    def forward(self, laf: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            LAF: :math:`(B, N, 2, 3)`
            img: :math:`(B, 1, H, W)`

        Returns:
            LAF_out: :math:`(B, N, 2, 3)`
        """
        KORNIA_CHECK_LAF(laf)
        KORNIA_CHECK_SHAPE(img, ["B", "1", "H", "W"])
        B, N = laf.shape[:2]
        PS: int = self.patch_size
        patches: torch.Tensor = extract_patches_from_pyramid(img, make_upright(laf), PS, True).view(-1, 1, PS, PS)
        xy = self.features(self._normalize_input(patches)).view(-1, 3)
        a1 = torch.cat([1.0 + xy[:, 0].reshape(-1, 1, 1), 0 * xy[:, 0].reshape(-1, 1, 1)], dim=2)
        a2 = torch.cat([xy[:, 1].reshape(-1, 1, 1), 1.0 + xy[:, 2].reshape(-1, 1, 1)], dim=2)
        new_laf_no_center = torch.cat([a1, a2], dim=1).reshape(B, N, 2, 2)
        new_laf = torch.cat([new_laf_no_center, laf[:, :, :, 2:3]], dim=3)
        scale_orig = get_laf_scale(laf)
        if self.preserve_orientation:
            ori_orig = get_laf_orientation(laf)
        ellipse_scale = get_laf_scale(new_laf)
        laf_out = scale_laf(make_upright(new_laf), scale_orig / ellipse_scale)
        if self.preserve_orientation:
            laf_out = set_laf_orientation(laf_out, ori_orig)
        return laf_out
