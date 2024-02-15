from __future__ import annotations

from kornia.losses.cauchy import CauchyLoss, cauchy_loss
from kornia.losses.charbonnier import CharbonnierLoss, charbonnier_loss
from kornia.losses.depth_smooth import InverseDepthSmoothnessLoss, inverse_depth_smoothness_loss
from kornia.losses.dice import DiceLoss, dice_loss
from kornia.losses.divergence import js_div_loss_2d, kl_div_loss_2d
from kornia.losses.focal import BinaryFocalLossWithLogits, FocalLoss, binary_focal_loss_with_logits, focal_loss
from kornia.losses.geman_mcclure import GemanMcclureLoss, geman_mcclure_loss
from kornia.losses.hausdorff import HausdorffERLoss, HausdorffERLoss3D
from kornia.losses.lovasz_hinge import LovaszHingeLoss, lovasz_hinge_loss
from kornia.losses.lovasz_softmax import LovaszSoftmaxLoss, lovasz_softmax_loss
from kornia.losses.ms_ssim import MS_SSIMLoss
from kornia.losses.psnr import PSNRLoss, psnr_loss
from kornia.losses.ssim import SSIMLoss, ssim_loss
from kornia.losses.ssim3d import SSIM3DLoss, ssim3d_loss
from kornia.losses.total_variation import TotalVariation, total_variation
from kornia.losses.tversky import TverskyLoss, tversky_loss
from kornia.losses.welsch import WelschLoss, welsch_loss

__all__ = [
    "inverse_depth_smoothness_loss",
    "InverseDepthSmoothnessLoss",
    "dice_loss",
    "DiceLoss",
    "js_div_loss_2d",
    "kl_div_loss_2d",
    "binary_focal_loss_with_logits",
    "BinaryFocalLossWithLogits",
    "focal_loss",
    "FocalLoss",
    "HausdorffERLoss",
    "HausdorffERLoss3D",
    "psnr_loss",
    "PSNRLoss",
    "ssim_loss",
    "SSIMLoss",
    "ssim3d_loss",
    "SSIM3DLoss",
    "total_variation",
    "TotalVariation",
    "tversky_loss",
    "TverskyLoss",
    "MS_SSIMLoss",
    "LovaszHingeLoss",
    "lovasz_hinge_loss",
    "LovaszSoftmaxLoss",
    "lovasz_softmax_loss",
    "WelschLoss",
    "welsch_loss",
    "CauchyLoss",
    "cauchy_loss",
    "GemanMcclureLoss",
    "geman_mcclure_loss",
    "CharbonnierLoss",
    "charbonnier_loss",
]
