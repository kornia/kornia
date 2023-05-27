from __future__ import annotations

from .cauchy import CauchyLoss, cauchy_loss
from .charbonnier import CharbonnierLoss, charbonnier_loss
from .depth_smooth import InverseDepthSmoothnessLoss, inverse_depth_smoothness_loss
from .dice import DiceLoss, dice_loss
from .divergence import js_div_loss_2d, kl_div_loss_2d
from .focal import BinaryFocalLossWithLogits, FocalLoss, binary_focal_loss_with_logits, focal_loss
from .geman_mcclure import GemanMcclureLoss, geman_mcclure_loss
from .hausdorff import HausdorffERLoss, HausdorffERLoss3D
from .lovasz_hinge import LovaszHingeLoss, lovasz_hinge_loss
from .lovasz_softmax import LovaszSoftmaxLoss, lovasz_softmax_loss
from .ms_ssim import MS_SSIMLoss
from .psnr import PSNRLoss, psnr_loss
from .ssim import SSIMLoss, ssim_loss
from .ssim3d import SSIM3DLoss, ssim3d_loss
from .total_variation import TotalVariation, total_variation
from .tversky import TverskyLoss, tversky_loss
from .welsch import WelschLoss, welsch_loss

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
