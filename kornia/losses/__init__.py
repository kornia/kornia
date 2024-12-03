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
    "BinaryFocalLossWithLogits",
    "CauchyLoss",
    "CharbonnierLoss",
    "DiceLoss",
    "FocalLoss",
    "GemanMcclureLoss",
    "HausdorffERLoss",
    "HausdorffERLoss3D",
    "InverseDepthSmoothnessLoss",
    "LovaszHingeLoss",
    "LovaszSoftmaxLoss",
    "MS_SSIMLoss",
    "PSNRLoss",
    "SSIM3DLoss",
    "SSIMLoss",
    "TotalVariation",
    "TverskyLoss",
    "WelschLoss",
    "binary_focal_loss_with_logits",
    "cauchy_loss",
    "charbonnier_loss",
    "dice_loss",
    "focal_loss",
    "geman_mcclure_loss",
    "inverse_depth_smoothness_loss",
    "js_div_loss_2d",
    "kl_div_loss_2d",
    "lovasz_hinge_loss",
    "lovasz_softmax_loss",
    "psnr_loss",
    "ssim3d_loss",
    "ssim_loss",
    "total_variation",
    "tversky_loss",
    "welsch_loss",
]
