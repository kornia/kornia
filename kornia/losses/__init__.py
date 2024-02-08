from __future__ import annotations

from .cauchy import CauchyLoss
from .cauchy import cauchy_loss
from .charbonnier import CharbonnierLoss
from .charbonnier import charbonnier_loss
from .depth_smooth import InverseDepthSmoothnessLoss
from .depth_smooth import inverse_depth_smoothness_loss
from .dice import DiceLoss
from .dice import dice_loss
from .divergence import js_div_loss_2d
from .divergence import kl_div_loss_2d
from .focal import BinaryFocalLossWithLogits
from .focal import FocalLoss
from .focal import binary_focal_loss_with_logits
from .focal import focal_loss
from .geman_mcclure import GemanMcclureLoss
from .geman_mcclure import geman_mcclure_loss
from .hausdorff import HausdorffERLoss
from .hausdorff import HausdorffERLoss3D
from .lovasz_hinge import LovaszHingeLoss
from .lovasz_hinge import lovasz_hinge_loss
from .lovasz_softmax import LovaszSoftmaxLoss
from .lovasz_softmax import lovasz_softmax_loss
from .ms_ssim import MS_SSIMLoss
from .psnr import PSNRLoss
from .psnr import psnr_loss
from .ssim import SSIMLoss
from .ssim import ssim_loss
from .ssim3d import SSIM3DLoss
from .ssim3d import ssim3d_loss
from .total_variation import TotalVariation
from .total_variation import total_variation
from .tversky import TverskyLoss
from .tversky import tversky_loss
from .welsch import WelschLoss
from .welsch import welsch_loss

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
