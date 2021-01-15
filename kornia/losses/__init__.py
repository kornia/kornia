from .ssim import SSIM, ssim, SSIMLoss, ssim_loss
from .dice import DiceLoss, dice_loss
from .tversky import TverskyLoss, tversky_loss
from .focal import FocalLoss, focal_loss, BinaryFocalLossWithLogits, binary_focal_loss_with_logits
from .depth_smooth import (
    InverseDepthSmoothnessLoss, inverse_depth_smoothness_loss
)
from .divergence import kl_div_loss_2d, js_div_loss_2d
from .total_variation import TotalVariation, total_variation
from .psnr import PSNRLoss, psnr_loss, psnr
