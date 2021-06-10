from .depth_smooth import inverse_depth_smoothness_loss, InverseDepthSmoothnessLoss
from .dice import dice_loss, DiceLoss
from .divergence import js_div_loss_2d, kl_div_loss_2d
from .focal import binary_focal_loss_with_logits, BinaryFocalLossWithLogits, focal_loss, FocalLoss
from .psnr import psnr, psnr_loss, PSNRLoss
from .ssim import SSIM, ssim, ssim_loss, SSIMLoss
from .total_variation import total_variation, TotalVariation
from .tversky import tversky_loss, TverskyLoss
