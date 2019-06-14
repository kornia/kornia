from .ssim import SSIM, ssim
from .dice import DiceLoss, dice_loss
from .tversky import TverskyLoss, tversky_loss
from .focal import FocalLoss, focal_loss
from .depth_smooth import (
    InverseDepthSmoothnessLoss, inverse_depth_smoothness_loss
)
from .divergence import kl_div_loss_2d, js_div_loss_2d
