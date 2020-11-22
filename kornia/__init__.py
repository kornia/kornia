# Make sure that kornia is running on Python 3.6.0 or later
# (to avoid running into this bug: https://bugs.python.org/issue29246)
import sys
import warnings
if sys.version_info < (3, 6, 0):
    raise RuntimeError("Kornia requires Python 3.6.0 or later")

try:
    from .version import __version__  # noqa: F401
except ImportError:
    pass


def deprecation_warning(name: str, replacement: str) -> None:
    warnings.warn(f"`{name}` is no longer maintained and will be removed from the future versions. "
                  f"Please use {replacement} instead.", category=DeprecationWarning)


from kornia import nn
from kornia import color
from kornia import contrib
from kornia import enhance
from kornia import feature
from kornia import filters
from kornia import geometry
from kornia import jit
from kornia import losses
from kornia import utils
from kornia import augmentation

# Exposes package functional to top level
from kornia.augmentation.functional import *
from kornia.color import (
    rgb_to_grayscale,
    bgr_to_grayscale,
    bgr_to_rgb,
    rgb_to_bgr,
    rgb_to_rgba,
    bgr_to_rgba,
    rgba_to_rgb,
    rgba_to_bgr,
    rgb_to_hsv,
    hsv_to_rgb,
    rgb_to_hls,
    rgb_to_yuv,
    yuv_to_rgb,
    hls_to_rgb,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
    rgb_to_xyz,
    xyz_to_rgb,
    rgb_to_luv,
    luv_to_rgb,
)
from kornia.enhance import (
    normalize,
    normalize_min_max,
    denormalize,
    zca_mean,
    zca_whiten,
    linear_transform,
    adjust_brightness,
    adjust_contrast,
    adjust_gamma,
    adjust_hue,
    adjust_saturation,
)
from kornia.contrib import (
    extract_tensor_patches,
    max_blur_pool2d,
)
from kornia.feature import (
    nms2d,
    harris_response,
    hessian_response,
    gftt_response,
    SIFTDescriptor
)
from kornia.filters import (
    get_gaussian_kernel1d,
    get_gaussian_erf_kernel1d,
    get_gaussian_discrete_kernel1d,
    get_gaussian_kernel2d,
    get_laplacian_kernel1d,
    get_laplacian_kernel2d,
    get_motion_kernel2d,
    gaussian_blur2d,
    laplacian,
    sobel,
    spatial_gradient,
    box_blur,
    median_blur,
    motion_blur,
    filter2D,
    filter3D,
)
from kornia.losses import (
    ssim,
    dice_loss,
    tversky_loss,
    inverse_depth_smoothness_loss,
    total_variation,
    psnr_loss,
    kl_div_loss_2d,
    js_div_loss_2d,
)
from kornia.utils import (
    one_hot,
    create_meshgrid,
    create_meshgrid3d,
    tensor_to_image,
    image_to_tensor,
    save_pointcloud_ply,
    load_pointcloud_ply,
)

from kornia.geometry import *
from kornia.constants import *
