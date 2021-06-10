# Make sure that kornia is running on Python 3.6.0 or later
# (to avoid running into this bug: https://bugs.python.org/issue29246)
import sys

if sys.version_info < (3, 6, 0):
    raise RuntimeError("Kornia requires Python 3.6.0 or later")

try:
    from .version import __version__  # noqa: F401
except ImportError:
    pass

from kornia import augmentation, color, contrib, enhance, feature, filters, geometry, jit, losses, morphology, utils
from kornia.color import (
    bgr_to_grayscale,
    bgr_to_rgb,
    bgr_to_rgba,
    hls_to_rgb,
    hsv_to_rgb,
    lab_to_rgb,
    luv_to_rgb,
    rgb_to_bgr,
    rgb_to_grayscale,
    rgb_to_hls,
    rgb_to_hsv,
    rgb_to_lab,
    rgb_to_luv,
    rgb_to_rgba,
    rgb_to_xyz,
    rgb_to_ycbcr,
    rgb_to_yuv,
    rgba_to_bgr,
    rgba_to_rgb,
    xyz_to_rgb,
    ycbcr_to_rgb,
    yuv_to_rgb,
)
from kornia.constants import *
from kornia.contrib import extract_tensor_patches, max_blur_pool2d
from kornia.enhance import (
    adjust_brightness,
    adjust_contrast,
    adjust_gamma,
    adjust_hue,
    adjust_saturation,
    denormalize,
    linear_transform,
    normalize,
    normalize_min_max,
    zca_mean,
    zca_whiten,
)
from kornia.feature import gftt_response, harris_response, hessian_response, MKDDescriptor, nms2d, SIFTDescriptor
from kornia.filters import (
    box_blur,
    canny,
    filter2d,
    filter3d,
    gaussian_blur2d,
    get_gaussian_discrete_kernel1d,
    get_gaussian_erf_kernel1d,
    get_gaussian_kernel1d,
    get_gaussian_kernel2d,
    get_laplacian_kernel1d,
    get_laplacian_kernel2d,
    get_motion_kernel2d,
    get_motion_kernel3d,
    laplacian,
    median_blur,
    motion_blur,
    sobel,
    spatial_gradient,
    unsharp_mask,
)
from kornia.geometry import *
from kornia.losses import (
    dice_loss,
    inverse_depth_smoothness_loss,
    js_div_loss_2d,
    kl_div_loss_2d,
    psnr_loss,
    ssim,
    total_variation,
    tversky_loss,
)
from kornia.testing import xla_is_available
from kornia.utils import (
    create_meshgrid,
    image_to_tensor,
    load_pointcloud_ply,
    one_hot,
    save_pointcloud_ply,
    tensor_to_image,
)

# Exposes package functional to top level


# Exposes package functional to top level
