# Make sure that kornia is running on Python 3.6.0 or later
# (to avoid running into this bug: https://bugs.python.org/issue29246)
import sys
if sys.version_info < (3, 6, 0):
    raise RuntimeError("Kornia requires Python 3.6.0 or later")

from .version import __version__

from kornia import color
from kornia import contrib
from kornia import feature
from kornia import filters
from kornia import geometry
from kornia import losses
from kornia import utils
from kornia import augmentation

# Exposes package functional to top level

from kornia.color import (
    rgb_to_grayscale,
    bgr_to_grayscale,
    bgr_to_rgb,
    rgb_to_bgr,
    rgb_to_hsv,
    hsv_to_rgb,
    rgb_to_hls,
    hls_to_rgb,
    normalize,
    denormalize,
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
    non_maxima_suppression2d,
    harris_response,
    SIFTDescriptor
)
from kornia.filters import (
    get_gaussian_kernel1d,
    get_gaussian_kernel2d,
    get_laplacian_kernel1d,
    get_laplacian_kernel2d,
    gaussian_blur2d,
    laplacian,
    sobel,
    spatial_gradient,
    box_blur,
    median_blur,
    filter2D,
)
from kornia.losses import (
    ssim,
    dice_loss,
    tversky_loss,
    inverse_depth_smoothness_loss,
    total_variation,
    psnr_loss
)
from kornia.utils import (
    one_hot,
    create_meshgrid,
    tensor_to_image,
    image_to_tensor,
    save_pointcloud_ply,
    load_pointcloud_ply,
)
from kornia.augmentation import (
    random_hflip,
)

from kornia.geometry import *
from kornia.geometry import pi
