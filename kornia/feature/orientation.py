from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import kornia


class PassLAF(kornia.nn.PassLAF):
    """Dummy module to use instead of local feature orientation or affine shape estimator"""

    def __init__(self) -> None:
        super(PassLAF, self).__init__()
        kornia.deprecation_warning("kornia.feature.PassLAF", "kornia.nn.PassLAF")


class PatchDominantGradientOrientation(kornia.nn.PatchDominantGradientOrientation):
    """Module, which estimates the dominant gradient orientation of the given patches, in radians.
    Zero angle points towards right.

    Args:
            patch_size: int, default = 32
            num_angular_bins: int, default is 36
            eps: float, for safe division, and arctan, default is 1e-8"""

    def __init__(self,
                 patch_size: int = 32,
                 num_angular_bins: int = 36, eps: float = 1e-8):
        super(PatchDominantGradientOrientation, self).__init__(patch_size, num_angular_bins, eps)
        kornia.deprecation_warning(
            "kornia.feature.PatchDominantGradientOrientation", "kornia.nn.PatchDominantGradientOrientation")


class LAFOrienter(kornia.nn.LAFOrienter):
    """Module, which extracts patches using input images and local affine frames (LAFs),
    then runs :class:`~kornia.feature.PatchDominantGradientOrientation`
    on patches and then rotates the LAFs by the estimated angles

    Args:
            patch_size: int, default = 32
            num_angular_bins: int, default is 36"""

    def __init__(self,
                 patch_size: int = 32,
                 num_angular_bins: int = 36):
        super(LAFOrienter, self).__init__(patch_size, num_angular_bins)
        kornia.deprecation_warning(
            "kornia.feature.LAFOrienter", "kornia.nn.LAFOrienter")
