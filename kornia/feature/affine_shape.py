from typing import Tuple

import torch

import kornia


class PatchAffineShapeEstimator(kornia.nn.PatchAffineShapeEstimator):
    """Module, which estimates the second moment matrix of the patch gradients in order to determine the
    affine shape of the local feature as in :cite:`baumberg2000`.

    Args:
        patch_size: int, default = 19
        eps: float, for safe division, default is 1e-10"""

    def __init__(self, patch_size: int = 19, eps: float = 1e-10):
        super(PatchAffineShapeEstimator, self).__init__(patch_size, eps)
        kornia.deprecation_warning(
            "kornia.feature.PatchAffineShapeEstimator", "kornia.nn.PatchAffineShapeEstimator")


class LAFAffineShapeEstimator(kornia.nn.LAFAffineShapeEstimator):
    """Module, which extracts patches using input images and local affine frames (LAFs),
    then runs :class:`~kornia.feature.PatchAffineShapeEstimator` on patches to estimate LAFs shape.
    Then original LAF shape is replaced with estimated one. The original LAF orientation is not preserved,
    so it is recommended to first run LAFAffineShapeEstimator and then LAFOrienter.

    Args:
            patch_size: int, default = 32"""

    def __init__(self,
                 patch_size: int = 32) -> None:
        super(LAFAffineShapeEstimator, self).__init__(patch_size)
        kornia.deprecation_warning(
            "kornia.feature.LAFAffineShapeEstimator", "kornia.nn.LAFAffineShapeEstimator")
