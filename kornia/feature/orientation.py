# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core.check import KORNIA_CHECK_LAF, KORNIA_CHECK_SHAPE
from kornia.core.download import hf_url, load_state_dict_from_url
from kornia.filters import SpatialGradient, get_gaussian_discrete_kernel1d, get_gaussian_kernel2d
from kornia.geometry import rad2deg

from ._layout import _patch_conv_layout
from .laf import extract_patches_from_pyramid, get_laf_orientation, set_laf_orientation
from .siftdesc import _gradient_magnitude_orientation

urls: Dict[str, str | list[str]] = {}
urls["orinet"] = [
    hf_url("orinet", "OriNet.pth"),
    "https://github.com/ducha-aiki/affnet/raw/master/pretrained/OriNet.pth",
]


class PassLAF(nn.Module):
    """Return LAFs unchanged, in place of an orientation or affine shape estimator.

    ``out = PassLAF()(laf, img)`` takes ``(B, N, 2, 3)`` LAFs and an image and returns its ``laf`` argument itself, not
    a copy, without reading ``img``. It is the default ``ori_module`` and ``aff_module`` of
    :class:`ScaleSpaceDetector`, :class:`MultiResolutionDetector` and :class:`KeyNetDetector`, and the ``ori_module``
    that keeps features upright. To estimate the orientation, use :class:`LAFOrienter`; to estimate the affine shape,
    :class:`LAFAffineShapeEstimator` or :class:`LAFAffNetShapeEstimator`.

    See Also:
        - :class:`LAFOrienter`: rotates each LAF by the orientation estimated on its image patch.
        - :class:`LAFAffineShapeEstimator`: adapts each LAF's shape to the second-moment matrix of its patch.
        - :class:`LAFAffNetShapeEstimator`: adapts each LAF's shape with the learned AffNet.

    """

    def forward(self, laf: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """Run forward.

        Args:
            laf: ``(B, N, 2, 3)``
            img: not used.

        Returns:
            LAF, unchanged ``(B, N, 2, 3)``

        """
        return laf


class PatchDominantGradientOrientation(nn.Module):
    r"""Estimate the orientation of patches from the peak of their histogram of gradient directions.

    ``angles = PatchDominantGradientOrientation(patch_size)(patches)`` takes ``(B, 1, patch_size, patch_size)``
    patches and returns the direction of each patch's dominant gradient, ``(B,)`` in radians, 0 for a gradient that
    points to the right: the peak, refined between bins, of a ``num_angular_bins``-bin histogram of the gradient
    directions, each pixel weighted by its gradient magnitude and a Gaussian centred on the patch, as in SIFT's
    orientation assignment. It returns one angle per patch, that of the highest peak. To orient LAFs on an image, run
    it through :class:`LAFOrienter`, whose default it is; for a learned estimate, use :class:`OriNet` instead.

    Convention:
        - The angle is that of the dominant gradient, which points toward brighter values, counter-clockwise on
          screen with image y pointing down: :math:`-\operatorname{atan2}(g_y, g_x)` in pixel coordinates. A gradient
          pointing down the patch gives :math:`-\pi / 2`. :class:`~kornia.feature.LAFOrienter` adds the angle to the
          LAF orientation.
        - The angle is not wrapped into :math:`(-\pi, \pi]`: it can exceed :math:`\pi` by up to
          ``pi / num_angular_bins``.
        - A patch whose Sobel gradients are all exactly zero, such as an all-zero patch, gets the angle
          :math:`\pi` from its empty histogram. A nearly flat patch gets an arbitrary angle set by rounding noise: a
          constant float64 patch can have nonzero rounding-level gradients, and so does a patch sampled from a flat
          image region by :func:`~kornia.feature.extract_patches_from_pyramid`.
        - Known defect: in float16 the gradients with respect to a textured patch and to the smoothing weights are
          NaN, because the histogram is cast back to float16 before the peak is refined
          (`#6455 <https://github.com/kornia/kornia/issues/6455>`_).

    Args:
        patch_size: size of the (square) input patch.
        num_angular_bins: number of histogram bins.
        eps: for safe division, and arctan.

    Note:
        Each gradient pixel is accumulated into its two neighbouring histogram bins with
        ``scatter_add_``. On CUDA that accumulation uses atomics, so two calls on identical input
        can differ at the ulp level, and a patch whose two strongest bins are nearly tied can
        return the neighbouring peak. ``torch.use_deterministic_algorithms(True)`` selects the
        deterministic kernel and restores run-to-run reproducibility.

    See Also:
        - :class:`OriNet`: a learned orientation of 32 x 32 patches.
        - :class:`LAFOrienter`: samples each LAF's patch, runs this module on it by default and rotates the LAF.
        - :class:`PatchAffineShapeEstimator`: the affine shape of a patch, from the second moments of the same Sobel
          gradients.

    """

    def __init__(self, patch_size: int = 32, num_angular_bins: int = 36, eps: float = 1e-8) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_ang_bins = num_angular_bins
        self.gradient = SpatialGradient("sobel", 1)
        self.eps = eps
        self.angular_smooth = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False, padding_mode="circular")
        with torch.no_grad():
            self.angular_smooth.weight[:] = get_gaussian_discrete_kernel1d(5, 1.6)
        sigma: float = float(self.patch_size) / 6.0
        # non-persistent: derived from `patch_size`, so it stays out of `state_dict()` but follows `.to()`
        self.register_buffer(
            "weighting",
            get_gaussian_kernel2d((self.patch_size, self.patch_size), (sigma, sigma), True),
            persistent=False,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(patch_size={self.patch_size}, num_ang_bins={self.num_ang_bins}, eps={self.eps})"
        )

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        """Run forward.

        Args:
            patch: ``(B, 1, patch_size, patch_size)``; another size raises ``TypeError``.

        Returns:
            angle in radians: ``(B,)``

        """
        KORNIA_CHECK_SHAPE(patch, ["B", "1", "H", "W"])
        _, CH, W, H = patch.size()
        if (W != self.patch_size) or (H != self.patch_size) or (CH != 1):
            raise TypeError(
                f"input shape should be must be [Bx1x{self.patch_size}x{self.patch_size}]. Got {patch.size()}"
            )
        # cast into a local; rebinding `self.weighting` would make the buffer's dtype/device depend on
        # whichever tensor was passed last, and is a `torch.compile` guard hazard.
        weighting = self.weighting.to(patch.dtype).to(patch.device)
        angular_smooth_weight = self.angular_smooth.weight.to(patch.dtype).to(patch.device)
        angular_smooth_bias = self.angular_smooth.bias
        if angular_smooth_bias is not None:
            angular_smooth_bias = angular_smooth_bias.to(patch.dtype).to(patch.device)
        grads: torch.Tensor = self.gradient(patch)
        # unpack the edges
        gx: torch.Tensor = grads[:, :, 0]
        gy: torch.Tensor = grads[:, :, 1]

        # float16 is lifted to float32 inside: the squared gradient of a flat patch underflows
        # and `sqrt(0 + eps)` with an underflowed `eps` is NaN in the backward and, through
        # `atan2`, in the forward as well. bfloat16 holds both and takes the plain expression.
        mag, ori = _gradient_magnitude_orientation(gx, gy, self.eps)
        mag = mag * weighting

        o_big = float(self.num_ang_bins) * (ori + math.pi) / (2.0 * math.pi)
        bo0_big = torch.floor(o_big)
        wo1_big = o_big - bo0_big
        bo0_big = bo0_big % self.num_ang_bins
        bo1_big = (bo0_big + 1) % self.num_ang_bins
        wo0_big = (1.0 - wo1_big) * mag
        wo1_big = wo1_big * mag
        # Each pixel votes into just two bins. Accumulate those votes directly instead
        # of scanning the entire patch once per angular bin. Like average pooling,
        # accumulate half-precision inputs in float32, then divide before casting back.
        accumulation_dtype = torch.float64 if patch.dtype == torch.float64 else torch.float32
        ang_bins = torch.zeros(patch.shape[0], self.num_ang_bins, device=patch.device, dtype=accumulation_dtype)
        # The `%` after the integer conversion only keeps the index in range: a NaN orientation
        # casts to an arbitrary integer. A patch with NaN pixels returns a finite but arbitrary
        # angle, as it did before this accumulation; the `max`/`where` below do not propagate NaN.
        ang_bins.scatter_add_(
            1, bo0_big.flatten(1).long() % self.num_ang_bins, wo0_big.flatten(1).to(accumulation_dtype)
        )
        ang_bins.scatter_add_(
            1, bo1_big.flatten(1).long() % self.num_ang_bins, wo1_big.flatten(1).to(accumulation_dtype)
        )
        ang_bins = (ang_bins / (W * H)).to(patch.dtype).view(-1, 1, self.num_ang_bins)
        angular_smooth_padding = self.angular_smooth.padding
        if self.angular_smooth.padding_mode != "zeros":
            ang_bins = F.pad(
                ang_bins,
                (angular_smooth_padding[0], angular_smooth_padding[0]),
                mode=self.angular_smooth.padding_mode,
            )
            angular_smooth_padding = (0,)
        ang_bins = F.conv1d(
            ang_bins,
            angular_smooth_weight,
            angular_smooth_bias,
            self.angular_smooth.stride,
            angular_smooth_padding,
            self.angular_smooth.dilation,
            self.angular_smooth.groups,
        ).view(-1, self.num_ang_bins)
        values, indices = ang_bins.max(1)
        indices_left = (self.num_ang_bins + indices - 1) % self.num_ang_bins
        indices_right = (indices + 1) % self.num_ang_bins
        left = torch.gather(ang_bins, 1, indices_left.reshape(-1, 1)).reshape(-1)
        center = values
        right = torch.gather(ang_bins, 1, indices_right.reshape(-1, 1)).reshape(-1)
        # A flat patch has an empty histogram (a zero-gradient pixel contributes no magnitude), and a
        # uniform one has no peak: the parabolic refinement is `0 / 0` there, so it is skipped.
        denom = left + right - 2.0 * center
        has_peak = denom != 0
        # Guard the division itself: masking 0 / 0 afterwards still gives NaN smoothing-weight gradients.
        safe_denom = torch.where(has_peak, denom, torch.ones_like(denom))
        c_subpix = torch.where(has_peak, 0.5 * (left - right) / safe_denom, torch.zeros_like(denom))
        return -((2.0 * math.pi * (indices.to(patch.dtype) + c_subpix) / float(self.num_ang_bins)) - math.pi)


class OriNet(nn.Module):
    """Estimate the orientation of 32 x 32 patches with the learned OriNet CNN.

    ``angles = OriNet()(patches)`` builds the network with its pretrained weights, in eval mode, and takes
    ``(B, 1, 32, 32)`` patches and returns a learned canonical orientation per patch, ``(B,)`` in radians, in the
    sense of the angle of :class:`PatchDominantGradientOrientation`; its zero is learned, so on a patch it need not
    match the dominant gradient direction. ``pretrained=False`` leaves the weights random. To orient LAFs, pass it to
    :class:`LAFOrienter` as ``angle_detector``, as :class:`KeyNetHardNet` and :class:`KeyNetAffNetHardNet` do. For an
    estimate that needs no weights, use :class:`PatchDominantGradientOrientation` instead. This is based on the
    original code from the paper "Repeatability Is Not Enough: Learning Discriminative Affine Regions via
    Discriminability". See :cite:`AffNet2018` for more details.

    Convention:
        - With the pretrained weights, the angle turns with the patch content, in the sense of
          :class:`~kornia.feature.PatchDominantGradientOrientation`: content turned counter-clockwise on screen
          increases the angle. See :class:`~kornia.feature.LAFOrienter` for its use.
        - Each patch is normalised first: its mean is subtracted and the result is divided by its standard deviation
          plus ``1e-6``. An offset of its intensity therefore does not change the angle, and a positive gain changes it
          only through that ``1e-6``: little on a textured patch, more as the patch flattens, up to ``pi`` rad on a
          nearly flat one. An inverted patch generally gets a different angle.

    Args:
        pretrained: Download and set pretrained weights to the model.
        eps: to avoid division by zero in atan2.

    Returns:
        Angle in radians.

    Shape:
        - Input: (B, 1, 32, 32)
        - Output: (B)

    See Also:
        - :class:`PatchDominantGradientOrientation`: the hand-crafted dominant gradient direction, with no weights.
        - :class:`LAFOrienter`: runs it on each LAF's patch, as ``angle_detector``, and rotates the LAF.
        - :class:`LAFAffNetShapeEstimator`: AffNet, the affine shape network of the same paper.

    Examples:
        >>> input = torch.rand(16, 1, 32, 32)
        >>> orinet = OriNet(pretrained=False).eval()
        >>> angle = orinet(input) # 16

    """

    def __init__(self, pretrained: bool = True, eps: float = 1e-8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(64, 2, kernel_size=8, stride=1, padding=1, bias=True),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.eps = eps
        # use torch.hub to load pretrained model
        if pretrained:
            pretrained_dict = load_state_dict_from_url(urls["orinet"], map_location=torch.device("cpu"))
            self.load_state_dict(pretrained_dict["state_dict"], strict=False)
            self.eval()

    @staticmethod
    def _normalize_input(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Subtract each patch's mean and divide by its standard deviation plus ``eps``."""
        # `torch.std_mean` decomposes to `prims.sum`, which has no ONNX lowering; the split form is equivalent.
        mp = torch.mean(x, dim=(-3, -2, -1), keepdim=True)
        sp = torch.std(x, dim=(-3, -2, -1), keepdim=True)
        # WARNING: we need to .detach() input, otherwise the gradients produced by
        # the patches extractor with F.grid_sample are very noisy, making the detector
        # training totally unstable.
        return (x - mp.detach()) / (sp.detach() + eps)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        """Run forward.

        Args:
            patch: ``(B, 1, 32, 32)``, the patch size the weights were trained on.

        Returns:
            angle in radians: ``(B,)``

        """
        xy = self.features(_patch_conv_layout(self._normalize_input(patch))).view(-1, 2)
        return torch.atan2(xy[:, 0] + 1e-8, xy[:, 1] + self.eps)


class LAFOrienter(nn.Module):
    """Orient LAFs by the angle that a patch orientation estimator finds in each LAF's patch.

    ``lafs = LAFOrienter(patch_size)(laf, img)`` takes ``(B, N, 2, 3)`` LAFs in pixels and their grayscale images
    ``(B, 1, H, W)``, samples each LAF's ``patch_size`` patch with :func:`extract_patches_from_pyramid`, runs
    ``angle_detector`` on it, :class:`PatchDominantGradientOrientation` with ``num_angular_bins`` bins by default, and
    returns the ``(B, N, 2, 3)`` LAFs rotated by the angles found. Passed to a detector as ``ori_module``, it runs after
    the ``aff_module``. To keep the LAFs upright, pass :class:`PassLAF` instead; to rotate them by known angles, use
    :func:`rotate_laf`.

    Convention:
        - Each LAF's patch (:func:`~kornia.feature.extract_patches_from_pyramid`) gives an angle, measured in the
          patch, and the output orientation is the input orientation plus that angle, converted to degrees, modulo
          360 degrees. :func:`~kornia.feature.get_laf_orientation` reads it back in ``[-180, 180]``. The centre, the
          scale and the upright shape (:func:`~kornia.feature.make_upright`) of the LAF are kept.
        - For an isotropic LAF, the default angle detector therefore turns the LAF's x axis onto the dominant
          gradient, whatever the input orientation: content turned 30 degrees counter-clockwise on screen turns the
          output orientation by +30, the sign of :func:`~kornia.feature.laf_from_center_scale_ori`
          (see :ref:`Local features <local-features-conventions>`). For a mirrored LAF (negative determinant) the x
          axis still turns onto the gradient, and the orientation then changes by -30.

    Args:
        patch_size: side of the square patch sampled for each LAF.
        num_angular_bins: number of histogram bins of the default angle detector; not used when ``angle_detector``
          is given.
        angle_detector: Patch orientation estimator, e.g. :class:`~kornia.feature.PatchDominantGradientOrientation`
          or :class:`~kornia.feature.OriNet`.

    See Also:
        - :class:`PatchDominantGradientOrientation`: the default ``angle_detector``, the dominant gradient direction.
        - :class:`OriNet`: a learned ``angle_detector`` for 32 x 32 patches.
        - :class:`PassLAF`: keeps the input orientation, for upright features.
        - :func:`rotate_laf`: rotates LAFs by known angles, as this module does with the estimated ones.
        - :class:`LAFAffineShapeEstimator`: estimates the affine shape, which a detector runs before this module.

    """  # pylint: disable

    def __init__(
        self, patch_size: int = 32, num_angular_bins: int = 36, angle_detector: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_ang_bins = num_angular_bins
        self.angle_detector: nn.Module
        if angle_detector is None:
            self.angle_detector = PatchDominantGradientOrientation(self.patch_size, self.num_ang_bins)
        else:
            self.angle_detector = angle_detector

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(patch_size={self.patch_size}, angle_detector={self.angle_detector})"

    def forward(self, laf: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """Run forward.

        Args:
            laf: ``(B, N, 2, 3)``
            img: ``(B, 1, H, W)``

        Returns:
            LAF_out: ``(B, N, 2, 3)``

        """
        KORNIA_CHECK_LAF(laf)
        KORNIA_CHECK_SHAPE(img, ["B", "C", "H", "W"])
        if laf.size(0) != img.size(0):
            raise ValueError(f"Batch size of laf and img should be the same. Got {img.size(0)}, {laf.size(0)}")
        B, N = laf.shape[:2]
        patches: torch.Tensor = extract_patches_from_pyramid(img, laf, self.patch_size).view(
            -1, 1, self.patch_size, self.patch_size
        )
        angles_radians: torch.Tensor = self.angle_detector(patches).view(B, N)
        prev_angle = get_laf_orientation(laf).view_as(angles_radians)
        laf_out: torch.Tensor = set_laf_orientation(laf, rad2deg(angles_radians) + prev_angle)
        return laf_out
