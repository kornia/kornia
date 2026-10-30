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

"""Build Gaussian, Laplacian and scale-space pyramids, and blur and resize by one level.

:func:`pyrdown` and :func:`pyrup`, and the pyramids built from them, take channels-first images ``(B, C, H, W)``,
blur with the 5x5 binomial kernel (the outer product of ``[1, 4, 6, 4, 1] / 16`` with itself) and resize bilinearly:
``pyrdown`` blurs and then resizes to ``floor(side / factor)``, and ``pyrup`` resizes to twice the size and then
blurs. They default to ``align_corners=False`` and pad the blur by ``border_type``, ``'reflect'`` by default, under
which the blurred image needs at least 3 pixels per side. :class:`ScalePyramid` blurs with Gaussians of its own sigmas
and resizes with ``align_corners=True``.

Which function to use:

- One level down or up: :func:`pyrdown` or :class:`PyrDown`, :func:`pyrup` or :class:`PyrUp`.
- A list of Gaussian levels, each half the size of the previous: :func:`build_pyramid`. Band-pass levels that
  :func:`pyrup` and addition rebuild the image from: :func:`build_laplacian_pyramid`.
- Octaves of progressively blurred levels with their sigmas, for scale-space keypoint detection:
  :class:`ScalePyramid`.
- The integer helpers that :func:`build_laplacian_pyramid` pads with: :func:`is_powerof_two` and
  :func:`find_next_powerof_two`.
- To keep every second pixel of a blurred feature map in a network, use :func:`~kornia.filters.blur_pool2d`; to
  change the size without blurring, :func:`~kornia.geometry.transform.resize`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE
from kornia.filters import filter2d, gaussian_blur2d
from kornia.filters.gaussian import _gaussian_blur2d_cpu, _gaussian_blur2d_cpu_eligible

__all__ = [
    "PyrDown",
    "PyrUp",
    "ScalePyramid",
    "build_laplacian_pyramid",
    "build_pyramid",
    "find_next_powerof_two",
    "is_powerof_two",
    "pyrdown",
    "pyrup",
]


def _get_pyramid_gaussian_kernel() -> torch.Tensor:
    """Return a pre-computed gaussian kernel."""
    return (
        torch.tensor(
            [
                [
                    [1.0, 4.0, 6.0, 4.0, 1.0],
                    [4.0, 16.0, 24.0, 16.0, 4.0],
                    [6.0, 24.0, 36.0, 24.0, 6.0],
                    [4.0, 16.0, 24.0, 16.0, 4.0],
                    [1.0, 4.0, 6.0, 4.0, 1.0],
                ]
            ]
        )
        / 256.0
    )


class PyrDown(nn.Module):
    r"""Blur images with the 5x5 pyramid kernel and downsample them by a factor (module form).

    Build ``down = PyrDown()``, optionally with ``border_type``, ``align_corners`` and ``factor``, then
    ``out = down(input)`` blurs ``(B, C, H, W)`` images and resizes them to ``floor(H / factor)`` by
    ``floor(W / factor)``, as :func:`pyrdown` does. For the step up, use :class:`PyrUp`; for every level at once,
    :func:`build_pyramid`.

    Convention:
        - See the convention block of :func:`~kornia.geometry.transform.pyrdown`.

    Args:
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.
        factor: the downsampling factor

    Return:
        the downsampled torch.Tensor.

    Shape:
        - Input: ``(B, C, H, W)``
        - Output: ``(B, C, H // 2, W // 2)`` at the default ``factor=2.0``

    See Also:
        - :func:`pyrdown`: the one-call form.
        - :class:`PyrUp`: the module that goes one level up.

    Examples:
        >>> input = torch.rand(1, 2, 4, 4)
        >>> output = PyrDown()(input)  # 1x2x2x2

    """

    def __init__(self, border_type: str = "reflect", align_corners: bool = False, factor: float = 2.0) -> None:
        super().__init__()
        self.border_type: str = border_type
        self.align_corners: bool = align_corners
        self.factor: float = factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Build the next lower-resolution Gaussian pyramid level.

        The input is smoothed before resizing so high-frequency content is not
        directly subsampled. This is the standard pyramid-down step used by
        many multi-scale image algorithms.

        Args:
            input: Image tensor with shape ``(B, C, H, W)``, where
                :math:`B` is batch size, :math:`C` is channels, :math:`H` is
                height, and :math:`W` is width.

        Returns:
            Downsampled tensor from :func:`pyrdown`. The channel and batch
            dimensions are preserved, while spatial dimensions are reduced
            according to ``self.factor``.
        """
        return pyrdown(input, self.border_type, self.align_corners, self.factor)


class PyrUp(nn.Module):
    r"""Upsample images to twice their size and blur them with the 5x5 pyramid kernel (module form).

    Build ``up = PyrUp()``, optionally with ``border_type`` and ``align_corners``, then ``out = up(input)`` resizes
    ``(B, C, H, W)`` images to ``(B, C, H * 2, W * 2)`` and blurs them, as :func:`pyrup` does. For the step down, use
    :class:`PyrDown`.

    Convention:
        - See the convention block of :func:`~kornia.geometry.transform.pyrup`.

    Args:
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.

    Return:
        the upsampled torch.Tensor.

    Shape:
        - Input: ``(B, C, H, W)``
        - Output: ``(B, C, H * 2, W * 2)``

    See Also:
        - :func:`pyrup`: the one-call form.
        - :class:`PyrDown`: the module that goes one level down.

    Examples:
        >>> input = torch.rand(1, 2, 4, 4)
        >>> output = PyrUp()(input)  # 1x2x8x8

    """

    def __init__(self, border_type: str = "reflect", align_corners: bool = False) -> None:
        super().__init__()
        self.border_type: str = border_type
        self.align_corners: bool = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Build the next higher-resolution Gaussian pyramid level.

        The operation upsamples the image and applies pyramid smoothing so the
        enlarged result is spatially consistent with Gaussian pyramid
        reconstruction.

        Args:
            input: Image tensor with shape ``(B, C, H, W)``, where
                :math:`B` is batch size, :math:`C` is channels, :math:`H` is
                height, and :math:`W` is width.

        Returns:
            Upsampled tensor from :func:`pyrup`, typically with doubled height
            and width, preserving batch and channel dimensions.
        """
        return pyrup(input, self.border_type, self.align_corners)


class ScalePyramid(nn.Module):
    r"""Build octaves of progressively blurred images, with their sigmas, for scale-space detection.

    Build ``sp = ScalePyramid(n_levels, init_sigma, min_size)``, then ``pyr, sigmas, pixel_dists = sp(input)`` takes
    ``(B, C, H, W)`` images and returns three lists with one entry per octave. ``pyr[o]`` stacks
    ``L = n_levels + extra_levels`` levels as ``(B, C, L, H_o, W_o)``, each blurred ``2 ** (1 / n_levels)`` times more
    than the previous, and ``sigmas[o]`` and ``pixel_dists[o]`` are ``(B, L)``. Each octave after the
    first starts from level ``n_levels`` of the previous one (for ``extra_levels >= 1``,
    `#6400 <https://github.com/kornia/kornia/issues/6400>`_), resized to half its size, and
    octaves are added while that halved size stays above ``min_size`` on its shorter side. It is the pyramid
    :class:`~kornia.feature.ScaleSpaceDetector` searches; for a plain list of halved images, use :func:`build_pyramid`.

    Convention:
        - input: ``(B, C, H, W)``
        - ``forward`` returns ``(pyr, sigmas, pixel_dists)``, one entry per octave: ``pyr[o]`` is
          ``(B, C, L, H_o, W_o)`` with ``L = n_levels + extra_levels``, and ``sigmas[o]`` and
          ``pixel_dists[o]`` are ``(B, L)`` nominal values (the targeted blur, not a measurement)
        - ``sigmas`` is octave-relative: the blur in input pixels is ``sigmas[o] * pixel_dists[o]``
        - an ``init_sigma`` below the assumed input blur (``0.5``, or ``1.0`` with
          ``double_image=True``) leaves the first level at that blur, and the octave is built and
          labelled from it, as if ``init_sigma`` were the assumed input blur
        - the internal resizes use ``align_corners=True``; there is no ``align_corners`` parameter. They keep the
          first and last pixel centres, so pixel :math:`(u, v)` of octave ``o`` is input pixel
          :math:`(u (W - 1) / (W_o - 1), v (H - 1) / (H_o - 1))`; ``pixel_dists`` is the nominal spacing
          :math:`2^o` (:math:`2^{o - 1}` with ``double_image=True``), not this map

    Args:
        n_levels: number of the levels in octave.
        init_sigma: initial blur level.
        min_size: the minimum size of the octave in pixels: an octave after the first is built only when its shorter
          side is larger than ``min_size``.
        double_image: add 2x upscaled image as 1st level of pyramid. OpenCV SIFT does this.
        extra_levels: number of levels per octave beyond ``n_levels``; the default 3 is what DoG non-maximum
          suppression needs. Keep it at least 1: the next octave starts from level ``n_levels``, which
          ``extra_levels=0`` leaves out of the octave, and then starts from the least blurred level instead
          (`#6400 <https://github.com/kornia/kornia/issues/6400>`_).

    Returns:
        1st output: images, the levels of each octave.
        2nd output: sigmas, the nominal blur of each level in the octave's own pixels.
        3rd output: pixel_dists, the nominal pixel spacing of each level relative to the input.

    Shape:
        - Input: ``(B, C, H, W)``
        - Output 1st: ``[(B, C, L, H, W), (B, C, L, H // 2, W // 2), ...]``, starting from ``(B, C, L, H * 2, W * 2)``
          with ``double_image=True``
        - Output 2nd: ``[(B, L), (B, L), (B, L), ...]``
        - Output 3rd: ``[(B, L), (B, L), (B, L), ...]``

    See Also:
        - :func:`build_pyramid`: a list of halved images, one blur per level, without sigmas.
        - :func:`build_laplacian_pyramid`: band-pass levels for reconstruction rather than detection.
        - :class:`~kornia.feature.ScaleSpaceDetector`: detects keypoints in this pyramid.

    Examples:
        >>> input = torch.rand(2, 4, 100, 100)
        >>> sp, sigmas, pds = ScalePyramid(3, 15)(input)

    """

    def __init__(
        self,
        n_levels: int = 3,
        init_sigma: float = 1.6,
        min_size: int = 15,
        double_image: bool = False,
        extra_levels: int = 3,
    ) -> None:
        super().__init__()
        # 3 extra levels are needed for DoG nms.
        self.n_levels = n_levels
        self.extra_levels = extra_levels
        self.init_sigma = init_sigma
        self.min_size = min_size
        self.border = min_size // 2 - 1
        self.sigma_step = 2 ** (1.0 / float(self.n_levels))
        self.double_image = double_image
        # Blur of the first level of every octave: an init_sigma below the assumed input blur
        # leaves the first level at the input blur.
        self._base_sigma = max(init_sigma, 1.0 if double_image else 0.5)
        self._precompute_gauss_kernels(n_levels, extra_levels, init_sigma, double_image)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_levels={self.n_levels}, "
            f"init_sigma={self.init_sigma}, "
            f"min_size={self.min_size}, "
            f"extra_levels={self.extra_levels}, "
            f"border={self.border}, "
            f"sigma_step={self.sigma_step}, "
            f"double_image={self.double_image})"
        )

    @staticmethod
    def _make_gaussian_kernel1d(sigma: float, ksize: int) -> torch.Tensor:
        """Normalized 1D Gaussian kernel as a float32 tensor."""
        # MPS has no float64: compute on CPU, then move the rounded kernel to the default device.
        x = torch.arange(ksize, device="cpu", dtype=torch.float64) - ksize // 2
        kernel = torch.exp(-0.5 * x**2 / sigma**2)
        return (kernel / kernel.sum()).float().to(torch.empty(0, dtype=torch.float32).device)

    def _precompute_gauss_kernels(
        self, n_levels: int, extra_levels: int, init_sigma: float, double_image: bool
    ) -> None:
        """Register precomputed 1D Gaussian kernels as buffers so they move with .to()."""
        # Initial blur: from camera sigma (0.5 / 1.0) up to init_sigma
        cur_sigma_init = 1.0 if double_image else 0.5
        if init_sigma > cur_sigma_init:
            sigma = max(math.sqrt(init_sigma**2 - cur_sigma_init**2), 0.01)
            ksize = self.get_kernel_size(sigma)
            self.register_buffer("_gk_init", self._make_gaussian_kernel1d(sigma, ksize))
        else:
            self.register_buffer("_gk_init", None)

        # Level-to-level kernels inside an octave.
        # cur_sigma_oct resets to the first level's blur at the start of every octave, so
        # these delta_sigma values are the SAME for every octave — precompute once.
        cur_s = self._base_sigma
        for lvl in range(n_levels + extra_levels - 1):
            delta = cur_s * math.sqrt(self.sigma_step**2 - 1.0)
            ksize = self.get_kernel_size(delta)
            self.register_buffer(f"_gk_{lvl}", self._make_gaussian_kernel1d(delta, ksize))
            cur_s *= self.sigma_step

    def _blur_fast(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Separable Gaussian blur with a precomputed 1D kernel — no validation overhead."""
        ksize = kernel.shape[0]
        pad = ksize // 2
        _B, C, _H, _W = x.shape
        k = kernel.to(device=x.device, dtype=x.dtype)
        if _gaussian_blur2d_cpu_eligible(x) and not (torch.is_grad_enabled() and k.requires_grad):
            return _gaussian_blur2d_cpu(x, k.view(1, -1), k.view(1, -1), "reflect")
        # Depthwise separable: same kernel applied independently to every channel.
        k_h = k.view(1, 1, 1, ksize).expand(C, 1, 1, ksize).contiguous()
        tmp = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), k_h, groups=C)
        k_v = k.view(1, 1, ksize, 1).expand(C, 1, ksize, 1).contiguous()
        return F.conv2d(F.pad(tmp, (0, 0, pad, pad), mode="reflect"), k_v, groups=C)

    def get_kernel_size(self, sigma: float) -> int:
        """Choose an odd Gaussian kernel size for a given blur sigma.

        Args:
            sigma: Gaussian standard deviation measured in pixels.

        Returns:
            Odd integer kernel size large enough to cover the effective support
            used by this scale-pyramid implementation.
        """
        ksize = int(2.0 * 4.0 * sigma + 1.0)

        #  matches OpenCV, but may cause padding problem for small images
        #  PyTorch does not allow to F.pad more than original size.
        #  Therefore there is a hack in forward function

        if ksize % 2 == 0:
            ksize += 1
        return ksize

    def get_first_level(self, input: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        """Create the first image level and its scale metadata.

        The method optionally doubles image resolution, then applies initial
        Gaussian blur so the first level reaches ``self.init_sigma``. An
        ``init_sigma`` below the assumed input blur leaves the level unblurred.

        Args:
            input: Image tensor with shape ``(B, C, H, W)``.

        Returns:
            Tuple ``(cur_level, cur_sigma, pixel_distance)``. ``cur_level`` is
            the first image level, ``cur_sigma`` is its effective blur sigma,
            and ``pixel_distance`` is the level's nominal pixel spacing in input
            pixels (``0.5`` when doubled). It is not the coordinate map: see the
            class Convention for where a level pixel lies in the input.
        """
        pixel_distance = 1.0
        cur_sigma = 0.5
        # Same as in OpenCV up to interpolation difference
        if self.double_image:
            x = F.interpolate(input, scale_factor=2.0, mode="bilinear", align_corners=True)
            pixel_distance = 0.5
            cur_sigma *= 2.0
        else:
            x = input

        if self.init_sigma > cur_sigma:
            sigma = max(math.sqrt(self.init_sigma**2 - cur_sigma**2), 0.01)
            ksize = self.get_kernel_size(sigma)
            min_dim = min(x.size(2), x.size(3))
            if self._gk_init is not None and ksize <= min_dim:
                cur_level = self._blur_fast(x, self._gk_init)
            else:
                ksize = min(ksize, min_dim if min_dim % 2 == 1 else min_dim - 1)
                cur_level = gaussian_blur2d(x, (ksize, ksize), (sigma, sigma))
            cur_sigma = self.init_sigma
        else:
            cur_level = x
        return cur_level, cur_sigma, pixel_distance

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Construct a Gaussian scale pyramid over several octaves.

        Each octave stores multiple progressively blurred versions of the same
        image resolution. The next octave starts from a downsampled image, so
        feature detectors can search both spatial location and scale.

        Args:
            x: Input image tensor with shape ``(B, C, H, W)``, where
                :math:`B` is batch size, :math:`C` is channels, :math:`H` is
                height, and :math:`W` is width.

        Returns:
            Three lists with one entry per octave. ``pyr`` contains stacked
            image levels, ``sigmas`` contains the octave-relative nominal blur
            sigma for each level (multiply by the matching ``pixel_dists``
            entry for the nominal absolute blur in original-image pixels), and
            ``pixel_dists`` contains the nominal pixel spacing of each level
            relative to the original image. It is not the coordinate map: see
            the class Convention for where an octave pixel lies in the input.
        """
        bs, _, _, _ = x.size()
        cur_level, cur_sigma, pixel_distance = self.get_first_level(x)

        sigmas = [torch.full((bs, self.n_levels + self.extra_levels), cur_sigma, device=x.device, dtype=x.dtype)]
        pixel_dists = [
            torch.full((bs, self.n_levels + self.extra_levels), pixel_distance, device=x.device, dtype=x.dtype)
        ]
        pyr = [[cur_level]]

        while True:
            # Build octave levels incrementally: each level is a Gaussian blur of
            # the previous one.  This matches VLFeat's level-to-level construction
            # and avoids discrete truncation artefacts in DoG differences.
            cur_sigma_oct = self._base_sigma  # sigma of pyr[-1][0] in current octave pixels
            for level_idx in range(1, self.n_levels + self.extra_levels):
                kernel = getattr(self, f"_gk_{level_idx - 1}")
                min_dim = min(pyr[-1][-1].size(2), pyr[-1][-1].size(3))
                if kernel.shape[0] <= min_dim:
                    new_level = self._blur_fast(pyr[-1][-1], kernel)
                else:
                    # Tiny image: clamp kernel size and fall back to the validated path.
                    delta_sigma = cur_sigma_oct * math.sqrt(self.sigma_step**2 - 1.0)
                    ksize = min_dim if min_dim % 2 == 1 else min_dim - 1
                    new_level = gaussian_blur2d(pyr[-1][-1], (ksize, ksize), (delta_sigma, delta_sigma))
                cur_sigma_oct *= self.sigma_step
                pyr[-1].append(new_level)
                sigmas[-1][:, level_idx] = cur_sigma_oct
                pixel_dists[-1][:, level_idx] = pixel_distance

            # Seed next octave: bilinear 2x downscale of the (-extra_levels) level.
            # Bilinear resampling is more accurate than integer-stride decimation.
            _pyr = pyr[-1][-self.extra_levels]
            H, W = _pyr.shape[2], _pyr.shape[3]
            if min(H // 2, W // 2) <= self.min_size:
                break
            H_new, W_new = H // 2, W // 2
            nextOctaveFirstLevel = F.interpolate(_pyr, size=(H_new, W_new), mode="bilinear", align_corners=True)
            pixel_distance *= 2.0
            pyr.append([nextOctaveFirstLevel])
            sigmas.append(
                torch.full((bs, self.n_levels + self.extra_levels), self._base_sigma, device=x.device, dtype=x.dtype)
            )
            pixel_dists.append(
                torch.full((bs, self.n_levels + self.extra_levels), pixel_distance, device=x.device, dtype=x.dtype)
            )

        output_pyr = [torch.stack(i, 2) for i in pyr]
        return output_pyr, sigmas, pixel_dists


def pyrdown(
    input: torch.Tensor, border_type: str = "reflect", align_corners: bool = False, factor: float = 2.0
) -> torch.Tensor:
    r"""Blur images with the 5x5 pyramid kernel and downsample them by a factor, one pyramid level down.

    ``out = pyrdown(input)`` takes images ``(B, C, H, W)``, blurs them with the 5x5 binomial kernel and resizes them
    bilinearly to ``(B, C, floor(H / factor), floor(W / factor))``, half the size at the default ``factor=2.0``. It
    resamples bilinearly where OpenCV's ``pyrDown`` keeps every second pixel, and rounds an odd side down where OpenCV
    rounds it up. For the step up, use :func:`pyrup`; for every level at once, :func:`build_pyramid`; to keep
    every second pixel of a blurred map, :func:`~kornia.filters.blur_pool2d`.

    .. image:: _static/img/pyrdown.png

    Convention:
        - input: ``(B, C, H, W)``
        - output spatial size is ``floor(side / factor)`` per dimension
          (:math:`5 \times 5 \to 2 \times 2` at the default ``factor=2.0`` —
          note OpenCV's ``pyrDown`` rounds up instead)
        - Dynamo ONNX uses float32 reciprocal-scale arithmetic for dynamic sides, which can differ from eager by one
          pixel for some fractional factors (for example, side 33 and factor 1.1)
        - border_type: ``'reflect'`` by default (blur before downsample), which needs ``H`` and ``W`` of at least 3
        - align_corners: ``False`` by default

    Args:
        input: the torch.Tensor to be downsampled.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.
        factor: the downsampling factor

    Return:
        the downsampled torch.Tensor.

    See Also:
        - :class:`PyrDown`: the module form.
        - :func:`pyrup`: one level up, to twice the size.
        - :func:`build_pyramid`: the list of levels from repeated ``pyrdown`` calls.
        - :func:`~kornia.filters.blur_pool2d`: a zero-padded binomial blur sampled at a stride, from pixel 0;
          ``blur_pool2d(x, 5, 2)`` blurs with this kernel and returns ``ceil(H / 2)`` rows.

    Examples:
        >>> input = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        >>> pyrdown(input, align_corners=True)
        tensor([[[[ 3.7500,  5.2500],
                  [ 9.7500, 11.2500]]]])

    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])

    kernel: torch.Tensor = _get_pyramid_gaussian_kernel()
    # blur image
    x_blur: torch.Tensor = filter2d(input, kernel, border_type)

    # Keep the output size dynamic in ONNX while preserving eager's floor rule.
    _, _, height, width = input.shape
    if isinstance(height, torch.Tensor):
        # The TorchScript tracer represents sizes as tensors; float64 avoids
        # rounding a fractional quotient before truncating it.
        divisor = torch.tensor([factor], dtype=torch.float64)
        size = [torch.floor(side.reshape(1).double() / divisor).long()[0] for side in (height, width)]
    elif isinstance(height, torch.SymInt) and torch.onnx.is_in_onnx_export():
        # Dynamo ONNX lowering supports dynamic SymInt sizes through reciprocal
        # scale. Its float32 arithmetic may differ from eager for some factors.
        size = [torch.sym_int(side * (1.0 / factor)) for side in (height, width)]
    else:
        size = [int(float(height) / factor), int(float(width) / factor)]
    out: torch.Tensor = F.interpolate(x_blur, size=size, mode="bilinear", align_corners=align_corners)
    return out


def pyrup(input: torch.Tensor, border_type: str = "reflect", align_corners: bool = False) -> torch.Tensor:
    r"""Upsample images to twice their size and blur them with the 5x5 pyramid kernel, one pyramid level up.

    ``out = pyrup(input)`` takes images ``(B, C, H, W)``, resizes them bilinearly to ``(B, C, H * 2, W * 2)`` and then
    blurs them with the 5x5 binomial kernel; it has no ``factor``. With ``pyrdown`` it builds the levels of
    :func:`build_laplacian_pyramid`. For the step down, use :func:`pyrdown`; to change the size without blurring,
    :func:`~kornia.geometry.transform.resize`.

    .. image:: _static/img/pyrup.png

    Convention:
        - input: ``(B, C, H, W)``
        - output spatial size is exactly doubled per dimension (no ``factor``
          parameter, unlike :func:`pyrdown`; :math:`3 \times 3 \to 6 \times 6`)
        - border_type: ``'reflect'`` by default (blur after upsample)
        - align_corners: ``False`` by default

    Args:
        input: the torch.Tensor to be upsampled.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.

    Return:
        the upsampled torch.Tensor.

    See Also:
        - :class:`PyrUp`: the module form.
        - :func:`pyrdown`: one level down, by any factor.
        - :func:`build_laplacian_pyramid`: subtracts ``pyrup`` of each Gaussian level from the next finer one.

    Examples:
        >>> input = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
        >>> pyrup(input, align_corners=True)
        tensor([[[[0.7500, 0.8750, 1.1250, 1.2500],
                  [1.0000, 1.1250, 1.3750, 1.5000],
                  [1.5000, 1.6250, 1.8750, 2.0000],
                  [1.7500, 1.8750, 2.1250, 2.2500]]]])

    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])

    kernel: torch.Tensor = _get_pyramid_gaussian_kernel()
    # upsample torch.Tensor
    _, _, height, width = input.shape
    # TODO: use kornia.geometry.resize/rescale
    x_up: torch.Tensor = F.interpolate(
        input, size=(height * 2, width * 2), mode="bilinear", align_corners=align_corners
    )

    # blurs upsampled torch.Tensor
    x_blur: torch.Tensor = filter2d(x_up, kernel, border_type)
    return x_blur


def build_pyramid(
    input: torch.Tensor, max_level: int, border_type: str = "reflect", align_corners: bool = False
) -> list[torch.Tensor]:
    r"""Build a Gaussian pyramid: the image and its successively halved levels, as a list.

    ``levels = build_pyramid(input, max_level)`` takes images ``(B, C, H, W)`` and returns a list of ``max_level``
    tensors: the input itself, then each previous level through :func:`pyrdown`, ``(B, C, H // 2, W // 2)`` and so on.
    With ``max_level > 1`` and the default ``'reflect'`` border every level but the last needs 3 pixels per side, so
    ``H`` and ``W`` must be at least ``3 * 2 ** (max_level - 2)``. For band-pass levels, use
    :func:`build_laplacian_pyramid`; for octaves of several blurs each, with their sigmas, :class:`ScalePyramid`.

    .. image:: _static/img/build_pyramid.png

    Convention:
        - input: ``(B, C, H, W)``
        - ``max_level`` is the total number of levels returned (including the
          unchanged original as level 0), not a 0-based index of the last level
        - each subsequent level is produced by :func:`pyrdown` (``floor(side / 2)``
          per dimension)

    Args:
        input : the torch.Tensor to be used to construct the pyramid.
        max_level: a positive integer giving the number of pyramid levels to return, including the
          unchanged original image as level 0.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.

    Shape:
        - Input: ``(B, C, H, W)``
        - Output ``[(B, C, H, W), (B, C, H // 2, W // 2), ...]``

    See Also:
        - :func:`build_laplacian_pyramid`: the differences of these levels, ending with the coarsest one.
        - :class:`ScalePyramid`: octaves of several blurred levels each, with their sigmas, for keypoint detection.
        - :func:`pyrdown`: one level of this pyramid.

    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    KORNIA_CHECK(
        isinstance(max_level, int) and not isinstance(max_level, bool) and max_level > 0,
        lambda: f"Invalid max_level, it must be a positive integer. Got: {max_level}",
    )

    # create empty list and append the original image
    pyramid: list[torch.Tensor] = []
    pyramid.append(input)

    # iterate and downsample
    for _ in range(max_level - 1):
        img_curr: torch.Tensor = pyramid[-1]
        img_down: torch.Tensor = pyrdown(img_curr, border_type, align_corners)
        pyramid.append(img_down)

    return pyramid


def is_powerof_two(x: int) -> bool:
    r"""Check whether an integer is a positive power of two.

    ``is_powerof_two(x)`` returns ``True`` for 1, 2, 4, 8, ... and ``False`` for every other integer, ``0`` and
    negative numbers included. :func:`build_laplacian_pyramid` uses it to decide whether to pad an image.

    Args:
        x: the integer to check.

    Returns:
        ``True`` if ``x`` is a positive power of two, ``False`` otherwise (including for ``0``).

    See Also:
        - :func:`find_next_powerof_two`: the smallest power of two not below ``x``.

    Example:
        >>> is_powerof_two(64)
        True
        >>> is_powerof_two(100)
        False

    """
    return bool(x) and (not (x & (x - 1)))


def find_next_powerof_two(x: int) -> int:
    r"""Find the smallest power of two that is greater than or equal to a positive integer.

    ``find_next_powerof_two(x)`` returns ``x`` itself when it is a power of two, and the next one above otherwise.
    It is defined for positive integers only. :func:`build_laplacian_pyramid` pads each side of an image to it.

    Args:
        x: a positive integer.

    Returns:
        the smallest power of two not below ``x``.

    See Also:
        - :func:`is_powerof_two`: whether ``x`` is already a power of two.

    Example:
        >>> find_next_powerof_two(100)
        128
        >>> find_next_powerof_two(64)
        64

    """
    return 1 << (x - 1).bit_length()


def build_laplacian_pyramid(
    input: torch.Tensor, max_level: int, border_type: str = "reflect", align_corners: bool = False
) -> list[torch.Tensor]:
    r"""Build a Laplacian pyramid: band-pass differences of Gaussian levels, ending with the coarsest level.

    ``levels = build_laplacian_pyramid(input, max_level)`` takes images ``(B, C, H, W)`` and returns a list of
    ``max_level`` tensors: level ``i`` is Gaussian level ``i`` of :func:`build_pyramid` minus :func:`pyrup` of Gaussian
    level ``i + 1``, and the last is the coarsest Gaussian level. Starting from ``x = levels[-1]``,
    ``x = pyrup(x, border_type, align_corners) + levels[i]`` for ``i`` from ``max_level - 2`` down to 0, with the
    ``border_type`` and ``align_corners`` the pyramid was built with, rebuilds the input up to float rounding. The
    input is first reflect-padded at the right and bottom to the next power of two per side when neither side is a
    power of two or a side does not halve evenly ``max_level - 1`` times (see Convention); every level is then shaped
    from the padded size, and the rebuilt image is the padded one. For the Gaussian levels themselves, use
    :func:`build_pyramid`.

    See :cite:`burt1987laplacian` for more details.

    Convention:
        - input: ``(B, C, H, W)``
        - ``max_level`` is the total number of levels returned, not a 0-based
          index of the last level
        - levels ``0`` through ``max_level - 2`` are band-pass residuals: level
          ``i`` = :func:`build_pyramid` Gaussian level ``i`` minus a
          :func:`pyrup`-expanded Gaussian level ``i + 1``; only the last
          element (index ``max_level - 1``) is the unsubtracted final Gaussian
          level — level 0 is **not** the unchanged original image whenever
          ``max_level > 1``
        - the input is reflect-padded up to the next power of two (per dimension)
          when neither side is already a power of two, or when ``max_level > 1``
          and a side is not divisible by ``2 ** (max_level - 1)``. When
          padding is applied, every returned level — including level 0 — is
          shaped from the padded size
        - border_type: ``'reflect'`` by default
        - align_corners: ``False`` by default

    Args:
        input : the torch.Tensor to be used to construct the pyramid with shape ``(B, C, H, W)``.
        max_level: a positive integer giving the number of pyramid levels to return (see Convention
          above for what each level contains).
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.

    Return:
        Output: ``[(B, C, H', W'), (B, C, H' / 2, W' / 2), ...]``, where
        ``(H', W')`` equals the input ``(H, W)`` unless padding was
        applied (see Convention above), in which case it is the padded size.

    See Also:
        - :func:`build_pyramid`: the Gaussian levels these are differences of.
        - :func:`pyrup`: the upsampling step between levels, also the one that rebuilds the image.
        - :class:`ScalePyramid`: octaves of blurred levels for keypoint detection, with their sigmas.

    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    KORNIA_CHECK(
        isinstance(max_level, int) and not isinstance(max_level, bool) and max_level > 0,
        lambda: f"Invalid max_level, it must be a positive integer. Got: {max_level}",
    )

    h = input.size()[2]
    w = input.size()[3]
    require_padding = not (is_powerof_two(w) or is_powerof_two(h))
    if max_level > 1:
        # pyrdown halves each side with floor division, while pyrup doubles it.
        # Every side subtracted from an upsampled level must therefore be even.
        stride = 1 << (max_level - 1)
        require_padding = require_padding or h % stride != 0 or w % stride != 0

    if require_padding:
        # in case of arbitrary shape torch.Tensor image need to be padded.
        # Reference: https://stackoverflow.com/a/29967555
        padding = (0, find_next_powerof_two(w) - w, 0, find_next_powerof_two(h) - h)
        input = F.pad(input, padding, "reflect")

    # create gaussian pyramid
    gaussian_pyramid: list[torch.Tensor] = build_pyramid(input, max_level, border_type, align_corners)
    # create empty list
    laplacian_pyramid: list[torch.Tensor] = []

    # iterate and compute difference of adjacent layers in a gaussian pyramid
    for i in range(max_level - 1):
        img_expand: torch.Tensor = pyrup(gaussian_pyramid[i + 1], border_type, align_corners)
        laplacian: torch.Tensor = gaussian_pyramid[i] - img_expand
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid
