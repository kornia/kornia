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

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.filters import filter2d, gaussian_blur2d
from kornia.filters.filter import filter2d_separable
from kornia.filters.kernels import get_gaussian_kernel1d

__all__ = [
    "PyrDown",
    "PyrUp",
    "ScalePyramid",
    "build_laplacian_pyramid",
    "build_pyramid",
    "pyrdown",
    "pyrup",
    "upscale_double",
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
    r"""Blur a torch.Tensor and downsamples it.

    Args:
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.
        factor: the downsampling factor

    Return:
        the downsampled torch.Tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H / 2, W / 2)`

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
        return pyrdown(input, self.border_type, self.align_corners, self.factor)


class PyrUp(nn.Module):
    r"""Upsample a torch.Tensor and then blurs it.

    Args:
        borde_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.

    Return:
        the upsampled torch.Tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H * 2, W * 2)`

    Examples:
        >>> input = torch.rand(1, 2, 4, 4)
        >>> output = PyrUp()(input)  # 1x2x8x8

    """

    def __init__(self, border_type: str = "reflect", align_corners: bool = False) -> None:
        super().__init__()
        self.border_type: str = border_type
        self.align_corners: bool = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return pyrup(input, self.border_type, self.align_corners)


class ScalePyramid(nn.Module):
    r"""Create an scale pyramid of image, usually used for local feature detection.

    Images are consequently smoothed with Gaussian blur and downscaled.

    Args:
        n_levels: number of the levels in octave.
        init_sigma: initial blur level.
        min_size: the minimum size of the octave in pixels.
        double_image: add 2x upscaled image as 1st level of pyramid. OpenCV SIFT does this.

    Returns:
        1st output: images
        2nd output: sigmas (coefficients for scale conversion)
        3rd output: pixelDists (coefficients for coordinate conversion)

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output 1st: :math:`[(B, C, NL, H, W), (B, C, NL, H/2, W/2), ...]`
        - Output 2nd: :math:`[(B, NL), (B, NL), (B, NL), ...]`
        - Output 3rd: :math:`[(B, NL), (B, NL), (B, NL), ...]`

    Examples:
        >>> input = torch.rand(2, 4, 100, 100)
        >>> sp, sigmas, pds = ScalePyramid(3, 15)(input)

    """

    def __init__(
        self, n_levels: int = 3, init_sigma: float = 1.6, min_size: int = 15, double_image: bool = False
    ) -> None:
        super().__init__()
        # 3 extra levels are needed for DoG nms.
        self.n_levels = n_levels
        self.extra_levels: int = 3
        self.init_sigma = init_sigma
        self.min_size = min_size
        self.border = min_size // 2 - 1
        self.sigma_step = 2 ** (1.0 / float(self.n_levels))
        self.double_image = double_image
        # Pre-compute incremental 1-D Gaussian kernels for each level transition.
        # Sigmas/kernel-sizes are fixed for a given ScalePyramid config, so we
        # compute them once and cache them (dtype/device are handled in forward).
        self._precompute_kernels()

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

    def _precompute_kernels(self) -> None:
        """Pre-compute blur 1-D kernels for all level transitions."""
        n = self.n_levels + self.extra_levels - 1  # levels 1..n+extra
        cur_sigma = self.init_sigma

        # --- Incremental kernels (each level from the previous) ---
        self._cached_sigmas: list[float] = []
        self._cached_ksizes: list[int] = []
        for _ in range(n):
            sigma = cur_sigma * math.sqrt(self.sigma_step**2 - 1.0)
            ksize = self.get_kernel_size(sigma)
            self._cached_sigmas.append(sigma)
            self._cached_ksizes.append(ksize)
            cur_sigma *= self.sigma_step
        self._cached_kernels: list[torch.Tensor] = [
            get_gaussian_kernel1d(k, torch.tensor([[s]]))  # (1, k)
            for s, k in zip(self._cached_sigmas, self._cached_ksizes)
        ]

        # --- Cumulative kernels (each level directly from the base level L0) ---
        # Blurring L0 (at init_sigma) once with these kernels yields the same
        # result as chaining the incremental blurs above, but allows all levels
        # to be computed in a single batched grouped convolution.
        cumul_sigmas: list[float] = []
        cumul_ksizes: list[int] = []
        for i in range(1, n + 1):
            sigma_i = self.init_sigma * self.sigma_step**i
            sigma_incr = max(math.sqrt(sigma_i**2 - self.init_sigma**2), 0.01)
            cumul_sigmas.append(sigma_incr)
            cumul_ksizes.append(self.get_kernel_size(sigma_incr))
        self._cumul_sigmas: list[float] = cumul_sigmas
        self._cumul_ksizes: list[int] = cumul_ksizes
        # Raw 1-D kernels at their natural sizes (float32 on CPU)
        self._cumul_kernels: list[torch.Tensor] = [
            get_gaussian_kernel1d(k, torch.tensor([[s]])).squeeze(0)  # (1, k) → (k,)
            for s, k in zip(cumul_sigmas, cumul_ksizes)
        ]
        # Padded to the same max size so they can be stacked for a grouped conv.
        kmax: int = max(cumul_ksizes)
        self._cumul_kmax: int = kmax
        padded: list[torch.Tensor] = []
        for k in self._cumul_kernels:
            pad = (kmax - k.shape[0]) // 2
            padded.append(F.pad(k, (pad, pad)))
        stacked = torch.stack(padded)                  # (n, kmax)
        self._K_cumul_h: torch.Tensor = stacked.unsqueeze(1).unsqueeze(1)   # (n, 1, 1, kmax)
        self._K_cumul_v: torch.Tensor = stacked.unsqueeze(1).unsqueeze(-1)  # (n, 1, kmax, 1)

    def get_kernel_size(self, sigma: float) -> int:
        ksize = int(2.0 * 4.0 * sigma + 1.0)

        #  matches OpenCV, but may cause padding problem for small images
        #  PyTorch does not allow to F.pad more than original size.
        #  Therefore there is a hack in forward function

        if ksize % 2 == 0:
            ksize += 1
        return ksize

    def _compute_octave_levels_batched(self, base: torch.Tensor) -> list[torch.Tensor] | None:
        """Compute all n_levels+extra_levels-1 blurred levels from *base* in one batched grouped conv.

        Instead of sequential incremental blurs (L0→L1→L2→…), this applies all cumulative
        kernels to L0 simultaneously using a depthwise grouped convolution.  Each input channel
        is processed by a different Gaussian kernel, so all levels are produced in a single
        forward pass — better cache utilisation and fewer kernel launches than n sequential calls.

        Returns ``None`` when the image is smaller than the largest cumulative kernel (the
        incremental fallback in ``forward`` handles that case).
        """
        B, C, H, W = base.shape
        n = self.n_levels + self.extra_levels - 1
        min_dim = min(H, W)
        if min_dim <= self._cumul_kmax:
            return None  # image too small — caller falls back to incremental

        dev, dtype = base.device, base.dtype
        K_h = self._K_cumul_h.to(device=dev, dtype=dtype)  # (n, 1, 1, kmax)
        K_v = self._K_cumul_v.to(device=dev, dtype=dtype)  # (n, 1, kmax, 1)

        # Reshape to (B*C, n, H, W): treat channel and batch dims as the "batch"
        # then replicate to n copies (one per target level) for grouped depthwise conv.
        base_bc = base.view(B * C, 1, H, W)
        base_bc_n = base_bc.expand(-1, n, -1, -1).contiguous()  # (B*C, n, H, W)

        pad = self._cumul_kmax // 2
        # Horizontal pass
        out_h = F.conv2d(F.pad(base_bc_n, (pad, pad, 0, 0), mode="reflect"), K_h, groups=n)
        # Vertical pass
        out_v = F.conv2d(F.pad(out_h, (0, 0, pad, pad), mode="reflect"), K_v, groups=n)

        # Split (B*C, n, H, W) → n tensors of shape (B, C, H, W)
        out_v = out_v.view(B, C, n, H, W)
        return [base] + [out_v[:, :, i] for i in range(n)]

    def get_first_level(self, input: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        pixel_distance = 1.0
        cur_sigma = 0.5
        # Same as in OpenCV up to interpolation difference
        if self.double_image:
            x = upscale_double(input)
            pixel_distance = 0.5
            cur_sigma *= 2.0
        else:
            x = input

        if self.init_sigma > cur_sigma:
            sigma = max(math.sqrt(self.init_sigma**2 - cur_sigma**2), 0.01)
            ksize = self.get_kernel_size(sigma)
            cur_level = gaussian_blur2d(x, (ksize, ksize), (sigma, sigma))
            cur_sigma = self.init_sigma
        else:
            cur_level = x
        return cur_level, cur_sigma, pixel_distance

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        bs, _, _, _ = x.size()
        cur_level, cur_sigma, pixel_distance = self.get_first_level(x)

        sigmas = [torch.full((bs, self.n_levels + self.extra_levels), cur_sigma, device=x.device, dtype=x.dtype)]
        pixel_dists = [torch.full((bs, self.n_levels + self.extra_levels), pixel_distance, device=x.device, dtype=x.dtype)]
        pyr = [[cur_level]]
        oct_idx = 0
        while True:
            base = pyr[-1][0]

            # Fast path: compute all levels from the octave base in one batched
            # grouped convolution.  Falls back to the incremental path for very
            # small images where the largest cumulative kernel would exceed the
            # image dimension.
            octave_levels = self._compute_octave_levels_batched(base)

            if octave_levels is not None:
                pyr[-1] = octave_levels
                cur_sigma = self.init_sigma
                for level_idx in range(1, self.n_levels + self.extra_levels):
                    cur_sigma *= self.sigma_step
                    sigmas[-1][:, level_idx] = cur_sigma
                    pixel_dists[-1][:, level_idx] = pixel_distance
            else:
                # Incremental fallback for small images.
                cur_level = base
                for level_idx in range(1, self.n_levels + self.extra_levels):
                    cache_idx = level_idx - 1
                    sigma = self._cached_sigmas[cache_idx]
                    ksize = self._cached_ksizes[cache_idx]
                    kernel = self._cached_kernels[cache_idx]

                    min_dim = min(cur_level.size(2), cur_level.size(3))
                    if ksize > min_dim:
                        ksize = min_dim
                        if ksize % 2 == 0:
                            ksize -= 1
                        kernel = get_gaussian_kernel1d(ksize, torch.tensor([[sigma]]))

                    kernel = kernel.to(device=cur_level.device, dtype=cur_level.dtype)
                    cur_level = filter2d_separable(cur_level, kernel, kernel, "reflect")
                    cur_sigma *= self.sigma_step
                    pyr[-1].append(cur_level)
                    sigmas[-1][:, level_idx] = cur_sigma
                    pixel_dists[-1][:, level_idx] = pixel_distance

            _pyr = pyr[-1][-self.extra_levels]
            nextOctaveFirstLevel = _pyr[:, :, ::2, ::2]

            pixel_distance *= 2.0
            cur_sigma = self.init_sigma
            if min(nextOctaveFirstLevel.size(2), nextOctaveFirstLevel.size(3)) <= self.min_size:
                break
            pyr.append([nextOctaveFirstLevel])
            sigmas.append(torch.full((bs, self.n_levels + self.extra_levels), cur_sigma, device=x.device, dtype=x.dtype))
            pixel_dists.append(torch.full((bs, self.n_levels + self.extra_levels), pixel_distance, device=x.device, dtype=x.dtype))
            oct_idx += 1

        output_pyr = [torch.stack(i, 2) for i in pyr]

        return output_pyr, sigmas, pixel_dists


def pyrdown(
    input: torch.Tensor, border_type: str = "reflect", align_corners: bool = False, factor: float = 2.0
) -> torch.Tensor:
    r"""Blur a torch.Tensor and downsamples it.

    .. image:: _static/img/pyrdown.png

    Args:
        input: the torch.Tensor to be downsampled.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.
        factor: the downsampling factor

    Return:
        the downsampled torch.Tensor.

    Examples:
        >>> input = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        >>> pyrdown(input, align_corners=True)
        tensor([[[[ 3.7500,  5.2500],
                  [ 9.7500, 11.2500]]]])

    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])

    kernel: torch.Tensor = _get_pyramid_gaussian_kernel()
    _, _, height, width = input.shape
    # blur image
    x_blur: torch.Tensor = filter2d(input, kernel, border_type)

    # TODO: use kornia.geometry.resize/rescale
    # downsample.
    out: torch.Tensor = F.interpolate(
        x_blur,
        size=(int(float(height) / factor), int(float(width) // factor)),
        mode="bilinear",
        align_corners=align_corners,
    )
    return out


def pyrup(input: torch.Tensor, border_type: str = "reflect", align_corners: bool = False) -> torch.Tensor:
    r"""Upsample a torch.Tensor and then blurs it.

    .. image:: _static/img/pyrup.png

    Args:
        input: the torch.Tensor to be downsampled.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.

    Return:
        the downsampled torch.Tensor.

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
    r"""Construct the Gaussian pyramid for a torch.Tensor image.

    .. image:: _static/img/build_pyramid.png

    The function constructs a vector of images and builds the Gaussian pyramid
    by recursively applying pyrDown to the previously built pyramid layers.

    Args:
        input : the torch.Tensor to be used to construct the pyramid.
        max_level: 0-based index of the last (the smallest) pyramid layer.
          It must be non-negative.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output :math:`[(B, C, H, W), (B, C, H/2, W/2), ...]`

    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    KORNIA_CHECK(
        isinstance(max_level, int) or max_level < 0,
        f"Invalid max_level, it must be a positive integer. Got: {max_level}",
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
    # check if number x is a power of two
    return bool(x) and (not (x & (x - 1)))


def find_next_powerof_two(x: int) -> int:
    return 1 << (x - 1).bit_length()


def build_laplacian_pyramid(
    input: torch.Tensor, max_level: int, border_type: str = "reflect", align_corners: bool = False
) -> list[torch.Tensor]:
    r"""Construct the Laplacian pyramid for a torch.Tensor image.

    The function constructs a vector of images and builds the Laplacian pyramid
    by recursively computing the difference after applying
    pyrUp to the adjacent layer in its Gaussian pyramid.

    See :cite:`burt1987laplacian` for more details.

    Args:
        input : the torch.Tensor to be used to construct the pyramid with shape :math:`(B, C, H, W)`.
        max_level: 0-based index of the last (the smallest) pyramid layer.
          It must be non-negative.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.

    Return:
        Output: :math:`[(B, C, H, W), (B, C, H/2, W/2), ...]`

    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    KORNIA_CHECK(
        isinstance(max_level, int) or max_level < 0,
        f"Invalid max_level, it must be a positive integer. Got: {max_level}",
    )

    h = input.size()[2]
    w = input.size()[3]
    require_padding = not (is_powerof_two(w) or is_powerof_two(h))

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


def upscale_double(x: torch.Tensor) -> torch.Tensor:
    r"""Upscale image by the factor of 2, even indices maps to original indices.

    Odd indices are linearly interpolated from the even torch.ones.

    Args:
        x: input image.

    Shape:
        - Input: :math:`(*, H, W)`
        - Output :math:`(*, H, W)`

    """
    KORNIA_CHECK_IS_TENSOR(x)
    KORNIA_CHECK_SHAPE(x, ["*", "H", "W"])
    double_shape = x.shape[:-2] + (x.shape[-2] * 2, x.shape[-1] * 2)
    upscaled = torch.zeros(double_shape, device=x.device, dtype=x.dtype)
    upscaled[..., ::2, ::2] = x
    upscaled[..., ::2, 1::2][..., :-1] = (upscaled[..., ::2, ::2][..., :-1] + upscaled[..., ::2, 2::2]) / 2
    upscaled[..., ::2, -1] = upscaled[..., ::2, -2]
    upscaled[..., 1::2, :][..., :-1, :] = (upscaled[..., ::2, :][..., :-1, :] + upscaled[..., 2::2, :]) / 2
    upscaled[..., -1, :] = upscaled[..., -2, :]
    return upscaled
