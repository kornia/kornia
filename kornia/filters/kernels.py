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
from typing import Any, Optional, Union

import torch

from kornia.core._compat import deprecated
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE


def _check_kernel_size(kernel_size: tuple[int, ...] | int, min_value: int = 0, allow_even: bool = False) -> None:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,)

    fmt = "even or odd" if allow_even else "odd"
    for size in kernel_size:
        KORNIA_CHECK(
            isinstance(size, int) and (((size % 2 == 1) or allow_even) and size > min_value),
            f"Kernel size must be an {fmt} integer bigger than {min_value}. Gotcha {size} on {kernel_size}",
        )


def _check_laplacian_kernel_size(kernel_size: tuple[int, ...] | int) -> None:
    # The Laplacian centre is 1 - prod(kernel_size), so a single tap is the all-zero kernel (0 / 0 once normalized).
    if isinstance(kernel_size, int):
        sizes: tuple[int, ...] = (kernel_size,)
    else:
        sizes = kernel_size

    KORNIA_CHECK(
        any(size != 1 for size in sizes),
        f"A Laplacian kernel needs a size of at least 3 along one axis: a single tap is all zeros. Got {kernel_size}",
    )


def _unpack_2d_ks(kernel_size: tuple[int, int] | int) -> tuple[int, int]:
    if isinstance(kernel_size, int):
        ky = kx = kernel_size
    else:
        KORNIA_CHECK(len(kernel_size) == 2, "2D Kernel size should have a length of 2.")
        ky, kx = kernel_size

    ky = int(ky)
    kx = int(kx)

    return (ky, kx)


def _unpack_3d_ks(kernel_size: tuple[int, int, int] | int) -> tuple[int, int, int]:
    if isinstance(kernel_size, int):
        kz = ky = kx = kernel_size
    else:
        KORNIA_CHECK(len(kernel_size) == 3, "3D Kernel size should have a length of 3.")
        kz, ky, kx = kernel_size

    kz = int(kz)
    ky = int(ky)
    kx = int(kx)

    return (kz, ky, kx)


def normalize_kernel2d(input: torch.Tensor) -> torch.Tensor:
    r"""Normalize both derivative and smoothing kernel."""
    KORNIA_CHECK_SHAPE(input, ["*", "H", "W"])

    norm = input.abs().sum(dim=-1).sum(dim=-1)

    return input / (norm[..., None, None])


def _normalize_kernel2d_2nd_order(input: torch.Tensor) -> torch.Tensor:
    r"""Scale a stack of second order derivative kernels ``(dxx, dxy, dyy)`` to derivative estimates.

    Each kernel is divided by the magnitude of its response to the quadratic whose second derivative it
    estimates, ``x**2 / 2`` for ``dxx``, ``x * y`` for ``dxy`` and ``y**2 / 2`` for ``dyy``, so the three
    channels come out in the same units. Dividing every kernel by its own absolute sum instead, as
    :func:`normalize_kernel2d` does, scales the mixed kernel differently from the pure ones.
    """
    KORNIA_CHECK_SHAPE(input, ["3", "H", "W"])

    h, w = input.shape[-2:]
    y = torch.arange(h, device=input.device, dtype=torch.float32) - (h - 1) / 2
    x = torch.arange(w, device=input.device, dtype=torch.float32) - (w - 1) / 2
    y, x = torch.meshgrid(y, x, indexing="ij")
    quadratics = torch.stack([x * x / 2, x * y, y * y / 2])

    norm = (input * quadratics).sum(dim=(-2, -1)).abs().to(input.dtype)

    return input / norm[:, None, None]


def gaussian(
    window_size: int,
    sigma: torch.Tensor | float,
    *,
    mean: Optional[Union[torch.Tensor, float]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Compute the gaussian values based on the window and sigma values.

    Convention:
        See the Convention block on :func:`~kornia.filters.get_gaussian_kernel1d`, which validates the size and
        calls this function; ``gaussian`` does not validate it. In a floating ``dtype``, for an even ``window_size``
        the Gaussian is centred at ``mean - 0.5``: the default ``mean`` centres the kernel on the middle of the
        window, halfway between the two middle samples, and an explicit ``mean=m`` centres it at ``m - 0.5``.

    Args:
        window_size: the size which drives the filter amount.
        sigma: gaussian standard deviation. If a tensor, should be in a shape :math:`(B, 1)`.
        mean: Mean of the Gaussian function (center); see the Convention block for an even
            ``window_size``. If not provided, it defaults to ``window_size // 2``. If a tensor,
            should be in a shape :math:`(B, 1)`.
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. Dtype desired for compute.

    Returns:
        A tensor with shape :math:`(B, \text{kernel_size})`, with Gaussian values.

    .. note::
        A ``sigma`` of zero -- and any ``sigma`` too small for the window to hold a representable
        weight -- returns the unit-impulse limit of the kernel: all the mass on the sample nearest
        the mean, split evenly between the two centre taps when ``window_size`` is even. The
        gradient with respect to ``sigma`` is zero there, matching the continuous limit.

    """
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]], device=device, dtype=dtype)

    KORNIA_CHECK_IS_TENSOR(sigma)
    KORNIA_CHECK_SHAPE(sigma, ["B", "1"])
    batch_size = sigma.shape[0]

    mean = float(window_size // 2) if mean is None else mean
    if isinstance(mean, float):
        mean = torch.tensor([[mean]], device=sigma.device, dtype=sigma.dtype)

    KORNIA_CHECK_IS_TENSOR(mean)
    KORNIA_CHECK_SHAPE(mean, ["B", "1"])

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - mean).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    # Measure the squared distance from the nearest sample rather than from the mean. The shift
    # cancels in the normalization, but the nearest sample now always weighs exp(0) = 1, so a
    # small sigma (or a mean far off the grid) cannot underflow every sample to 0 and divide 0 / 0.
    dist = x.pow(2.0)
    if window_size > 0:
        dist = dist - dist.min(-1, keepdim=True)[0]

    # A zero denominator is the unit-impulse limit of the kernel: sigma == 0 exactly, or a sigma
    # whose square underflows the dtype (float16 below ~1e-4). Either way the nearest samples
    # divide 0 / 0. Repairing that after the fact leaves the division on the graph and the sigma
    # gradient still comes back NaN, so divide by a stand-in and select the impulse out of that
    # finite arm instead. The gradient is then the 0 of the continuous sigma -> 0 limit. A NaN
    # sigma is not caught by the comparison and still propagates, as it did before.
    denominator = 2 * sigma.pow(2.0)
    is_impulse = denominator == 0
    safe_denominator = torch.where(is_impulse, torch.ones_like(denominator), denominator)
    gauss = torch.exp(-dist / safe_denominator)
    gauss = torch.where(is_impulse, (dist == 0).to(dist.dtype), gauss)

    return gauss / gauss.sum(-1, keepdim=True)


def gaussian_discrete_erf(
    window_size: int,
    sigma: torch.Tensor | float,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Discrete Gaussian by interpolating the error function.

    Adapted from: https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    Args:
        window_size: the size which drives the filter amount.
        sigma: gaussian standard deviation. If a tensor, should be in a shape :math:`(B, 1)`
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. Dtype desired for compute.

    Returns:
        A tensor withshape :math:`(B, \text{kernel_size})`, with discrete Gaussian values computed by approximation of
        the error function.

    """
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]], device=device, dtype=dtype)

    KORNIA_CHECK_SHAPE(sigma, ["B", "1"])
    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    t = 0.70710678 / sigma.abs()
    # t = torch.tensor(2, device=sigma.device, dtype=sigma.dtype).sqrt() / (sigma.abs() * 2)

    gauss = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())
    gauss = gauss.clamp(min=0)

    return gauss / gauss.sum(-1, keepdim=True)


def _modified_bessel_0(x: torch.Tensor, scaled: bool = False) -> torch.Tensor:
    """Adapted from:https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py.

    Both polynomial branches are evaluated on the full tensor and merged with ``torch.where`` (instead
    of boolean-mask indexing) so that the kernel traces without data-dependent control flow.
    """
    ax = torch.abs(x)
    idx_a = ax < 3.75

    # small-argument branch (|x| < 3.75)
    x_a = torch.where(idx_a, x, torch.zeros_like(x))
    y = (x_a / 3.75) * (x_a / 3.75)
    out_a = 1.0 + y * (
        3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2))))
    )

    # large-argument branch; clamp keeps the unused lanes finite (|x| = 0 would divide by zero)
    ax_b = torch.where(idx_a, torch.full_like(ax, 3.75), ax)
    y = 3.75 / ax_b
    ans = 0.916281e-2 + y * (-0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1 + y * 0.392377e-2)))
    coef = 0.39894228 + y * (0.1328592e-1 + y * (0.225319e-2 + y * (-0.157565e-2 + y * ans)))
    out_b = coef / ax_b.sqrt() if scaled else (ax_b.exp() / ax_b.sqrt()) * coef
    if scaled:
        out_a = out_a * (-x_a.abs()).exp()

    return torch.where(idx_a, out_a, out_b)


def _modified_bessel_1(x: torch.Tensor, scaled: bool = False) -> torch.Tensor:
    """Adapted from:https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py.

    Branch-free like :func:`_modified_bessel_0`.
    """
    ax = torch.abs(x)
    idx_a = ax < 3.75

    # small-argument branch (|x| < 3.75)
    x_a = torch.where(idx_a, x, torch.zeros_like(x))
    y = (x_a / 3.75) * (x_a / 3.75)
    ans = 0.51498869 + y * (0.15084934 + y * (0.2658733e-1 + y * (0.301532e-2 + y * 0.32411e-3)))
    out_a = x_a.abs() * (0.5 + y * (0.87890594 + y * ans))

    # large-argument branch; clamp keeps the unused lanes finite (|x| = 0 would divide by zero)
    ax_b = torch.where(idx_a, torch.full_like(ax, 3.75), ax)
    y = 3.75 / ax_b
    ans = 0.2282967e-1 + y * (-0.2895312e-1 + y * (0.1787654e-1 - y * 0.420059e-2))
    ans = 0.39894228 + y * (-0.3988024e-1 + y * (-0.362018e-2 + y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans))))
    ans = ans / ax_b.sqrt() if scaled else ans * ax_b.exp() / ax_b.sqrt()
    if scaled:
        out_a = out_a * (-x_a.abs()).exp()
    out_b = torch.where(x < 0, -ans, ans)

    return torch.where(idx_a, out_a, out_b)


def _modified_bessel_i(n: int, x: torch.Tensor, scaled: bool = False, max_order: Optional[int] = None) -> torch.Tensor:
    """Adapted from: https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py."""
    KORNIA_CHECK(n >= 2, "n must be greater than 1.99")

    # I_n(0) = 0 for n >= 1. The zero lanes are computed on a safe placeholder and masked out at the
    # end instead of being compacted away, so the recurrence has no data-dependent control flow.
    is_zero_mask = torch.isclose(x, torch.tensor(0.0, device=x.device, dtype=x.dtype))
    order = n if max_order is None else max_order
    # A ~1e-7 approximation mismatch here can make finite-difference gradcheck fail exactly at the branch switch.
    use_forward = x.abs() > order * order / 4 if scaled else torch.zeros_like(x, dtype=torch.bool)
    x_nz = torch.where(is_zero_mask | use_forward, torch.ones_like(x), x)

    tox = 2.0 / x_nz.abs()

    ans = torch.zeros_like(x)
    bip = torch.zeros_like(x)
    bi = torch.ones_like(x)

    m = int(2 * (order + int(math.sqrt(40.0 * order))))
    for j in range(m, 0, -1):
        bim = torch.addcmul(bip, tox, bi, value=j)
        bip, bi = bi, bim

        # Rescale only the lanes that grow past 1e10; the other lanes pass through unchanged.
        scale_mask = bi.abs() > 1.0e10
        ans = torch.where(scale_mask, ans * 1e-10, ans)
        bi = torch.where(scale_mask, bi * 1e-10, bi)
        bip = torch.where(scale_mask, bip * 1e-10, bip)

        if j == n:
            ans = bip

    out = ans * _modified_bessel_0(x_nz, scaled=scaled) / bi
    if (n % 2) == 1:
        out = torch.where(x_nz < 0.0, -out, out)

    if scaled:
        # Upward recurrence is stable when all requested orders are small relative to sqrt(x).
        x_up = torch.where(use_forward, x.abs(), torch.full_like(x, order * order / 4))
        previous = _modified_bessel_0(x_up, scaled=True)
        current = _modified_bessel_1(x_up, scaled=True)
        for k in range(1, n):
            previous, current = current, previous - (2.0 * k / x_up) * current
        if (n % 2) == 1:
            current = torch.where(x < 0, -current, current)
        out = torch.where(use_forward, current, out)

    return torch.where(is_zero_mask, torch.zeros_like(x), out)


def gaussian_discrete(
    window_size: int,
    sigma: torch.Tensor | float,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Discrete Gaussian kernel based on the modified Bessel functions.

    Adapted from: https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    Args:
        window_size: the size which drives the filter amount.
        sigma: gaussian standard deviation. If a tensor, should be in a shape :math:`(B, 1)`
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. Dtype desired for compute.

    Returns:
        A tensor withshape :math:`(B, \text{kernel_size})`, with discrete Gaussian values computed by modified Bessel
        function.

    """
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]], device=device, dtype=dtype)

    KORNIA_CHECK_SHAPE(sigma, ["B", "1"])

    output_dtype = sigma.dtype
    if sigma.dtype in (torch.float16, torch.bfloat16):
        sigma = sigma.float()
    sigma2 = sigma * sigma
    tail = int(window_size // 2) + 1
    bessels = [
        _modified_bessel_0(sigma2, scaled=True),
        _modified_bessel_1(sigma2, scaled=True),
        *(_modified_bessel_i(k, sigma2, scaled=True, max_order=tail) for k in range(2, tail)),
    ]
    # The exp(-sigma²) factor is already included in the scaled Bessel terms.
    out = torch.cat(bessels[:0:-1] + bessels, -1)

    return (out / out.sum(-1, keepdim=True)).to(output_dtype)


def laplacian_1d(
    window_size: int, *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    r"""Return the 1D Laplacian kernel of :func:`~kornia.filters.get_laplacian_kernel1d` without checking the size.

    Convention:
        See the Convention block on :func:`~kornia.filters.get_laplacian_kernel1d`, which validates the size and
        calls this function. ``laplacian_1d`` accepts any positive size; an even one puts the negative tap at
        ``window_size // 2``, off the middle of the kernel.

    Args:
        window_size: the number of taps.
        device: tensor device desired to create the kernel
        dtype: tensor dtype desired to create the kernel

    Returns:
        1D tensor with shape :math:`(\text{window_size},)`.

    """
    # TODO: add default dtype as None when kornia relies on torch > 1.12
    filter_1d = torch.ones(window_size, device=device, dtype=dtype)
    middle = window_size // 2
    filter_1d[middle] = 1 - window_size
    return filter_1d


def get_box_kernel1d(
    kernel_size: int, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    r"""Return a 1-D box filter.

    Convention:
        - For a floating ``dtype`` every tap is ``1 / kernel_size``; an even ``kernel_size`` is accepted.
        - Known defects:

          - the kernel is a stride-0 view of a single value, so writing one tap in place changes every tap
            (`#5160 <https://github.com/kornia/kornia/issues/5160>`_).
          - an integer ``dtype`` truncates ``1 / kernel_size``, so every tap is 0 once ``kernel_size`` is above 1
            (`#5155 <https://github.com/kornia/kornia/issues/5155>`_).

    Args:
        kernel_size: the size of the kernel.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.

    Returns:
        A tensor with shape :math:`(1, \text{kernel\_size})`, filled with the value
        :math:`\frac{1}{\text{kernel\_size}}` for a floating ``dtype``.

    """
    scale = torch.tensor(1.0 / kernel_size, device=device, dtype=dtype)
    return scale.expand(1, kernel_size)


def get_box_kernel2d(
    kernel_size: tuple[int, int] | int, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    r"""Return a 2-D box filter.

    Convention:
        See the Convention block on :func:`~kornia.filters.get_box_kernel1d`; ``kernel_size`` is ``(k_y, k_x)``.

    Args:
        kernel_size: the size of the kernel, an integer or ``(k_y, k_x)``.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.

    Returns:
        A tensor with shape :math:`(1, \text{kernel\_size}[0], \text{kernel\_size}[1])`,
        filled with the value :math:`\frac{1}{\text{kernel\_size}[0] \times \text{kernel\_size}[1]}` for a
        floating ``dtype``.

    """
    ky, kx = _unpack_2d_ks(kernel_size)
    scale = torch.tensor(1.0 / (kx * ky), device=device, dtype=dtype)
    return scale.expand(1, ky, kx)


def get_binary_kernel2d(
    window_size: tuple[int, int] | int, *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    r"""Create a binary kernel to extract the patches.

    If the window size is HxW will create a (H*W)x1xHxW kernel.

    Convention:
        Channel ``i`` is one-hot at ``(i // k_x, i % k_x)`` for ``window_size=(k_y, k_x)``, in row-major order, so
        :func:`torch.nn.functional.conv2d` of an image with the kernel stacks each pixel's neighbourhood in that
        order, the layout :func:`~kornia.filters.median_blur` builds on.

    Args:
        window_size: the window size, an integer or ``(k_y, k_x)``.
        device: tensor device desired to create the kernel
        dtype: tensor dtype desired to create the kernel

    Returns:
        the kernel with shape :math:`(k_y k_x, 1, k_y, k_x)`.

    """
    # TODO: add default dtype as None when kornia relies on torch > 1.12

    ky, kx = _unpack_2d_ks(window_size)

    window_range = kx * ky

    kernel = torch.zeros((window_range, window_range), device=device, dtype=dtype)
    idx = torch.arange(window_range, device=device)
    kernel[idx, idx] += 1.0
    return kernel.view(window_range, 1, ky, kx)


def get_sobel_kernel_3x3(*, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Return a sobel kernel of 3x3."""
    return torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device, dtype=dtype)


def get_sobel_kernel_5x5_2nd_order(
    *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Return a 2nd order sobel kernel of 5x5."""
    return torch.tensor(
        [
            [1.0, 0.0, -2.0, 0.0, 1.0],
            [4.0, 0.0, -8.0, 0.0, 4.0],
            [6.0, 0.0, -12.0, 0.0, 6.0],
            [4.0, 0.0, -8.0, 0.0, 4.0],
            [1.0, 0.0, -2.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )


def _get_sobel_kernel_5x5_2nd_order_xy(
    *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Return a 2nd order sobel kernel of 5x5."""
    return torch.tensor(
        [
            [1.0, 2.0, 0.0, -2.0, -1.0],
            [2.0, 4.0, 0.0, -4.0, -2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-2.0, -4.0, 0.0, 4.0, 2.0],
            [-1.0, -2.0, 0.0, 2.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )


def get_diff_kernel_3x3(*, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Return a first order derivative kernel of 3x3."""
    return torch.tensor([[-0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [-0.0, 0.0, 0.0]], device=device, dtype=dtype)


def get_diff_kernel3d(device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Return a first order derivative kernel of 3x3x3."""
    kernel = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]],
            ],
        ],
        device=device,
        dtype=dtype,
    )
    return kernel[:, None, ...]


def get_diff_kernel3d_2nd_order(
    device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Return second order derivative kernels of 3x3x3 for ``(dxx, dyy, dzz, dxy, dyz, dxz)``."""
    kernel = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.25, 0.0, -0.25], [0.0, 0.0, 0.0], [-0.25, 0.0, 0.25]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.25, 0.0], [0.0, 0.0, 0.0], [0.0, -0.25, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, -0.25, 0.0], [0.0, 0.0, 0.0], [0.0, 0.25, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.25, 0.0, -0.25], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [-0.25, 0.0, 0.25], [0.0, 0.0, 0.0]],
            ],
        ],
        device=device,
        dtype=dtype,
    )
    return kernel[:, None, ...]


def get_sobel_kernel2d(*, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Return 1st order gradient for sobel operator.

    Convention:
        See the Convention block on :func:`~kornia.filters.get_spatial_gradient_kernel2d`; this is
        ``get_spatial_gradient_kernel2d('sobel', 1)``, of shape :math:`(2, 3, 3)`.
    """
    kernel_x = get_sobel_kernel_3x3(device=device, dtype=dtype)
    kernel_y = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])


def get_diff_kernel2d(*, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Return 1st order gradient for diff operator.

    Convention:
        See the Convention block on :func:`~kornia.filters.get_spatial_gradient_kernel2d`; this is
        ``get_spatial_gradient_kernel2d('diff', 1)``, of shape :math:`(2, 3, 3)`.
    """
    kernel_x = get_diff_kernel_3x3(device=device, dtype=dtype)
    kernel_y = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])


def get_sobel_kernel2d_2nd_order(
    *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Return 2nd order gradient for sobel operator."""
    gxx = get_sobel_kernel_5x5_2nd_order(device=device, dtype=dtype)
    gyy = gxx.transpose(0, 1)
    gxy = _get_sobel_kernel_5x5_2nd_order_xy(device=device, dtype=dtype)
    return torch.stack([gxx, gxy, gyy])


def get_diff_kernel2d_2nd_order(
    *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Return 2nd order gradient for diff operator."""
    gxx = torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]], device=device, dtype=dtype)
    gyy = gxx.transpose(0, 1)
    gxy = torch.tensor([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]], device=device, dtype=dtype)
    return torch.stack([gxx, gxy, gyy])


def get_spatial_gradient_kernel2d(
    mode: str,
    order: int,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Return kernel for 1st or 2nd order image gradients.

    Uses one of the following operators: sobel, diff.

    Convention:
        - ``order=1`` stacks :math:`(\partial_x, \partial_y)` along the first axis and ``order=2`` stacks
          :math:`(\partial_{xx}, \partial_{xy}, \partial_{yy})`. Correlated with an image, as
          :func:`~kornia.filters.filter2d` does, an ``order=1`` channel is positive where the values increase with
          the column (x) or the row (y), and an ``order=2`` channel is positive on :math:`x^2`, :math:`xy` and
          :math:`y^2` respectively.
        - The kernels are raw integer stencils, not derivative estimates. Per unit slope Sobel answers 8 and
          ``'diff'`` answers 2; per unit second derivative Sobel answers 64 on all three channels, while ``'diff'``
          answers 1 on :math:`\partial_{xx}` and :math:`\partial_{yy}` and 4 on :math:`\partial_{xy}`.
        - Known defect: ``mode`` is checked case-insensitively but used as given, so ``'Sobel'`` raises
          (`#5156 <https://github.com/kornia/kornia/issues/5156>`_).

    Args:
        mode: ``'sobel'`` or ``'diff'``.
        order: the derivative order, 1 or 2.
        device: tensor device desired to create the kernel
        dtype: tensor dtype desired to create the kernel

    Returns:
        the kernels with shape :math:`(2, 3, 3)` for ``order=1``, and :math:`(3, 5, 5)` (Sobel) or
        :math:`(3, 3, 3)` (diff) for ``order=2``.

    """
    KORNIA_CHECK(mode.lower() in {"sobel", "diff"}, f"Mode should be `sobel` or `diff`. Got {mode}")
    KORNIA_CHECK(order in {1, 2}, f"Order should be 1 or 2. Got {order}")

    if mode == "sobel" and order == 1:
        kernel: torch.Tensor = get_sobel_kernel2d(device=device, dtype=dtype)
    elif mode == "sobel" and order == 2:
        kernel = get_sobel_kernel2d_2nd_order(device=device, dtype=dtype)
    elif mode == "diff" and order == 1:
        kernel = get_diff_kernel2d(device=device, dtype=dtype)
    elif mode == "diff" and order == 2:
        kernel = get_diff_kernel2d_2nd_order(device=device, dtype=dtype)
    else:
        raise NotImplementedError(f"Not implemented for order {order} on mode {mode}")

    return kernel


def get_spatial_gradient_kernel3d(
    mode: str, order: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    r"""Return kernel for 1st or 2nd order gradients of a volume.

    Convention:
        - Only ``mode='diff'`` exists; ``'sobel'`` passes the mode check and raises ``NotImplementedError``.
        - ``order=1`` stacks :math:`(\partial_x, \partial_y, \partial_z)` and ``order=2`` stacks
          :math:`(\partial_{xx}, \partial_{yy}, \partial_{zz}, \partial_{xy}, \partial_{yz}, \partial_{xz})`, an
          order that differs from the 2d :math:`(\partial_{xx}, \partial_{xy}, \partial_{yy})`. The stack has a
          singleton second axis that :func:`~kornia.filters.get_spatial_gradient_kernel2d` lacks.
        - In a floating ``dtype`` every channel is in derivative units: it answers 1 to a unit slope or a unit
          second derivative, unlike the raw 2d stencils.
        - Known defects:

          - ``mode`` is checked case-insensitively but used as given, so ``'Diff'`` raises
            (`#5156 <https://github.com/kornia/kornia/issues/5156>`_).
          - a signed integer ``dtype`` truncates the half and quarter taps to 0, so the first-order channels and
            the mixed second-order ones are all zero (`#5155 <https://github.com/kornia/kornia/issues/5155>`_).

    Args:
        mode: ``'diff'``.
        order: the derivative order, 1 or 2.
        device: tensor device desired to create the kernel
        dtype: tensor dtype desired to create the kernel

    Returns:
        the kernels with shape :math:`(3, 1, 3, 3, 3)` for ``order=1`` and :math:`(6, 1, 3, 3, 3)` for ``order=2``.

    """
    KORNIA_CHECK(mode.lower() in {"sobel", "diff"}, f"Mode should be `sobel` or `diff`. Got {mode}")
    KORNIA_CHECK(order in {1, 2}, f"Order should be 1 or 2. Got {order}")

    if mode == "diff" and order == 1:
        kernel = get_diff_kernel3d(device=device, dtype=dtype)
    elif mode == "diff" and order == 2:
        kernel = get_diff_kernel3d_2nd_order(device=device, dtype=dtype)
    else:
        raise NotImplementedError(f"Not implemented 3d gradient kernel for order {order} on mode {mode}")

    return kernel


def get_gaussian_kernel1d(
    kernel_size: int,
    sigma: float | torch.Tensor,
    force_even: bool = False,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Return Gaussian filter coefficients.

    Convention:
        - In a floating ``dtype`` the kernel samples :math:`\exp(-n^2 / 2\sigma^2)` at the offsets :math:`n` of its
          taps from its centre, integers for an odd size and half-integers for an even one, and is normalized to
          sum 1; :ref:`Filtering <filtering-conventions>` names the matching scipy and OpenCV kernels.
          :func:`~kornia.filters.get_gaussian_erf_kernel1d` integrates the Gaussian over each pixel instead, and
          :func:`~kornia.filters.get_gaussian_discrete_kernel1d` is the discrete Gaussian.
        - ``force_even=True`` also accepts an even ``kernel_size``, and in a floating ``dtype`` the kernel is then
          symmetric about the middle of the window, ``(kernel_size - 1) / 2``.
        - A tensor ``sigma`` of shape :math:`(B, 1)` gives one kernel per row.
        - Known defects:

          - a Python ``int`` ``sigma`` raises, while the 2d and 3d builders accept integers
            (`#5157 <https://github.com/kornia/kornia/issues/5157>`_).
          - an integer ``dtype`` truncates a fractional ``sigma``, and uint8 also wraps the negative offsets, so the
            taps before the centre are wrong (`#5155 <https://github.com/kornia/kornia/issues/5155>`_).

    Args:
        kernel_size: filter size. It should be odd and positive.
        sigma: gaussian standard deviation.
        force_even: overrides requirement for odd kernel size.
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. Dtype desired for compute.

    Returns:
        gaussian filter coefficients with shape :math:`(B, \text{kernel_size})`.

    Examples:
        >>> get_gaussian_kernel1d(3, 2.5)
        tensor([[0.3243, 0.3513, 0.3243]])
        >>> get_gaussian_kernel1d(5, 1.5)
        tensor([[0.1201, 0.2339, 0.2921, 0.2339, 0.1201]])
        >>> get_gaussian_kernel1d(5, torch.tensor([[1.5], [0.7]]))
        tensor([[0.1201, 0.2339, 0.2921, 0.2339, 0.1201],
                [0.0096, 0.2054, 0.5699, 0.2054, 0.0096]])

        A ``sigma`` of zero, or one too small for the window to hold a representable weight,
        returns the unit impulse rather than a kernel of NaN:

        >>> get_gaussian_kernel1d(5, 0.0)
        tensor([[0., 0., 1., 0., 0.]])
        >>> get_gaussian_kernel1d(4, 0.0, force_even=True)
        tensor([[0.0000, 0.5000, 0.5000, 0.0000]])

    """
    _check_kernel_size(kernel_size, allow_even=force_even)

    return gaussian(kernel_size, sigma, device=device, dtype=dtype)


def get_gaussian_discrete_kernel1d(
    kernel_size: int,
    sigma: float | torch.Tensor,
    force_even: bool = False,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Return Gaussian filter coefficients based on the modified Bessel functions.

    Adapted from: https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py.

    Convention:
        - See the Convention block on :func:`~kornia.filters.get_gaussian_kernel1d`. In a floating ``dtype`` this
          kernel is Lindeberg's discrete Gaussian :math:`e^{-\sigma^2} I_{|n|}(\sigma^2)`, with :math:`I_n` the
          modified Bessel function of the first kind, normalized over the window: the smoothing kernel of discrete
          scale space. A float16 or bfloat16 kernel is computed in float32 and rounded to its ``dtype``.
        - Known defect: the tap count is not always ``kernel_size``. ``kernel_size=1`` gives 3 taps, and an even
          size with ``force_even=True`` gives one more than asked
          (`#5158 <https://github.com/kornia/kornia/issues/5158>`_).

    Args:
        kernel_size: filter size. It should be odd and positive.
        sigma: gaussian standard deviation. If a tensor, should be in a shape :math:`(B, 1)`
        force_even: overrides requirement for odd kernel size.
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. Dtype desired for compute.

    Returns:
        1D tensor with gaussian filter coefficients. With shape :math:`(B, \text{kernel_size})` for an odd
        ``kernel_size`` greater than 1 (see the known defects).

    Examples:
        >>> get_gaussian_discrete_kernel1d(3, 2.5)
        tensor([[0.3235, 0.3531, 0.3235]])
        >>> get_gaussian_discrete_kernel1d(5, 1.5)
        tensor([[0.1096, 0.2323, 0.3161, 0.2323, 0.1096]])
        >>> get_gaussian_discrete_kernel1d(5, torch.tensor([[1.5],[2.4]]))
        tensor([[0.1096, 0.2323, 0.3161, 0.2323, 0.1096],
                [0.1635, 0.2170, 0.2389, 0.2170, 0.1635]])

    """
    _check_kernel_size(kernel_size, allow_even=force_even)

    return gaussian_discrete(kernel_size, sigma, device=device, dtype=dtype)


def get_gaussian_erf_kernel1d(
    kernel_size: int,
    sigma: float | torch.Tensor,
    force_even: bool = False,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Return Gaussian filter coefficients by interpolating the error function.

    Adapted from: https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py.

    Convention:
        - See the Convention block on :func:`~kornia.filters.get_gaussian_kernel1d`. In a floating ``dtype`` this
          kernel integrates the Gaussian over each pixel, :math:`\Phi((n + 1/2) / \sigma) - \Phi((n - 1/2) / \sigma)`
          with :math:`\Phi` the normal CDF, so it blurs more than the sampled kernel: for :math:`\sigma` of
          about 1 or more, on a window wide enough for the tails, its variance is :math:`\sigma^2 + 1/12`.
        - Known defect: with ``force_even=True`` an even kernel is centred on tap ``kernel_size // 2`` instead of
          the middle of the window, so it is not symmetric (`#5158 <https://github.com/kornia/kornia/issues/5158>`_).

    Args:
        kernel_size: filter size. It should be odd and positive.
        sigma: gaussian standard deviation. If a tensor, should be in a shape :math:`(B, 1)`
        force_even: overrides requirement for odd kernel size.
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. Dtype desired for compute.

    Returns:
        1D tensor with gaussian filter coefficients. Shape :math:`(B, \text{kernel_size})`

    Examples:
        >>> get_gaussian_erf_kernel1d(3, 2.0)
        tensor([[0.3195, 0.3611, 0.3195]])
        >>> get_gaussian_erf_kernel1d(5, 1.5)
        tensor([[0.1226, 0.2331, 0.2887, 0.2331, 0.1226]])
        >>> get_gaussian_erf_kernel1d(5, torch.tensor([[1.5], [2.1]]))
        tensor([[0.1226, 0.2331, 0.2887, 0.2331, 0.1226],
                [0.1574, 0.2198, 0.2456, 0.2198, 0.1574]])

    """
    _check_kernel_size(kernel_size, allow_even=force_even)

    return gaussian_discrete_erf(kernel_size, sigma, device=device, dtype=dtype)


def get_gaussian_kernel2d(
    kernel_size: tuple[int, int] | int,
    sigma: tuple[float, float] | torch.Tensor,
    force_even: bool = False,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Return Gaussian filter matrix coefficients.

    Convention:
        See the Convention block on :func:`~kornia.filters.get_gaussian_kernel1d`. ``kernel_size`` is
        ``(k_y, k_x)`` and ``sigma`` is :math:`(\sigma_y, \sigma_x)`, y first as in the output
        :math:`(B, k_y, k_x)`, which is the outer product of the two 1d kernels. A tensor ``sigma`` of shape
        :math:`(B, 2)` gives one kernel per row.

    Args:
        kernel_size: filter sizes in the y and x direction. Sizes should be odd and positive.
        sigma: gaussian standard deviation in the y and x.
        force_even: overrides requirement for odd kernel size.
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. Dtype desired for compute.

    Returns:
        2D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(B, \text{kernel_size}_y, \text{kernel_size}_x)`

    Examples:
        >>> get_gaussian_kernel2d((5, 5), (1.5, 1.5))
        tensor([[[0.0144, 0.0281, 0.0351, 0.0281, 0.0144],
                 [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                 [0.0351, 0.0683, 0.0853, 0.0683, 0.0351],
                 [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                 [0.0144, 0.0281, 0.0351, 0.0281, 0.0144]]])
        >>> get_gaussian_kernel2d((3, 5), (1.5, 1.5))
        tensor([[[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                 [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                 [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]]])
        >>> get_gaussian_kernel2d((5, 5), torch.tensor([[1.5, 1.5]]))
        tensor([[[0.0144, 0.0281, 0.0351, 0.0281, 0.0144],
                 [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                 [0.0351, 0.0683, 0.0853, 0.0683, 0.0351],
                 [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                 [0.0144, 0.0281, 0.0351, 0.0281, 0.0144]]])

    """
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], device=device, dtype=dtype)

    KORNIA_CHECK_IS_TENSOR(sigma)
    KORNIA_CHECK_SHAPE(sigma, ["B", "2"])

    ksize_y, ksize_x = _unpack_2d_ks(kernel_size)
    sigma_y, sigma_x = sigma[:, 0, None], sigma[:, 1, None]

    kernel_y = get_gaussian_kernel1d(ksize_y, sigma_y, force_even, device=device, dtype=dtype)[..., None]
    kernel_x = get_gaussian_kernel1d(ksize_x, sigma_x, force_even, device=device, dtype=dtype)[..., None]

    return kernel_y * kernel_x.view(-1, 1, ksize_x)


def get_gaussian_kernel3d(
    kernel_size: tuple[int, int, int] | int,
    sigma: tuple[float, float, float] | torch.Tensor,
    force_even: bool = False,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Return Gaussian filter matrix coefficients.

    Convention:
        See the Convention block on :func:`~kornia.filters.get_gaussian_kernel1d`. ``kernel_size`` is
        ``(k_z, k_y, k_x)`` and ``sigma`` is :math:`(\sigma_z, \sigma_y, \sigma_x)`, z first as in the output
        :math:`(B, k_z, k_y, k_x)`, which is the outer product of the three 1d kernels. A tensor ``sigma`` of shape
        :math:`(B, 3)` gives one kernel per row.

    Args:
        kernel_size: filter sizes in the z, y and x direction. Sizes should be odd and positive.
        sigma: gaussian standard deviation in the z, y and x direction.
        force_even: overrides requirement for odd kernel size.
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. Dtype desired for compute.

    Returns:
        3D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(B, \text{kernel_size}_z, \text{kernel_size}_y, \text{kernel_size}_x)`

    Examples:
        >>> get_gaussian_kernel3d((3, 3, 3), (1.5, 1.5, 1.5))
        tensor([[[[0.0292, 0.0364, 0.0292],
                  [0.0364, 0.0455, 0.0364],
                  [0.0292, 0.0364, 0.0292]],
        <BLANKLINE>
                 [[0.0364, 0.0455, 0.0364],
                  [0.0455, 0.0568, 0.0455],
                  [0.0364, 0.0455, 0.0364]],
        <BLANKLINE>
                 [[0.0292, 0.0364, 0.0292],
                  [0.0364, 0.0455, 0.0364],
                  [0.0292, 0.0364, 0.0292]]]])
        >>> torch.allclose(get_gaussian_kernel3d((3, 3, 3), (1.5, 1.5, 1.5)).sum(), torch.tensor(1.0))
        True
        >>> get_gaussian_kernel3d((3, 3, 3), (1.5, 1.5, 1.5)).shape
        torch.Size([1, 3, 3, 3])
        >>> get_gaussian_kernel3d((3, 7, 5), torch.tensor([[1.5, 1.5, 1.5]])).shape
        torch.Size([1, 3, 7, 5])

    """
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], device=device, dtype=dtype)

    KORNIA_CHECK_IS_TENSOR(sigma)
    KORNIA_CHECK_SHAPE(sigma, ["B", "3"])

    ksize_z, ksize_y, ksize_x = _unpack_3d_ks(kernel_size)
    sigma_z, sigma_y, sigma_x = sigma[:, 0, None], sigma[:, 1, None], sigma[:, 2, None]

    kernel_z = get_gaussian_kernel1d(ksize_z, sigma_z, force_even, device=device, dtype=dtype)
    kernel_y = get_gaussian_kernel1d(ksize_y, sigma_y, force_even, device=device, dtype=dtype)
    kernel_x = get_gaussian_kernel1d(ksize_x, sigma_x, force_even, device=device, dtype=dtype)

    return kernel_z.view(-1, ksize_z, 1, 1) * kernel_y.view(-1, 1, ksize_y, 1) * kernel_x.view(-1, 1, 1, ksize_x)


def get_laplacian_kernel1d(
    kernel_size: int, *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    r"""Return the coefficients of a 1D Laplacian filter.

    Convention:
        - In a floating ``dtype``, or a signed integer one wide enough to hold ``1 - kernel_size``, the kernel is
          all ones with the centre tap set to ``1 - kernel_size``, so it sums to 0. Size 3 is the second difference
          ``[1, -2, 1]``; a larger size is not a wider second difference, and size 5 answers 5 to a unit second
          derivative.
        - The negative centre makes the response positive where the values curve upwards.
        - Known defect: uint8 wraps the negative centre, to 252 for size 5
          (`#5155 <https://github.com/kornia/kornia/issues/5155>`_).

    Args:
        kernel_size: filter size. It should be odd and at least 3.
        device: tensor device desired to create the kernel
        dtype: tensor dtype desired to create the kernel

    Returns:
        1D tensor with laplacian filter coefficients.

    Raises:
        BaseError: if ``kernel_size`` is even, not positive, or ``1``: a single tap is the all-zero kernel.

    Shape:
        - Output: :math:`(\text{kernel_size})`

    Examples:
        >>> get_laplacian_kernel1d(3)
        tensor([ 1., -2.,  1.])
        >>> get_laplacian_kernel1d(5)
        tensor([ 1.,  1., -4.,  1.,  1.])

    """
    # TODO: add default dtype as None when kornia relies on torch > 1.12

    _check_kernel_size(kernel_size)
    _check_laplacian_kernel_size(kernel_size)

    return laplacian_1d(kernel_size, device=device, dtype=dtype)


def get_laplacian_kernel2d(
    kernel_size: tuple[int, int] | int, *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    r"""Return Laplacian filter matrix coefficients.

    Convention:
        See the Convention block on :func:`~kornia.filters.get_laplacian_kernel1d`: in a floating ``dtype``, or a
        signed integer one wide enough to hold :math:`1 - k_y k_x`, all ones with the centre set to
        :math:`1 - k_y k_x` for ``kernel_size=(k_y, k_x)``. Size 3 is the 8-neighbour stencil, which answers 3 to a
        unit :math:`\partial_{xx}` or :math:`\partial_{yy}`, so it estimates :math:`3 \nabla^2`; size 5 estimates
        :math:`25 \nabla^2`. :ref:`Filtering <filtering-conventions>` compares it with the scipy, OpenCV and
        scikit-image Laplacians.

    Args:
        kernel_size: filter size should be odd, and at least 3 along one axis.
        device: tensor device desired to create the kernel
        dtype: tensor dtype desired to create the kernel

    Returns:
        2D tensor with laplacian filter matrix coefficients.

    Raises:
        BaseError: if a size is even or not positive, if ``kernel_size`` is a sequence of other than 2 sizes, or if
            it is ``1`` or ``(1, 1)``: a :math:`1 \times 1` kernel is the all-zero kernel.

    Shape:
        - Output: :math:`(\text{kernel_size}_y, \text{kernel_size}_x)`

    Examples:
        >>> get_laplacian_kernel2d(3)
        tensor([[ 1.,  1.,  1.],
                [ 1., -8.,  1.],
                [ 1.,  1.,  1.]])
        >>> get_laplacian_kernel2d(5)
        tensor([[  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1., -24.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.]])

    """
    # TODO: add default dtype as None when kornia relies on torch > 1.12

    ky, kx = _unpack_2d_ks(kernel_size)
    _check_kernel_size((ky, kx))
    _check_laplacian_kernel_size((ky, kx))

    kernel = torch.ones((ky, kx), device=device, dtype=dtype)
    mid_x = kx // 2
    mid_y = ky // 2

    kernel[mid_y, mid_x] = 1 - kernel.sum()
    return kernel


def get_pascal_kernel_2d(
    kernel_size: tuple[int, int] | int,
    norm: bool = True,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Generate pascal filter kernel by kernel size.

    Args:
        kernel_size: height and width of the kernel.
        norm: if to normalize the kernel or not. Default: True.
        device: tensor device desired to create the kernel
        dtype: tensor dtype desired to create the kernel

    Returns:
        if kernel_size is an integer the kernel will be shaped as :math:`(kernel_size, kernel_size)`
        otherwise the kernel will be shaped as :math: `kernel_size`

    Examples:
    >>> get_pascal_kernel_2d(1)
    tensor([[1.]])
    >>> get_pascal_kernel_2d(4)
    tensor([[0.0156, 0.0469, 0.0469, 0.0156],
            [0.0469, 0.1406, 0.1406, 0.0469],
            [0.0469, 0.1406, 0.1406, 0.0469],
            [0.0156, 0.0469, 0.0469, 0.0156]])
    >>> get_pascal_kernel_2d(4, norm=False)
    tensor([[1., 3., 3., 1.],
            [3., 9., 9., 3.],
            [3., 9., 9., 3.],
            [1., 3., 3., 1.]])

    """
    ky, kx = _unpack_2d_ks(kernel_size)
    ax = get_pascal_kernel_1d(kx, device=device, dtype=dtype)
    ay = get_pascal_kernel_1d(ky, device=device, dtype=dtype)

    filt = ay[:, None] * ax[None, :]
    if norm:
        filt = filt / torch.sum(filt)
    return filt


def get_pascal_kernel_1d(
    kernel_size: int, norm: bool = False, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Generate Yang Hui triangle (Pascal's triangle) by a given number.

    Args:
        kernel_size: height and width of the kernel.
        norm: if to normalize the kernel or not. Default: False.
        device: tensor device desired to create the kernel
        dtype: tensor dtype desired to create the kernel

    Returns:
        kernel shaped as :math:`(kernel_size,)`

    Examples:
    >>> get_pascal_kernel_1d(1)
    tensor([1.])
    >>> get_pascal_kernel_1d(2)
    tensor([1., 1.])
    >>> get_pascal_kernel_1d(3)
    tensor([1., 2., 1.])
    >>> get_pascal_kernel_1d(4)
    tensor([1., 3., 3., 1.])
    >>> get_pascal_kernel_1d(5)
    tensor([1., 4., 6., 4., 1.])
    >>> get_pascal_kernel_1d(6)
    tensor([ 1.,  5., 10., 10.,  5.,  1.])

    """
    pre: list[float] = []
    cur: list[float] = []
    for i in range(kernel_size):
        cur = [1.0] * (i + 1)

        for j in range(1, i // 2 + 1):
            value = pre[j - 1] + pre[j]
            cur[j] = value
            if i != 2 * j:
                cur[-j - 1] = value
        pre = cur

    out = torch.tensor(cur, device=device, dtype=dtype)

    if norm:
        out = out / out.sum()

    return out


def get_canny_nms_kernel(device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Return 3x3 kernels for the Canny Non-maximal suppression.

    Not used by :func:`~kornia.filters.canny`, which compares the neighbours by slicing, so that ties compare exactly.
    """
    return torch.tensor(
        [
            [[[0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
        ],
        device=device,
        dtype=dtype,
    )


def get_hysteresis_kernel(device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Return the 3x3 kernels for the Canny hysteresis."""
    return torch.tensor(
        [
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        ],
        device=device,
        dtype=dtype,
    )


def get_hanning_kernel1d(
    kernel_size: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    r"""Return Hanning (also known as Hann) kernel.

    .. math::  w(n) = 0.5 - 0.5 \cos\left(\frac{2\pi n}{M - 1}\right) \qquad 0 \leq n \leq M - 1

    Convention:
        This is the symmetric window of size :math:`M`: both end taps are 0, so only :math:`M - 2` samples carry
        weight, and it is not normalized, its taps summing to :math:`(M - 1) / 2`. An even size is accepted and is
        symmetric about :math:`(M - 1) / 2`. :ref:`Filtering <filtering-conventions>` names the matching numpy and
        torch windows.

    Args:
        kernel_size: The size of the kernel, an integer greater than 2.
        device: tensor device desired to create the kernel
        dtype: tensor dtype desired to create the kernel

    Returns:
        1D tensor with Hanning filter coefficients. Shape :math:`(\text{kernel_size})`.

    Examples:
        >>> get_hanning_kernel1d(4)
        tensor([0.0000, 0.7500, 0.7500, 0.0000])

    """
    _check_kernel_size(kernel_size, 2, allow_even=True)

    x = torch.arange(kernel_size, device=device, dtype=dtype)
    return 0.5 - 0.5 * torch.cos(2.0 * math.pi * x / float(kernel_size - 1))


def get_hanning_kernel2d(
    kernel_size: tuple[int, int] | int,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Return 2d Hanning kernel.

    Convention:
        See the Convention block on :func:`~kornia.filters.get_hanning_kernel1d`. For ``kernel_size=(k_y, k_x)``
        the kernel is the outer product of the 1d windows of sizes ``k_y`` (along ``H``) and ``k_x`` (along ``W``).

    Args:
        kernel_size: The size of the kernel for the filter, an integer or ``(k_y, k_x)``, each greater than 2.
        device: tensor device desired to create the kernel
        dtype: tensor dtype desired to create the kernel

    Returns:
        2D tensor with Hanning filter coefficients. Shape :math:`(k_y, k_x)`.

    """
    kernel_size = _unpack_2d_ks(kernel_size)
    _check_kernel_size(kernel_size, 2, allow_even=True)

    ky = get_hanning_kernel1d(kernel_size[0], device, dtype)[None].T
    kx = get_hanning_kernel1d(kernel_size[1], device, dtype)[None]
    return ky @ kx


@deprecated(replace_with="get_gaussian_kernel1d", version="0.6.10")
def get_gaussian_kernel1d_t(*args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: D103
    return get_gaussian_kernel1d(*args, **kwargs)


@deprecated(replace_with="get_gaussian_kernel2d", version="0.6.10")
def get_gaussian_kernel2d_t(*args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: D103
    return get_gaussian_kernel2d(*args, **kwargs)


@deprecated(replace_with="get_gaussian_kernel3d", version="0.6.10")
def get_gaussian_kernel3d_t(*args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: D103
    return get_gaussian_kernel3d(*args, **kwargs)
