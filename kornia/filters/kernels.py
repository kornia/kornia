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
from typing import Optional, Union

import torch

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.core.ops import _device_literal


def _check_kernel_size(kernel_size: tuple[int, ...] | int, min_value: int = 0, allow_even: bool = False) -> None:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,)

    fmt = "even or odd" if allow_even else "odd"
    invalid = [
        size
        for size in kernel_size
        if not (isinstance(size, int) and (((size % 2 == 1) or allow_even) and size > min_value))
    ]
    KORNIA_CHECK(
        not invalid,
        lambda: f"Kernel size must be an {fmt} integer bigger than {min_value}. Got {invalid[0]} on {kernel_size}",
    )


def _check_laplacian_kernel_size(kernel_size: tuple[int, ...] | int) -> None:
    # The Laplacian centre is 1 - prod(kernel_size), so a single tap is the all-zero kernel (0 / 0 once normalized).
    if isinstance(kernel_size, int):
        sizes: tuple[int, ...] = (kernel_size,)
    else:
        sizes = kernel_size

    KORNIA_CHECK(
        any(size != 1 for size in sizes),
        lambda: (
            "A Laplacian kernel needs a size of at least 3 along one axis: a single tap is all zeros. Got "
            f"{kernel_size}"
        ),
    )


def _check_kernel_dtype(dtype: Optional[torch.dtype], name: str, taps: Optional[tuple[int, int]] = None) -> None:
    # An integer or bool dtype truncates fractional taps and wraps the ones outside its range (#5155). ``taps`` is
    # the (lowest, highest) tap of a kernel whose taps are all integers; without it the kernel has fractional taps.
    if dtype is None or dtype.is_floating_point or dtype.is_complex:
        return
    KORNIA_CHECK(
        taps is not None,
        lambda: (
            f"{name} has fractional taps, which an integer dtype cannot hold. Got dtype={dtype}; use a floating dtype."
        ),
    )
    low, high = (0, 1) if dtype == torch.bool else (torch.iinfo(dtype).min, torch.iinfo(dtype).max)
    KORNIA_CHECK(
        low <= taps[0] and taps[1] <= high,
        lambda: (
            f"{name} has taps from {taps[0]} to {taps[1]}, which dtype={dtype} cannot hold. Use a signed integer "
            "dtype wide enough for them, or a floating one."
        ),
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
    r"""Divide each 2D kernel by the sum of its absolute values.

    ``kernel = normalize_kernel2d(input)`` scales every ``(H, W)`` kernel of a ``(*, H, W)`` tensor to an absolute sum
    of 1, so a non-negative kernel sums to 1 and a derivative kernel keeps its sign. An all-zero kernel becomes NaN.

    Args:
        input: kernels with shape ``(*, H, W)``.

    Returns:
        the normalized kernels, with the shape of ``input``.

    See Also:
        - :func:`~kornia.filters.filter2d`: normalizes its kernel this way with ``normalized=True``.

    """
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
    r"""Sample a Gaussian on any window around a chosen mean, summing to 1.

    ``kernel = gaussian(window_size, sigma)`` returns the Gaussian of standard deviation ``sigma`` sampled at the
    ``window_size`` taps, as a ``(B, window_size)`` tensor. Use it for an off-centre ``mean``;
    :func:`get_gaussian_kernel1d` is the checked call for a centred kernel.

    Convention:
        See the Convention block on :func:`~kornia.filters.get_gaussian_kernel1d`, which validates the size and
        calls this function; ``gaussian`` does not validate it. In a floating ``dtype``, for an even ``window_size``
        the Gaussian is centred at ``mean - 0.5``: the default ``mean`` centres the kernel on the middle of the
        window, halfway between the two middle samples, and an explicit ``mean=m`` centres it at ``m - 0.5``.

    Args:
        window_size: number of taps.
        sigma: non-negative standard deviation in taps, a float or a ``(B, 1)`` tensor.
        mean: centre in taps, a float or a ``(B, 1)`` tensor with the ``B`` of ``sigma``; ``window_size // 2`` by
            default. See the Convention block for an even ``window_size``.
        device: device of the kernel when ``sigma`` is a Python number.
        dtype: dtype of the kernel when ``sigma`` is a Python number.

    Returns:
        the kernels, each summing to 1, with shape ``(B, window_size)``.

    .. note::
        Float16 and bfloat16 kernels are computed in float32 and rounded once at return.

        A ``sigma`` of zero, or one too small for the window to hold a representable weight, puts all the mass on
        the sample nearest the mean: the unit impulse. With an even ``window_size`` the default ``mean`` falls between
        two samples, so a small ``sigma`` splits the mass between them, and a ``sigma`` of zero is not defined.

    See Also:
        - :func:`get_gaussian_kernel1d`: the checked call.

    """
    output_dtype = sigma.dtype if isinstance(sigma, torch.Tensor) else dtype or torch.get_default_dtype()
    if isinstance(sigma, (int, float)):
        _check_kernel_dtype(dtype, "A Gaussian kernel")
        compute_dtype = torch.float32 if output_dtype in (torch.float16, torch.bfloat16) else output_dtype
        sigma = torch.tensor([[float(sigma)]], device=device, dtype=compute_dtype)

    KORNIA_CHECK_IS_TENSOR(sigma)
    KORNIA_CHECK_SHAPE(sigma, ["B", "1"])
    if not (sigma.dtype.is_floating_point or sigma.dtype.is_complex):
        # as in `gaussian_discrete`: in a uint8 sigma's dtype the offsets below the mean would wrap (#5155)
        sigma = sigma.to(torch.get_default_dtype())
    if output_dtype in (torch.float16, torch.bfloat16):
        sigma = sigma.float()
        if isinstance(mean, torch.Tensor):
            mean = mean.float()
    batch_size = sigma.shape[0]

    mean = float(window_size // 2) if mean is None else mean
    if isinstance(mean, (int, float)):
        # Filled on the device in eager mode, so a CUDA ``sigma`` gets no host copy that waits for queued work.
        mean = _device_literal([[float(mean)]], device=sigma.device, dtype=sigma.dtype)

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
    # whose square underflows the working dtype. Either way the nearest samples
    # divide 0 / 0. Repairing that after the fact leaves the division on the graph and the sigma
    # gradient still comes back NaN, so divide by a stand-in and select the impulse out of that
    # finite arm instead. The gradient is then the 0 of the continuous sigma -> 0 limit. A NaN
    # sigma is not caught by the comparison and still propagates, as it did before.
    denominator = 2 * sigma.pow(2.0)
    is_impulse = denominator == 0
    safe_denominator = torch.where(is_impulse, torch.ones_like(denominator), denominator)
    gauss = torch.exp(-dist / safe_denominator)
    gauss = torch.where(is_impulse, (dist == 0).to(dist.dtype), gauss)

    kernel = gauss / gauss.sum(-1, keepdim=True)
    return kernel.to(output_dtype) if output_dtype in (torch.float16, torch.bfloat16) else kernel


def gaussian_discrete_erf(
    window_size: int,
    sigma: torch.Tensor | float,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Integrate a Gaussian over each tap of a window with the error function, summing to 1.

    ``kernel = gaussian_discrete_erf(window_size, sigma)`` returns the Gaussian of standard deviation ``sigma``
    integrated over the unit-wide interval of each tap, as a ``(B, window_size)`` tensor.

    Args:
        window_size: number of taps, even or odd.
        sigma: non-negative standard deviation in taps, a float or a ``(B, 1)`` tensor.
        device: device of the kernel when ``sigma`` is a Python number.
        dtype: dtype of the kernel when ``sigma`` is a Python number.

    Returns:
        the kernels, each summing to 1, with shape ``(B, window_size)``.

    See Also:
        - :func:`get_gaussian_erf_kernel1d`: the checked call, which rejects an even size unless ``force_even``.

    """
    # Adapted from https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    if isinstance(sigma, (int, float)):
        _check_kernel_dtype(dtype, "A Gaussian kernel")
        sigma = torch.tensor([[float(sigma)]], device=device, dtype=dtype)

    KORNIA_CHECK_SHAPE(sigma, ["B", "1"])
    batch_size = sigma.shape[0]

    # Centre the window at (window_size - 1) / 2: the middle tap of an odd window, and halfway between the two
    # middle taps of an even one, as the sampled kernel in `gaussian` is.
    centre = (window_size - 1) / 2
    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - centre).expand(batch_size, -1)

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
    r"""Compute the discrete Gaussian of a window from modified Bessel functions, summing to 1.

    ``kernel = gaussian_discrete(window_size, sigma)`` returns Lindeberg's discrete Gaussian at the integer offsets
    of the ``window_size`` taps from the centre tap, as a ``(B, window_size)`` tensor.

    Args:
        window_size: number of taps, odd.
        sigma: non-negative standard deviation in taps, a float or a ``(B, 1)`` tensor.
        device: device of the kernel when ``sigma`` is a Python number.
        dtype: dtype of the kernel when ``sigma`` is a Python number.

    Returns:
        the kernels, each summing to 1, with shape ``(B, window_size)``.

    See Also:
        - :func:`get_gaussian_discrete_kernel1d`: the checked call, which also takes ``force_even``.

    """
    # Adapted from https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    if isinstance(sigma, (int, float)):
        _check_kernel_dtype(dtype, "A Gaussian kernel")
        sigma = torch.tensor([[float(sigma)]], device=device, dtype=dtype)

    KORNIA_CHECK_SHAPE(sigma, ["B", "1"])
    # The discrete Gaussian is defined at the integer offsets of the taps from the centre tap, so there is no even
    # window: the window must be odd, and holds the offsets 0 .. window_size // 2 on either side of the centre.
    KORNIA_CHECK(window_size % 2 == 1, lambda: f"The discrete Gaussian kernel needs an odd window. Got {window_size}")

    output_dtype = sigma.dtype
    if not (sigma.dtype.is_floating_point or sigma.dtype.is_complex):
        # An integer sigma tensor gives a floating kernel, as it does in `gaussian` and `gaussian_discrete_erf`;
        # casting the normalized taps back to its dtype would truncate them all to 0 (#5155).
        output_dtype = torch.get_default_dtype()
        sigma = sigma.to(output_dtype)
    elif sigma.dtype in (torch.float16, torch.bfloat16):
        sigma = sigma.float()
    sigma2 = sigma * sigma
    tail = int(window_size // 2) + 1
    bessels = [_modified_bessel_0(sigma2, scaled=True)]
    if tail > 1:
        bessels.append(_modified_bessel_1(sigma2, scaled=True))
    bessels.extend(_modified_bessel_i(k, sigma2, scaled=True, max_order=tail) for k in range(2, tail))
    # The exp(-sigma²) factor is already included in the scaled Bessel terms.
    out = torch.cat(bessels[:0:-1] + bessels, -1)

    return (out / out.sum(-1, keepdim=True)).to(output_dtype)


def laplacian_1d(
    window_size: int, *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    r"""Build a 1D Laplacian kernel of any positive size, without validating it.

    ``kernel = laplacian_1d(window_size)`` returns ones with ``1 - window_size`` at index ``window_size // 2``. Call
    :func:`get_laplacian_kernel1d` instead, which rejects an even size and a single tap.

    Convention:
        See the Convention block on :func:`~kornia.filters.get_laplacian_kernel1d`. An even size puts the negative tap
        at ``window_size // 2``, off the middle of the kernel, and size 1 gives ``[0.]``.

    Args:
        window_size: number of taps.
        device: device of the kernel.
        dtype: dtype of the kernel.

    Returns:
        the kernel, with shape ``(window_size,)``.

    See Also:
        - :func:`get_laplacian_kernel1d`: the same kernel with its size checked.

    """
    # TODO: add default dtype as None when kornia relies on torch > 1.12
    _check_kernel_dtype(dtype, "The Laplacian kernel", (1 - window_size, 1))
    filter_1d = torch.ones(window_size, device=device, dtype=dtype)
    middle = window_size // 2
    filter_1d[middle] = 1 - window_size
    return filter_1d


def get_box_kernel1d(
    kernel_size: int, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    r"""Return a 1D box kernel: equal taps summing to 1.

    ``kernel = get_box_kernel1d(kernel_size)`` returns the averaging kernel of ``kernel_size`` taps, as a
    ``(1, kernel_size)`` tensor.

    Convention:
        - An even ``kernel_size`` is accepted.
        - The kernel is a contiguous tensor, so an in-place edit changes only the taps it addresses.
        - An integer or bool ``dtype`` raises: it would truncate ``1 / kernel_size`` to 0
          (`#5155 <https://github.com/kornia/kornia/issues/5155>`_).

    Args:
        kernel_size: number of taps.
        device: device of the kernel.
        dtype: dtype of the kernel.

    Returns:
        the kernel, every tap :math:`1 / \text{kernel\_size}`.

    See Also:
        - :func:`get_box_kernel2d`: the 2D kernel.
        - :func:`box_blur`: the blur that applies it.

    """
    _check_kernel_dtype(dtype, "The box kernel")
    return torch.full((1, kernel_size), 1.0 / kernel_size, device=device, dtype=dtype)


def get_box_kernel2d(
    kernel_size: tuple[int, int] | int, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    r"""Return a 2D box kernel: equal taps summing to 1.

    ``kernel = get_box_kernel2d(kernel_size)`` returns the averaging kernel of ``(k_y, k_x)`` taps, as a
    ``(1, k_y, k_x)`` tensor.

    Convention:
        See the Convention block on :func:`~kornia.filters.get_box_kernel1d`.

    Args:
        kernel_size: size ``(k_y, k_x)``, or one integer for both.
        device: device of the kernel.
        dtype: dtype of the kernel.

    Returns:
        the kernel, every tap :math:`1 / (k_y k_x)`.

    See Also:
        - :func:`get_box_kernel1d`: the 1D kernel.
        - :func:`box_blur`: the blur that applies it.

    """
    ky, kx = _unpack_2d_ks(kernel_size)
    _check_kernel_dtype(dtype, "The box kernel")
    return torch.full((1, ky, kx), 1.0 / (kx * ky), device=device, dtype=dtype)


def get_binary_kernel2d(
    window_size: tuple[int, int] | int, *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    r"""Return one-hot kernels that pick out every pixel of a window.

    ``kernels = get_binary_kernel2d(window_size)`` returns one kernel per position of a ``(k_y, k_x)`` window, as a
    ``(k_y * k_x, 1, k_y, k_x)`` tensor. :func:`torch.nn.functional.conv2d` with them stacks each pixel's
    neighbourhood along the channel axis.

    Convention:
        Channel ``i`` is one-hot at ``(i // k_x, i % k_x)``, in row-major order, the layout
        :func:`~kornia.filters.median_blur` builds on.

    Args:
        window_size: size ``(k_y, k_x)``, or one integer for both.
        device: device of the kernel.
        dtype: dtype of the kernel.

    Returns:
        the kernels, with shape ``(k_y * k_x, 1, k_y, k_x)``.

    See Also:
        - :func:`median_blur`: takes the median over the stacked neighbourhood.

    """
    # TODO: add default dtype as None when kornia relies on torch > 1.12

    ky, kx = _unpack_2d_ks(window_size)

    window_range = kx * ky

    kernel = torch.zeros((window_range, window_range), device=device, dtype=dtype)
    idx = torch.arange(window_range, device=device)
    kernel[idx, idx] += 1.0
    return kernel.view(window_range, 1, ky, kx)


def get_sobel_kernel_3x3(*, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Return the 3x3 Sobel kernel of the x derivative.

    ``kernel = get_sobel_kernel_3x3()`` is the integer stencil ``[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]``; its transpose
    is the y derivative.
    """
    return torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device, dtype=dtype)


def get_sobel_kernel_5x5_2nd_order(
    *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    r"""Return the 5x5 Sobel kernel of the second x derivative.

    ``kernel = get_sobel_kernel_5x5_2nd_order()`` is the ``(5, 5)`` integer stencil of :math:`\partial_{xx}`; its
    transpose is :math:`\partial_{yy}`.
    """
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
    """Return the 3x3 central-difference kernel of the x derivative.

    ``kernel = get_diff_kernel_3x3()`` is ``[-1, 0, 1]`` in the middle row of a ``(3, 3)`` tensor; its transpose is
    the y derivative.
    """
    return torch.tensor([[-0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [-0.0, 0.0, 0.0]], device=device, dtype=dtype)


def get_diff_kernel3d(device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    r"""Return the 3D central-difference kernels of the first derivatives.

    ``kernels = get_diff_kernel3d()`` returns the kernels of :math:`(\partial_x, \partial_y, \partial_z)` with shape
    ``(3, 1, 3, 3, 3)`` and taps :math:`\pm 0.5`. :func:`get_spatial_gradient_kernel3d` calls it for ``order=1`` and
    documents the units and ``dtype``.
    """
    _check_kernel_dtype(dtype, "The 3d first-order difference kernel")
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
    r"""Return the 3D central-difference kernels of the second derivatives.

    ``kernels = get_diff_kernel3d_2nd_order()`` returns the kernels of
    :math:`(\partial_{xx}, \partial_{yy}, \partial_{zz}, \partial_{xy}, \partial_{yz}, \partial_{xz})` with shape
    ``(6, 1, 3, 3, 3)``. :func:`get_spatial_gradient_kernel3d` calls it for ``order=2`` and documents the units and
    ``dtype``.
    """
    _check_kernel_dtype(dtype, "The 3d second-order difference kernel")
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
    r"""Return the 2D Sobel kernels of the first derivatives (x, y).

    ``kernels = get_sobel_kernel2d()`` returns the kernels of :math:`(\partial_x, \partial_y)` stacked as a
    ``(2, 3, 3)`` tensor.

    Convention:
        See the Convention block on :func:`~kornia.filters.get_spatial_gradient_kernel2d`; this is
        ``get_spatial_gradient_kernel2d('sobel', 1)``.

    See Also:
        - :func:`get_diff_kernel2d`: the same stack of central-difference kernels.
        - :func:`get_spatial_gradient_kernel2d`: also the second order and the ``'diff'`` mode.

    """
    kernel_x = get_sobel_kernel_3x3(device=device, dtype=dtype)
    kernel_y = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])


def get_diff_kernel2d(*, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    r"""Return the 2D central-difference kernels of the first derivatives (x, y).

    ``kernels = get_diff_kernel2d()`` returns the kernels of :math:`(\partial_x, \partial_y)` stacked as a
    ``(2, 3, 3)`` tensor.

    Convention:
        See the Convention block on :func:`~kornia.filters.get_spatial_gradient_kernel2d`; this is
        ``get_spatial_gradient_kernel2d('diff', 1)``.

    See Also:
        - :func:`get_sobel_kernel2d`: the same stack of Sobel kernels.
        - :func:`get_spatial_gradient_kernel2d`: also the second order and the ``'sobel'`` mode.

    """
    kernel_x = get_diff_kernel_3x3(device=device, dtype=dtype)
    kernel_y = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])


def get_sobel_kernel2d_2nd_order(
    *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    r"""Return the 2D Sobel kernels of the second derivatives.

    ``kernels = get_sobel_kernel2d_2nd_order()`` returns the kernels of
    :math:`(\partial_{xx}, \partial_{xy}, \partial_{yy})` stacked as a ``(3, 5, 5)`` tensor.
    """
    gxx = get_sobel_kernel_5x5_2nd_order(device=device, dtype=dtype)
    gyy = gxx.transpose(0, 1)
    gxy = _get_sobel_kernel_5x5_2nd_order_xy(device=device, dtype=dtype)
    return torch.stack([gxx, gxy, gyy])


def get_diff_kernel2d_2nd_order(
    *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    r"""Return the 2D central-difference kernels of the second derivatives.

    ``kernels = get_diff_kernel2d_2nd_order()`` returns the kernels of
    :math:`(\partial_{xx}, \partial_{xy}, \partial_{yy})` stacked as a ``(3, 3, 3)`` tensor.
    """
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
    r"""Return the Sobel or difference kernels of the first or second image derivatives.

    ``kernels = get_spatial_gradient_kernel2d(mode, order)`` returns the stencils that :func:`spatial_gradient`
    correlates with an image.

    Convention:
        - ``order=1`` stacks :math:`(\partial_x, \partial_y)` along the first axis and ``order=2`` stacks
          :math:`(\partial_{xx}, \partial_{xy}, \partial_{yy})`. Correlated with an image, as
          :func:`~kornia.filters.filter2d` does, an ``order=1`` channel is positive where the values increase with
          the column (x) or the row (y), and an ``order=2`` channel is positive on :math:`x^2`, :math:`xy` and
          :math:`y^2` respectively.
        - The kernels are raw integer stencils, not derivative estimates. Per unit slope Sobel answers 8 and
          ``'diff'`` answers 2; per unit second derivative Sobel answers 64 on all three channels, while ``'diff'``
          answers 1 on :math:`\partial_{xx}` and :math:`\partial_{yy}` and 4 on :math:`\partial_{xy}`.
        - ``mode`` is case-insensitive: ``'Sobel'`` is ``'sobel'``.

    Args:
        mode: ``'sobel'`` or ``'diff'``.
        order: derivative order, 1 or 2.
        device: device of the kernels.
        dtype: dtype of the kernels.

    Returns:
        the kernels with shape ``(2, 3, 3)`` for ``order=1``, and ``(3, 5, 5)`` (Sobel) or
        ``(3, 3, 3)`` (diff) for ``order=2``.

    See Also:
        - :func:`get_spatial_gradient_kernel3d`: the 3D kernels, ``'diff'`` only.
        - :func:`spatial_gradient`: the filter that applies them, normalized by default.

    """
    KORNIA_CHECK(mode.lower() in {"sobel", "diff"}, f"Mode should be `sobel` or `diff`. Got {mode}")
    KORNIA_CHECK(order in {1, 2}, lambda: f"Order should be 1 or 2. Got {order}")
    # the check is case-insensitive, so dispatch on the lower-case spelling as well
    mode = mode.lower()

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
    r"""Return the central-difference kernels of the first or second derivatives of a volume.

    ``kernels = get_spatial_gradient_kernel3d(mode, order)`` returns the ``(3, 3, 3)`` stencils that
    :func:`spatial_gradient3d` correlates with a volume.

    Convention:
        - Only ``mode='diff'`` exists; ``'sobel'`` passes the mode check and raises ``NotImplementedError``.
        - ``order=1`` stacks :math:`(\partial_x, \partial_y, \partial_z)` and ``order=2`` stacks
          :math:`(\partial_{xx}, \partial_{yy}, \partial_{zz}, \partial_{xy}, \partial_{yz}, \partial_{xz})`, an
          order that differs from the 2d :math:`(\partial_{xx}, \partial_{xy}, \partial_{yy})`. The stack has a
          singleton second axis that :func:`~kornia.filters.get_spatial_gradient_kernel2d` lacks.
        - In a floating ``dtype`` every channel is in derivative units: it answers 1 to a unit slope or a unit
          second derivative, unlike the raw 2d stencils.
        - ``mode`` is case-insensitive: ``'Diff'`` is ``'diff'``.
        - An integer or bool ``dtype`` raises: it would truncate the half and quarter taps to 0, leaving the
          first-order channels and the mixed second-order ones all zero
          (`#5155 <https://github.com/kornia/kornia/issues/5155>`_).

    Args:
        mode: ``'diff'``.
        order: derivative order, 1 or 2.
        device: device of the kernels.
        dtype: dtype of the kernels.

    Returns:
        the kernels with shape ``(3, 1, 3, 3, 3)`` for ``order=1`` and ``(6, 1, 3, 3, 3)`` for ``order=2``.

    See Also:
        - :func:`get_spatial_gradient_kernel2d`: the 2D kernels, with the ``'sobel'`` mode.
        - :func:`spatial_gradient3d`: the filter that applies them.

    """
    KORNIA_CHECK(mode.lower() in {"sobel", "diff"}, f"Mode should be `sobel` or `diff`. Got {mode}")
    KORNIA_CHECK(order in {1, 2}, lambda: f"Order should be 1 or 2. Got {order}")
    # the check is case-insensitive, so dispatch on the lower-case spelling as well
    mode = mode.lower()

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
    r"""Return a 1D Gaussian kernel sampled at the taps, summing to 1.

    ``kernel = get_gaussian_kernel1d(kernel_size, sigma)`` returns the Gaussian of standard deviation ``sigma``
    sampled at the ``kernel_size`` taps, as a ``(B, kernel_size)`` tensor. It is the usual
    Gaussian kernel, the one :func:`gaussian_blur2d` applies.

    Convention:
        - In a floating ``dtype`` the kernel samples :math:`\exp(-n^2 / 2\sigma^2)` at the offsets :math:`n` of its
          taps from its centre, integers for an odd size and half-integers for an even one, and is normalized to
          sum 1; :ref:`Filtering <filtering-conventions>` names the matching scipy and OpenCV kernels.
        - ``force_even=True`` also accepts an even ``kernel_size``, and in a floating ``dtype`` the kernel is then
          symmetric about the middle of the window, ``(kernel_size - 1) / 2``. No tap sits on that middle, so a
          ``sigma`` of zero is not defined for an even ``kernel_size``.
        - A tensor ``sigma`` of shape ``(B, 1)`` gives one kernel per row.
        - A Python integer ``sigma`` is treated as its floating-point equivalent. With ``dtype=None``, both use
          the default floating-point dtype.
        - An integer or bool ``dtype`` raises: the taps are fractional, and an integer ``dtype`` would also
          truncate a fractional ``sigma`` (`#5155 <https://github.com/kornia/kornia/issues/5155>`_).

    Args:
        kernel_size: number of taps, odd and positive.
        sigma: non-negative standard deviation in taps, a float or a ``(B, 1)`` tensor.
        force_even: whether to accept an even ``kernel_size``.
        device: device of the kernel when ``sigma`` is a Python number.
        dtype: dtype of the kernel when ``sigma`` is a Python number.

    Returns:
        the kernels, with shape ``(B, kernel_size)``.

    See Also:
        - :func:`get_gaussian_erf_kernel1d`: integrates the Gaussian over each pixel, so it blurs more.
        - :func:`get_gaussian_discrete_kernel1d`: Lindeberg's discrete Gaussian, for discrete scale space.
        - :func:`gaussian`: the same samples around a chosen ``mean``, without the size check.
        - :func:`get_gaussian_kernel2d`: the outer product of two of these kernels.
        - :func:`gaussian_blur2d`: the blur that applies it along each axis.

    Examples:
        >>> get_gaussian_kernel1d(3, 2.5)
        tensor([[0.3243, 0.3513, 0.3243]])
        >>> get_gaussian_kernel1d(5, 1.5)
        tensor([[0.1201, 0.2339, 0.2921, 0.2339, 0.1201]])
        >>> get_gaussian_kernel1d(5, torch.tensor([[1.5], [0.7]]))
        tensor([[0.1201, 0.2339, 0.2921, 0.2339, 0.1201],
                [0.0096, 0.2054, 0.5699, 0.2054, 0.0096]])

        A ``sigma`` of zero returns the unit impulse:

        >>> get_gaussian_kernel1d(5, 0.0)
        tensor([[0., 0., 1., 0., 0.]])

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
    r"""Return a 1D discrete Gaussian kernel from modified Bessel functions, summing to 1.

    ``kernel = get_gaussian_discrete_kernel1d(kernel_size, sigma)`` takes the arguments of
    :func:`get_gaussian_kernel1d` and returns Lindeberg's discrete Gaussian, the smoothing kernel of discrete scale
    space, as a ``(B, kernel_size)`` tensor.

    Convention:
        - See the Convention block on :func:`~kornia.filters.get_gaussian_kernel1d`. In a floating ``dtype`` the
          kernel is :math:`e^{-\sigma^2} I_{|n|}(\sigma^2)`, with :math:`I_n` the modified Bessel function of the
          first kind, normalized over the window. A float16 or bfloat16 kernel is computed in float32 and rounded to
          its ``dtype``.
        - ``kernel_size=1`` is the single tap ``[1.0]``. The discrete Gaussian is defined at integer offsets from the
          centre tap, so there is no even window: an even ``kernel_size`` raises with and without ``force_even``.

    Args:
        kernel_size: number of taps, odd and positive.
        sigma: non-negative standard deviation in taps, a float or a ``(B, 1)`` tensor.
        force_even: accepted for parity with the other Gaussian builders, and ignored.
        device: device of the kernel when ``sigma`` is a Python number.
        dtype: dtype of the kernel when ``sigma`` is a Python number.

    Returns:
        the kernels, with shape ``(B, kernel_size)``.

    See Also:
        - :func:`get_gaussian_kernel1d`: samples the Gaussian, the usual choice, and takes an even size.
        - :func:`get_gaussian_erf_kernel1d`: integrates the Gaussian over each pixel.

    Examples:
        >>> get_gaussian_discrete_kernel1d(3, 2.5)
        tensor([[0.3235, 0.3531, 0.3235]])
        >>> get_gaussian_discrete_kernel1d(5, 1.5)
        tensor([[0.1096, 0.2323, 0.3161, 0.2323, 0.1096]])
        >>> get_gaussian_discrete_kernel1d(5, torch.tensor([[1.5],[2.4]]))
        tensor([[0.1096, 0.2323, 0.3161, 0.2323, 0.1096],
                [0.1635, 0.2170, 0.2389, 0.2170, 0.1635]])

    """
    # Adapted from https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
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
    r"""Return a 1D Gaussian kernel integrated per pixel with the error function.

    ``kernel = get_gaussian_erf_kernel1d(kernel_size, sigma)`` takes the arguments of :func:`get_gaussian_kernel1d`
    and returns the Gaussian integrated over the unit-wide interval of each tap, as a ``(B, kernel_size)`` tensor.

    Convention:
        - See the Convention block on :func:`~kornia.filters.get_gaussian_kernel1d`. In a floating ``dtype`` the
          kernel is :math:`\Phi((n + 1/2) / \sigma) - \Phi((n - 1/2) / \sigma)` with :math:`\Phi` the normal CDF,
          so it blurs more than the sampled kernel: for :math:`\sigma` of about 1 or more, on a window wide enough
          for the tails, its variance is :math:`\sigma^2 + 1/12`.
        - With ``force_even=True`` an even kernel is symmetric about the middle of the window,
          ``(kernel_size - 1) / 2``, as the sampled kernel is.

    Args:
        kernel_size: number of taps, odd and positive.
        sigma: non-negative standard deviation in taps, a float or a ``(B, 1)`` tensor.
        force_even: whether to accept an even ``kernel_size``.
        device: device of the kernel when ``sigma`` is a Python number.
        dtype: dtype of the kernel when ``sigma`` is a Python number.

    Returns:
        the kernels, with shape ``(B, kernel_size)``.

    See Also:
        - :func:`get_gaussian_kernel1d`: samples the Gaussian at the taps, the usual choice.
        - :func:`get_gaussian_discrete_kernel1d`: Lindeberg's discrete Gaussian.

    Examples:
        >>> get_gaussian_erf_kernel1d(3, 2.0)
        tensor([[0.3195, 0.3611, 0.3195]])
        >>> get_gaussian_erf_kernel1d(5, 1.5)
        tensor([[0.1226, 0.2331, 0.2887, 0.2331, 0.1226]])
        >>> get_gaussian_erf_kernel1d(5, torch.tensor([[1.5], [2.1]]))
        tensor([[0.1226, 0.2331, 0.2887, 0.2331, 0.1226],
                [0.1574, 0.2198, 0.2456, 0.2198, 0.1574]])

    """
    # Adapted from https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
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
    r"""Return a 2D Gaussian kernel as the outer product of two 1D kernels, summing to 1.

    ``kernel = get_gaussian_kernel2d(kernel_size, sigma)`` multiplies the :func:`get_gaussian_kernel1d` kernels of
    ``kernel_size=(k_y, k_x)`` and ``sigma=(sigma_y, sigma_x)``, y first, and returns a ``(B, k_y, k_x)`` tensor.

    Convention:
        See the Convention block on :func:`~kornia.filters.get_gaussian_kernel1d`. A tensor ``sigma`` of shape
        ``(B, 2)`` gives one kernel per row.

    Args:
        kernel_size: size ``(k_y, k_x)``, or one integer for both; odd and positive.
        sigma: non-negative standard deviations ``(sigma_y, sigma_x)``, or a ``(B, 2)`` tensor.
        force_even: whether to accept an even size.
        device: device of the kernel when ``sigma`` is a tuple.
        dtype: dtype of the kernel when ``sigma`` is a tuple.

    Returns:
        the kernels, with shape ``(B, k_y, k_x)``.

    See Also:
        - :func:`get_gaussian_kernel1d`: the 1D kernel.
        - :func:`get_gaussian_kernel3d`: the 3D kernel.
        - :func:`gaussian_blur2d`: the blur that applies it.

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
        _check_kernel_dtype(dtype, "A Gaussian kernel")
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
    r"""Return a 3D Gaussian kernel as the outer product of three 1D kernels, summing to 1.

    ``kernel = get_gaussian_kernel3d(kernel_size, sigma)`` multiplies the :func:`get_gaussian_kernel1d` kernels of
    ``kernel_size=(k_z, k_y, k_x)`` and ``sigma=(sigma_z, sigma_y, sigma_x)``, z first, and returns a
    ``(B, k_z, k_y, k_x)`` tensor.

    Convention:
        See the Convention block on :func:`~kornia.filters.get_gaussian_kernel1d`. A tensor ``sigma`` of shape
        ``(B, 3)`` gives one kernel per row.

    Args:
        kernel_size: size ``(k_z, k_y, k_x)``, or one integer for all three; odd and positive.
        sigma: non-negative standard deviations ``(sigma_z, sigma_y, sigma_x)``, or a ``(B, 3)`` tensor.
        force_even: whether to accept an even size.
        device: device of the kernel when ``sigma`` is a tuple.
        dtype: dtype of the kernel when ``sigma`` is a tuple.

    Returns:
        the kernels, with shape ``(B, k_z, k_y, k_x)``.

    See Also:
        - :func:`get_gaussian_kernel2d`: the 2D kernel.
        - :func:`filter3d`: filters a volume with it.

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
        _check_kernel_dtype(dtype, "A Gaussian kernel")
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
    r"""Return a 1D Laplacian kernel: ones with a negative centre, summing to 0.

    ``kernel = get_laplacian_kernel1d(kernel_size)`` returns ``kernel_size`` taps, all ones except the centre tap
    ``1 - kernel_size``, as a ``(kernel_size,)`` tensor.

    Convention:
        - Size 3 is the second difference ``[1, -2, 1]``; a larger size is not a wider second difference, and size 5
          answers 5 to a unit second derivative.
        - The negative centre makes the response positive where the values curve upwards.
        - An integer or bool ``dtype`` that cannot hold ``1 - kernel_size`` raises, so uint8 and bool always do,
          rather than wrap the centre (to 252 for size 5 in uint8)
          (`#5155 <https://github.com/kornia/kornia/issues/5155>`_).

    Args:
        kernel_size: number of taps, odd and at least 3.
        device: device of the kernel.
        dtype: dtype of the kernel.

    Returns:
        the kernel, with shape ``(kernel_size,)``.

    Raises:
        BaseError: if ``kernel_size`` is even, not positive, or ``1``: a single tap is the all-zero kernel.

    See Also:
        - :func:`get_laplacian_kernel2d`: the 2D kernel.
        - :func:`laplacian_1d`: the same kernel without the size checks.

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
    r"""Return a 2D Laplacian kernel: ones with a negative centre, summing to 0.

    ``kernel = get_laplacian_kernel2d(kernel_size)`` returns ``(k_y, k_x)`` taps, all ones except the centre tap
    :math:`1 - k_y k_x`, as a ``(k_y, k_x)`` tensor.

    Convention:
        See the Convention block on :func:`~kornia.filters.get_laplacian_kernel1d`. Size 3 is the 8-neighbour
        stencil, which answers 3 to a unit :math:`\partial_{xx}` or :math:`\partial_{yy}`, so it estimates
        :math:`3 \nabla^2`; size 5 estimates :math:`25 \nabla^2`. :ref:`Filtering <filtering-conventions>` compares
        it with the scipy, OpenCV and scikit-image Laplacians.

    Args:
        kernel_size: size ``(k_y, k_x)``, or one integer for both; odd, and at least 3 along one axis.
        device: device of the kernel.
        dtype: dtype of the kernel.

    Returns:
        the kernel, with shape ``(k_y, k_x)``.

    Raises:
        BaseError: if a size is even or not positive, if ``kernel_size`` is a sequence of other than 2 sizes, or if
            it is ``1`` or ``(1, 1)``: a :math:`1 \times 1` kernel is the all-zero kernel.

    See Also:
        - :func:`get_laplacian_kernel1d`: the 1D kernel.
        - :func:`laplacian`: the filter that applies it, normalized by default.

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
    _check_kernel_dtype(dtype, "The Laplacian kernel", (1 - ky * kx, 1))

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
    """Return a 2D binomial (Pascal) kernel, normalized by default.

    ``kernel = get_pascal_kernel_2d(kernel_size)`` returns the outer product of two Pascal-triangle rows, as a
    ``(k_y, k_x)`` tensor for ``kernel_size=(k_y, k_x)``.

    Args:
        kernel_size: size ``(k_y, k_x)``, or one integer for both.
        norm: whether to divide the kernel by its sum.
        device: device of the kernel.
        dtype: dtype of the kernel.

    Returns:
        the kernel, with shape ``(k_y, k_x)``.

    See Also:
        - :func:`get_pascal_kernel_1d`: one row of the triangle.
        - :func:`blur_pool2d`: the anti-aliased downsampling that applies it.

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
    ax = get_pascal_kernel_1d(kx, norm=norm, device=device, dtype=dtype)
    ay = get_pascal_kernel_1d(ky, norm=norm, device=device, dtype=dtype)

    return ay[:, None] * ax[None, :]


def get_pascal_kernel_1d(
    kernel_size: int, norm: bool = False, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Return a row of Pascal's triangle as a 1D binomial kernel.

    ``kernel = get_pascal_kernel_1d(kernel_size)`` returns the binomial coefficients of ``kernel_size`` taps, as a
    ``(kernel_size,)`` tensor. ``norm`` defaults to ``False`` here and to ``True`` in :func:`get_pascal_kernel_2d`.

    Args:
        kernel_size: number of taps.
        norm: whether to divide the kernel by its sum.
        device: device of the kernel.
        dtype: dtype of the kernel.

    Returns:
        the kernel, with shape ``(kernel_size,)``.

    See Also:
        - :func:`get_pascal_kernel_2d`: the 2D kernel.

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

    if norm and (dtype is None or dtype.is_floating_point or dtype.is_complex):
        # Normalize before casting: even the row sum can overflow in the requested dtype.
        total = sum(cur)
        return torch.tensor([value / total for value in cur], device=device, dtype=dtype)

    out = torch.tensor(cur, device=device, dtype=dtype)

    if norm:
        out = out / out.sum()

    return out


def get_canny_nms_kernel(device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Return the eight 3x3 kernels that subtract each neighbour from the centre pixel.

    ``kernels = get_canny_nms_kernel()`` returns a ``(8, 1, 3, 3)`` tensor with ``1`` at the centre and ``-1`` at one
    neighbour per kernel. :func:`canny` does not use it: it compares the neighbours by slicing, so that ties compare
    exactly.
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
    """Return the eight 3x3 kernels that select each neighbour of the centre pixel.

    ``kernels = get_hysteresis_kernel()`` returns a ``(8, 1, 3, 3)`` tensor with a single ``1`` at one neighbour per
    kernel, which :func:`canny` correlates with its edge map in the hysteresis step.
    """
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
    r"""Return a symmetric 1D Hanning (Hann) window.

    ``window = get_hanning_kernel1d(kernel_size)`` returns the window of :math:`M` = ``kernel_size`` taps, as a
    ``(kernel_size,)`` tensor:

    .. math::  w(n) = 0.5 - 0.5 \cos\left(\frac{2\pi n}{M - 1}\right) \qquad 0 \leq n \leq M - 1

    Convention:
        Both end taps are 0, so only :math:`M - 2` samples carry weight, and the window is not normalized, its taps
        summing to :math:`(M - 1) / 2`. An even size is accepted and is symmetric about :math:`(M - 1) / 2`.
        :ref:`Filtering <filtering-conventions>` names the matching numpy and torch windows.

    Args:
        kernel_size: number of taps, an integer greater than 2.
        device: device of the kernel.
        dtype: dtype of the kernel.

    Returns:
        the window, with shape ``(kernel_size,)``.

    See Also:
        - :func:`get_hanning_kernel2d`: the 2D window.

    Examples:
        >>> get_hanning_kernel1d(4).round(decimals=4)
        tensor([0.0000, 0.7500, 0.7500, 0.0000])

    """
    _check_kernel_size(kernel_size, 2, allow_even=True)

    x = torch.arange(kernel_size, device=device, dtype=dtype)
    if dtype is None:
        # The int64 ramp is converted to the default dtype explicitly, as the multiplication below promoted it on the
        # CPU, without the int64-tensor-times-float ``aten.mul.Scalar`` that the torch 2.14 dynamo ONNX exporter cannot
        # translate with onnxscript 0.7.1.
        x = x.to(torch.get_default_dtype())
    return 0.5 - 0.5 * torch.cos(2.0 * math.pi * x / float(kernel_size - 1))


def get_hanning_kernel2d(
    kernel_size: tuple[int, int] | int,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Return a 2D Hanning (Hann) window as the outer product of two 1D windows.

    ``window = get_hanning_kernel2d(kernel_size)`` multiplies the :func:`get_hanning_kernel1d` windows of
    ``kernel_size=(k_y, k_x)``, along ``H`` and ``W``, and returns a ``(k_y, k_x)`` tensor.

    Convention:
        See the Convention block on :func:`~kornia.filters.get_hanning_kernel1d`.

    Args:
        kernel_size: size ``(k_y, k_x)``, or one integer for both; each greater than 2.
        device: device of the kernel.
        dtype: dtype of the kernel.

    Returns:
        the window, with shape ``(k_y, k_x)``.

    See Also:
        - :func:`get_hanning_kernel1d`: the 1D window.

    """
    kernel_size = _unpack_2d_ks(kernel_size)
    _check_kernel_size(kernel_size, 2, allow_even=True)

    ky = get_hanning_kernel1d(kernel_size[0], device, dtype)[None].T
    kx = get_hanning_kernel1d(kernel_size[1], device, dtype)[None]
    return ky @ kx
