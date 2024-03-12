from __future__ import annotations

import math
from math import sqrt
from typing import Any, Optional, Union

import torch

from kornia.core import Device, Dtype, Tensor, concatenate, cos, stack, tensor, where, zeros, zeros_like
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.utils import deprecated


def _check_kernel_size(kernel_size: tuple[int, ...] | int, min_value: int = 0, allow_even: bool = False) -> None:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,)

    fmt = "even or odd" if allow_even else "odd"
    for size in kernel_size:
        KORNIA_CHECK(
            isinstance(size, int) and (((size % 2 == 1) or allow_even) and size > min_value),
            f"Kernel size must be an {fmt} integer bigger than {min_value}. Gotcha {size} on {kernel_size}",
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


def normalize_kernel2d(input: Tensor) -> Tensor:
    r"""Normalize both derivative and smoothing kernel."""
    KORNIA_CHECK_SHAPE(input, ["*", "H", "W"])

    norm = input.abs().sum(dim=-1).sum(dim=-1)

    return input / (norm[..., None, None])


def gaussian(
    window_size: int,
    sigma: Tensor | float,
    *,
    mean: Optional[Union[Tensor, float]] = None,
    device: Optional[Device] = None,
    dtype: Optional[Dtype] = None,
) -> Tensor:
    """Compute the gaussian values based on the window and sigma values.

    Args:
        window_size: the size which drives the filter amount.
        sigma: gaussian standard deviation. If a tensor, should be in a shape :math:`(B, 1)`
        mean: Mean of the Gaussian function (center). If not provided, it defaults to window_size // 2.
        If a tensor, should be in a shape :math:`(B, 1)`
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. Dtype desired for compute.
    Returns:
        A tensor withshape :math:`(B, \text{kernel_size})`, with Gaussian values.
    """

    if isinstance(sigma, float):
        sigma = tensor([[sigma]], device=device, dtype=dtype)

    KORNIA_CHECK_IS_TENSOR(sigma)
    KORNIA_CHECK_SHAPE(sigma, ["B", "1"])
    batch_size = sigma.shape[0]

    mean = float(window_size // 2) if mean is None else mean
    if isinstance(mean, float):
        mean = tensor([[mean]], device=sigma.device, dtype=sigma.dtype)

    KORNIA_CHECK_IS_TENSOR(mean)
    KORNIA_CHECK_SHAPE(mean, ["B", "1"])

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - mean).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def gaussian_discrete_erf(
    window_size: int, sigma: Tensor | float, *, device: Optional[Device] = None, dtype: Optional[Dtype] = None
) -> Tensor:
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
        sigma = tensor([[sigma]], device=device, dtype=dtype)

    KORNIA_CHECK_SHAPE(sigma, ["B", "1"])
    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    t = 0.70710678 / sigma.abs()
    # t = tensor(2, device=sigma.device, dtype=sigma.dtype).sqrt() / (sigma.abs() * 2)

    gauss = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())
    gauss = gauss.clamp(min=0)

    return gauss / gauss.sum(-1, keepdim=True)


def _modified_bessel_0(x: Tensor) -> Tensor:
    r"""Adapted from:

    https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    """
    ax = torch.abs(x)

    out = zeros_like(x)
    idx_a = ax < 3.75

    if idx_a.any():
        y = (x[idx_a] / 3.75) * (x[idx_a] / 3.75)
        out[idx_a] = 1.0 + y * (
            3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2))))
        )

    idx_b = ~idx_a
    if idx_b.any():
        y = 3.75 / ax[idx_b]
        ans = 0.916281e-2 + y * (-0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1 + y * 0.392377e-2)))
        coef = 0.39894228 + y * (0.1328592e-1 + y * (0.225319e-2 + y * (-0.157565e-2 + y * ans)))
        out[idx_b] = (ax[idx_b].exp() / ax[idx_b].sqrt()) * coef

    return out


def _modified_bessel_1(x: Tensor) -> Tensor:
    r"""Adapted from:

    https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    """
    ax = torch.abs(x)

    out = zeros_like(x)
    idx_a = ax < 3.75

    if idx_a.any():
        y = (x[idx_a] / 3.75) * (x[idx_a] / 3.75)
        ans = 0.51498869 + y * (0.15084934 + y * (0.2658733e-1 + y * (0.301532e-2 + y * 0.32411e-3)))
        out[idx_a] = ax[idx_a] * (0.5 + y * (0.87890594 + y * ans))

    idx_b = ~idx_a
    if idx_b.any():
        y = 3.75 / ax[idx_b]
        ans = 0.2282967e-1 + y * (-0.2895312e-1 + y * (0.1787654e-1 - y * 0.420059e-2))
        ans = 0.39894228 + y * (-0.3988024e-1 + y * (-0.362018e-2 + y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans))))
        ans = ans * ax[idx_b].exp() / ax[idx_b].sqrt()
        out[idx_b] = where(x[idx_b] < 0, -ans, ans)

    return out


def _modified_bessel_i(n: int, x: Tensor) -> Tensor:
    r"""Adapted from:

    https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    """
    KORNIA_CHECK(n >= 2, "n must be greater than 1.99")

    if (x == 0.0).all():
        return x

    batch_size = x.shape[0]

    tox = 2.0 / x.abs()
    ans = zeros(batch_size, 1, device=x.device, dtype=x.dtype)
    bip = zeros(batch_size, 1, device=x.device, dtype=x.dtype)
    bi = torch.ones(batch_size, 1, device=x.device, dtype=x.dtype)

    m = int(2 * (n + int(sqrt(40.0 * n))))
    for j in range(m, 0, -1):
        bim = bip + float(j) * tox * bi
        bip = bi
        bi = bim
        idx = bi.abs() > 1.0e10

        if idx.any():
            ans[idx] = ans[idx] * 1.0e-10
            bi[idx] = bi[idx] * 1.0e-10
            bip[idx] = bip[idx] * 1.0e-10

        if j == n:
            ans = bip

    out = ans * _modified_bessel_0(x) / bi

    if (n % 2) == 1:
        out = where(x < 0.0, -out, out)

    # TODO: skip the previous computation for x == 0, instead of forcing here
    out = where(x == 0.0, x, out)

    return out


def gaussian_discrete(
    window_size: int, sigma: Tensor | float, *, device: Optional[Device] = None, dtype: Optional[Dtype] = None
) -> Tensor:
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
        sigma = tensor([[sigma]], device=device, dtype=dtype)

    KORNIA_CHECK_SHAPE(sigma, ["B", "1"])

    sigma2 = sigma * sigma
    tail = int(window_size // 2) + 1
    bessels = [
        _modified_bessel_0(sigma2),
        _modified_bessel_1(sigma2),
        *(_modified_bessel_i(k, sigma2) for k in range(2, tail)),
    ]
    # NOTE: on monain is exp(-sig)
    # https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/layers/convutils.py#L128
    out = concatenate(bessels[:0:-1] + bessels, -1) * sigma2.exp()

    return out / out.sum(-1, keepdim=True)


def laplacian_1d(window_size: int, *, device: Optional[Device] = None, dtype: Dtype = torch.float32) -> Tensor:
    """One could also use the Laplacian of Gaussian formula to design the filter."""
    # TODO: add default dtype as None when kornia relies on torch > 1.12
    filter_1d = torch.ones(window_size, device=device, dtype=dtype)
    middle = window_size // 2
    filter_1d[middle] = 1 - window_size
    return filter_1d


def get_box_kernel1d(kernel_size: int, *, device: Optional[Device] = None, dtype: Optional[Dtype] = None) -> Tensor:
    r"""Utility function that returns a 1-D box filter.

    Args:
        kernel_size: the size of the kernel.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.
    Returns:
        A tensor with shape :math:`(1, \text{kernel\_size})`, filled with the value
        :math:`\frac{1}{\text{kernel\_size}}`.
    """
    scale = tensor(1.0 / kernel_size, device=device, dtype=dtype)
    return scale.expand(1, kernel_size)


def get_box_kernel2d(
    kernel_size: tuple[int, int] | int, *, device: Optional[Device] = None, dtype: Optional[Dtype] = None
) -> Tensor:
    r"""Utility function that returns a 2-D box filter.

    Args:
        kernel_size: the size of the kernel.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.
    Returns:
        A tensor with shape :math:`(1, \text{kernel\_size}[0], \text{kernel\_size}[1])`,
        filled with the value :math:`\frac{1}{\text{kernel\_size}[0] \times \text{kernel\_size}[1]}`.
    """
    ky, kx = _unpack_2d_ks(kernel_size)
    scale = tensor(1.0 / (kx * ky), device=device, dtype=dtype)
    return scale.expand(1, ky, kx)


def get_binary_kernel2d(
    window_size: tuple[int, int] | int, *, device: Optional[Device] = None, dtype: Dtype = torch.float32
) -> Tensor:
    """Create a binary kernel to extract the patches.

    If the window size is HxW will create a (H*W)x1xHxW kernel.
    """
    # TODO: add default dtype as None when kornia relies on torch > 1.12

    ky, kx = _unpack_2d_ks(window_size)

    window_range = kx * ky

    kernel = zeros((window_range, window_range), device=device, dtype=dtype)
    idx = torch.arange(window_range, device=device)
    kernel[idx, idx] += 1.0
    return kernel.view(window_range, 1, ky, kx)


def get_sobel_kernel_3x3(*, device: Optional[Device] = None, dtype: Optional[Dtype] = None) -> Tensor:
    """Utility function that returns a sobel kernel of 3x3."""
    return tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device, dtype=dtype)


def get_sobel_kernel_5x5_2nd_order(*, device: Optional[Device] = None, dtype: Optional[Dtype] = None) -> Tensor:
    """Utility function that returns a 2nd order sobel kernel of 5x5."""
    return tensor(
        [
            [-1.0, 0.0, 2.0, 0.0, -1.0],
            [-4.0, 0.0, 8.0, 0.0, -4.0],
            [-6.0, 0.0, 12.0, 0.0, -6.0],
            [-4.0, 0.0, 8.0, 0.0, -4.0],
            [-1.0, 0.0, 2.0, 0.0, -1.0],
        ],
        device=device,
        dtype=dtype,
    )


def _get_sobel_kernel_5x5_2nd_order_xy(*, device: Optional[Device] = None, dtype: Optional[Dtype] = None) -> Tensor:
    """Utility function that returns a 2nd order sobel kernel of 5x5."""
    return tensor(
        [
            [-1.0, -2.0, 0.0, 2.0, 1.0],
            [-2.0, -4.0, 0.0, 4.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 4.0, 0.0, -4.0, -2.0],
            [1.0, 2.0, 0.0, -2.0, -1.0],
        ],
        device=device,
        dtype=dtype,
    )


def get_diff_kernel_3x3(*, device: Optional[Device] = None, dtype: Optional[Dtype] = None) -> Tensor:
    """Utility function that returns a first order derivative kernel of 3x3."""
    return tensor([[-0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [-0.0, 0.0, 0.0]], device=device, dtype=dtype)


def get_diff_kernel3d(device: Optional[Device] = None, dtype: Optional[Dtype] = None) -> Tensor:
    """Utility function that returns a first order derivative kernel of 3x3x3."""
    kernel = tensor(
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


def get_diff_kernel3d_2nd_order(device: Optional[Device] = None, dtype: Optional[Dtype] = None) -> Tensor:
    """Utility function that returns a first order derivative kernel of 3x3x3."""
    kernel = tensor(
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
                [[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, -1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            ],
        ],
        device=device,
        dtype=dtype,
    )
    return kernel[:, None, ...]


def get_sobel_kernel2d(*, device: Optional[Device] = None, dtype: Optional[Dtype] = None) -> Tensor:
    kernel_x = get_sobel_kernel_3x3(device=device, dtype=dtype)
    kernel_y = kernel_x.transpose(0, 1)
    return stack([kernel_x, kernel_y])


def get_diff_kernel2d(*, device: Optional[Device] = None, dtype: Optional[Dtype] = None) -> Tensor:
    kernel_x = get_diff_kernel_3x3(device=device, dtype=dtype)
    kernel_y = kernel_x.transpose(0, 1)
    return stack([kernel_x, kernel_y])


def get_sobel_kernel2d_2nd_order(*, device: Optional[Device] = None, dtype: Optional[Dtype] = None) -> Tensor:
    gxx = get_sobel_kernel_5x5_2nd_order(device=device, dtype=dtype)
    gyy = gxx.transpose(0, 1)
    gxy = _get_sobel_kernel_5x5_2nd_order_xy(device=device, dtype=dtype)
    return stack([gxx, gxy, gyy])


def get_diff_kernel2d_2nd_order(*, device: Optional[Device] = None, dtype: Optional[Dtype] = None) -> Tensor:
    gxx = tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]], device=device, dtype=dtype)
    gyy = gxx.transpose(0, 1)
    gxy = tensor([[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, -1.0]], device=device, dtype=dtype)
    return stack([gxx, gxy, gyy])


def get_spatial_gradient_kernel2d(
    mode: str, order: int, *, device: Optional[Device] = None, dtype: Optional[Dtype] = None
) -> Tensor:
    r"""Function that returns kernel for 1st or 2nd order image gradients, using one of the following operators:

    sobel, diff.
    """
    KORNIA_CHECK(mode.lower() in {"sobel", "diff"}, f"Mode should be `sobel` or `diff`. Got {mode}")
    KORNIA_CHECK(order in {1, 2}, f"Order should be 1 or 2. Got {order}")

    if mode == "sobel" and order == 1:
        kernel: Tensor = get_sobel_kernel2d(device=device, dtype=dtype)
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
    mode: str, order: int, device: Optional[Device] = None, dtype: Optional[Dtype] = None
) -> Tensor:
    r"""Function that returns kernel for 1st or 2nd order scale pyramid gradients, using one of the following
    operators: sobel, diff."""
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
    sigma: float | Tensor,
    force_even: bool = False,
    *,
    device: Optional[Device] = None,
    dtype: Optional[Dtype] = None,
) -> Tensor:
    r"""Function that returns Gaussian filter coefficients.

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
    """
    _check_kernel_size(kernel_size, allow_even=force_even)

    return gaussian(kernel_size, sigma, device=device, dtype=dtype)


def get_gaussian_discrete_kernel1d(
    kernel_size: int,
    sigma: float | Tensor,
    force_even: bool = False,
    *,
    device: Optional[Device] = None,
    dtype: Optional[Dtype] = None,
) -> Tensor:
    """Function that returns Gaussian filter coefficients based on the modified Bessel functions.

    Adapted from: https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py.

    Args:
        kernel_size: filter size. It should be odd and positive.
        sigma: gaussian standard deviation. If a tensor, should be in a shape :math:`(B, 1)`
        force_even: overrides requirement for odd kernel size.
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. Dtype desired for compute.

    Returns:
        1D tensor with gaussian filter coefficients. With shape :math:`(B, \text{kernel_size})`

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
    sigma: float | Tensor,
    force_even: bool = False,
    *,
    device: Optional[Device] = None,
    dtype: Optional[Dtype] = None,
) -> Tensor:
    """Function that returns Gaussian filter coefficients by interpolating the error function.

    Adapted from: https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py.

    Args:
        kernel_size: filter size. It should be odd and positive.
        sigma: gaussian standard deviation. If a tensor, should be in a shape :math:`(B, 1)`
        force_even: overrides requirement for odd kernel size.
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. Dtype desired for compute.

    Returns:
        1D tensor with gaussian filter coefficients. Shape :math:`(B, \text{kernel_size})`

    Examples:
        >>> get_gaussian_erf_kernel1d(3, 2.5)
        tensor([[0.3245, 0.3511, 0.3245]])
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
    sigma: tuple[float, float] | Tensor,
    force_even: bool = False,
    *,
    device: Optional[Device] = None,
    dtype: Optional[Dtype] = None,
) -> Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size: filter sizes in the y and x direction. Sizes should be odd and positive.
        sigma: gaussian standard deviation in the y and x.
        force_even: overrides requirement for odd kernel size.
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. Dtype desired for compute.

    Returns:
        2D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(B, \text{kernel_size}_x, \text{kernel_size}_y)`

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
        sigma = tensor([sigma], device=device, dtype=dtype)

    KORNIA_CHECK_IS_TENSOR(sigma)
    KORNIA_CHECK_SHAPE(sigma, ["B", "2"])

    ksize_y, ksize_x = _unpack_2d_ks(kernel_size)
    sigma_y, sigma_x = sigma[:, 0, None], sigma[:, 1, None]

    kernel_y = get_gaussian_kernel1d(ksize_y, sigma_y, force_even, device=device, dtype=dtype)[..., None]
    kernel_x = get_gaussian_kernel1d(ksize_x, sigma_x, force_even, device=device, dtype=dtype)[..., None]

    return kernel_y * kernel_x.view(-1, 1, ksize_x)


def get_gaussian_kernel3d(
    kernel_size: tuple[int, int, int] | int,
    sigma: tuple[float, float, float] | Tensor,
    force_even: bool = False,
    *,
    device: Optional[Device] = None,
    dtype: Optional[Dtype] = None,
) -> Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size: filter sizes in the z, y and x direction. Sizes should be odd and positive.
        sigma: gaussian standard deviation in the z, y and x direction.
        force_even: overrides requirement for odd kernel size.
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. Dtype desired for compute.

    Returns:
        3D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(B, \text{kernel_size}_x, \text{kernel_size}_y,  \text{kernel_size}_z)`

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
        >>> get_gaussian_kernel3d((3, 3, 3), (1.5, 1.5, 1.5)).sum()
        tensor(1.)
        >>> get_gaussian_kernel3d((3, 3, 3), (1.5, 1.5, 1.5)).shape
        torch.Size([1, 3, 3, 3])
        >>> get_gaussian_kernel3d((3, 7, 5), torch.tensor([[1.5, 1.5, 1.5]])).shape
        torch.Size([1, 3, 7, 5])
    """
    if isinstance(sigma, tuple):
        sigma = tensor([sigma], device=device, dtype=dtype)

    KORNIA_CHECK_IS_TENSOR(sigma)
    KORNIA_CHECK_SHAPE(sigma, ["B", "3"])

    ksize_z, ksize_y, ksize_x = _unpack_3d_ks(kernel_size)
    sigma_z, sigma_y, sigma_x = sigma[:, 0, None], sigma[:, 1, None], sigma[:, 2, None]

    kernel_z = get_gaussian_kernel1d(ksize_z, sigma_z, force_even, device=device, dtype=dtype)
    kernel_y = get_gaussian_kernel1d(ksize_y, sigma_y, force_even, device=device, dtype=dtype)
    kernel_x = get_gaussian_kernel1d(ksize_x, sigma_x, force_even, device=device, dtype=dtype)

    return kernel_z.view(-1, ksize_z, 1, 1) * kernel_y.view(-1, 1, ksize_y, 1) * kernel_x.view(-1, 1, 1, ksize_x)


def get_laplacian_kernel1d(
    kernel_size: int, *, device: Optional[Device] = None, dtype: Dtype = torch.float32
) -> Tensor:
    r"""Function that returns the coefficients of a 1D Laplacian filter.

    Args:
        kernel_size: filter size. It should be odd and positive.
        device: tensor device desired to create the kernel
        dtype: tensor dtype desired to create the kernel

    Returns:
        1D tensor with laplacian filter coefficients.

    Shape:
        - Output: math:`(\text{kernel_size})`

    Examples:
        >>> get_laplacian_kernel1d(3)
        tensor([ 1., -2.,  1.])
        >>> get_laplacian_kernel1d(5)
        tensor([ 1.,  1., -4.,  1.,  1.])
    """
    # TODO: add default dtype as None when kornia relies on torch > 1.12

    _check_kernel_size(kernel_size)

    return laplacian_1d(kernel_size, device=device, dtype=dtype)


def get_laplacian_kernel2d(
    kernel_size: tuple[int, int] | int, *, device: Optional[Device] = None, dtype: Dtype = torch.float32
) -> Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size: filter size should be odd.
        device: tensor device desired to create the kernel
        dtype: tensor dtype desired to create the kernel

    Returns:
        2D tensor with laplacian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

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

    kernel = torch.ones((ky, kx), device=device, dtype=dtype)
    mid_x = kx // 2
    mid_y = ky // 2

    kernel[mid_y, mid_x] = 1 - kernel.sum()
    return kernel


def get_pascal_kernel_2d(
    kernel_size: tuple[int, int] | int,
    norm: bool = True,
    *,
    device: Optional[Device] = None,
    dtype: Optional[Dtype] = None,
) -> Tensor:
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
    kernel_size: int, norm: bool = False, *, device: Optional[Device] = None, dtype: Optional[Dtype] = None
) -> Tensor:
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

    out = tensor(cur, device=device, dtype=dtype)

    if norm:
        out = out / out.sum()

    return out


def get_canny_nms_kernel(device: Optional[Device] = None, dtype: Optional[Dtype] = None) -> Tensor:
    """Utility function that returns 3x3 kernels for the Canny Non-maximal suppression."""
    return tensor(
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


def get_hysteresis_kernel(device: Optional[Device] = None, dtype: Optional[Dtype] = None) -> Tensor:
    """Utility function that returns the 3x3 kernels for the Canny hysteresis."""
    return tensor(
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


def get_hanning_kernel1d(kernel_size: int, device: Optional[Device] = None, dtype: Optional[Dtype] = None) -> Tensor:
    """Returns Hanning (also known as Hann) kernel, used in signal processing and KCF tracker.

    .. math::  w(n) = 0.5 - 0.5cos\\left(\\frac{2\\pi{n}}{M-1}\\right)
               \\qquad 0 \\leq n \\leq M-1

    See further in numpy docs https://numpy.org/doc/stable/reference/generated/numpy.hanning.html

    Args:
        kernel_size: The size the of the kernel. It should be positive.
        device: tensor device desired to create the kernel
        dtype: tensor dtype desired to create the kernel

    Returns:
        1D tensor with Hanning filter coefficients. Shape math:`(\text{kernel_size})`
        .. math::  w(n) = 0.5 - 0.5cos\\left(\\frac{2\\pi{n}}{M-1}\\right)

    Examples:
        >>> get_hanning_kernel1d(4)
        tensor([0.0000, 0.7500, 0.7500, 0.0000])
    """
    _check_kernel_size(kernel_size, 2, allow_even=True)

    x = torch.arange(kernel_size, device=device, dtype=dtype)
    x = 0.5 - 0.5 * cos(2.0 * math.pi * x / float(kernel_size - 1))
    return x


def get_hanning_kernel2d(
    kernel_size: tuple[int, int] | int, device: Optional[Device] = None, dtype: Optional[Dtype] = None
) -> Tensor:
    """Returns 2d Hanning kernel, used in signal processing and KCF tracker.

    Args:
        kernel_size: The size of the kernel for the filter. It should be positive.
        device: tensor device desired to create the kernel
        dtype: tensor dtype desired to create the kernel

    Returns:
        2D tensor with Hanning filter coefficients. Shape: math:`(\text{kernel_size[0], kernel_size[1]})`
        .. math::  w(n) = 0.5 - 0.5cos\\left(\\frac{2\\pi{n}}{M-1}\\right)
    """
    kernel_size = _unpack_2d_ks(kernel_size)
    _check_kernel_size(kernel_size, 2, allow_even=True)

    ky = get_hanning_kernel1d(kernel_size[0], device, dtype)[None].T
    kx = get_hanning_kernel1d(kernel_size[1], device, dtype)[None]
    kernel2d = ky @ kx

    return kernel2d


@deprecated(replace_with="get_gaussian_kernel1d", version="6.9.10")
def get_gaussian_kernel1d_t(*args: Any, **kwargs: Any) -> Tensor:
    return get_gaussian_kernel1d(*args, **kwargs)


@deprecated(replace_with="get_gaussian_kernel2d", version="6.9.10")
def get_gaussian_kernel2d_t(*args: Any, **kwargs: Any) -> Tensor:
    return get_gaussian_kernel2d(*args, **kwargs)


@deprecated(replace_with="get_gaussian_kernel3d", version="6.9.10")
def get_gaussian_kernel3d_t(*args: Any, **kwargs: Any) -> Tensor:
    return get_gaussian_kernel3d(*args, **kwargs)
