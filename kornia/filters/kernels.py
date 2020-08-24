from typing import Tuple, List, Union

import torch
import torch.nn as nn

from kornia.geometry.transform.affwarp import rotate


def normalize_kernel2d(input: torch.Tensor) -> torch.Tensor:
    r"""Normalizes both derivative and smoothing kernel.
    """
    if len(input.size()) < 2:
        raise TypeError("input should be at least 2D tensor. Got {}"
                        .format(input.size()))
    norm: torch.Tensor = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))


def gaussian(window_size, sigma):
    x = torch.arange(window_size).float() - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp((-x.pow(2.0) / float(2 * sigma ** 2)))
    return gauss / gauss.sum()


def laplacian_1d(window_size) -> torch.Tensor:
    r"""One could also use the Laplacian of Gaussian formula
        to design the filter.
    """

    filter_1d = torch.ones(window_size)
    filter_1d[window_size // 2] = 1 - window_size
    laplacian_1d: torch.Tensor = filter_1d
    return laplacian_1d


def get_box_kernel2d(kernel_size: Tuple[int, int]) -> torch.Tensor:
    r"""Utility function that returns a box filter."""
    kx: float = float(kernel_size[0])
    ky: float = float(kernel_size[1])
    scale: torch.Tensor = torch.tensor(1.) / torch.tensor([kx * ky])
    tmp_kernel: torch.Tensor = torch.ones(1, kernel_size[0], kernel_size[1])
    return scale.to(tmp_kernel.dtype) * tmp_kernel


def get_binary_kernel2d(window_size: Tuple[int, int]) -> torch.Tensor:
    r"""Creates a binary kernel to extract the patches. If the window size
    is HxW will create a (H*W)xHxW kernel.
    """
    window_range: int = window_size[0] * window_size[1]
    kernel: torch.Tensor = torch.zeros(window_range, window_range)
    for i in range(window_range):
        kernel[i, i] += 1.0
    return kernel.view(window_range, 1, window_size[0], window_size[1])


def get_sobel_kernel_3x3() -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3"""
    return torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.],
    ])


def get_sobel_kernel_5x5_2nd_order() -> torch.Tensor:
    """Utility function that returns a 2nd order sobel kernel of 5x5"""
    return torch.tensor([
        [-1., 0., 2., 0., -1.],
        [-4., 0., 8., 0., -4.],
        [-6., 0., 12., 0., -6.],
        [-4., 0., 8., 0., -4.],
        [-1., 0., 2., 0., -1.]
    ])


def _get_sobel_kernel_5x5_2nd_order_xy() -> torch.Tensor:
    """Utility function that returns a 2nd order sobel kernel of 5x5"""
    return torch.tensor([
        [-1., -2., 0., 2., 1.],
        [-2., -4., 0., 4., 2.],
        [0., 0., 0., 0., 0.],
        [2., 4., 0., -4., -2.],
        [1., 2., 0., -2., -1.]
    ])


def get_diff_kernel_3x3() -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3"""
    return torch.tensor([
        [-0., 0., 0.],
        [-1., 0., 1.],
        [-0., 0., 0.],
    ])


def get_diff_kernel3d(device=torch.device('cpu'), dtype=torch.float) -> torch.Tensor:
    """Utility function that returns a first order derivative kernel of 3x3x3"""
    kernel: torch.Tensor = torch.tensor([[[[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]],

                                          [[0.0, 0.0, 0.0],
                                           [-0.5, 0.0, 0.5],
                                           [0.0, 0.0, 0.0]],

                                          [[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]],
                                          ],
                                         [[[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]],

                                          [[0.0, -0.5, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.5, 0.0]],

                                          [[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]],
                                          ],
                                         [[[0.0, 0.0, 0.0],
                                           [0.0, -0.5, 0.0],
                                           [0.0, 0.0, 0.0]],

                                          [[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]],

                                          [[0.0, 0.0, 0.0],
                                           [0.0, 0.5, 0.0],
                                           [0.0, 0.0, 0.0]],
                                          ],
                                         ], device=device, dtype=dtype)
    return kernel.unsqueeze(1)


def get_diff_kernel3d_2nd_order(device=torch.device('cpu'), dtype=torch.float) -> torch.Tensor:
    """Utility function that returns a first order derivative kernel of 3x3x3"""
    kernel: torch.Tensor = torch.tensor([[[[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]],

                                          [[0.0, 0.0, 0.0],
                                           [1.0, -2.0, 1.0],
                                           [0.0, 0.0, 0.0]],

                                          [[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]],
                                          ],
                                         [[[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]],

                                          [[0.0, 1.0, 0.0],
                                           [0.0, -2.0, 0.0],
                                           [0.0, 1.0, 0.0]],

                                          [[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]],
                                          ],
                                         [[[0.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0],
                                           [0.0, 0.0, 0.0]],

                                          [[0.0, 0.0, 0.0],
                                           [0.0, -2.0, 0.0],
                                           [0.0, 0.0, 0.0]],

                                          [[0.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0],
                                           [0.0, 0.0, 0.0]],
                                          ],
                                         [[[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]],

                                          [[1.0, 0.0, -1.0],
                                           [0.0, 0.0, 0.0],
                                           [-1.0, 0.0, 1.0]],

                                          [[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]],
                                          ],
                                         [[[0.0, 1.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, -1.0, 0.0]],

                                          [[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]],

                                          [[0.0, -1.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0]],
                                          ],
                                         [[[0.0, 0.0, 0.0],
                                           [1.0, 0.0, -1.0],
                                           [0.0, 0.0, 0.0]],

                                          [[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]],

                                          [[0.0, 0.0, 0.0],
                                           [-1.0, 0.0, 1.0],
                                           [0.0, 0.0, 0.0]],
                                          ],
                                         ], device=device, dtype=dtype)
    return kernel.unsqueeze(1)


def get_sobel_kernel2d() -> torch.Tensor:
    kernel_x: torch.Tensor = get_sobel_kernel_3x3()
    kernel_y: torch.Tensor = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])


def get_diff_kernel2d() -> torch.Tensor:
    kernel_x: torch.Tensor = get_diff_kernel_3x3()
    kernel_y: torch.Tensor = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])


def get_sobel_kernel2d_2nd_order() -> torch.Tensor:
    gxx: torch.Tensor = get_sobel_kernel_5x5_2nd_order()
    gyy: torch.Tensor = gxx.transpose(0, 1)
    gxy: torch.Tensor = _get_sobel_kernel_5x5_2nd_order_xy()
    return torch.stack([gxx, gxy, gyy])


def get_diff_kernel2d_2nd_order() -> torch.Tensor:
    gxx: torch.Tensor = torch.tensor([
        [0., 0., 0.],
        [1., -2., 1.],
        [0., 0., 0.],
    ])
    gyy: torch.Tensor = gxx.transpose(0, 1)
    gxy: torch.Tensor = torch.tensor([
        [-1., 0., 1.],
        [0., 0., 0.],
        [1., 0., -1.],
    ])
    return torch.stack([gxx, gxy, gyy])


def get_spatial_gradient_kernel2d(mode: str, order: int) -> torch.Tensor:
    r"""Function that returns kernel for 1st or 2nd order image gradients,
    using one of the following operators: sobel, diff"""
    if mode not in ['sobel', 'diff']:
        raise TypeError("mode should be either sobel\
                         or diff. Got {}".format(mode))
    if order not in [1, 2]:
        raise TypeError("order should be either 1 or 2\
                         Got {}".format(order))
    if mode == 'sobel' and order == 1:
        kernel: torch.Tensor = get_sobel_kernel2d()
    elif mode == 'sobel' and order == 2:
        kernel = get_sobel_kernel2d_2nd_order()
    elif mode == 'diff' and order == 1:
        kernel = get_diff_kernel2d()
    elif mode == 'diff' and order == 2:
        kernel = get_diff_kernel2d_2nd_order()
    else:
        raise NotImplementedError("")
    return kernel


def get_spatial_gradient_kernel3d(mode: str, order: int, device=torch.device('cpu'), dtype=torch.float) -> torch.Tensor:
    r"""Function that returns kernel for 1st or 2nd order scale pyramid gradients,
    using one of the following operators: sobel, diff"""
    if mode not in ['sobel', 'diff']:
        raise TypeError("mode should be either sobel\
                         or diff. Got {}".format(mode))
    if order not in [1, 2]:
        raise TypeError("order should be either 1 or 2\
                         Got {}".format(order))
    if mode == 'sobel':
        raise NotImplementedError("Sobel kernel for 3d gradient is not implemented yet")
    elif mode == 'diff' and order == 1:
        kernel = get_diff_kernel3d(device, dtype)
    elif mode == 'diff' and order == 2:
        kernel = get_diff_kernel3d_2nd_order(device, dtype)
    else:
        raise NotImplementedError("")
    return kernel


def get_gaussian_kernel1d(kernel_size: int,
                          sigma: float,
                          force_even: bool = False) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients.

    Args:
        kernel_size (int): filter size. It should be odd and positive.
        sigma (float): gaussian standard deviation.
        force_even (bool): overrides requirement for odd kernel size.

    Returns:
        Tensor: 1D tensor with gaussian filter coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size})`

    Examples::

        >>> kornia.image.get_gaussian_kernel(3, 2.5)
        tensor([0.3243, 0.3513, 0.3243])

        >>> kornia.image.get_gaussian_kernel(5, 1.5)
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """
    if (not isinstance(kernel_size, int) or (
            (kernel_size % 2 == 0) and not force_even) or (
            kernel_size <= 0)):
        raise TypeError(
            "kernel_size must be an odd positive integer. "
            "Got {}".format(kernel_size)
        )
    window_1d: torch.Tensor = gaussian(kernel_size, sigma)
    return window_1d


def get_gaussian_kernel2d(
        kernel_size: Tuple[int, int],
        sigma: Tuple[float, float],
        force_even: bool = False) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size (Tuple[int, int]): filter sizes in the x and y direction.
         Sizes should be odd and positive.
        sigma (Tuple[int, int]): gaussian standard deviation in the x and y
         direction.
        force_even (bool): overrides requirement for odd kernel size.

    Returns:
        Tensor: 2D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples::

        >>> kornia.image.get_gaussian_kernel2d((3, 3), (1.5, 1.5))
        tensor([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])

        >>> kornia.image.get_gaussian_kernel2d((3, 5), (1.5, 1.5))
        tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
    """
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError(
            "kernel_size must be a tuple of length two. Got {}".format(
                kernel_size
            )
        )
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError(
            "sigma must be a tuple of length two. Got {}".format(sigma)
        )
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel1d(ksize_x, sigma_x, force_even)
    kernel_y: torch.Tensor = get_gaussian_kernel1d(ksize_y, sigma_y, force_even)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t()
    )
    return kernel_2d


def get_laplacian_kernel1d(kernel_size: int) -> torch.Tensor:
    r"""Function that returns the coefficients of a 1D Laplacian filter.

    Args:
        kernel_size (int): filter size. It should be odd and positive.

    Returns:
        Tensor (float): 1D tensor with laplacian filter coefficients.

    Shape:
        - Output: math:`(\text{kernel_size})`

    Examples::
        >>> kornia.image.get_laplacian_kernel(3)
        tensor([ 1., -2.,  1.])

        >>> kornia.image.get_laplacian_kernel(5)
        tensor([ 1.,  1., -4.,  1.,  1.])

    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or \
            kernel_size <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}"
                        .format(kernel_size))
    window_1d: torch.Tensor = laplacian_1d(kernel_size)
    return window_1d


def get_laplacian_kernel2d(kernel_size: int) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size (int): filter size should be odd.

    Returns:
        Tensor: 2D tensor with laplacian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples::

        >>> kornia.image.get_laplacian_kernel2d(3)
        tensor([[ 1.,  1.,  1.],
                [ 1., -8.,  1.],
                [ 1.,  1.,  1.]])

        >>> kornia.image.get_laplacian_kernel2d(5)
        tensor([[  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1., -24.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.]])

    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or \
            kernel_size <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}"
                        .format(kernel_size))

    kernel = torch.ones((kernel_size, kernel_size))
    mid = kernel_size // 2
    kernel[mid, mid] = 1 - kernel_size ** 2
    kernel_2d: torch.Tensor = kernel
    return kernel_2d


def get_motion_kernel2d(kernel_size: int, angle: Union[torch.Tensor, float],
                        direction: Union[torch.Tensor, float] = 0.) -> torch.Tensor:
    r"""Function that returns motion blur filter.

    Args:
        kernel_size (int): motion kernel width and height. It should be odd and positive.
        angle (torch.Tensor, float): angle of the motion blur in degrees (anti-clockwise rotation).
        direction (float): forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle),
            while higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a
            uniformly (but still angled) motion blur.

    Returns:
        torch.Tensor: the motion blur kernel.

    Shape:
        - Output: :math:`(ksize, ksize)`

    Examples::
        >>> kornia.filters.get_motion_kernel2d(5, 0., 0.)
        tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
        >>> kornia.filters.get_motion_kernel2d(3, 215., -0.5)
            tensor([[0.0000, 0.0412, 0.0732],
                    [0.1920, 0.3194, 0.0804],
                    [0.2195, 0.0743, 0.0000]])
    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or kernel_size < 3:
        raise TypeError("ksize must be an odd integer >= than 3")

    if not isinstance(angle, torch.Tensor):
        angle = torch.tensor(angle)

    assert angle.dim() == 0, f"angle must be a 0-dim tensor. Got {angle}."

    if not isinstance(direction, torch.Tensor):
        direction = torch.tensor(direction)

    assert direction.dim() == 0, f"direction must be a 0-dim tensor. Got {direction}."

    kernel_tuple: Tuple[int, int] = (kernel_size, kernel_size)
    # direction from [-1, 1] to [0, 1] range
    direction = (torch.clamp(direction, -1., 1.).item() + 1.) / 2.
    kernel = torch.zeros(kernel_tuple, dtype=torch.float)
    kernel[kernel_tuple[0] // 2, :] = torch.linspace(direction, 1. - direction, steps=kernel_tuple[0])
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    # rotate (counterclockwise) kernel by given angle
    kernel = rotate(kernel, angle)
    kernel = kernel[0][0]
    kernel = kernel / kernel.sum()
    return kernel
