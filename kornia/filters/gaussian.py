from typing import Tuple

import torch
import torch.nn as nn

from .filter import filter2d, filter2d_separable
from .kernels import get_gaussian_kernel1d_t, get_gaussian_kernel2d_t


def gaussian_blur2d(
    input: torch.Tensor,
    kernel_size: Tuple[int, int],
    sigma: Tuple[float, float],
    border_type: str = 'reflect',
    separable: bool = True,
) -> torch.Tensor:
    r"""Create an operator that blurs a tensor using a Gaussian filter.

    .. image:: _static/img/gaussian_blur2d.png

    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It supports batched operation.

    Arguments:
        input: the input tensor with shape :math:`(B,C,H,W)`.
        kernel_size: the size of the kernel.
        sigma: the standard deviation of the kernel.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        separable: run as composition of two 1d-convolutions.

    Returns:
        the blurred tensor with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       gaussian_blur.html>`__.

    Examples:
        >>> input = torch.rand(2, 4, 5, 5)
        >>> output = gaussian_blur2d(input, (3, 3), (1.5, 1.5))
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """
    device, dtype = input.device, input.dtype
    sigma_t = torch.tensor(sigma, device=device, dtype=dtype).unsqueeze(0)

    return gaussian_blur2d_t(input, kernel_size, sigma_t, border_type, separable)


def gaussian_blur2d_t(
    input: torch.Tensor,
    kernel_size: Tuple[int, int],
    sigma: torch.Tensor,
    border_type: str = 'reflect',
    separable: bool = True,
) -> torch.Tensor:
    r"""Create an operator that blurs a tensor using a Gaussian filter.

    .. image:: _static/img/gaussian_blur2d.png

    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It supports batched operation.

    Arguments:
        input: the input tensor with shape :math:`(B,C,H,W)`.
        kernel_size: the size of the kernel.
        sigma: the standard deviation of the kernel with shape :math:`(B,2)`.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        separable: run as composition of two 1d-convolutions.

    Returns:
        the blurred tensor with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       gaussian_blur.html>`__.

    Examples:
        >>> input = torch.rand(2, 4, 5, 5)
        >>> output = gaussian_blur2d_t(input, (3, 3), torch.tensor([[1.5, 1.5]]))
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """
    device, dtype = input.device, input.dtype
    sigma = sigma.to(device=device, dtype=dtype)

    if separable:
        kernel_x: torch.Tensor = get_gaussian_kernel1d_t(kernel_size[1], sigma[:, 1])
        kernel_y: torch.Tensor = get_gaussian_kernel1d_t(kernel_size[0], sigma[:, 0])
        out = filter2d_separable(input, kernel_x, kernel_y, border_type)
    else:
        kernel: torch.Tensor = get_gaussian_kernel2d_t(kernel_size, sigma)
        out = filter2d(input, kernel, border_type)
    return out


class GaussianBlur2d(nn.Module):
    r"""Create an operator that blurs a tensor using a Gaussian filter.

    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It supports batched operation.

    Arguments:
        kernel_size: the size of the kernel.
        sigma: the standard deviation of the kernel.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        separable: run as composition of two 1d-convolutions.

    Returns:
        the blurred tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::

        >>> input = torch.rand(2, 4, 5, 5)
        >>> gauss = GaussianBlur2d((3, 3), (1.5, 1.5))
        >>> output = gauss(input)  # 2x4x5x5
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """

    kernel: torch.Tensor
    kernel_x: torch.Tensor
    kernel_y: torch.Tensor

    def __init__(
        self,
        kernel_size: Tuple[int, int],
        sigma: Tuple[float, float],
        border_type: str = 'reflect',
        separable: bool = True,
    ) -> None:
        super().__init__()
        self._kernel_size: Tuple[int, int] = kernel_size
        self._sigma: Tuple[float, float] = sigma
        self.border_type = border_type
        self._separable = separable
        self._create_kernel()

    @property
    def kernel_size(self):
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, value: Tuple[int, int]):
        self._kernel_size = value
        self._create_kernel()

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value: Tuple[float, float]):
        self._sigma = value
        self._create_kernel()

    @property
    def separable(self):
        return self._separable

    @separable.setter
    def separable(self, value: bool):
        self._separable = value
        self._create_kernel(clear_buffers=True)

    def _create_kernel(self, clear_buffers=False):
        """Create the kernel and store as buffer.

        In the case that we are changing from separable non-separable we first clear any buffers so as not to waste
        memory using clear_buffers
        """
        if clear_buffers:
            existing_buffers = [x[0] for x in self.named_buffers()]
            for x in existing_buffers:
                delattr(self, x)

        if self._separable:
            kernel_x: torch.Tensor = get_gaussian_kernel1d(self._kernel_size[1], self._sigma[1])
            kernel_y: torch.Tensor = get_gaussian_kernel1d(self._kernel_size[0], self._sigma[0])
            self.register_buffer("kernel_x", kernel_x)
            self.register_buffer("kernel_y", kernel_y)
        else:
            kernel: torch.Tensor = get_gaussian_kernel2d(self._kernel_size, self._sigma)
            self.register_buffer("kernel", kernel)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + '(kernel_size='
            + str(self._kernel_size)
            + ', '
            + 'sigma='
            + str(self._sigma)
            + ', '
            + 'border_type='
            + self.border_type
            + ', '
            + 'separable='
            + str(self._separable)
            + ')'
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._separable:
            return filter2d_separable(input, self.kernel_x[None], self.kernel_y[None], self.border_type)
        return filter2d(input, self.kernel[None], self.border_type)
