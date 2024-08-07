from __future__ import annotations

from typing import Any

from kornia.core import ImageModule as Module
from kornia.core import Tensor, tensor
from kornia.core.check import KORNIA_CHECK_IS_TENSOR
from kornia.utils import deprecated

from .filter import filter2d, filter2d_separable
from .kernels import _unpack_2d_ks, get_gaussian_kernel1d, get_gaussian_kernel2d


def gaussian_blur2d(
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    sigma: tuple[float, float] | Tensor,
    border_type: str = "reflect",
    separable: bool = True,
) -> Tensor:
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
       See a working example `here <https://kornia.github.io/tutorials/nbs/gaussian_blur.html>`__.

    Examples:
        >>> input = torch.rand(2, 4, 5, 5)
        >>> output = gaussian_blur2d(input, (3, 3), (1.5, 1.5))
        >>> output.shape
        torch.Size([2, 4, 5, 5])

        >>> output = gaussian_blur2d(input, (3, 3), torch.tensor([[1.5, 1.5]]))
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """
    KORNIA_CHECK_IS_TENSOR(input)

    if isinstance(sigma, tuple):
        sigma = tensor([sigma], device=input.device, dtype=input.dtype)
    else:
        KORNIA_CHECK_IS_TENSOR(sigma)
        sigma = sigma.to(device=input.device, dtype=input.dtype)

    if separable:
        ky, kx = _unpack_2d_ks(kernel_size)
        bs = sigma.shape[0]
        kernel_x = get_gaussian_kernel1d(kx, sigma[:, 1].view(bs, 1))
        kernel_y = get_gaussian_kernel1d(ky, sigma[:, 0].view(bs, 1))
        out = filter2d_separable(input, kernel_x, kernel_y, border_type)
    else:
        kernel = get_gaussian_kernel2d(kernel_size, sigma)
        out = filter2d(input, kernel, border_type)

    return out


class GaussianBlur2d(Module):
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

    def __init__(
        self,
        kernel_size: tuple[int, int] | int,
        sigma: tuple[float, float] | Tensor,
        border_type: str = "reflect",
        separable: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.border_type = border_type
        self.separable = separable

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(kernel_size={self.kernel_size}, "
            f"sigma={self.sigma}, "
            f"border_type={self.border_type}, "
            f"separable={self.separable})"
        )

    def forward(self, input: Tensor) -> Tensor:
        return gaussian_blur2d(input, self.kernel_size, self.sigma, self.border_type, self.separable)


@deprecated(replace_with="gaussian_blur2d", version="6.9.10")
def gaussian_blur2d_t(*args: Any, **kwargs: Any) -> Tensor:
    return gaussian_blur2d(*args, **kwargs)
