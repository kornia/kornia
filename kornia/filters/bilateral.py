from __future__ import annotations

from typing import Optional

from kornia.core import ImageModule as Module
from kornia.core import Tensor, pad
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE

from .kernels import _unpack_2d_ks, get_gaussian_kernel2d
from .median import _compute_zero_padding


def _bilateral_blur(
    input: Tensor,
    guidance: Optional[Tensor],
    kernel_size: tuple[int, int] | int,
    sigma_color: float | Tensor,
    sigma_space: tuple[float, float] | Tensor,
    border_type: str = "reflect",
    color_distance_type: str = "l1",
) -> Tensor:
    "Single implementation for both Bilateral Filter and Joint Bilateral Filter"

    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    if guidance is not None:
        # NOTE: allow guidance and input having different number of channels
        KORNIA_CHECK_IS_TENSOR(guidance)
        KORNIA_CHECK_SHAPE(guidance, ["B", "C", "H", "W"])
        KORNIA_CHECK(
            (guidance.shape[0] == input.shape[0]) and (guidance.shape[-2:] == input.shape[-2:]),
            "guidance and input should have the same batch size and spatial dimensions",
        )

    if isinstance(sigma_color, Tensor):
        KORNIA_CHECK_SHAPE(sigma_color, ["B"])
        sigma_color = sigma_color.to(device=input.device, dtype=input.dtype).view(-1, 1, 1, 1, 1)

    ky, kx = _unpack_2d_ks(kernel_size)
    pad_y, pad_x = _compute_zero_padding(kernel_size)

    padded_input = pad(input, (pad_x, pad_x, pad_y, pad_y), mode=border_type)
    unfolded_input = padded_input.unfold(2, ky, 1).unfold(3, kx, 1).flatten(-2)  # (B, C, H, W, Ky x Kx)

    if guidance is None:
        guidance = input
        unfolded_guidance = unfolded_input
    else:
        padded_guidance = pad(guidance, (pad_x, pad_x, pad_y, pad_y), mode=border_type)
        unfolded_guidance = padded_guidance.unfold(2, ky, 1).unfold(3, kx, 1).flatten(-2)  # (B, C, H, W, Ky x Kx)

    diff = unfolded_guidance - guidance.unsqueeze(-1)
    if color_distance_type == "l1":
        color_distance_sq = diff.abs().sum(1, keepdim=True).square()
    elif color_distance_type == "l2":
        color_distance_sq = diff.square().sum(1, keepdim=True)
    else:
        raise ValueError("color_distance_type only accepts l1 or l2")
    color_kernel = (-0.5 / sigma_color**2 * color_distance_sq).exp()  # (B, 1, H, W, Ky x Kx)

    space_kernel = get_gaussian_kernel2d(kernel_size, sigma_space, device=input.device, dtype=input.dtype)
    space_kernel = space_kernel.view(-1, 1, 1, 1, kx * ky)

    kernel = space_kernel * color_kernel
    out = (unfolded_input * kernel).sum(-1) / kernel.sum(-1)
    return out


def bilateral_blur(
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    sigma_color: float | Tensor,
    sigma_space: tuple[float, float] | Tensor,
    border_type: str = "reflect",
    color_distance_type: str = "l1",
) -> Tensor:
    r"""Blur a tensor using a Bilateral filter.

    .. image:: _static/img/bilateral_blur.png

    The operator is an edge-preserving image smoothing filter. The weight
    for each pixel in a neighborhood is determined not only by its distance
    to the center pixel, but also the difference in intensity or color.

    Arguments:
        input: the input tensor with shape :math:`(B,C,H,W)`.
        kernel_size: the size of the kernel.
        sigma_color: the standard deviation for intensity/color Gaussian kernel.
          Smaller values preserve more edges.
        sigma_space: the standard deviation for spatial Gaussian kernel.
          This is similar to ``sigma`` in :func:`gaussian_blur2d()`.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        color_distance_type: the type of distance to calculate intensity/color
          difference. Only ``'l1'`` or ``'l2'`` is allowed. Use ``'l1'`` to
          match OpenCV implementation. Use ``'l2'`` to match Matlab implementation.
          Default: ``'l1'``.

    Returns:
        the blurred tensor with shape :math:`(B, C, H, W)`.

    Examples:
        >>> input = torch.rand(2, 4, 5, 5)
        >>> output = bilateral_blur(input, (3, 3), 0.1, (1.5, 1.5))
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """
    return _bilateral_blur(input, None, kernel_size, sigma_color, sigma_space, border_type, color_distance_type)


def joint_bilateral_blur(
    input: Tensor,
    guidance: Tensor,
    kernel_size: tuple[int, int] | int,
    sigma_color: float | Tensor,
    sigma_space: tuple[float, float] | Tensor,
    border_type: str = "reflect",
    color_distance_type: str = "l1",
) -> Tensor:
    r"""Blur a tensor using a Joint Bilateral filter.

    .. image:: _static/img/joint_bilateral_blur.png

    This operator is almost identical to a Bilateral filter. The only difference
    is that the color Gaussian kernel is computed based on another image called
    a guidance image. See :func:`bilateral_blur()` for more information.

    Arguments:
        input: the input tensor with shape :math:`(B,C,H,W)`.
        guidance: the guidance tensor with shape :math:`(B,C,H,W)`.
        kernel_size: the size of the kernel.
        sigma_color: the standard deviation for intensity/color Gaussian kernel.
          Smaller values preserve more edges.
        sigma_space: the standard deviation for spatial Gaussian kernel.
          This is similar to ``sigma`` in :func:`gaussian_blur2d()`.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        color_distance_type: the type of distance to calculate intensity/color
          difference. Only ``'l1'`` or ``'l2'`` is allowed. Use ``'l1'`` to
          match OpenCV implementation.

    Returns:
        the blurred tensor with shape :math:`(B, C, H, W)`.

    Examples:
        >>> input = torch.rand(2, 4, 5, 5)
        >>> guidance = torch.rand(2, 4, 5, 5)
        >>> output = joint_bilateral_blur(input, guidance, (3, 3), 0.1, (1.5, 1.5))
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """
    return _bilateral_blur(input, guidance, kernel_size, sigma_color, sigma_space, border_type, color_distance_type)


# trick to make mypy not throw errors about difference in .forward() signatures of subclass and superclass
class _BilateralBlur(Module):
    def __init__(
        self,
        kernel_size: tuple[int, int] | int,
        sigma_color: float | Tensor,
        sigma_space: tuple[float, float] | Tensor,
        border_type: str = "reflect",
        color_distance_type: str = "l1",
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.border_type = border_type
        self.color_distance_type = color_distance_type

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(kernel_size={self.kernel_size}, "
            f"sigma_color={self.sigma_color}, "
            f"sigma_space={self.sigma_space}, "
            f"border_type={self.border_type}, "
            f"color_distance_type={self.color_distance_type})"
        )


class BilateralBlur(_BilateralBlur):
    r"""Blur a tensor using a Bilateral filter.

    The operator is an edge-preserving image smoothing filter. The weight
    for each pixel in a neighborhood is determined not only by its distance
    to the center pixel, but also the difference in intensity or color.

    Arguments:
        kernel_size: the size of the kernel.
        sigma_color: the standard deviation for intensity/color Gaussian kernel.
          Smaller values preserve more edges.
        sigma_space: the standard deviation for spatial Gaussian kernel.
          This is similar to ``sigma`` in :func:`gaussian_blur2d()`.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        color_distance_type: the type of distance to calculate intensity/color
          difference. Only ``'l1'`` or ``'l2'`` is allowed. Use ``'l1'`` to
          match OpenCV implementation. Use ``'l2'`` to match Matlab implementation.
          Default: ``'l1'``.

    Returns:
        the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:
        >>> input = torch.rand(2, 4, 5, 5)
        >>> blur = BilateralBlur((3, 3), 0.1, (1.5, 1.5))
        >>> output = blur(input)
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """

    def forward(self, input: Tensor) -> Tensor:
        return bilateral_blur(
            input, self.kernel_size, self.sigma_color, self.sigma_space, self.border_type, self.color_distance_type
        )


class JointBilateralBlur(_BilateralBlur):
    r"""Blur a tensor using a Joint Bilateral filter.

    This operator is almost identical to a Bilateral filter. The only difference
    is that the color Gaussian kernel is computed based on another image called
    a guidance image. See :class:`BilateralBlur` for more information.

    Arguments:
        kernel_size: the size of the kernel.
        sigma_color: the standard deviation for intensity/color Gaussian kernel.
          Smaller values preserve more edges.
        sigma_space: the standard deviation for spatial Gaussian kernel.
          This is similar to ``sigma`` in :func:`gaussian_blur2d()`.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        color_distance_type: the type of distance to calculate intensity/color
          difference. Only ``'l1'`` or ``'l2'`` is allowed. Use ``'l1'`` to
          match OpenCV implementation.

    Returns:
        the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`, :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:
        >>> input = torch.rand(2, 4, 5, 5)
        >>> guidance = torch.rand(2, 4, 5, 5)
        >>> blur = JointBilateralBlur((3, 3), 0.1, (1.5, 1.5))
        >>> output = blur(input, guidance)
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """

    def forward(self, input: Tensor, guidance: Tensor) -> Tensor:
        return joint_bilateral_blur(
            input,
            guidance,
            self.kernel_size,
            self.sigma_color,
            self.sigma_space,
            self.border_type,
            self.color_distance_type,
        )
