from __future__ import annotations

import torch
from torch.nn.functional import interpolate

from kornia.core import ImageModule as Module
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE

from .blur import box_blur
from .kernels import _unpack_2d_ks


def _preprocess_fast_guided_blur(
    guidance: Tensor, input: Tensor, kernel_size: tuple[int, int] | int, subsample: int = 1
) -> tuple[Tensor, Tensor, tuple[int, int]]:
    ky, kx = _unpack_2d_ks(kernel_size)
    if subsample > 1:
        s = 1 / subsample
        guidance_sub = interpolate(guidance, scale_factor=s, mode="nearest")
        input_sub = guidance_sub if input is guidance else interpolate(input, scale_factor=s, mode="nearest")
        ky, kx = ((k - 1) // subsample + 1 for k in (ky, kx))
    else:
        guidance_sub = guidance
        input_sub = input
    return guidance_sub, input_sub, (ky, kx)


def _guided_blur_grayscale_guidance(
    guidance: Tensor,
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    eps: float | Tensor,
    border_type: str = "reflect",
    subsample: int = 1,
) -> Tensor:
    guidance_sub, input_sub, kernel_size = _preprocess_fast_guided_blur(guidance, input, kernel_size, subsample)

    mean_I = box_blur(guidance_sub, kernel_size, border_type)
    corr_I = box_blur(guidance_sub.square(), kernel_size, border_type)
    var_I = corr_I - mean_I.square()

    if input is guidance:
        mean_p = mean_I
        cov_Ip = var_I

    else:
        mean_p = box_blur(input_sub, kernel_size, border_type)
        corr_Ip = box_blur(guidance_sub * input_sub, kernel_size, border_type)
        cov_Ip = corr_Ip - mean_I * mean_p

    if isinstance(eps, Tensor):
        eps = eps.view(-1, 1, 1, 1)  # N -> NCHW

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box_blur(a, kernel_size, border_type)
    mean_b = box_blur(b, kernel_size, border_type)

    if subsample > 1:
        mean_a = interpolate(mean_a, scale_factor=subsample, mode="bilinear")
        mean_b = interpolate(mean_b, scale_factor=subsample, mode="bilinear")

    return mean_a * guidance + mean_b


def _guided_blur_multichannel_guidance(
    guidance: Tensor,
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    eps: float | Tensor,
    border_type: str = "reflect",
    subsample: int = 1,
) -> Tensor:
    guidance_sub, input_sub, kernel_size = _preprocess_fast_guided_blur(guidance, input, kernel_size, subsample)
    B, C, H, W = guidance_sub.shape

    mean_I = box_blur(guidance_sub, kernel_size, border_type).permute(0, 2, 3, 1)
    II = (guidance_sub.unsqueeze(1) * guidance_sub.unsqueeze(2)).flatten(1, 2)
    corr_I = box_blur(II, kernel_size, border_type).permute(0, 2, 3, 1)
    var_I = corr_I.reshape(B, H, W, C, C) - mean_I.unsqueeze(-2) * mean_I.unsqueeze(-1)

    if guidance is input:
        mean_p = mean_I
        cov_Ip = var_I

    else:
        mean_p = box_blur(input_sub, kernel_size, border_type).permute(0, 2, 3, 1)
        Ip = (input_sub.unsqueeze(1) * guidance_sub.unsqueeze(2)).flatten(1, 2)
        corr_Ip = box_blur(Ip, kernel_size, border_type).permute(0, 2, 3, 1)
        cov_Ip = corr_Ip.reshape(B, H, W, C, -1) - mean_p.unsqueeze(-2) * mean_I.unsqueeze(-1)

    if isinstance(eps, Tensor):
        _eps = torch.eye(C, device=guidance.device, dtype=guidance.dtype).view(1, 1, 1, C, C) * eps.view(-1, 1, 1, 1, 1)
    else:
        _eps = guidance.new_full((C,), eps).diag().view(1, 1, 1, C, C)
    a = torch.linalg.solve(var_I + _eps, cov_Ip)  # B, H, W, C_guidance, C_input
    b = mean_p - (mean_I.unsqueeze(-2) @ a).squeeze(-2)  # B, H, W, C_input

    mean_a = box_blur(a.flatten(-2).permute(0, 3, 1, 2), kernel_size, border_type)
    mean_b = box_blur(b.permute(0, 3, 1, 2), kernel_size, border_type)

    if subsample > 1:
        mean_a = interpolate(mean_a, scale_factor=subsample, mode="bilinear")
        mean_b = interpolate(mean_b, scale_factor=subsample, mode="bilinear")
    mean_a = mean_a.view(B, C, -1, H * subsample, W * subsample)

    # einsum might not be contiguous, thus mean_b is the first argument
    return mean_b + torch.einsum("BCHW,BCcHW->BcHW", guidance, mean_a)


def guided_blur(
    guidance: Tensor,
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    eps: float | Tensor,
    border_type: str = "reflect",
    subsample: int = 1,
) -> Tensor:
    r"""Blur a tensor using a Guided filter.

    .. image:: _static/img/guided_blur.png

    The operator is an edge-preserving image smoothing filter. See :cite:`he2010guided`
    and :cite:`he2015fast` for details. Guidance and input can have different number of channels.

    Arguments:
        guidance: the guidance tensor with shape :math:`(B,C,H,W)`.
        input: the input tensor with shape :math:`(B,C,H,W)`.
        kernel_size: the size of the kernel.
        eps: regularization parameter. Smaller values preserve more edges.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        subsample: subsampling factor for Fast Guided filtering. Default: 1 (no subsampling)

    Returns:
        the blurred tensor with same shape as `input` :math:`(B, C, H, W)`.

    Examples:
        >>> guidance = torch.rand(2, 3, 5, 5)
        >>> input = torch.rand(2, 4, 5, 5)
        >>> output = guided_blur(guidance, input, 3, 0.1)
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """

    KORNIA_CHECK_IS_TENSOR(guidance)
    KORNIA_CHECK_SHAPE(guidance, ["B", "C", "H", "W"])
    if input is not guidance:
        KORNIA_CHECK_IS_TENSOR(input)
        KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
        KORNIA_CHECK(
            (guidance.shape[0] == input.shape[0]) and (guidance.shape[-2:] == input.shape[-2:]),
            "guidance and input should have the same batch size and spatial dimensions",
        )

    if guidance.shape[1] == 1:
        return _guided_blur_grayscale_guidance(guidance, input, kernel_size, eps, border_type, subsample)
    else:
        return _guided_blur_multichannel_guidance(guidance, input, kernel_size, eps, border_type, subsample)


class GuidedBlur(Module):
    r"""Blur a tensor using a Guided filter.

    The operator is an edge-preserving image smoothing filter. See :cite:`he2010guided`
    and :cite:`he2015fast` for details. Guidance and input can have different number of channels.

    Arguments:
        kernel_size: the size of the kernel.
        eps: regularization parameter. Smaller values preserve more edges.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        subsample: subsampling factor for Fast Guided filtering. Default: 1 (no subsampling)

    Returns:
        the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`, :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:
        >>> guidance = torch.rand(2, 3, 5, 5)
        >>> input = torch.rand(2, 4, 5, 5)
        >>> blur = GuidedBlur(3, 0.1)
        >>> output = blur(guidance, input)
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """

    def __init__(
        self, kernel_size: tuple[int, int] | int, eps: float, border_type: str = "reflect", subsample: int = 1
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.eps = eps
        self.border_type = border_type
        self.subsample = subsample

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(kernel_size={self.kernel_size}, "
            f"eps={self.eps}, "
            f"border_type={self.border_type}, "
            f"subsample={self.subsample})"
        )

    def forward(self, guidance: Tensor, input: Tensor) -> Tensor:
        return guided_blur(guidance, input, self.kernel_size, self.eps, self.border_type, self.subsample)
