from __future__ import annotations

import torch

from kornia.core import Module, Tensor

from .blur import box_blur


def _guided_blur_grayscale_guidance(
    guidance: Tensor,
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    eps: float | Tensor,
    border_type: str = 'reflect',
) -> Tensor:
    if isinstance(eps, Tensor):
        eps = eps.view(-1, 1, 1, 1)  # N -> NCHW

    mean_I = box_blur(guidance, kernel_size, border_type)
    corr_I = box_blur(guidance.square(), kernel_size, border_type)
    var_I = corr_I - mean_I.square()

    if guidance is input:
        mean_p = mean_I
        cov_Ip = var_I

    else:
        mean_p = box_blur(input, kernel_size, border_type)
        corr_Ip = box_blur(guidance * input, kernel_size, border_type)
        cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box_blur(a, kernel_size, border_type)
    mean_b = box_blur(b, kernel_size, border_type)
    return mean_a * guidance + mean_b


def _guided_blur_multichannel_guidance(
    guidance: Tensor,
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    eps: float | Tensor,
    border_type: str = 'reflect',
) -> Tensor:
    B, C, H, W = guidance.shape

    mean_I = box_blur(guidance, kernel_size, border_type).permute(0, 2, 3, 1)
    II = (guidance.unsqueeze(1) * guidance.unsqueeze(2)).flatten(1, 2)
    corr_I = box_blur(II, kernel_size, border_type).permute(0, 2, 3, 1)
    var_I = corr_I.reshape(B, H, W, C, C) - mean_I.unsqueeze(-2) * mean_I.unsqueeze(-1)

    if guidance is input:
        mean_p = mean_I
        cov_Ip = var_I

    else:
        mean_p = box_blur(input, kernel_size, border_type).permute(0, 2, 3, 1)
        Ip = (input.unsqueeze(1) * guidance.unsqueeze(2)).flatten(1, 2)
        corr_Ip = box_blur(Ip, kernel_size, border_type).permute(0, 2, 3, 1)
        cov_Ip = corr_Ip.reshape(B, H, W, C, -1) - mean_p.unsqueeze(-2) * mean_I.unsqueeze(-1)

    if isinstance(eps, Tensor):
        _eps = torch.eye(C, device=guidance.device, dtype=guidance.dtype).view(1, 1, 1, C, C) * eps.view(-1, 1, 1, 1, 1)
    else:
        _eps = guidance.new_full((C,), eps).diag().view(1, 1, 1, C, C)
    a = torch.linalg.solve(var_I + _eps, cov_Ip)  # B, H, W, C_guidance, C_input
    b = mean_p - (mean_I.unsqueeze(-2) @ a).squeeze(-2)  # B, H, W, C_input

    mean_a = box_blur(a.flatten(-2).permute(0, 3, 1, 2), kernel_size, border_type).view(B, C, -1, H, W)
    mean_b = box_blur(b.permute(0, 3, 1, 2), kernel_size, border_type)

    # einsum might not be contiguous, thus mean_b is the first argument
    return mean_b + torch.einsum("BCHW,BCcHW->BcHW", guidance, mean_a)


def guided_blur(
    guidance: Tensor,
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    eps: float | Tensor,
    border_type: str = 'reflect',
) -> Tensor:
    if guidance.shape[1] == 1:
        return _guided_blur_grayscale_guidance(guidance, input, kernel_size, eps, border_type)
    else:
        return _guided_blur_multichannel_guidance(guidance, input, kernel_size, eps, border_type)


def fast_guided_blur(
    guidance: Tensor,
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    eps: float | Tensor,
    border_type: str = 'reflect',
) -> Tensor:
    raise NotImplementedError


class GuidedBlur(Module):
    def __init__(self, kernel_size: tuple[int, int] | int, eps: float | Tensor, border_type: str = 'reflect'):
        super().__init__()
        self.kernel_size = kernel_size
        if isinstance(eps, Tensor):
            self.register_buffer("eps", eps)
        else:
            self.eps = eps
        self.border_type = border_type

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(kernel_size={self.kernel_size}, "
            f"eps={self.eps}, "
            f"border_type={self.border_type})"
        )

    def forward(self, guidance: Tensor, input: Tensor) -> Tensor:
        return guided_blur(guidance, input, self.kernel_size, self.eps, self.border_type)


class FastGuidedBlur(GuidedBlur):
    def forward(self, guidance: Tensor, input: Tensor) -> Tensor:
        return fast_guided_blur(guidance, input, self.kernel_size, self.eps, self.border_type)
