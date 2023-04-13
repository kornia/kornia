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
):
    B, C, H, W = guidance.shape

    mean_I = box_blur(guidance, kernel_size, border_type)
    corr_I = box_blur((guidance.unsqueeze(1) * guidance.unsqueeze(2)).flatten(1, 2), kernel_size, border_type)
    var_I = corr_I.reshape(B, C, C, H, W) - mean_I.unsqueeze(1) * mean_I.unsqueeze(2)

    if guidance is input:
        mean_p = mean_I
        cov_Ip = var_I

    else:
        mean_p = box_blur(input, kernel_size, border_type)
        corr_Ip = box_blur((input.unsqueeze(1) * guidance.unsqueeze(2)).flatten(1, 2), kernel_size, border_type)
        cov_Ip = corr_Ip.reshape(B, C, -1, H, W) - mean_p.unsqueeze(1) * mean_I.unsqueeze(2)

    if isinstance(eps, Tensor):
        eps = eps.view(-1, 1, 1, 1)  # N -> N, C, C, H, W

    _eps = torch.eye(C, device=guidance.device, dtype=guidance.dtype).view(1, C, C, 1, 1) * eps
    a1 = (var_I + _eps).permute(0, 3, 4, 1, 2)  # B, H, W, C_guidance, C_guidance
    a2 = cov_Ip.permute(0, 3, 4, 1, 2)  # B, H, W, C_guidance, C_input
    a = torch.linalg.solve(a1, a2)  # B, H, W, C_guidance, C_input
    b = mean_p.permute(0, 2, 3, 1) - (a.transpose(-1, -2) @ mean_I.permute(0, 2, 3, 1).unsqueeze(-1)).squeeze(-1)
    # B, H, W, C_input

    a = a.flatten(-2).permute(0, 3, 1, 2)
    b = b.permute(0, 3, 1, 2)

    mean_a = box_blur(a, kernel_size, border_type).permute(0, 2, 3, 1).view(B, H, W, C, -1)
    mean_b = box_blur(b, kernel_size, border_type)
    return (guidance.permute(0, 2, 3, 1).unsqueeze(-2) @ mean_a).squeeze(-2).permute(0, 3, 1, 2) + mean_b


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
