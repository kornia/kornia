from __future__ import annotations

from kornia.core import Module, Tensor


def guided_blur(input: Tensor, guidance: Tensor, kernel_size: tuple[int, int] | int, eps: float | Tensor):
    raise NotImplementedError


class GuidedBlur(Module):
    def __init__(self, kernel_size: tuple[int, int] | int, eps: float | Tensor):
        super().__init__()
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, input: Tensor, guidance: Tensor):
        return guided_blur(input, guidance, self.kernel_size, self.eps)
