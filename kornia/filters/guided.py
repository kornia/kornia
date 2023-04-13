from __future__ import annotations

from kornia.core import Module, Tensor

from .blur import box_blur


def guided_blur(
    guidance: Tensor,
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    eps: float | Tensor,
    border_type: str = 'replicate',
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


class GuidedBlur(Module):
    def __init__(self, kernel_size: tuple[int, int] | int, eps: float | Tensor, border_type: str = 'replicate'):
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
