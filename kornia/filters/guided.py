from __future__ import annotations

from kornia.core import Module, Tensor

from .blur import box_blur


def guided_blur(
    input: Tensor,
    guidance: Tensor,
    kernel_size: tuple[int, int] | int,
    eps: float | Tensor,
    border_type: str = 'reflect',
):
    if isinstance(eps, Tensor):
        eps = eps.view(-1, 1, 1, 1)  # N -> NCHW

    mean_i = box_blur(input, kernel_size, border_type)
    corr_i = box_blur(input.square(), kernel_size, border_type)
    var_i = corr_i - mean_i.square()

    if guidance is input:
        mean_g = mean_i
        corr_ig = corr_i
        cov_ig = var_i

    else:
        mean_g = box_blur(guidance, kernel_size, border_type)
        corr_ig = box_blur(input * guidance, kernel_size, border_type)
        cov_ig = corr_ig - mean_i * mean_g

    a = cov_ig / (var_i + eps)
    b = mean_g - a * mean_i

    mean_a = box_blur(a, kernel_size, border_type)
    mean_b = box_blur(b, kernel_size, border_type)

    return mean_a * input + mean_b


class GuidedBlur(Module):
    def __init__(self, kernel_size: tuple[int, int] | int, eps: float | Tensor, border_type: str = 'reflect'):
        super().__init__()
        self.kernel_size = kernel_size
        self.eps = eps
        self.border_type = border_type

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(kernel_size={self.kernel_size}, "
            f"eps={self.eps}, "
            f"border_type={self.border_type})"
        )

    def forward(self, input: Tensor, guidance: Tensor):
        return guided_blur(input, guidance, self.kernel_size, self.eps, self.border_type)
