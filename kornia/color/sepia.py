import torch

from kornia.core import ImageModule as Module
from kornia.core import Tensor


def sepia_from_rgb(input: Tensor, rescale: bool = True, eps: float = 1e-6) -> Tensor:
    r"""Apply to a tensor the sepia filter.

    Args:
        input: the input tensor with shape of :math:`(*, C, H, W)`.
        rescale: If True, the output tensor will be rescaled (max values be 1. or 255).
        eps: scalar to enforce numerical stability.

    Returns:
        Tensor: The sepia tensor of same size and numbers of channels
        as the input with shape :math:`(*, C, H, W)`.

    Example:
        >>> input = torch.ones(3, 1, 1)
        >>> sepia_from_rgb(input, rescale=False)
        tensor([[[1.3510]],
        <BLANKLINE>
                [[1.2030]],
        <BLANKLINE>
                [[0.9370]]])
    """
    if len(input.shape) < 3 or input.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {input.shape}")

    r = input[..., 0, :, :]
    g = input[..., 1, :, :]
    b = input[..., 2, :, :]

    r_out = 0.393 * r + 0.769 * g + 0.189 * b
    g_out = 0.349 * r + 0.686 * g + 0.168 * b
    b_out = 0.272 * r + 0.534 * g + 0.131 * b

    sepia_out = torch.stack([r_out, g_out, b_out], dim=-3)

    if rescale:
        max_values = sepia_out.amax(dim=-1).amax(dim=-1)
        sepia_out = sepia_out / (max_values[..., None, None] + eps)

    return sepia_out


class Sepia(Module):
    r"""Module that apply the sepia filter to tensors.

    Args:
        input: the input tensor with shape of :math:`(*, C, H, W)`.
        rescale: If True, the output tensor will be rescaled (max values be 1. or 255).
        eps: scalar to enforce numerical stability.

    Returns:
        Tensor: The sepia tensor of same size and numbers of channels
        as the input with shape :math:`(*, C, H, W)`.

    Example:
        >>>
        >>> input = torch.ones(3, 1, 1)
        >>> Sepia(rescale=False)(input)
        tensor([[[1.3510]],
        <BLANKLINE>
                [[1.2030]],
        <BLANKLINE>
                [[0.9370]]])
    """

    def __init__(self, rescale: bool = True, eps: float = 1e-6) -> None:
        self.rescale = rescale
        self.eps = eps
        super().__init__()

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(rescale={self.rescale}, eps={self.eps})"

    def forward(self, input: Tensor) -> Tensor:
        return sepia_from_rgb(input, rescale=self.rescale, eps=self.eps)
