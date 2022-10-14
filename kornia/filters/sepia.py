import torch
import torch.nn as nn


def sepia(input: torch.Tensor, rescale: bool = True) -> torch.Tensor:
    r"""Apply to a tensor the sepia filter.

    Args:
        input: the input tensor with shape of :math:`(B, C, H, W)`.
        rescale: If True, the output tensor will be rescaled (max values be 1. or 255).

    Returns:
        - torch.Tensor: The sepia tensor of same size and numbers of channels
        as the input with shape :math:`(B, C, H, W)`.

    Example:
        >>> input = torch.ones(2, 3, 1, 1)
        >>> sepia(input)
        tensor([[[[1.0000]],
                 [[0.8905]],
                 [[0.6936]]],
                [[[1.0000]],
                 [[0.8905]],
                 [[0.6936]]]])
    """
    if len(input.shape) < 3 or input.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {input.shape}")

    # 8 bit images
    if input.dtype == torch.uint8:
        if input.device != torch.device('cpu'):
            raise TypeError(
                f"Input device must be `cpu` to apply sepia to a `uint8` tensor. Got {input.device}"
            )  # issue pytorch#44428

        weights = torch.tensor([[100, 196, 48], [88, 174, 42], [69, 136, 33]], device=input.device, dtype=torch.uint8)
    # floating point images
    elif input.dtype in (torch.float16, torch.float32, torch.float64):
        weights = torch.tensor(
            [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]],
            device=input.device,
            dtype=input.dtype,
        )
    else:
        raise TypeError(f"Unknown data type: {input.dtype}")

    input_reshaped = input.movedim(-3, -1)

    sepia_out = torch.matmul(input_reshaped, weights.T)
    if rescale:
        max_values = sepia_out.amax(dim=-1).amax(dim=-1)
        sepia_out = torch.div(sepia_out, (max_values.unsqueeze(-1).unsqueeze(-1)))

    sepia_out = sepia_out.movedim(-1, -3).contiguous()

    return sepia_out


class Sepia(nn.Module):
    r"""Module that apply the sepia filter to tensors.

    Args:
        input: the input tensor with shape of :math:`(B, C, H, W)`.
        rescale: If True, the output tensor will be rescaled (max values be 1. or 255).

    Returns:
        - torch.Tensor: The sepia tensor of same size and numbers of channels
        as the input with shape :math:`(B, C, H, W)`.

    Example:
        >>>
        >>> input = torch.ones(2, 3, 1, 1)
        >>> Sepia()(input)
        tensor([[[[1.0000]],
                 [[0.8905]],
                 [[0.6936]]],
                [[[1.0000]],
                 [[0.8905]],
                 [[0.6936]]]])
    """

    def __init__(self, rescale: bool = True) -> None:
        self.rescale = True
        super().__init__()

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(rescale={self.rescale})'

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return sepia(input, rescale=self.rescale)
