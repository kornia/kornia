import torch
import torch.nn as nn


def sepia(input: torch.Tensor, rescale: bool = True) -> torch.Tensor:
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
        sepia_out = torch.div(sepia_out, sepia_out.max())

    sepia_out = sepia_out.movedim(-1, -3).contiguous()

    return sepia_out


class Sepia(nn.Module):
    def __init__(self, rescale: bool = True) -> None:
        self.rescale = True
        super().__init__()

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(rescale={self.rescale})'

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return sepia(input, rescale=self.rescale)
