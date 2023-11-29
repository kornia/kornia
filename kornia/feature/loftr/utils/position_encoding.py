import math
from typing import Tuple

import torch

from kornia.core import Module, Tensor, cos, sin, zeros


class PositionEncodingSine(Module):
    """This is a sinusoidal position encoding that generalized to 2-dimensional images."""

    pe: Tensor

    def __init__(self, d_model: int, max_shape: Tuple[int, int] = (256, 256), temp_bug_fix: bool = True) -> None:
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatibility.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()
        self.d_model = d_model
        self.temp_bug_fix = temp_bug_fix

        pe = self._create_position_encoding(max_shape)
        self.register_buffer("pe", pe, persistent=False)  # [1, C, H, W]

    def _create_position_encoding(self, max_shape: Tuple[int, int]) -> Tensor:
        """Creates a position encoding from scratch.

        For 1/8 feature map (which is standard): If the input image size is H, W (both divisible by 8), the max_shape
        should be (H//8, W//8).
        """
        pe = zeros((self.d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if self.temp_bug_fix:
            div_term = torch.exp(
                torch.arange(0, self.d_model // 2, 2).float() * (-math.log(10000.0) / (self.d_model // 2))
            )
        else:  # a buggy implementation (for backward compatibility only)
            div_term = torch.exp(
                torch.arange(0, self.d_model // 2, 2).float() * (-math.log(10000.0) / self.d_model // 2)
            )
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = sin(x_position * div_term)
        pe[1::4, :, :] = cos(x_position * div_term)
        pe[2::4, :, :] = sin(y_position * div_term)
        pe[3::4, :, :] = cos(y_position * div_term)
        return pe.unsqueeze(0)

    def update_position_encoding_size(self, max_shape: Tuple[int, int]) -> None:
        """Updates position encoding to new max_shape.

        For 1/8 feature map (which is standard): If the input image size is H, W (both divisible by 8), the max_shape
        should be (H//8, W//8).
        """
        self.pe = self._create_position_encoding(max_shape).to(self.pe.device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [N, C, H, W]
        """
        if x.size(2) > self.pe.size(2) or x.size(3) > self.pe.size(3):
            max_shape = (max(x.size(2), self.pe.size(2)), max(x.size(3), self.pe.size(3)))
            self.update_position_encoding_size(max_shape)

        return x + self.pe[:, :, : x.size(2), : x.size(3)]
