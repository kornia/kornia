from __future__ import annotations

from torch import nn

from kornia.core import Module, Tensor

from .blocks import ThinUnetDownBlock, ThinUnetUpBlock


class Unet(Module):
    def __init__(self, in_features: int = 1, up: list[int] = [], down: list[int] = [], size: int = 5) -> None:
        super().__init__()

        if not len(down) == len(up) + 1:
            raise ValueError("`down` must be 1 item longer than `up`")

        self.up = up
        self.down = down
        self.in_features = in_features

        down_dims = [in_features, *down]
        self.path_down = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(down_dims[:-1], down_dims[1:])):
            down_block = ThinUnetDownBlock(d_in, d_out, size=size, is_first=i == 0)
            self.path_down.append(down_block)

        bot_dims = [down[-1], *up]
        hor_dims = down_dims[-2::-1]
        self.path_up = nn.ModuleList()
        for i, (d_bot, d_hor, d_out) in enumerate(zip(bot_dims, hor_dims, up)):
            up_block = ThinUnetUpBlock(d_bot, d_hor, d_out, size=size)
            self.path_up.append(up_block)

        self.n_params = 0
        for param in self.parameters():
            self.n_params += param.numel()

    def forward(self, inp: Tensor) -> Tensor:
        if inp.size(1) != self.in_features:
            fmt = "Expected {} feature channels in input, got {}"
            msg = fmt.format(self.in_features, inp.size(1))
            raise ValueError(msg)

        input_size_divisor = 2 ** len(self.up)
        if (inp.size(2) % input_size_divisor != 0) or (inp.size(3) % input_size_divisor != 0):
            raise ValueError(
                f"Input image shape must be divisible by {input_size_divisor} (got {inp.size()}). "
                "This is not inherent to DISK, but to the U-Net architecture used in pretrained models. "
                "Please pad if necessary."
            )

        features = [inp]
        for layer in self.path_down:
            features.append(layer(features[-1]))

        f_bot = features[-1]
        features_horizontal = features[-2::-1]

        for layer, f_hor in zip(self.path_up, features_horizontal):
            f_bot = layer(f_bot, f_hor)

        return f_bot
