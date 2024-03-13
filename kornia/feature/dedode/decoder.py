from typing import Any, Optional, Tuple

import torch
from torch import nn

from kornia.core import Tensor


class Decoder(nn.Module):
    def __init__(self, layers: Any, *args, super_resolution: bool = False, num_prototypes: int = 1, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.layers = layers
        self.scales = self.layers.keys()
        self.super_resolution = super_resolution
        self.num_prototypes = num_prototypes

    def forward(
        self, features: Tensor, context: Optional[Tensor] = None, scale: Optional[int] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if context is not None:
            features = torch.cat((features, context), dim=1)
        stuff = self.layers[scale](features)
        logits, context = stuff[:, : self.num_prototypes], stuff[:, self.num_prototypes :]
        return logits, context


class ConvRefiner(nn.Module):
    def __init__(  # type: ignore[no-untyped-def]
        self,
        in_dim=6,
        hidden_dim=16,
        out_dim=2,
        dw=True,
        kernel_size=5,
        hidden_blocks=5,
        amp=True,
        residual=False,
        amp_dtype=torch.float16,
    ):
        super().__init__()
        self.block1 = self.create_block(
            in_dim,
            hidden_dim,
            dw=False,
            kernel_size=1,
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    kernel_size=kernel_size,
                )
                for hb in range(hidden_blocks)
            ]
        )
        self.hidden_blocks = self.hidden_blocks
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.residual = residual

    def create_block(  # type: ignore[no-untyped-def]
        self,
        in_dim,
        out_dim,
        dw=True,
        kernel_size=5,
        bias=True,
        norm_type=nn.BatchNorm2d,
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            if out_dim % in_dim != 0:
                raise Exception("outdim must be divisible by indim for depthwise")
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
            bias=bias,
        )
        norm = norm_type(out_dim) if norm_type is nn.BatchNorm2d else norm_type(num_channels=out_dim)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)

    def forward(self, feats: Tensor) -> Tensor:
        b, c, hs, ws = feats.shape
        with torch.autocast("cuda", enabled=self.amp, dtype=self.amp_dtype):
            x0 = self.block1(feats)
            x = self.hidden_blocks(x0)
            if self.residual:
                x = (x + x0) / 1.4
            x = self.out_conv(x)
            return x
