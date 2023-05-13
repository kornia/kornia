"""Based on code from
https://github.com/PaddlePaddle/PaddleDetection/blob/ec37e66685f3bc5a38cd13f60685acea175922e1/
ppdet/modeling/backbones/hgnet_v2.py."""

from __future__ import annotations

from typing import NamedTuple

from torch import nn

from kornia.core import Module, Tensor, concatenate
from kornia.core.check import KORNIA_CHECK

from .common import ConvNormAct


class StemBlock(Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super().__init__()
        self.stem1 = ConvNormAct(in_channels, mid_channels, 3, 2)
        self.stem2a = ConvNormAct(mid_channels, mid_channels // 2, 2)
        self.stem2b = ConvNormAct(mid_channels // 2, mid_channels, 2)
        self.stem3 = ConvNormAct(mid_channels * 2, mid_channels, 3, 2)
        self.stem4 = ConvNormAct(mid_channels, out_channels, 1)
        self.pool = nn.MaxPool2d(2, 1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem1(x)
        x = concatenate([self.pool(x), self.stem2b(self.stem2a(x))], 1)
        x = self.stem4(self.stem3(x))
        return x


# Separable conv
class LightConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, 1, act="none")  # point-wise
        self.conv2 = ConvNormAct(out_channels, out_channels, kernel_size, groups=out_channels)  # depth-wise


class StageConfig(NamedTuple):
    in_channels: int
    mid_channels: int
    out_channels: int
    num_blocks: int
    downsample: bool
    light_block: bool
    kernel_size: int
    layer_num: int


class HGBlock(Module):
    def __init__(self, in_channels: int, cfg: StageConfig, identity: bool):
        super().__init__()
        self.identity = identity

        layer_cls = LightConvNormAct if cfg.light_block else ConvNormAct
        self.layers = nn.ModuleList()
        for i in range(cfg.layer_num):
            in_c = in_channels if i == 0 else cfg.mid_channels
            self.layers.append(layer_cls(in_c, cfg.mid_channels, cfg.kernel_size))

        total_channels = in_channels + cfg.mid_channels * cfg.layer_num
        self.aggregation_squeeze_conv = ConvNormAct(total_channels, cfg.out_channels // 2, 1)
        self.aggregation_excitation_conv = ConvNormAct(cfg.out_channels // 2, cfg.out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        out, feats = x, []
        for layer in self.layers:
            out = layer(out)
            feats.append(out)
        out = concatenate(feats, 1)
        out = self.aggregation_squeeze_conv(out)
        out = self.aggregation_excitation_conv(out)
        return x + out if self.identity else out


class HGStage(nn.Sequential):
    def __init__(self, cfg: StageConfig):
        super().__init__()
        self.downsample = ConvNormAct(cfg.in_channels, cfg.in_channels, 3, 2, "none") if cfg.downsample else None
        blocks = []
        for i in range(cfg.num_blocks):
            in_c = cfg.in_channels if i == 0 else cfg.out_channels
            blocks.append(HGBlock(in_c, cfg, i > 0))
        self.blocks = nn.Sequential(*blocks)


class PPHGNetV2(Module):
    def __init__(self, stem_channels: list[int], stage_configs: list[StageConfig]):
        KORNIA_CHECK(len(stem_channels) == 3)
        KORNIA_CHECK(len(stage_configs) == 4)
        super().__init__()
        self.out_channels = tuple(config.out_channels for config in stage_configs[-3:])
        self.stem = StemBlock(*stem_channels)
        self.stages = nn.ModuleList()
        for cfg in stage_configs:
            self.stages.append(HGStage(cfg))

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        s2 = self.stages[0](x)
        s3 = self.stages[1](s2)
        s4 = self.stages[2](s3)
        s5 = self.stages[3](s4)
        return [s3, s4, s5]

    @staticmethod
    def from_config(variant: str):
        arch_configs = dict(
            L=dict(
                stem_channels=[3, 32, 48],
                stage_configs=[
                    StageConfig(48, 48, 128, 1, False, False, 3, 6),
                    StageConfig(128, 96, 512, 1, True, False, 3, 6),
                    StageConfig(512, 192, 1024, 3, True, True, 5, 6),
                    StageConfig(1024, 384, 2048, 1, True, True, 5, 6),
                ],
            ),
            X=dict(
                stem_channels=[3, 32, 64],
                stage_configs=[
                    StageConfig(64, 64, 128, 1, False, False, 3, 6),
                    StageConfig(128, 128, 512, 2, True, False, 3, 6),
                    StageConfig(512, 256, 1024, 5, True, True, 5, 6),
                    StageConfig(1024, 512, 2048, 2, True, True, 5, 6),
                ],
            ),
        )
        KORNIA_CHECK(variant in arch_configs, "Only variant L and X are supported")
        return PPHGNetV2(**arch_configs[variant])
