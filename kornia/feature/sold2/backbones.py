"""Implements several backbone networks."""

import functools
import operator
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import pixel_shuffle, softmax

from kornia.core import Module, Tensor


class HourglassConfig(NamedTuple):
    depth: int
    num_stacks: int
    num_blocks: int
    num_classes: int
    input_channels: int
    head: Type[Module]


# [Hourglass backbone classes]
class HourglassBackbone(Module):
    """Hourglass network, taken from https://github.com/zhou13/lcnn.

    Args:
        input_channel: number of input channels.
        depth: number of residual blocks per hourglass module.
        num_stacks: number of hourglass modules stacked together.
        num_blocks: number of layers in each residual block.
        num_classes: number of heads for the output of a hourglass module.
    """

    def __init__(
        self, input_channel: int = 1, depth: int = 4, num_stacks: int = 2, num_blocks: int = 1, num_classes: int = 5
    ) -> None:
        super().__init__()
        self.head = MultitaskHead
        self.net = hg(HourglassConfig(depth, num_stacks, num_blocks, num_classes, input_channel, head=self.head))

    def forward(self, input_images: Tensor) -> Tensor:
        return self.net(input_images)


class MultitaskHead(Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()

        m = int(input_channels / 4)
        head_size = [[2], [1], [2]]
        heads = []
        _iter: list[int] = functools.reduce(operator.iconcat, head_size, [])
        for output_channels in _iter:
            heads.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(m, output_channels, kernel_size=1),
                )
            )
        self.heads = nn.ModuleList(heads)

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([head(x) for head in self.heads], dim=1)


class Bottleneck2D(Module):
    def __init__(
        self, inplanes: int, planes: int, stride: Union[int, Tuple[int, int]] = 1, downsample: Optional[Module] = None
    ) -> None:
        super().__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(Module):
    def __init__(self, block: Type[Bottleneck2D], num_blocks: int, planes: int, depth: int, expansion: int = 2) -> None:
        super().__init__()
        self.depth = depth
        self.block = block
        self.expansion = expansion
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block: Type[Bottleneck2D], num_blocks: int, planes: int) -> Module:
        layers = []
        for _ in range(0, num_blocks):
            layers.append(block(planes * self.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block: Type[Bottleneck2D], num_blocks: int, planes: int, depth: int) -> nn.ModuleList:
        hgl = []
        for i in range(depth):
            res = []
            for _ in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hgl.append(nn.ModuleList(res))
        return nn.ModuleList(hgl)

    def _hour_glass_forward(self, n: int, x: Tensor) -> Tensor:
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, size=up1.shape[2:])
        out = up1 + up2
        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(Module):
    """Hourglass model from Newell et al ECCV 2016."""

    def __init__(
        self,
        block: Type[Bottleneck2D],
        head: Type[Module],
        depth: int,
        num_stacks: int,
        num_blocks: int,
        num_classes: int,
        input_channels: int,
        expansion: int = 2,
    ) -> None:
        super().__init__()

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.expansion = expansion
        self.conv1 = nn.Conv2d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # Build hourglass modules
        ch = self.num_feats * self.expansion
        hgl, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hgl.append(Hourglass(block, num_blocks, self.num_feats, depth))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(head(ch))
            if i < num_stacks - 1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1))
        self.hg = nn.ModuleList(hgl)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_residual(
        self, block: Type[Bottleneck2D], planes: int, blocks: int, stride: Union[int, Tuple[int, int]] = 1
    ) -> Module:
        downsample = None
        if stride != 1 or self.inplanes != planes * self.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * self.expansion, kernel_size=1, stride=stride))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * self.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes: int, outplanes: int) -> Module:
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1)
        return nn.Sequential(conv, bn, self.relu)

    def forward(self, x: Tensor) -> Tensor:
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        return y


def hg(cfg: HourglassConfig) -> HourglassNet:
    return HourglassNet(
        Bottleneck2D,
        head=cfg.head,
        depth=cfg.depth,
        num_stacks=cfg.num_stacks,
        num_blocks=cfg.num_blocks,
        num_classes=cfg.num_classes,
        input_channels=cfg.input_channels,
    )


# [Backbone decoders]
class SuperpointDecoder(Module):
    """Junction decoder based on the SuperPoint architecture.

    Args:
        input_feat_dim: channel size of the input features.
    Returns:
        the junction heatmap, with shape (B, H, W).
    """

    def __init__(self, input_feat_dim: int = 128, grid_size: int = 8) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        # Perform strided convolution when using lcnn backbone.
        self.convPa = nn.Conv2d(input_feat_dim, 256, kernel_size=3, stride=2, padding=1)
        self.convPb = nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)
        self.grid_size = grid_size

    def forward(self, input_features: Tensor) -> Tensor:
        feat = self.relu(self.convPa(input_features))
        semi = self.convPb(feat)

        # Convert from semi-dense to dense heatmap
        junc_prob = softmax(semi, dim=1)
        junc_pred = pixel_shuffle(junc_prob[:, :-1, :, :], self.grid_size)[:, 0]
        return junc_pred


class PixelShuffleDecoder(Module):
    """Pixel shuffle decoder used to predict the line heatmap.

    Args:
        input_feat_dim: channel size of the input features.
        num_upsample: how many upsamples are performed.
        output_channel: number of output channels.
    Returns:
        the (B, 1, H, W) line heatmap.
    """

    def __init__(self, input_feat_dim: int = 128, num_upsample: int = 2, output_channel: int = 2) -> None:
        super().__init__()
        # Get channel parameters
        self.channel_conf = self.get_channel_conf(num_upsample)

        # Define the pixel shuffle
        self.pixshuffle = nn.PixelShuffle(2)

        # Process the feature
        conv_block_lst = []
        # The input block
        conv_block_lst.append(
            nn.Sequential(
                nn.Conv2d(input_feat_dim, self.channel_conf[0], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.channel_conf[0]),
                nn.ReLU(inplace=True),
            )
        )

        # Intermediate block
        for channel in self.channel_conf[1:-1]:
            conv_block_lst.append(
                nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(channel),
                    nn.ReLU(inplace=True),
                )
            )

        # Output block
        conv_block_lst.append(
            nn.Sequential(nn.Conv2d(self.channel_conf[-1], output_channel, kernel_size=1, stride=1, padding=0))
        )
        self.conv_block_lst = nn.ModuleList(conv_block_lst)

    def get_channel_conf(self, num_upsample: int) -> List[int]:
        """Get num of channels based on number of upsampling."""
        if num_upsample == 2:
            return [256, 64, 16]
        return [256, 64, 16, 4]

    def forward(self, input_features: Tensor) -> Tensor:
        # Iterate til output block
        out = input_features
        for block in self.conv_block_lst[:-1]:
            out = block(out)
            out = self.pixshuffle(out)

        # Output layer
        out = self.conv_block_lst[-1](out)
        heatmap = softmax(out, dim=1)[:, 1, :, :]

        return heatmap


class SuperpointDescriptor(Module):
    """Descriptor decoder based on the SuperPoint arcihtecture.

    Args:
        input_feat_dim: channel size of the input features.
    Returns:
        the semi-dense descriptors with shape (B, 128, H/4, W/4).
    """

    def __init__(self, input_feat_dim: int = 128) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.convPa = nn.Conv2d(input_feat_dim, 256, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)

    def forward(self, input_features: Tensor) -> Tensor:
        feat = self.relu(self.convPa(input_features))
        semi = self.convPb(feat)

        return semi


# [Combination of all previous models in one]


class SOLD2Net(Module):
    """Full network for SOLD².

    Args:
        model_cfg: the configuration as a Dict.
    Returns:
        a Dict with the following values:
            junctions: heatmap of junctions.
            heatmap: line heatmap.
            descriptors: semi-dense descriptors.
    """

    def __init__(self, model_cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = model_cfg

        # Backbone
        self.backbone_net = HourglassBackbone(**self.cfg["backbone_cfg"])
        feat_channel = 256

        # Junction decoder
        self.junction_decoder = SuperpointDecoder(feat_channel, self.cfg["grid_size"])

        # Line heatmap decoder
        self.heatmap_decoder = PixelShuffleDecoder(feat_channel, num_upsample=2)

        # Descriptor decoder
        if "use_descriptor" in self.cfg:
            self.descriptor_decoder = SuperpointDescriptor(feat_channel)

    def forward(self, input_images: Tensor) -> Dict[str, Tensor]:
        # The backbone
        features = self.backbone_net(input_images)

        # junction decoder
        junctions = self.junction_decoder(features)

        # heatmap decoder
        heatmaps = self.heatmap_decoder(features)

        outputs = {"junctions": junctions, "heatmap": heatmaps}

        # Descriptor decoder
        if "use_descriptor" in self.cfg:
            outputs["descriptors"] = self.descriptor_decoder(features)

        return outputs
