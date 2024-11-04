# adapted from: https://github.com/xavysp/DexiNed/blob/d944b70eb6eaf40e22f8467c1e12919aa600d8e4/model.py

from __future__ import annotations

from collections import OrderedDict
from typing import ClassVar, Optional

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core import ImageModule as Module
from kornia.core import Tensor, concatenate
from kornia.core.check import KORNIA_CHECK

url: str = "http://cmp.felk.cvut.cz/~mishkdmy/models/DexiNed_BIPED_10.pth"


def weight_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d,)):
        # torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        # torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, mean=0.0)

        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    # for fusion layer
    if isinstance(m, (nn.ConvTranspose2d,)):
        # torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        # torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, std=0.1)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class CoFusion(Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, out_ch, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.norm_layer1 = nn.GroupNorm(4, 64)
        self.norm_layer2 = nn.GroupNorm(4, 64)

    def forward(self, x: Tensor) -> Tensor:
        # fusecat = torch.cat(x, dim=1)
        attn = self.relu(self.norm_layer1(self.conv1(x)))
        attn = self.relu(self.norm_layer2(self.conv2(attn)))
        attn = F.softmax(self.conv3(attn), dim=1)

        # return ((fusecat * attn).sum(1)).unsqueeze(1)
        return ((x * attn).sum(1))[:, None, ...]


class _DenseLayer(nn.Sequential):
    def __init__(self, input_features: int, out_features: int) -> None:
        super().__init__(
            OrderedDict(
                [
                    ("relu1", nn.ReLU(inplace=True)),
                    ("conv1", nn.Conv2d(input_features, out_features, kernel_size=3, stride=1, padding=2, bias=True)),
                    ("norm1", nn.BatchNorm2d(out_features)),
                    ("relu2", nn.ReLU(inplace=True)),
                    ("conv2", nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, bias=True)),
                    ("norm2", nn.BatchNorm2d(out_features)),
                ]
            )
        )

    def forward(self, x: list[Tensor]) -> list[Tensor]:
        x1, x2 = x[0], x[1]
        x3: Tensor = x1
        for mod in self:
            x3 = mod(x3)
        return [0.5 * (x3 + x2), x2]


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers: int, input_features: int, out_features: int) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_features, out_features)
            self.add_module("denselayer%d" % (i + 1), layer)
            input_features = out_features

    def forward(self, x: list[Tensor]) -> list[Tensor]:
        x_out = x
        for mod in self:
            x_out = mod(x_out)
        return x_out


class UpConvBlock(Module):
    def __init__(self, in_features: int, up_scale: int) -> None:
        super().__init__()
        self.up_factor = 2
        self.constant_features = 16

        layers = self.make_deconv_layers(in_features, up_scale)
        KORNIA_CHECK(layers is not None, "layers cannot be none")
        self.features = nn.Sequential(*layers)

    def make_deconv_layers(self, in_features: int, up_scale: int) -> nn.ModuleList:
        layers = nn.ModuleList([])
        all_pads = [0, 0, 1, 3, 7]
        for i in range(up_scale):
            kernel_size = 2**up_scale
            pad = all_pads[up_scale]  # kernel_size-1
            out_features = self.compute_out_features(i, up_scale)
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ConvTranspose2d(out_features, out_features, kernel_size, stride=2, padding=pad))
            in_features = out_features
        return layers

    def compute_out_features(self, idx: int, up_scale: int) -> int:
        return 1 if idx == up_scale - 1 else self.constant_features

    def forward(self, x: Tensor, out_shape: list[int]) -> Tensor:
        out = self.features(x)
        out = F.interpolate(out, out_shape, mode="bilinear")
        return out


class SingleConvBlock(Module):
    def __init__(self, in_features: int, out_features: int, stride: int, use_bs: bool = True) -> None:
        super().__init__()
        self.use_bn = use_bs
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride, bias=True)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class DoubleConvBlock(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        mid_features: int,
        out_features: Optional[int] = None,
        stride: int = 1,
        use_act: bool = True,
    ) -> None:
        super().__init__()
        if out_features is None:
            out_features = mid_features
        self.add_module("conv1", nn.Conv2d(in_features, mid_features, 3, padding=1, stride=stride))
        self.add_module("bn1", nn.BatchNorm2d(mid_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(mid_features, out_features, 3, padding=1))
        self.add_module("bn2", nn.BatchNorm2d(out_features))
        if use_act:
            self.add_module("relu2", nn.ReLU(inplace=True))


class DexiNed(Module):
    r"""Definition of the DXtrem network from :cite:`xsoria2020dexined`.

    Return:
        A list of tensor with the intermediate features which the last element
        is the edges map with shape :math:`(B,1,H,W)`.

    Example:
        >>> img = torch.rand(1, 3, 320, 320)
        >>> net = DexiNed(pretrained=False)
        >>> out = net(img)
        >>> out.shape
        torch.Size([1, 1, 320, 320])
    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 1, -1, -1]

    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        self.block_1 = DoubleConvBlock(3, 32, 64, stride=2)
        self.block_2 = DoubleConvBlock(64, 128, use_act=False)
        self.dblock_3 = _DenseBlock(2, 128, 256)  # [128,256,100,100]
        self.dblock_4 = _DenseBlock(3, 256, 512)
        self.dblock_5 = _DenseBlock(3, 512, 512)
        self.dblock_6 = _DenseBlock(3, 512, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # left skip connections, figure in Journal
        self.side_1 = SingleConvBlock(64, 128, 2)
        self.side_2 = SingleConvBlock(128, 256, 2)
        self.side_3 = SingleConvBlock(256, 512, 2)
        self.side_4 = SingleConvBlock(512, 512, 1)
        self.side_5 = SingleConvBlock(512, 256, 1)

        # right skip connections, figure in Journal paper
        self.pre_dense_2 = SingleConvBlock(128, 256, 2)
        self.pre_dense_3 = SingleConvBlock(128, 256, 1)
        self.pre_dense_4 = SingleConvBlock(256, 512, 1)
        self.pre_dense_5 = SingleConvBlock(512, 512, 1)
        self.pre_dense_6 = SingleConvBlock(512, 256, 1)

        # USNet
        self.up_block_1 = UpConvBlock(64, 1)
        self.up_block_2 = UpConvBlock(128, 1)
        self.up_block_3 = UpConvBlock(256, 2)
        self.up_block_4 = UpConvBlock(512, 3)
        self.up_block_5 = UpConvBlock(512, 4)
        self.up_block_6 = UpConvBlock(256, 4)
        self.block_cat = SingleConvBlock(6, 1, stride=1, use_bs=False)  # hed fusion method
        # self.block_cat = CoFusion(6,6)# cats fusion method

        if pretrained:
            self.load_from_file(url)
        else:
            self.apply(weight_init)

    def load_from_file(self, path_file: str) -> None:
        # use torch.hub to load pretrained model
        pretrained_dict = torch.hub.load_state_dict_from_url(path_file, map_location=torch.device("cpu"))
        self.load_state_dict(pretrained_dict, strict=True)
        self.eval()

    def get_features(self, x: Tensor) -> list[Tensor]:
        # Block 1
        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)

        # Block 2
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = block_2_down + block_1_side
        block_2_side = self.side_2(block_2_add)

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])
        block_3_down = self.maxpool(block_3)  # [128,256,50,50]
        block_3_add = block_3_down + block_2_side
        block_3_side = self.side_3(block_3_add)

        # Block 4
        block_2_resize_half = self.pre_dense_2(block_2_down)
        block_4_pre_dense = self.pre_dense_4(block_3_down + block_2_resize_half)
        block_4, _ = self.dblock_4([block_3_add, block_4_pre_dense])
        block_4_down = self.maxpool(block_4)
        block_4_add = block_4_down + block_3_side
        block_4_side = self.side_4(block_4_add)

        # Block 5
        block_5_pre_dense = self.pre_dense_5(block_4_down)  # block_5_pre_dense_512 +block_4_down
        block_5, _ = self.dblock_5([block_4_add, block_5_pre_dense])
        block_5_add = block_5 + block_4_side

        # Block 6
        block_6_pre_dense = self.pre_dense_6(block_5)
        block_6, _ = self.dblock_6([block_5_add, block_6_pre_dense])

        # upsampling blocks
        out_shape = x.shape[-2:]
        out_1 = self.up_block_1(block_1, out_shape)
        out_2 = self.up_block_2(block_2, out_shape)
        out_3 = self.up_block_3(block_3, out_shape)
        out_4 = self.up_block_4(block_4, out_shape)
        out_5 = self.up_block_5(block_5, out_shape)
        out_6 = self.up_block_6(block_6, out_shape)
        results = [out_1, out_2, out_3, out_4, out_5, out_6]
        return results

    def forward(self, x: Tensor) -> Tensor:
        features = self.get_features(x)

        # concatenate multiscale outputs
        block_cat = concatenate(features, 1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW

        return block_cat
