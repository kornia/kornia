# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# --------------------------------------------------------
# RepVGG: Making VGG-style ConvNets Great Again (https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf)
# Github source: https://github.com/DingXiaoH/RepVGG
# Licensed under The MIT License [see LICENSE for details]
# Modified from: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
# --------------------------------------------------------
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils import checkpoint

from kornia.core import Tensor


def conv_bn(
    in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, groups: int = 1
) -> nn.Sequential:
    """Convolutional block builder function."""
    result = nn.Sequential()
    result.add_module(
        "conv",
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
    )
    result.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        deploy: bool = False,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        if kernel_size != 3:
            raise AssertionError(kernel_size)
        if padding != 1:
            raise AssertionError(padding)

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            #   Note that RepVGG-D2se uses SE before nonlinearity. But RepVGGplus models uses SE after nonlinearity.
            # self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
            raise ValueError("SEBlock not supported")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode,
            )
        else:
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            )
            self.rbr_dense = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
            )
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_11,
                groups=groups,
            )

    def forward(self, inputs: Tensor) -> Tensor:
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_custom_L2(self) -> Tensor:
        """Optional. This may improve the accuracy and facilitates quantization in some cases.

        1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
        2.  Use like this.
            loss = criterion(....)
            for every RepVGGBlock blk:
                loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
            optimizer.zero_grad()
            loss.backward()
        """
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (
            (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt()))
            .reshape(-1, 1, 1, 1)
            .detach()
        )
        t1 = (
            (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt()))
            .reshape(-1, 1, 1, 1)
            .detach()
        )

        l2_loss_circle = (K3**2).sum() - (
            K3[:, :, 1:2, 1:2] ** 2
        ).sum()  # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1  # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (
            eq_kernel**2 / (t3**2 + t1**2)
        ).sum()  # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle

    def get_equivalent_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        """This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.

        You can get the equivalent kernel and bias at any time and do whatever you want,
        for example, apply some penalties or constraints during training, just like you do to the other models.
        May be useful for quantization or pruning.
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1: Optional[Tensor]) -> Union[Tensor, int]:
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: Optional[nn.Module]) -> Tuple[Union[Tensor, int], Union[Tensor, int]]:
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            if not isinstance(branch, nn.BatchNorm2d):
                raise AssertionError(branch)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self) -> None:
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True


class RepVGG(nn.Module):
    def __init__(
        self,
        num_blocks: List[int],
        num_classes: int = 1000,
        width_multiplier: Optional[List[float]] = None,
        override_groups_map: Optional[Dict[int, int]] = None,
        deploy: bool = False,
        use_se: bool = False,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if width_multiplier is None and len(width_multiplier) != 4:
            raise AssertionError(width_multiplier)
        self.deploy = deploy
        self.override_groups_map = override_groups_map or {}
        if 0 in self.override_groups_map:
            raise AssertionError(self.override_groups_map)
        self.use_se = use_se
        self.use_checkpoint = use_checkpoint

        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock(
            in_channels=1,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=2,
            padding=1,
            deploy=self.deploy,
            use_se=self.use_se,
        )
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=1)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)

    def _make_stage(self, planes: int, num_blocks: int, stride: int) -> nn.ModuleList:
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for _stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(
                RepVGGBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=3,
                    stride=_stride,
                    padding=1,
                    groups=cur_groups,
                    deploy=self.deploy,
                    use_se=self.use_se,
                )
            )
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> Tensor:
        out = self.stage0(x)
        for stage in (self.stage1, self.stage2, self.stage3):
            for block in stage:
                if self.use_checkpoint:
                    out = checkpoint.checkpoint(block, out)
                else:
                    out = block(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = dict.fromkeys(optional_groupwise_layers, 2)
g4_map = dict.fromkeys(optional_groupwise_layers, 4)


def create_RepVGG(deploy: bool = False, use_checkpoint: bool = False) -> RepVGG:
    """Create RepVGG."""
    return RepVGG(
        num_blocks=[2, 4, 14, 1],
        num_classes=1000,
        width_multiplier=[1, 1, 1, 2.5],
        override_groups_map=None,
        deploy=deploy,
        use_checkpoint=use_checkpoint,
    )


def repvgg_model_convert(model: torch.nn.Module, save_path: Optional[str] = None, do_copy: bool = True) -> nn.Module:
    """Use this for converting a RepVGG model or a bigger model with RepVGG as its component.

    Use like this
    model = create_RepVGG_A0(deploy=False)
    train model or load weights
    repvgg_model_convert(model, save_path='repvgg_deploy.pth')
    If you want to preserve the original model, call with do_copy=True

    ================ for using RepVGG as the backbone of a bigger model, e.g., PSPNet, the pseudo code will be like
    train_backbone = create_RepVGG_B2(deploy=False)
    train_backbone.load_state_dict(torch.load('RepVGG-B2-train.pth'))
    train_pspnet = build_pspnet(backbone=train_backbone)
    segmentation_train(train_pspnet)
    deploy_pspnet = repvgg_model_convert(train_pspnet)
    segmentation_test(deploy_pspnet)
    ================ example_pspnet.py shows an example
    """
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "switch_to_deploy"):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model
