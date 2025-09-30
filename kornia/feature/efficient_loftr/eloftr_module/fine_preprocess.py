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

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core import Module, Tensor
from kornia.feature.loftr.backbone.resnet_fpn import conv1x1, conv3x3


class FinePreprocess(Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        self.config = config
        block_dims = config["backbone"]["block_dims"]
        self.W = self.config["fine_window_size"]
        self.fine_d_model = block_dims[0]

        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def inter_fpn(self, feat_c: Tensor, x2: Tensor, x1: Tensor, stride: int) -> Tensor:
        feat_c = self.layer3_outconv(feat_c)
        feat_c = F.interpolate(feat_c, scale_factor=2.0, mode="bilinear", align_corners=False)

        x2 = self.layer2_outconv(x2)
        x2 = self.layer2_outconv2(x2 + feat_c)
        x2 = F.interpolate(x2, scale_factor=2.0, mode="bilinear", align_corners=False)

        x1 = self.layer1_outconv(x1)
        x1 = self.layer1_outconv2(x1 + x2)
        x1 = F.interpolate(x1, scale_factor=2.0, mode="bilinear", align_corners=False)
        return x1

    def forward(self, feat_c0: Tensor, feat_c1: Tensor, data: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
        W = self.W
        stride = data["hw0_f"][0] // data["hw0_c"][0]

        data.update({"W": W})
        if data["b_ids"].shape[0] == 0:
            feat0 = torch.empty(0, self.W**2, self.fine_d_model, device=feat_c0.device)
            feat1 = torch.empty(0, self.W**2, self.fine_d_model, device=feat_c0.device)
            return feat0, feat1

        if data["hw0_i"] == data["hw1_i"]:
            # 1/8 feat
            # feat_c = rearrange(torch.cat([feat_c0, feat_c1], 0), 'b (h w) c -> b c h w', h=data['hw0_c'][0])
            feat_c = torch.cat([feat_c0, feat_c1], 0)
            b, hw, c = feat_c.shape
            w = hw // data["hw0_c"][0]
            feat_c = feat_c.reshape(b, -1, w, c).permute(0, 3, 1, 2)

            x2 = data["feats_x2"]  # 1/4 feat
            x1 = data["feats_x1"]  # 1/2 feat
            del data["feats_x2"], data["feats_x1"]

            # 1. fine feature extraction
            x1 = self.inter_fpn(feat_c, x2, x1, stride)
            feat_f0, feat_f1 = torch.chunk(x1, 2, dim=0)

        else:  # handle different input shapes
            # feat_c0 = rearrange(feat_c0, 'b (h w) c -> b c h w', h=data['hw0_c'][0])  # 1/8 feat
            b0, hw0, c0 = feat_c0.shape
            w0 = hw0 // data["hw0_c"][0]
            feat_c0 = feat_c0.reshape(b0, -1, w0, c0).permute(0, 3, 1, 2)

            # feat_c1 = rearrange(feat_c1, 'b (h w) c -> b c h w', h=data['hw1_c'][0])
            b1, hw1, c1 = feat_c1.shape
            w1 = hw1 // data["hw1_c"][0]
            feat_c1 = feat_c1.reshape(b1, -1, w1, c1).permute(0, 3, 1, 2)

            x2_0, x2_1 = data["feats_x2_0"], data["feats_x2_1"]  # 1/4 feat
            x1_0, x1_1 = data["feats_x1_0"], data["feats_x1_1"]  # 1/2 feat
            del data["feats_x2_0"], data["feats_x1_0"], data["feats_x2_1"], data["feats_x1_1"]

            # 1. fine feature extraction
            feat_f0, feat_f1 = self.inter_fpn(feat_c0, x2_0, x1_0, stride), self.inter_fpn(feat_c1, x2_1, x1_1, stride)

        # 2. unfold(crop) all local windows
        feat_f0 = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=0)
        # feat_f0 = rearrange(feat_f0, 'n (c ww) l -> n l ww c', ww=W**2)
        n0, cww0, l0 = feat_f0.shape
        c0 = cww0 // W**2
        feat_f0 = feat_f0.reshape(n0, c0, -1, l0).permute(0, 3, 2, 1)

        feat_f1 = F.unfold(feat_f1, kernel_size=(W + 2, W + 2), stride=stride, padding=1)
        # feat_f1 = rearrange(feat_f1, 'n (c ww) l -> n l ww c', ww=(W+2)**2)
        n1, cww1, l1 = feat_f1.shape
        c1 = cww1 // (W + 2) ** 2
        feat_f1 = feat_f1.reshape(n1, c1, -1, l1).permute(0, 3, 2, 1)

        # 3. select only the predicted matches
        feat_f0 = feat_f0[data["b_ids"], data["i_ids"]]  # [n, ww, cf]
        feat_f1 = feat_f1[data["b_ids"], data["j_ids"]]

        return feat_f0, feat_f1
