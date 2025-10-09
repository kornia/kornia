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

from typing import Any, Dict

from torch import nn

from kornia.core import Module, Tensor

from .repvgg import create_RepVGG


class RepVGG_8_1_align(Module):
    """RepVGG backbone, output resolution are 1/8 and 1.

    Each block has 2 layers.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        backbone = create_RepVGG(False)

        self.layer0, self.layer1, self.layer2, self.layer3 = (
            backbone.stage0,
            backbone.stage1,
            backbone.stage2,
            backbone.stage3,
        )

        for layer in [self.layer0, self.layer1, self.layer2, self.layer3]:
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = self.layer0(x)  # 1/2
        for module in self.layer1:
            out = module(out)  # 1/2
        x1 = out
        for module in self.layer2:
            out = module(out)  # 1/4
        x2 = out
        for module in self.layer3:
            out = module(out)  # 1/8
        x3 = out

        return {"feats_c": x3, "feats_f": None, "feats_x2": x2, "feats_x1": x1}
