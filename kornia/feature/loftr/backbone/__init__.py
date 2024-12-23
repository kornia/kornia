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

from __future__ import annotations

from typing import Any

from .resnet_fpn import ResNetFPN_8_2, ResNetFPN_16_4


def build_backbone(config: dict[str, Any]) -> ResNetFPN_8_2 | ResNetFPN_16_4:
    """Build model backbone."""
    if config["backbone_type"] == "ResNetFPN":
        if config["resolution"] == (8, 2):
            return ResNetFPN_8_2(config["resnetfpn"])
        elif config["resolution"] == (16, 4):
            return ResNetFPN_16_4(config["resnetfpn"])

    raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} with res {config['resolution']} not supported.")
