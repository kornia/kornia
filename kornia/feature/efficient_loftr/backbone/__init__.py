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

from .backbone import RepVGG_8_1_align


def build_backbone(config: Any) -> RepVGG_8_1_align:
    """Building backbone model."""
    if config["backbone_type"] == "RepVGG":
        if config["align_corner"] is False:
            if config["resolution"] == (8, 1):
                return RepVGG_8_1_align(config["backbone"])
        else:
            raise ValueError(f"Efficient LOFTR.ALIGN_CORNER {config['align_corner']} not supported.")
    else:
        raise ValueError(f"Efficient LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")
