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

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py


from typing import List

from torch import nn

from .vision_transformer import (
    vit_base,
    vit_giant2,
    vit_large,
    vit_small,
)


class DinoV2(nn.Module):
    def __init__(
        self,
        name: str,
        out_layers: List[int],
        alt_start: int = -1,
        qknorm_start: int = -1,
        rope_start: int = -1,
        cat_token: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert name in {"vits", "vitb", "vitl", "vitg"}
        self.name = name
        self.out_layers = out_layers
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.cat_token = cat_token
        encoder_map = {
            "vits": vit_small,
            "vitb": vit_base,
            "vitl": vit_large,
            "vitg": vit_giant2,
        }
        encoder_fn = encoder_map[self.name]
        ffn_layer = "swiglufused" if self.name == "vitg" else "mlp"
        self.pretrained = encoder_fn(
            img_size=518,
            patch_size=14,
            ffn_layer=ffn_layer,
            alt_start=alt_start,
            qknorm_start=qknorm_start,
            rope_start=rope_start,
            cat_token=cat_token,
        )

    def forward(self, x, **kwargs):
        return self.pretrained.get_intermediate_layers(
            x,
            self.out_layers,
            **kwargs,
        )
