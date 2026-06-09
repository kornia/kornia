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

import re

import torch

__all__ = [
    "remap_hf_vision_state_dict",
]


_PREFIX_RE = re.compile(r"^(?:model\.)?visual\.")


def remap_hf_vision_state_dict(hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Filter and rename the HuggingFace state dict for Qwen3-VL's vision tower.

    The kornia ``Qwen3VLVisionModel`` already uses HuggingFace's submodule
    naming (``patch_embed``, ``blocks.{i}.attn.{qkv,proj}``, ``blocks.{i}.mlp.{linear_fc1,linear_fc2}``,
    ``norm1``/``norm2``, ``pos_embed``, ``merger``, ``deepstack_merger_list``),
    so this function only strips the ``model.visual.`` / ``visual.`` prefix
    and drops any non-vision keys. ``rotary_pos_emb.inv_freq`` is recomputed
    on construction and intentionally excluded.
    """
    out: dict[str, torch.Tensor] = {}
    for key, tensor in hf_state_dict.items():
        match = _PREFIX_RE.match(key)
        if match is None:
            continue
        new_key = key[match.end() :]
        if new_key.startswith("rotary_pos_emb.inv_freq"):
            continue
        out[new_key] = tensor
    return out
