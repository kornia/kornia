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

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from kornia.core import Tensor
from kornia.feature.loftr.utils.coarse_matching import BaseCoarseMatching

INF = 1e9


class CoarseMatching(BaseCoarseMatching):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        # general config
        self.thr = config["thr"]
        self.border_rm = config["border_rm"]
        self.temperature = config["dsmax_temperature"]
        self.skip_softmax = config["skip_softmax"]
        self.fp16matmul = config["fp16matmul"]
        # -- # for training fine-level LoFTR
        self.train_coarse_percent = config["train_coarse_percent"]
        self.train_pad_num_gt_min = config["train_pad_num_gt_min"]

    def forward(
        self,
        feat_c0: Tensor,
        feat_c1: Tensor,
        data: Dict[str, Tensor],
        mask_c0: Optional[Tensor] = None,
        mask_c1: Optional[Tensor] = None,
    ) -> None:
        """Run Forward.

        Args:
            feat_c0 : [N, L, C]
            feat_c1 : [N, S, C]
            data (dict)
            mask_c0 : [N, L] (optional)
            mask_c1 : [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' : [M'],
                'i_ids' : [M'],
                'j_ids' : [M'],
                'm_bids' : [M],
                'mkpts0_c' : [M, 2],
                'mkpts1_c' : [M, 2],
                'mconf' : [M]}
            NOTE: M' != M during training.
        """
        # normalize
        feat_c0, feat_c1 = (feat / feat.shape[-1] ** 0.5 for feat in [feat_c0, feat_c1])

        if self.fp16matmul:
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
            del feat_c0, feat_c1
            if mask_c0 is not None and mask_c1 is not None:
                sim_matrix = sim_matrix.masked_fill(~(mask_c0[..., None] * mask_c1[:, None]).bool(), -1e4)
        else:
            with torch.autocast(enabled=False, device_type="cuda"):
                sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
                del feat_c0, feat_c1
                if mask_c0 is not None and mask_c1 is not None:
                    sim_matrix = sim_matrix.float().masked_fill(~(mask_c0[..., None] * mask_c1[:, None]).bool(), -INF)
        if not self.skip_softmax:
            sim_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        data.update({"conf_matrix": sim_matrix})

        # predict coarse matches from conf_matrix
        data.update(**self.get_coarse_match(sim_matrix, data))
