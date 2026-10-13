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


def _unfold_at(feat: torch.Tensor, b_ids: torch.Tensor, ids: torch.Tensor, window: int, stride: int) -> torch.Tensor:
    """Windows ``ids`` of ``F.unfold(feat, window, stride=stride, padding=window // 2)``, as ``(M, window**2, C)``.

    Gathers only the windows of the matched cells, with the same values and layout as unfolding the whole map and
    indexing it; the unfolded map is ``window**2`` times larger than ``feat``.
    """
    pad = window // 2
    padded = F.pad(feat, (pad, pad, pad, pad))
    # cell ids index the unfolded grid row-major
    w_out = (feat.shape[3] + 2 * pad - window) // stride + 1
    offsets = torch.arange(window, device=ids.device)
    rows = ((ids // w_out) * stride)[:, None] + offsets
    cols = ((ids % w_out) * stride)[:, None] + offsets
    windows = padded[b_ids[:, None, None], :, rows[:, :, None], cols[:, None, :]]  # (M, window, window, C)
    return windows.reshape(ids.shape[0], window * window, feat.shape[1])


class FinePreprocess(nn.Module):
    """Preprocess feature maps for fine-level matching in LoFTR.

    Args:
        config: A dictionary containing configuration parameters for fine-level processing.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        self.config = config
        self.cat_c_feat = config["fine_concat_coarse_feat"]
        self.W = self.config["fine_window_size"]

        d_model_c = self.config["coarse"]["d_model"]
        d_model_f = self.config["fine"]["d_model"]
        self.d_model_f = d_model_f
        if self.cat_c_feat:
            self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
            self.merge_feat = nn.Linear(2 * d_model_f, d_model_f, bias=True)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(
        self,
        feat_f0: torch.Tensor,
        feat_f1: torch.Tensor,
        feat_c0: torch.Tensor,
        feat_c1: torch.Tensor,
        data: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run this LoFTR component forward.

        Args:
            feat_f0: Fine feature map of the first image with shape :math:`(B, C_f, H_f, W_f)`.
            feat_f1: Fine feature map of the second image with shape :math:`(B, C_f, H_f, W_f)`.
            feat_c0: Coarse feature map or transformed tokens of the first image with shape
                :math:`(B, C_c, H_c, W_c)` or :math:`(N_c, C_c)`.
            feat_c1: Coarse feature map or transformed tokens of the second image with shape
                :math:`(B, C_c, H_c, W_c)` or :math:`(N_c, C_c)`.
            data: Dictionary containing coarse match metadata and indices, and the pixel strides ``stride_c`` and
                ``stride_f`` of the coarse and the fine feature maps.

        Returns:
            Tuple of local cropped fine feature patches ``(feat0, feat1)`` each with shape :math:`(N_m, W^2, C)`.
        """
        W = self.W
        # fine cells per coarse cell (4): the window of coarse cell i is centred on fine cell stride * i
        stride = data["stride_c"] // data["stride_f"]

        data.update({"W": W})
        if data["b_ids"].shape[0] == 0:
            feat0 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
            feat1 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
            return feat0, feat1

        # 1. crop the local windows of the predicted matches
        feat_f0_unfold = _unfold_at(feat_f0, data["b_ids"], data["i_ids"], W, stride)  # [n, ww, cf]
        feat_f1_unfold = _unfold_at(feat_f1, data["b_ids"], data["j_ids"], W, stride)

        # option: use coarse-level loftr feature as context: concat and linear
        if self.cat_c_feat:
            feat_c_win = self.down_proj(
                torch.cat([feat_c0[data["b_ids"], data["i_ids"]], feat_c1[data["b_ids"], data["j_ids"]]], 0)
            )  # [2n, c]
            feat_cf_win = self.merge_feat(
                torch.cat(
                    [
                        torch.cat([feat_f0_unfold, feat_f1_unfold], 0),  # [2n, ww, cf]
                        feat_c_win.unsqueeze(1).repeat(1, W**2, 1),  # [2n, ww, cf]
                    ],
                    -1,
                )
            )
            feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_cf_win, 2, dim=0)

        return feat_f0_unfold, feat_f1_unfold
