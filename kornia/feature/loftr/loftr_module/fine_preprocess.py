from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core import Module, Tensor


class FinePreprocess(Module):
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
        self, feat_f0: Tensor, feat_f1: Tensor, feat_c0: Tensor, feat_c1: Tensor, data: Dict[str, Any]
    ) -> Tuple[Tensor, Tensor]:
        W = self.W
        stride = data["hw0_f"][0] // data["hw0_c"][0]

        data.update({"W": W})
        if data["b_ids"].shape[0] == 0:
            feat0 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
            feat1 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
            return feat0, feat1

        # 1. unfold(crop) all local windows
        feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=W // 2)

        # feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W**2)
        n0, cww0, l0 = feat_f0_unfold.shape
        c0 = cww0 // (W * W)
        feat_f0_unfold = feat_f0_unfold.reshape(n0, c0, -1, l0).permute(0, 3, 2, 1)

        feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=W // 2)
        # feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W**2)
        n1, cww1, l1 = feat_f1_unfold.shape
        c1 = cww1 // (W * W)
        feat_f1_unfold = feat_f1_unfold.reshape(n1, c1, -1, l1).permute(0, 3, 2, 1)

        # 2. select only the predicted matches
        feat_f0_unfold = feat_f0_unfold[data["b_ids"], data["i_ids"]]  # [n, ww, cf]
        feat_f1_unfold = feat_f1_unfold[data["b_ids"], data["j_ids"]]

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
