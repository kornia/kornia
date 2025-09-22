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

from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F

from kornia.core import Module, Tensor

INF = 1e9


def mask_border(m: Tensor, b: int, v: Union[Tensor, float, bool]) -> None:
    """Mask borders with value.

    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int) : boarder
        v (m.dtype) : mask value
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


def mask_border_with_padding(m: Tensor, bd: int, v: Union[Tensor, float, bool], p_m0: Tensor, p_m1: Tensor) -> None:
    """Apply masking to a padded boarder."""
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd :] = v
        m[b_idx, :, w0 - bd :] = v
        m[b_idx, :, :, h1 - bd :] = v
        m[b_idx, :, :, :, w1 - bd :] = v


def compute_max_candidates(p_m0: Tensor, p_m1: Tensor) -> Tensor:
    """Compute the max candidates of all pairs within a batch.

    Args:
        p_m0 : padded masks
        p_m1 : padded masks
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand


class CoarseMatching(Module):
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
        # N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)

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

    @torch.no_grad()
    def get_coarse_match(self, conf_matrix: Tensor, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Get coarse matching.

        Args:
            conf_matrix (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']

        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        axes_lengths = {
            "h0c": data["hw0_c"][0],
            "w0c": data["hw0_c"][1],
            "h1c": data["hw1_c"][0],
            "w1c": data["hw1_c"][1],
        }
        _device = conf_matrix.device
        # 1. confidence thresholding
        mask = conf_matrix > self.thr
        N = conf_matrix.shape[0]
        # mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
        #                  **axes_lengths)
        mask = mask.reshape(N, axes_lengths["h0c"], axes_lengths["w0c"], axes_lengths["h1c"], axes_lengths["w1c"])

        if "mask0" not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(mask, self.border_rm, False, data["mask0"], data["mask1"])
        # mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
        #                  **axes_lengths)
        mask = mask.reshape(N, axes_lengths["h0c"] * axes_lengths["w0c"], axes_lengths["h1c"] * axes_lengths["w1c"])

        # 2. mutual nearest
        mask = (
            mask
            * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0])
            * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])
        )

        # 3. find all valid coarse matches
        # this only works when at most one `True` in each row
        mask_v, all_j_ids = mask.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        mconf = conf_matrix[b_ids, i_ids, j_ids]

        # 4. Random sampling of training samples for fine-level LoFTR
        # (optional) pad samples with gt coarse-level matches
        if self.training:
            # NOTE:
            # The sampling is performed across all pairs in a batch without manually balancing
            # #samples for fine-level increases w.r.t. batch_size
            if "mask0" not in data:
                num_candidates_max = mask.size(0) * max(mask.size(1), mask.size(2))
            else:
                num_candidates_max = compute_max_candidates(data["mask0"], data["mask1"])
            num_matches_train = int(num_candidates_max * self.train_coarse_percent)
            num_matches_pred = len(b_ids)
            if self.train_pad_num_gt_min >= num_matches_train:
                raise AssertionError("min-num-gt-pad should be less than num-train-matches")

            # pred_indices is to select from prediction
            if num_matches_pred <= num_matches_train - self.train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=_device)
            else:
                pred_indices = torch.randint(
                    num_matches_pred, (num_matches_train - self.train_pad_num_gt_min,), device=_device
                )

            # gt_pad_indices is to select from gt padding. e.g. max(3787-4800, 200)
            gt_pad_indices = torch.randint(
                len(data["spv_b_ids"]),
                (max(num_matches_train - num_matches_pred, self.train_pad_num_gt_min),),
                device=_device,
            )
            mconf_gt = torch.zeros(len(data["spv_b_ids"]), device=_device)  # set conf of gt paddings to all zero

            b_ids, i_ids, j_ids, mconf = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]], dim=0),
                *zip(
                    [b_ids, data["spv_b_ids"]],
                    [i_ids, data["spv_i_ids"]],
                    [j_ids, data["spv_j_ids"]],
                    [mconf, mconf_gt],
                ),
            )

        # These matches select patches that feed into fine-level network
        coarse_matches = {"b_ids": b_ids, "i_ids": i_ids, "j_ids": j_ids}

        # 4. Update with matches in original image resolution
        scale = data["hw0_i"][0] / data["hw0_c"][0]

        scale0 = scale * data["scale0"][b_ids] if "scale0" in data else scale
        scale1 = scale * data["scale1"][b_ids] if "scale1" in data else scale
        mkpts0_c = torch.stack([i_ids % data["hw0_c"][1], i_ids // data["hw0_c"][1]], dim=1) * scale0
        mkpts1_c = torch.stack([j_ids % data["hw1_c"][1], j_ids // data["hw1_c"][1]], dim=1) * scale1

        m_bids = b_ids[mconf != 0]
        # These matches is the current prediction (for visualization)
        coarse_matches.update(
            {
                "m_bids": m_bids,  # mconf == 0 => gt matches
                "mkpts0_c": mkpts0_c[mconf != 0],
                "mkpts1_c": mkpts1_c[mconf != 0],
                "mconf": mconf[mconf != 0],
            }
        )

        return coarse_matches
