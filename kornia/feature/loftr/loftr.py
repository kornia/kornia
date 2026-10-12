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

import copy
from typing import Any, Optional

import torch
from torch import nn

from kornia.core.download import hf_url, load_state_dict_from_url

from .backbone import build_backbone
from .loftr_module import FinePreprocess, LocalFeatureTransformer
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching
from .utils.position_encoding import PositionEncodingSine

urls: dict[str, str | list[str]] = {}
urls["outdoor"] = [
    hf_url("loftr", "loftr_outdoor.ckpt"),
    "https://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt",
]
urls["indoor_new"] = [
    hf_url("loftr", "loftr_indoor_ds_new.ckpt"),
    "https://cmp.felk.cvut.cz/~mishkdmy/models/loftr_indoor_ds_new.ckpt",
]
urls["indoor"] = [
    hf_url("loftr", "loftr_indoor.ckpt"),
    "https://cmp.felk.cvut.cz/~mishkdmy/models/loftr_indoor.ckpt",
]

# Comments: the config below is the one corresponding to the pretrained models
# Some do not change there anything, unless you want to retrain it.

default_cfg = {
    "backbone_type": "ResNetFPN",
    "resolution": (8, 2),
    "fine_window_size": 5,
    "fine_concat_coarse_feat": True,
    "resnetfpn": {"initial_dim": 128, "block_dims": [128, 196, 256]},
    "coarse": {
        "d_model": 256,
        "d_ffn": 256,
        "nhead": 8,
        "layer_names": ["self", "cross", "self", "cross", "self", "cross", "self", "cross"],
        "attention": "linear",
        "temp_bug_fix": False,
    },
    "match_coarse": {
        "thr": 0.2,
        "border_rm": 2,
        "match_type": "dual_softmax",
        "dsmax_temperature": 0.1,
        "skh_iters": 3,
        "skh_init_bin_score": 1.0,
        "skh_prefilter": True,
        "train_coarse_percent": 0.4,
        "train_pad_num_gt_min": 200,
    },
    "fine": {"d_model": 128, "d_ffn": 128, "nhead": 8, "layer_names": ["self", "cross"], "attention": "linear"},
}


class LoFTR(nn.Module):
    r"""nn.Module, which finds correspondences between two images.

    This is based on the original code from paper "LoFTR: Detector-Free Local
    Feature Matching with Transformers". See :cite:`LoFTR2021` for more details.

    Convention:
        - ``image0`` and ``image1`` are grayscale :math:`(N, 1, H, W)` images in :math:`[0, 1]`, the input of the
          pretrained models. With the default ``config``, whose ``resolution`` (8, 2) every pretrained model uses,
          the two images can differ in size, and their heights and widths do not need to be multiples of 8 (the
          reference implementation expects multiples of 8).
        - ``keypoints0`` lie on the grid of the 1/8-resolution coarse feature map, at ``8 * (column, row)`` of a
          coarse cell, and ``keypoints1`` at ``8 * (column, row)`` of the matching cell of image1 plus the offset
          found by the fine stage, at most 4 pixels along each axis. Both are :math:`(x, y)` in the frame of the
          reference implementation, in which the pretrained models put the centre of pixel :math:`(x, y)` at
          :math:`(x + 0.5, y + 0.5)`: subtract 0.5 to get kornia's frame
          (:func:`~kornia.feature.laf_from_center_scale_ori`), in which pixel centres have integer coordinates.
        - ``confidence`` is the coarse dual-softmax score of the match; only matches above the ``thr`` of
          ``config["match_coarse"]``, 0.2 by default, are returned.
        - With ``resolution`` (16, 4) in ``config``, ``keypoints0`` lie on a 16-pixel grid, and most sizes that are
          not multiples of 16 raise an error.

    Args:
        config: Dict with initialization parameters. Do not pass it, unless you know what you are doing.
                The model works on its own deep copy, so the dict you pass is never modified.
        pretrained: Download and set pretrained weights to the model. Options: 'outdoor', 'indoor', 'indoor_new'.
                    'outdoor' is trained on the MegaDepth dataset and 'indoor' and 'indoor_new'
                    on the ScanNet. 'indoor_new' also switches the positional encoding of the copy
                    to ``temp_bug_fix=True``, which is what its checkpoint was trained with. ``None`` loads no
                    weights.

    Returns:
        Dictionary with image correspondences and confidence scores.

    Example:
        >>> img1 = torch.rand(1, 1, 320, 200)
        >>> img2 = torch.rand(1, 1, 128, 128)
        >>> input = {"image0": img1, "image1": img2}
        >>> loftr = LoFTR('outdoor')
        >>> out = loftr(input)

    """

    def __init__(self, pretrained: Optional[str] = "outdoor", config: dict[str, Any] = default_cfg) -> None:
        super().__init__()
        # Misc
        # Each instance owns its config: neither the module-level ``default_cfg`` nor a caller-passed dict is mutated.
        config = copy.deepcopy(config)
        if pretrained == "indoor_new":
            config["coarse"]["temp_bug_fix"] = True
        self.config = config
        # Modules
        self.backbone = build_backbone(config)
        self.pos_encoding = PositionEncodingSine(
            config["coarse"]["d_model"], temp_bug_fix=config["coarse"]["temp_bug_fix"]
        )
        self.loftr_coarse = LocalFeatureTransformer(config["coarse"])
        self.coarse_matching = CoarseMatching(config["match_coarse"])
        self.fine_preprocess = FinePreprocess(config)
        self.loftr_fine = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching()
        self.pretrained = pretrained
        if pretrained is not None:
            if pretrained not in urls:
                raise ValueError(f"pretrained should be None or one of {urls.keys()}")

            pretrained_dict = load_state_dict_from_url(urls[pretrained], map_location=torch.device("cpu"))
            self.load_state_dict(pretrained_dict["state_dict"])
        self.eval()

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run forward.

        Args:
            data: dictionary containing the input data in the following format:

        Keyword Args:
            image0: left image with shape :math:`(N, 1, H1, W1)`.
            image1: right image with shape :math:`(N, 1, H2, W2)`.
            mask0 (optional): left image mask. '0' indicates a padded position :math:`(N, H1, W1)`.
            mask1 (optional): right image mask. '0' indicates a padded position :math:`(N, H2, W2)`.

        Returns:
            - ``keypoints0``, matching keypoints from image0 :math:`(NC, 2)`.
            - ``keypoints1``, matching keypoints from image1 :math:`(NC, 2)`.
            - ``confidence``, the coarse matching score, see the Convention block :math:`(NC)`.
            - ``batch_indexes``, batch index of each match :math:`(NC)`.

        """
        # Pixel strides of the coarse and the fine feature map, (8, 2) by default. Every stride-2 stage of the backbone
        # has a centred kernel, so coarse cell (x, y) sits on pixel (8x, 8y) and fine cell (x, y) on pixel (2x, 2y)
        # for any image size. The maps have ceil(H / 8) and ceil(H / 2) rows, so H divided by a map size is not the
        # stride.
        stride_c, stride_f = self.config["resolution"]

        # 1. Local Feature CNN
        _data: dict[str, torch.Tensor | int | torch.Size] = {
            "bs": data["image0"].size(0),
            "hw0_i": data["image0"].shape[2:],
            "hw1_i": data["image1"].shape[2:],
            "stride_c": stride_c,
            "stride_f": stride_f,
        }

        if _data["hw0_i"] == _data["hw1_i"]:  # faster & better BN convergence
            feats_c, feats_f = self.backbone(torch.cat([data["image0"], data["image1"]], dim=0))
            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(_data["bs"]), feats_f.split(_data["bs"])
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(data["image0"]), self.backbone(data["image1"])

        _data.update(
            {
                "hw0_c": feat_c0.shape[2:],
                "hw1_c": feat_c1.shape[2:],
                "hw0_f": feat_f0.shape[2:],
                "hw1_f": feat_f1.shape[2:],
            }
        )

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]

        # feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        # feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')
        feat_c0 = self.pos_encoding(feat_c0).permute(0, 2, 3, 1)
        n, _h, _w, c = feat_c0.shape
        feat_c0 = feat_c0.reshape(n, -1, c)

        feat_c1 = self.pos_encoding(feat_c1).permute(0, 2, 3, 1)
        n1, _h1, _w1, c1 = feat_c1.shape
        feat_c1 = feat_c1.reshape(n1, -1, c1)

        mask_c0 = mask_c1 = None  # mask is useful in training
        # a coarse cell takes the mask value of the pixel it sits on
        if "mask0" in data:
            mask_c0 = data["mask0"][..., ::stride_c, ::stride_c].flatten(-2)
        if "mask1" in data:
            mask_c1 = data["mask1"][..., ::stride_c, ::stride_c].flatten(-2)
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, _data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, _data)
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, _data)

        rename_keys: dict[str, str] = {
            "mkpts0_f": "keypoints0",
            "mkpts1_f": "keypoints1",
            "mconf": "confidence",
            "b_ids": "batch_indexes",
        }
        out: dict[str, torch.Tensor] = {}
        for k, v in rename_keys.items():
            _d = _data[k]
            if isinstance(_d, torch.Tensor):
                out[v] = _d
            else:
                raise TypeError(f"Expected torch.Tensor for item `{k}`. Got {type(_d)}")
        return out

    def load_state_dict(self, state_dict: dict[str, Any], *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        """Load a checkpoint state dictionary into the module.

        Args:
            state_dict: Checkpoint state dictionary whose keys are adapted to this module layout.
            *args: Additional positional arguments forwarded to the parent `load_state_dict` implementation.
            **kwargs: Additional keyword arguments forwarded to the parent `load_state_dict` implementation.

        Returns:
            Result returned by the parent `load_state_dict` implementation.
        """
        for k in list(state_dict.keys()):
            if k.startswith("matcher."):
                state_dict[k.replace("matcher.", "", 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
