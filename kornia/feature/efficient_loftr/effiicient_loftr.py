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

from typing import Any, Dict, Optional

import torch

from kornia.core import Module, Tensor

from .backbone import build_backbone
from .eloftr_module import FinePreprocess, LocalFeatureTransformer
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching

urls: dict[str, str] = {}
urls["outdoor"] = "https://huggingface.co/kornia/Efficient_LOFTR/resolve/main/eloftr_outdoor_modified.ckpt"

default_cfg = {
    "backbone_type": "RepVGG",
    "align_corner": False,
    "resolution": (8, 1),
    "fine_window_size": 8,  # window_size in fine_level, must be even
    "mp": False,
    "replace_nan": True,
    "half": False,
    "backbone": {"block_dims": [64, 128, 256]},
    "coarse": {
        "d_model": 256,
        "d_ffn": 256,
        "nhead": 8,
        "layer_names": ["self", "cross", "self", "cross", "self", "cross", "self", "cross"],
        "agg_size0": 4,
        "agg_size1": 4,
        "no_flash": False,
        "rope": True,
        "npe": [832, 832, 832, 832],
    },
    "match_coarse": {
        "thr": 0.2,
        "border_rm": 2,
        "dsmax_temperature": 0.1,
        "skip_softmax": False,
        "fp16matmul": False,
        "train_coarse_percent": 0.2,
        "train_pad_num_gt_min": 200,
    },
    "match_fine": {"local_regress_temperature": 10.0, "local_regress_slicedim": 8},
}


def reparameter(matcher: Module) -> Module:
    """Helper function."""
    module = matcher.backbone.layer0
    if hasattr(module, "switch_to_deploy"):
        module.switch_to_deploy()
    for modules in [matcher.backbone.layer1, matcher.backbone.layer2, matcher.backbone.layer3]:
        for module in modules:
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
    for modules in [matcher.fine_preprocess.layer2_outconv2, matcher.fine_preprocess.layer1_outconv2]:
        for module in modules:
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
    return matcher


class EfficientLoFTR(Module):
    """Efficient LoFTR module get sparse-matching between two images.

    This is based on the original code from paper "Efficient LoFTR: Semi-Dense
    Local Feature Matching with Sparse-Like Speed". See :cite:`ELoFTR2024` for more details.

    Args:
        config: Dict with initialization parameters. Do not pass it, unless you know what you are doing.
        pretrained: Download and set pretrained weights to the model. Options: 'outdoor'

    Returns:
        Dictionary with image correspondences and confidence scores.

    Example:
        >>> img1 = torch.rand(1, 1, 320, 200)
        >>> img2 = torch.rand(1, 1, 128, 128)
        >>> input = {"image0": img1, "image1": img2}
        >>> eloftr = EfficientLoFTR()
        >>> out = eloftr(input)
    """

    def __init__(self, config: dict[str, Any] = default_cfg, pretrained: Optional[str] = "outdoor") -> None:
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = build_backbone(config)
        self.loftr_coarse = LocalFeatureTransformer(config)
        self.coarse_matching = CoarseMatching(config["match_coarse"])
        self.fine_preprocess = FinePreprocess(config)
        self.fine_matching = FineMatching(config)
        if pretrained is not None:
            if pretrained not in urls.keys():
                raise ValueError(f"pretrained should be None or one of {urls.keys()}")
            pretrained_dict = torch.hub.load_state_dict_from_url(
                urls[pretrained], map_location=torch.device("cpu"), weights_only=True
            )
            self.load_state_dict(pretrained_dict["state_dict"])
        self.eval()

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run forward function.

        Keyword Args:
            data:
                'image0': left image with shape :math: `(torch.Tensor): (N, 1, H1, W1)`
                'image1': right image with shape :math: `(torch.Tensor): (N, 1, H2, W2)`
                'mask0'(optional) : left image mask. '0' indicates a padded position :math: `(N, H1, W1)`.
                'mask1'(optional) : right image mask. '0' indicates a padded position :math: `(N, H2, W2)`.

        Returns:
            - ``keypoints0``, matching keypoints from image0 :math:`(NC, 2)`.
            - ``keypoints1``, matching keypoints from image1 :math:`(NC, 2)`.
            - ``confidence``, confidence score [0, 1] :math:`(NC)`.
            - ``batch_indexes``, batch indexes for the keypoints and lafs :math:`(NC)`.

        """
        # 1. Local Feature CNN
        _data: dict[str, Any] = {
            "bs": data["image0"].size(0),
            "hw0_i": data["image0"].shape[2:],
            "hw1_i": data["image1"].shape[2:],
        }

        if _data["hw0_i"] == _data["hw1_i"]:  # faster & better BN convergence
            ret_dict = self.backbone(torch.cat([data["image0"], data["image1"]], dim=0))
            feats_c = ret_dict["feats_c"]
            _data.update(
                {
                    "feats_x2": ret_dict["feats_x2"],
                    "feats_x1": ret_dict["feats_x1"],
                }
            )
            (feat_c0, feat_c1) = feats_c.split(_data["bs"])
        else:  # handle different input shapes
            ret_dict0, ret_dict1 = self.backbone(data["image0"]), self.backbone(data["image1"])
            feat_c0 = ret_dict0["feats_c"]
            feat_c1 = ret_dict1["feats_c"]
            _data.update(
                {
                    "feats_x2_0": ret_dict0["feats_x2"],
                    "feats_x1_0": ret_dict0["feats_x1"],
                    "feats_x2_1": ret_dict1["feats_x2"],
                    "feats_x1_1": ret_dict1["feats_x1"],
                }
            )

        mul = self.config["resolution"][0] // self.config["resolution"][1]
        _data.update(
            {
                "hw0_c": feat_c0.shape[2:],
                "hw1_c": feat_c1.shape[2:],
                "hw0_f": [feat_c0.shape[2] * mul, feat_c0.shape[3] * mul],
                "hw1_f": [feat_c1.shape[2] * mul, feat_c1.shape[3] * mul],
            }
        )

        # 2. coarse-level loftr module
        mask_c0 = mask_c1 = None  # mask is useful in training
        if "mask0" in data:
            mask_c0, mask_c1 = data["mask0"], data["mask1"]

        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # feat_c0 = rearrange(feat_c0, 'n c h w -> n (h w) c')
        feat_c0 = feat_c0.permute(0, 2, 3, 1)
        n, _, _, c = feat_c0.shape
        feat_c0 = feat_c0.reshape(n, -1, c)

        # feat_c1 = rearrange(feat_c1, 'n c h w -> n (h w) c')
        feat_c1 = feat_c1.permute(0, 2, 3, 1)
        n1, _, _, c1 = feat_c1.shape
        feat_c1 = feat_c1.reshape(n1, -1, c1)

        # detect NaN during mixed precision training
        # if self.config['replace_nan'] and (torch.any(torch.isnan(feat_c0)) or torch.any(torch.isnan(feat_c1))):
        #     detect_NaN(feat_c0, feat_c1)

        # 3. match coarse-level
        self.coarse_matching(
            feat_c0,
            feat_c1,
            _data,
            mask_c0=mask_c0.view(mask_c0.size(0), -1) if mask_c0 is not None else mask_c0,
            mask_c1=mask_c1.view(mask_c1.size(0), -1) if mask_c1 is not None else mask_c1,
        )

        # prevent fp16 overflow during mixed precision training
        # feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1] ** 0.5, [feat_c0, feat_c1])
        feat_c0, feat_c1 = (feat / feat.shape[-1] ** 0.5 for feat in [feat_c0, feat_c1])

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_c0, feat_c1, _data)

        # detect NaN during mixed precision training
        # if self.config['replace_nan'] and (torch.any(torch.isnan(feat_f0_unfold))
        #     or torch.any(torch.isnan(feat_f1_unfold))):
        #     detect_NaN(feat_f0_unfold, feat_f1_unfold)

        del feat_c0, feat_c1, mask_c0, mask_c1

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, _data)

        rename_keys: dict[str, str] = {
            "mkpts0_f": "keypoints0",
            "mkpts1_f": "keypoints1",
            "mconf": "confidence",
            "m_bids": "batch_indexes",
        }
        out: dict[str, Tensor] = {}
        for k, v in rename_keys.items():
            _d = _data[k]
            if isinstance(_d, Tensor):
                out[v] = _d
            else:
                raise TypeError(f"Expected Tensor for item `{k}`. Gotcha {type(_d)}")
        return out

    def load_state_dict(self, state_dict: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        for k in list(state_dict.keys()):
            if k.startswith("matcher."):
                state_dict[k.replace("matcher.", "", 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
