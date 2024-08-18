from __future__ import annotations

from typing import Any, Optional

import torch

from kornia.core import Module, Tensor
from kornia.geometry import resize

from .backbone import build_backbone
from .loftr_module import FinePreprocess, LocalFeatureTransformer
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching
from .utils.position_encoding import PositionEncodingSine

urls: dict[str, str] = {}
urls["outdoor"] = "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt"
urls["indoor_new"] = "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_indoor_ds_new.ckpt"
urls["indoor"] = "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_indoor.ckpt"

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


class LoFTR(Module):
    r"""Module, which finds correspondences between two images.

    This is based on the original code from paper "LoFTR: Detector-Free Local
    Feature Matching with Transformers". See :cite:`LoFTR2021` for more details.

    If the distance matrix dm is not provided, :py:func:`torch.cdist` is used.

    Args:
        config: Dict with initialization parameters. Do not pass it, unless you know what you are doing`.
        pretrained: Download and set pretrained weights to the model. Options: 'outdoor', 'indoor'.
                    'outdoor' is trained on the MegaDepth dataset and 'indoor'
                    on the ScanNet.

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
        self.config = config
        if pretrained == "indoor_new":
            self.config["coarse"]["temp_bug_fix"] = True
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
            if pretrained not in urls.keys():
                raise ValueError(f"pretrained should be None or one of {urls.keys()}")

            pretrained_dict = torch.hub.load_state_dict_from_url(urls[pretrained], map_location=torch.device("cpu"))
            self.load_state_dict(pretrained_dict["state_dict"])
        self.eval()

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """
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
            - ``confidence``, confidence score [0, 1] :math:`(NC)`.
            - ``batch_indexes``, batch indexes for the keypoints and lafs :math:`(NC)`.
        """
        # 1. Local Feature CNN
        _data: dict[str, Tensor | int | torch.Size] = {
            "bs": data["image0"].size(0),
            "hw0_i": data["image0"].shape[2:],
            "hw1_i": data["image1"].shape[2:],
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
        n, h, w, c = feat_c0.shape
        feat_c0 = feat_c0.reshape(n, -1, c)

        feat_c1 = self.pos_encoding(feat_c1).permute(0, 2, 3, 1)
        n1, h1, w1, c1 = feat_c1.shape
        feat_c1 = feat_c1.reshape(n1, -1, c1)

        mask_c0 = mask_c1 = None  # mask is useful in training
        if "mask0" in data:
            mask_c0 = resize(data["mask0"], _data["hw0_c"], interpolation="nearest").flatten(-2)
        if "mask1" in data:
            mask_c1 = resize(data["mask1"], _data["hw1_c"], interpolation="nearest").flatten(-2)
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
        out: dict[str, Tensor] = {}
        for k, v in rename_keys.items():
            _d = _data[k]
            if isinstance(_d, Tensor):
                out[v] = _d
            else:
                raise TypeError(f"Expected Tensor for item `{k}`. Gotcha {type(_d)}")
        return out

    def load_state_dict(self, state_dict: dict[str, Any], *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        for k in list(state_dict.keys()):
            if k.startswith("matcher."):
                state_dict[k.replace("matcher.", "", 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
