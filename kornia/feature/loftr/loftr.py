from typing import Dict, Optional

import torch
import torch.nn as nn

from .backbone import build_backbone
from .loftr_module import FinePreprocess, LocalFeatureTransformer
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching
from .utils.position_encoding import PositionEncodingSine

#from einops.einops import rearrange



urls: Dict[str, str] = {}
urls["outdoor"] = "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt"
urls["indoor"] = "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_indoor.ckpt"

default_cfg = {'backbone_type': 'ResNetFPN',
 'resolution': (8, 2),
 'fine_window_size': 5,
 'fine_concat_coarse_feat': True,
 'resnetfpn': {'initial_dim': 128, 'block_dims': [128, 196, 256]},
 'coarse': {'d_model': 256,
  'd_ffn': 256,
  'nhead': 8,
  'layer_names': ['self',
   'cross',
   'self',
   'cross',
   'self',
   'cross',
   'self',
   'cross'],
  'attention': 'linear',
  'temp_bug_fix': False},
 'match_coarse': {'thr': 0.2,
  'border_rm': 2,
  'match_type': 'dual_softmax',
  'dsmax_temperature': 0.1,
  'skh_iters': 3,
  'skh_init_bin_score': 1.0,
  'skh_prefilter': True,
  'train_coarse_percent': 0.4,
  'train_pad_num_gt_min': 200},
 'fine': {'d_model': 128,
  'd_ffn': 128,
  'nhead': 8,
  'layer_names': ['self', 'cross'],
  'attention': 'linear'}}


class LoFTR(nn.Module):
    def __init__(self, config=default_cfg, pretrained: Optional[str] = None):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = build_backbone(config)
        self.pos_encoding = PositionEncodingSine(
            config['coarse']['d_model'],
            temp_bug_fix=config['coarse']['temp_bug_fix'])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)
        self.loftr_fine = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching()

        if pretrained is not None:
            if pretrained not in urls.keys():
                raise ValueError(f"pretrained should be None or one of {urls.keys()}")
            pretrained_dict = torch.hub.load_state_dict_from_url(
                 urls[pretrained], map_location=lambda storage, loc: storage)
            self.load_state_dict(pretrained_dict['state_dict'])

    def forward(self, data):
        """
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_f.split(data['bs'])
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(data['image0']), self.backbone(data['image1'])

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

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
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
