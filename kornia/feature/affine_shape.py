from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.feature import HessianResp
from kornia.feature.laf import (
    denormalize_LAF,
    normalize_LAF,
    make_upright,
    scale_LAF,
    extract_patches_from_pyramid,
    ell2LAF,
    get_laf_scale)



class AffineShapeEstimator(nn.Module):
    '''As is Hessian-Affine paper
    '''
    def __init__(self, threshold = 0.001, patch_size = 19):
        super(AffineShapeEstimator, self).__init__()
        self.threshold = threshold;
        self.PS = patch_size
        self.gx =  nn.Conv2d(1, 1, kernel_size=(1,3), bias = False)
        self.gx.weight.data = torch.tensor([[[[0.5, 0, -0.5]]]]).float()
        self.gy =  nn.Conv2d(1, 1, kernel_size=(3,1), bias = False)
        self.gy.weight.data = torch.tensor([[[[0.5], [0], [-0.5]]]]).float()
        #self.gk = torch.from_numpy(CircularGaussKernel(kernlen = self.PS, sigma = (self.PS / 2) /3.0).astype(np.float32))
        return
    def forward(self, laf, img):
        B,N,_,_ = laf.size()
        B,CH,H,W = img.size()
        patches = extract_patches_from_pyramid(img,
                                 laf,
                                 self.PS).view(B*N,CH,PS,PS)
        gx = self.gx(F.pad(patches, (1, 1, 0, 0), 'replicate')).view(B,N,CH,PS,PS)
        gy = self.gy(F.pad(patches, (0, 0, 1, 1), 'replicate')).view(B,N,CH,PS,PS)
        ells = torch.cat([laf[:,:,:,2], 
                          gx.pow(2).mean(dim=2).mean(dim=2,keepdim=True),
                          (gx*gy).mean(dim=2).mean(dim=2,keepdim=True),
                          gy.pow(2).mean(dim=2).mean(dim=2,keepdim=True)],dim=2)
        scale_orig = get_laf_scale(laf)
        lafs_new = ell2LAF(ells)
        lafs_new = scale_LAF(make_upright(lafs_new, False), scale_orig)
        return lafs_new
