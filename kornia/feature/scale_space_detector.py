from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.feature.nms import SoftNMS3d
from kornia.feature import HessianResp
from kornia.feature.laf import (
    denormalize_LAF,
    normalize_LAF,
    maxima_coords_to_LAF)

from kornia.geometry.transform import ScalePyramid

class PassLAF(nn.Module):
    def forward(self, laf, img):
        return laf

class ScaleSpaceDetector(nn.Module):
    def __init__(self,
                 num_features = 500,
                 mrSize = 5.192,
                 scalepyr = None,
                 nms = None,
                 resp = None,
                 ori = None,
                 aff = None,
                 **kwargs):
        super(ScaleSpaceDetector, self).__init__()
        self.mrSize = mrSize
        self.num_features = num_features
        
        if scalepyr is not None:
            self.scalepyr = scalepyr
        else:
            self.scalepyr = ScalePyramid(3, 1.6, 10)
        
        if resp is not None:
            self.resp = resp
        else:
            self.resp = HessianResp(False,False,False)
            
        if nms is not None:
            self.nms = nms
        else:
            self.nms = SoftNMS3d((self.scalepyr.n_levels, 3, 3), 
                                 stride = 2, 
                                 padding = 1,
                                 temperature = 5)
        if ori is not None:
            self.ori = ori
        else:
            self.ori = PassLAF()
        
        if aff is not None:
            self.aff = aff
        else:
            self.aff = PassLAF()    
        return
    def __repr__(self):
        return
    
    def detect(self, img, num_feats = 2000):
        sp, sigmas, pix_dists = self.scalepyr(img)
        resps = []
        lafs = []
        for oct_idx,octave in enumerate(sp):
            sigmas_oct = sigmas[oct_idx]
            pix_dists_oct = pix_dists[oct_idx]
            B, L, CH, H, W = octave.size()
            
            # Run responce function
            oct_resp = self.resp(octave.view(B*L,CH,H,W)).view(B,L,H,W)
            # Responce correction because of blur. 
            oct_resp = self.resp.update_response(oct_resp, sigmas_oct[0:1,:])
            
            # Differentiable nms
            resp, maxima_coord, mask = self.nms(oct_resp, sigmas_oct[0:1,:])
            
            # Reshape to LAF format
            Br, Lr, Hr, Wr = resp.size()
            resp = resp.permute(0,2,3,1).view(Br, Hr*Wr, Lr)
            laf = maxima_coords_to_LAF(maxima_coord)
            nlaf = normalize_LAF(laf, oct_resp)
            
            # Lets cut out some features now
            if resp.size(1) > num_feats:
                # ToDo: make better use of mask.
                resp, idxs = torch.topk(resp, k = num_feats, dim = 1)
                good_lafs = []
                for i in range(B):
                    good_lafs.append(nlaf[i,idxs[i].squeeze(),:,:].unsqueeze(0))
                good_lafs = torch.cat(good_lafs, dim=0)
            else:
                good_lafs = nlaf
            resps.append(resp)
            lafs.append(good_lafs)
        
        # Sort and keep best n
        resps = torch.cat(resps, dim = 1)
        lafs = torch.cat(lafs, dim = 1)
        resps, idxs = torch.topk(resps, k = num_feats, dim = 1)
        new_lafs = []
        for i in range(B):
            new_lafs.append(lafs[i,idxs[i].squeeze(),:,:].unsqueeze(0))
        new_lafs = torch.cat(new_lafs, dim=0)
        return new_lafs, resps

    def forward(self, img):
        lafs, resps = self.detect(img, self.num_features)
        lafs = self.aff(lafs, img)
        lafs = self.ori(lafs, img)
        return denormalize_LAF(lafs,img), resps


