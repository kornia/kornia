from typing import Tuple
import kornia
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_sift_pooling_kernel(ksize:int = 25)-> torch.Tensor:
    """
    Returns a weighted pooling kernel for SIFT descriptor
    Args:
        ksize: (int): kernel_size

    Returns:
        torch.Tensor: kernel

    Shape:
        - Output: :math: `(ksize,ksize)`
    """
    ksize_2 = float(ksize)/2.0
    xc2 = ksize_2 - torch.abs(torch.arange(ksize).float()+0.5 - ksize_2)
    kernel = torch.ger(xc2,xc2)/(ksize_2**2)
    return kernel


def get_sift_bin_ksize_stride_pad(patch_size:int, num_spatial_bins:int)-> Tuple[int,int,int]:
    """
    Returns a tuple with SIFT parameters, given the patch size 
    and numberof spatial bins.
    Args:
        patch_size: (int)
        num_spatial_bins: (int)
        
    Returns:
        ksize, stride, pad: ints
    """
    ksize = 2*int(patch_size / (num_spatial_bins+1));
    stride = patch_size // num_spatial_bins
    pad = ksize // 4
    return ksize, stride, pad

def get_sift_weighting_kernel(ksize:int, 
                              sigma_type:str = 'vlfeat',
                              circ:bool = False):
    """
    Returns a weighted golbal kernel for SIFT descriptor
    Args:
        ksize: (int): kernel_size
        sigma_type: (str): type of sigma, 'vlfeat', or 'hesamp'
        circ: (bool): kernel_size

    Returns:
        torch.Tensor: kernel

    Shape:
        - Output: :math: `(ksize,ksize)`
    """
    ksize_2 = float(ksize) / 2.;
    r2 = float(ksize_2**2);
    if sigma_type == 'hesamp':
        sigma_mul_2 = 0.9 * r2;
    elif sigma_type == 'vlfeat':
        sigma_mul_2 = ksize**2
    else:
        raise ValueError('Unknown sigma_type', sigma_type, 'try hesamp or vlfeat')
    disq = 0;
    kernel = torch.zeros(ksize,ksize).float()
    for y in range(ksize):
        for x in range(ksize):
            disq = (y - ksize_2+0.5)**2 +  (x - ksize_2+0.5)**2;
            kernel[y,x] = math.exp(-disq / sigma_mul_2)
            if circ and (disq >= r2):
                kernel[y,x] = 0.
    return kernel

class SIFTDescriptor(nn.Module):
    """
    Module, which computes SIFT descriptors of given patches
    """
    def __repr__(self):
            return self.__class__.__name__ +\
             '(' + 'num_ang_bins=' + str(self.num_ang_bins) +\
             ', ' + 'num_spatial_bins=' + str(self.num_spatial_bins) +\
             ', ' + 'patch_size=' + str(self.patch_size) +\
             ', ' + 'rootsift=' + str(self.rootsift) +\
             ', ' + 'sigma_type=' + str(self.sigma_type) +\
             ', ' + 'mask_type=' + str(self.mask_type) +\
             ', ' + 'clipval=' + str(self.clipval) + ')'
    def __init__(self,
                 patch_size = 41, 
                 num_ang_bins = 8,
                 num_spatial_bins = 4,
                 clipval = 0.2,
                 rootsift = True,
                 mask_type = 'Gauss',
                 sigma_type = 'vlfeat'):
        super(SIFTDescriptor, self).__init__()
        self.eps = 1e-10
        self.num_ang_bins = num_ang_bins
        self.num_spatial_bins = num_spatial_bins
        self.clipval = clipval
        self.rootsift = rootsift
        self.mask_type = mask_type
        self.patch_size = patch_size
        self.sigma_type = sigma_type
        if self.mask_type == 'CircularGauss':
            self.gk = get_sift_weighting_kernel(ksize=patch_size,
                                                circ=True,
                                                sigma_type=sigma_type)
        elif self.mask_type == 'Gauss':
            self.gk = get_sift_weighting_kernel(ksize=patch_size,
                                                circ=False,
                                                sigma_type=sigma_type)
        elif self.mask_type == 'Uniform':
            self.gk = torch.ones(patch_size,patch_size).float() / float(patch_size**2)
        else:
            raise ValueError(masktype, 'is unknown mask type. Try Gauss, CircularGauss or Uniform')
            
        self.bin_ksize, self.bin_stride, self.pad = get_sift_bin_ksize_stride_pad(patch_size, num_spatial_bins)
        self.gx =  nn.Conv2d(1, 1, kernel_size=(1,3),  bias = False)
        self.gx.weight.data = torch.tensor([[[[-1.0, 0.0, 1.0]]]]).float()
        
        self.gy = nn.Conv2d(1, 1, kernel_size=(3,1),  bias = False)
        self.gy.weight.data = torch.tensor([[[[-1], [0], [1]]]]).float()
        
        nw = get_sift_pooling_kernel(ksize = self.bin_ksize).float()
        
        self.pk = nn.Conv2d(1, 1, kernel_size=(nw.size(0), nw.size(1)),
                            stride = (self.bin_stride, self.bin_stride),
                            padding = (self.pad , self.pad ),
                            bias = False)
        self.pk.weight.data = nw.reshape(1,1, nw.size(0),nw.size(1))
        return
    def get_pooling_kernel(self):
        return self.pk.weight.data
    def get_weighting_kernel(self):
        return self.gk.data
    def forward(self, x):
        N,CH,W,H = x.size()
        if (W!=self.patch_size) or (H!=self.patch_size) or\
            (CH!=1):
            raise TypeError(
            "input shape should be must be [Bx1x{}x{}]. "
            "Got {}".format(self.patch_size, self.patch_size, x.size())
        )   
        gx = self.gx(F.pad(x, (1, 1, 0, 0), 'replicate'))
        gy = self.gy(F.pad(x, (0, 0, 1, 1), 'replicate'))
        mag = torch.sqrt(gx * gx + gy * gy + self.eps)
        ori = torch.atan2(gy, gx + self.eps)
        mag  = mag * self.gk.expand_as(mag).to(mag.device)
        o_big = (ori + 2.0 * math.pi )/ (2.0 * math.pi) * float(self.num_ang_bins)
        bo0_big_ =  torch.floor(o_big)
        wo1_big_ = o_big - bo0_big_
        bo0_big =  bo0_big_ %  self.num_ang_bins
        bo1_big = (bo0_big + 1) % self.num_ang_bins
        wo0_big = (1.0 - wo1_big_) * mag
        wo1_big = wo1_big_ * mag
        ang_bins = []
        for i in range(0, self.num_ang_bins):
            out = self.pk((bo0_big == i).float() * wo0_big + (bo1_big == i).float() * wo1_big)
            ang_bins.append(out)
        ang_bins = torch.cat(ang_bins,1)
        ang_bins = ang_bins.view(ang_bins.size(0), -1)
        ang_bins = F.normalize(ang_bins, p=2)
        ang_bins = torch.clamp(ang_bins, 0., float(self.clipval))
        ang_bins = F.normalize(ang_bins, p=2)
        if self.rootsift:
            ang_bins = torch.sqrt(F.normalize(ang_bins,p=1) + 1e-10)
        return ang_bins

