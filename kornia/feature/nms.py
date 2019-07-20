from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.utils import create_meshgrid
from kornia.feature.laf import maxima_coords_to_LAF

class NonMaximaSuppression2d(nn.Module):
    r"""Applies hard non maxima suppression to feature map.
    """

    def __init__(self, kernel_size: Tuple[int, int]):
        super(NonMaximaSuppression2d, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.padding: Tuple[int,
                            int] = self._compute_zero_padding2d(kernel_size)
        self.max_pool2d = nn.MaxPool2d(kernel_size, stride=1,
                                       padding=self.padding)

    @staticmethod
    def _compute_zero_padding2d(
            kernel_size: Tuple[int, int]) -> Tuple[int, int]:
        assert isinstance(kernel_size, tuple), type(kernel_size)
        assert len(kernel_size) == 2, kernel_size

        def pad(x):
            return (x - 1) // 2  # zero padding function

        ky, kx = kernel_size     # we assume a cubic kernel
        return (pad(ky), pad(kx))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        assert len(x.shape) == 4, x.shape
        # find local maximum values
        x_max: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = \
            self.max_pool2d(x)

        # create mask for maximums in the original map
        x_mask: torch.Tensor = torch.where(
            x == x_max, torch.ones_like(x), torch.zeros_like(x))

        return x * x_mask  # return original masked by local max

class NonMaximaSuppression3d(nn.Module):
    r"""Applies hard non maxima suppression to feature map.
    """

    def __init__(self, kernel_size: Tuple[int, int, int]):
        super(NonMaximaSuppression3d, self).__init__()
        self.kernel_size: Tuple[int, int, int] = kernel_size
        self.padding: Tuple[int,
                            int,
                            int] = self._compute_zero_padding3d(kernel_size)
        self.max_pool3d = nn.MaxPool3d(kernel_size, stride=1,
                                       padding=self.padding)

    @staticmethod
    def _compute_zero_padding3d(
            kernel_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        assert isinstance(kernel_size, tuple), type(kernel_size)
        assert len(kernel_size) == 3, kernel_size

        def pad(x):
            return (x - 1) // 2  # zero padding function

        kz, ky, kx = kernel_size     # we assume a cubic kernel
        return (pad(kz), pad(ky), pad(kx))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        assert len(x.shape) == 4, x.shape
        # find local maximum values
        x_max: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = \
            self.max_pool3d(x)

        # create mask for maximums in the original map
        x_mask: torch.Tensor = torch.where(
            x == x_max, torch.ones_like(x), torch.zeros_like(x))

        return x * x_mask  # return original masked by local max

def get_softmaxpool_kernel(d:int, h:int, w:int, centered:bool = True)-> torch.Tensor:
    r"""Generates kernel for SoftNMS3d
    """
    grid2d = create_meshgrid(h, w, False)
    if centered:
        grid2d[:,:,:,0] = grid2d[:,:,:,0] - w//2
        grid2d[:,:,:,1] = grid2d[:,:,:,1] - h//2
    if centered:
        z = torch.linspace(0, d-1, d).view(d,1,1,1) - d//2
    else:
        z = torch.linspace(0, d-1, d).view(d,1,1,1)
    grid3d = torch.cat([z.repeat(1,h,w,1).contiguous(),# type: ignore
                        grid2d.repeat(d,1,1,1)],dim = 3)
    grid3d = grid3d.permute(3,0,1,2)
    return grid3d


class SoftNMS3d(nn.Module):
    r"""Applies soft non maxima suppression to feature map
    across spatial dimentions and channels (scale)
    Args:
        kernel_size: (Tuple[int,int,int])
        stride: (int)
        padding: (int)
        temperature: (float) for softmax, default 5
        mrSize: float: scale multiplier
        
    Returns:
        responces: (torch.Tensor). extrema responces
        maxima_coords: (torch.Tensor). coordinates of nms window maxima.
        format is (scale, y, x)
        mask: (torch.Tensor) if the center of the nms window is true maxima
        
        
    Shape:
        - Input: :math:`(B, CH, H, W)`
        - Output:  :math:`(B, 1, H, W)`,  :math:`(B, 3, H, W)`, :math:`(B, 1, H, W)` 
    """
    def __init__(self, 
                 kernel_size:Tuple[int,int,int] = (3,3,3), 
                 stride:int=1,
                 padding:int=0, 
                 temperature:float = 5,
                 only_hard_peaks:bool = False):
        super(SoftNMS3d, self).__init__()
        self.ksize = kernel_size
        self.stride = stride
        self.pd = padding
        self.T = temperature
        self.only_hard_peaks = only_hard_peaks
        self.register_buffer('kernel', get_softmaxpool_kernel(*self.ksize).float())
        self.register_buffer('kernel_ones', torch.ones_like(self.kernel).float())
        self.register_buffer('kernel_resp', torch.ones(1,*self.kernel.shape[1:]).float())
        kernel_global_coord =  torch.zeros(2,2,*self.kernel.shape[2:]).float()
        kernel_global_coord.data[0, 0, self.ksize[1]//2, self.ksize[2]//2] =  1 # type: ignore
        kernel_global_coord.data[1, 1, self.ksize[1]//2, self.ksize[2]//2] =  1 # type: ignore
        self.register_buffer('kernel_global_coord', kernel_global_coord)
        return
    def forward(self, x:torch.Tensor, sigmas:torch.Tensor = None)-> Tuple: # type: ignore
        self.kernel_ones.to(x.dtype)
        self.kernel.to(x.dtype)
        if sigmas is not None:
            sigmas.to(x.dtype)
            self.kernel[0,:,:,:]  = self.kernel[0,:,:,:]*0 +sigmas.view(1,-1,1,1)
        n,ch,h,w = x.size()
        grid_global = create_meshgrid(h,w, False).permute(0,3,1,2)
        grid_global_pooled = F.conv2d(grid_global.to(x.device).flip(1), 
                                      self.kernel_global_coord, 
                                      stride = self.stride,
                                      padding = self.pd)
        resp_exp = (x  * self.T).exp()
        den = F.conv2d(resp_exp, 
                       self.kernel_ones,
                       stride = self.stride,
                       padding = self.pd) + 1e-12
        maxima_coords = F.conv2d(resp_exp,
                                 self.kernel,
                                 stride = self.stride,
                                 padding = self.pd) / den
        maxima_coords[:,1:,:,:] = maxima_coords[:,1:,:,:] + grid_global_pooled
        resp_maxima = F.conv2d(resp_exp*x,
                               self.kernel_resp,
                               stride = self.stride,
                               padding = self.pd) / den[:,0:1,:,:]
        if self.only_hard_peaks:
            mask = non_maxima_suppression3d(x, self.ksize) > 0
        else:
            mask = resp_maxima > 0
        return resp_maxima, maxima_coords, mask



# functiona api


def non_maxima_suppression2d(
        input: torch.Tensor, kernel_size: Tuple[int, int]) -> torch.Tensor:
    r"""Applies non maxima suppression to filter.

    See :class:`~kornia.feature.NonMaximaSuppression2d` for details.
    """
    return NonMaximaSuppression2d(kernel_size)(input)

def non_maxima_suppression3d(
        input: torch.Tensor, kernel_size: Tuple[int, int, int]) -> torch.Tensor:
    r"""Applies non maxima suppression to filter.

    See :class:`~kornia.feature.NonMaximaSuppression3d` for details.
    """
    return NonMaximaSuppression3d(kernel_size)(input)
