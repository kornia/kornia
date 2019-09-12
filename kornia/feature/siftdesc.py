from typing import Tuple
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import get_gaussian_kernel2d
from kornia.filters import spatial_gradient
from kornia.geometry import pi


def get_sift_pooling_kernel(ksize: int = 25) -> torch.Tensor:
    """
    Returns a weighted pooling kernel for SIFT descriptor
    Args:
        ksize: (int): kernel_size

    Returns:
        torch.Tensor: kernel

    Shape:
        - Output: :math: `(ksize,ksize)`
    """
    ks_2: float = float(ksize) / 2.0
    xc2: torch.Tensor = ks_2 - (torch.arange(ksize).float() + 0.5 - ks_2).abs()  # type: ignore # noqa
    kernel: torch.Tensor = torch.ger(xc2, xc2) / (ks_2**2)
    return kernel


def get_sift_bin_ksize_stride_pad(patch_size: int,
                                  num_spatial_bins: int) -> Tuple:
    """
    Returns a tuple with SIFT parameters, given the patch size
    and number of spatial bins.
    Args:
        patch_size: (int)
        num_spatial_bins: (int)

    Returns:
        ksize, stride, pad: ints
    """
    ksize: int = 2 * int(patch_size / (num_spatial_bins + 1))
    stride: int = patch_size // num_spatial_bins
    pad: int = ksize // 4
    return ksize, stride, pad


class SIFTDescriptor(nn.Module):
    """
    Module, which computes SIFT descriptors of given patches
    Args:
        patch_size: (int): Input patch size in pixels
        (41 is default)
        num_ang_bins: (int): Number of angular bins.
        (8 is default)
        num_spatial_bins: (int): Number of spatial bins
        (4 is default)
        clipval: (float): default 0.2
        rootsift: (bool): if True, RootSIFT (Arandjelović et. al, 2012)
        is computed

    Returns:
        Tensor: SIFT descriptor of the patches

    Shape:
        - Input: :math:`(B, 1, num_spatial_bins, num_spatial_bins)`
        - Output: :math:`(B, num_ang_bins * num_spatial_bins ** 2)`

    Examples::
        >>> input = torch.rand(23, 1, 32, 32)
        >>> SIFT = kornia.SIFTDescriptor(32, 8, 4)
        >>> descs = SIFT(input) # 23x128
    """

    def __repr__(self) -> str:
        return self.__class__.__name__ +\
            '(' + 'num_ang_bins=' + str(self.num_ang_bins) +\
            ', ' + 'num_spatial_bins=' + str(self.num_spatial_bins) +\
            ', ' + 'patch_size=' + str(self.patch_size) +\
            ', ' + 'rootsift=' + str(self.rootsift) +\
            ', ' + 'clipval=' + str(self.clipval) + ')'

    def __init__(self,
                 patch_size: int = 41,
                 num_ang_bins: int = 8,
                 num_spatial_bins: int = 4,
                 rootsift: bool = True,
                 clipval: float = 0.2,
                 ) -> None:
        super(SIFTDescriptor, self).__init__()
        self.eps = 1e-10
        self.num_ang_bins = num_ang_bins
        self.num_spatial_bins = num_spatial_bins
        self.clipval = clipval
        self.rootsift = rootsift
        self.patch_size = patch_size

        ks: int = self.patch_size
        sigma: float = float(ks) / math.sqrt(2.0)
        self.gk = get_gaussian_kernel2d((ks, ks), (sigma, sigma), True)

        (self.bin_ksize,
         self.bin_stride,
         self.pad) = get_sift_bin_ksize_stride_pad(patch_size,
                                                   num_spatial_bins)

        nw = get_sift_pooling_kernel(ksize=self.bin_ksize).float()
        self.pk = nn.Conv2d(1, 1, kernel_size=(nw.size(0), nw.size(1)),
                            stride=(self.bin_stride, self.bin_stride),
                            padding=(self.pad, self.pad),
                            bias=False)
        self.pk.weight.data.copy_(nw.reshape(1, 1, nw.size(0), nw.size(1)))  # type: ignore  # noqa
        return

    def get_pooling_kernel(self) -> torch.Tensor:
        return self.pk.weight.detach()

    def get_weighting_kernel(self) -> torch.Tensor:
        return self.gk.detach()

    def forward(self, input):
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect Bx1xHxW. Got: {}"
                             .format(input.shape))
        B, CH, W, H = input.size()
        if (W != self.patch_size) or (H != self.patch_size) or (CH != 1):
            raise TypeError(
                "input shape should be must be [Bx1x{}x{}]. "
                "Got {}".format(self.patch_size, self.patch_size, input.size()))
        self.pk = self.pk.to(input.dtype).to(input.device)

        grads: torch.Tensor = spatial_gradient(input, 'diff')
        # unpack the edges
        gx: torch.Tensor = grads[:, :, 0]
        gy: torch.Tensor = grads[:, :, 1]

        mag: torch.Tensor = torch.sqrt(gx * gx + gy * gy + self.eps)
        ori: torch.Tensor = torch.atan2(gy, gx + self.eps) + 2.0 * pi
        mag = mag * self.gk.expand_as(mag).type_as(mag).to(mag.device)
        o_big: torch.Tensor = float(self.num_ang_bins) * ori / (2.0 * pi)

        bo0_big_: torch.Tensor = torch.floor(o_big)
        wo1_big_: torch.Tensor = (o_big - bo0_big_)
        bo0_big: torch.Tensor = bo0_big_ % self.num_ang_bins
        bo1_big: torch.Tensor = (bo0_big + 1) % self.num_ang_bins
        wo0_big: torch.Tensor = (1.0 - wo1_big_) * mag  # type: ignore
        wo1_big: torch.Tensor = wo1_big_ * mag

        ang_bins = []
        for i in range(0, self.num_ang_bins):
            out = self.pk((bo0_big == i).to(input.dtype) * wo0_big +  # noqa
                          (bo1_big == i).to(input.dtype) * wo1_big)
            ang_bins.append(out)
        ang_bins = torch.cat(ang_bins, dim=1)
        ang_bins = ang_bins.view(B, -1)
        ang_bins = F.normalize(ang_bins, p=2)
        ang_bins = torch.clamp(ang_bins, 0., float(self.clipval))
        ang_bins = F.normalize(ang_bins, p=2)
        if self.rootsift:
            ang_bins = torch.sqrt(F.normalize(ang_bins, p=1) + self.eps)
        return ang_bins


def sift_describe(input: torch.Tensor,
                  patch_size: int = 41,
                  num_ang_bins: int = 8,
                  num_spatial_bins: int = 4,
                  rootsift: bool = True,
                  clipval: float = 0.2,
                  ) -> torch.Tensor:
    r"""Computes the sift descriptor.

    See :class:`~kornia.feature.SIFTDescriptor` for details.
    """

    return SIFTDescriptor(patch_size,
                          num_ang_bins,
                          num_spatial_bins,
                          rootsift,
                          clipval)(input)
