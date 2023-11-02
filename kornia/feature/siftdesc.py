import math
from typing import Tuple

import torch
from torch import nn

from kornia.core import Module, Tensor, concatenate, eye, normalize
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.filters import get_gaussian_kernel2d, spatial_gradient
from kornia.geometry.conversions import pi


def _get_reshape_kernel(kd: int, ky: int, kx: int) -> Tensor:
    """Utility function, which returns neigh2channels conv kernel."""
    numel: int = kd * ky * kx
    weight = eye(numel)
    return weight.view(numel, kd, ky, kx)


def get_sift_pooling_kernel(ksize: int = 25) -> Tensor:
    r"""Return a weighted pooling kernel for SIFT descriptor.

    Args:
        ksize: kernel_size.

    Returns:
        the pooling kernel with shape :math:`(ksize, ksize)`.
    """
    ks_2: float = float(ksize) / 2.0
    xc2 = ks_2 - (torch.arange(ksize).float() + 0.5 - ks_2).abs()
    kernel = torch.ger(xc2, xc2) / (ks_2**2)
    return kernel


def get_sift_bin_ksize_stride_pad(patch_size: int, num_spatial_bins: int) -> Tuple[int, int, int]:
    r"""Return a tuple with SIFT parameters.

    Args:
        patch_size: the given patch size.
        num_spatial_bins: the ggiven number of spatial bins.

    Returns:
        ksize, stride, pad.
    """
    ksize: int = 2 * int(patch_size / (num_spatial_bins + 1))
    stride: int = patch_size // num_spatial_bins
    pad: int = ksize // 4
    out_size: int = (patch_size + 2 * pad - (ksize - 1) - 1) // stride + 1
    if out_size != num_spatial_bins:
        raise ValueError(
            f"Patch size {patch_size} is incompatible with             requested number of spatial bins"
            f" {num_spatial_bins}             for SIFT descriptor. Usually it happens when patch size is too small     "
            "       for num_spatial_bins specified"
        )
    return ksize, stride, pad


class SIFTDescriptor(Module):
    r"""Module which computes SIFT descriptors of given patches.

    Args:
        patch_size: Input patch size in pixels.
        num_ang_bins: Number of angular bins.
        num_spatial_bins: Number of spatial bins.
        clipval: clipping value to reduce single-bin dominance
        rootsift: if ``True``, RootSIFT (Arandjelović et. al, 2012) is computed.

    Returns:
        SIFT descriptor of the patches with shape.

    Shape:
        - Input: :math:`(B, 1, \text{num_spatial_bins}, \text{num_spatial_bins})`
        - Output: :math:`(B, \text{num_ang_bins * num_spatial_bins ** 2})`

    Example:
        >>> input = torch.rand(23, 1, 32, 32)
        >>> SIFT = SIFTDescriptor(32, 8, 4)
        >>> descs = SIFT(input) # 23x128
    """

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_ang_bins={self.num_ang_bins}, "
            f"num_spatial_bins={self.num_spatial_bins}, "
            f"patch_size={self.patch_size}, "
            f"rootsift={self.rootsift}, "
            f"clipval={self.clipval})"
        )

    def __init__(
        self,
        patch_size: int = 41,
        num_ang_bins: int = 8,
        num_spatial_bins: int = 4,
        rootsift: bool = True,
        clipval: float = 0.2,
    ) -> None:
        super().__init__()
        self.eps = 1e-10
        self.num_ang_bins = num_ang_bins
        self.num_spatial_bins = num_spatial_bins
        self.clipval = clipval
        self.rootsift = rootsift
        self.patch_size = patch_size

        ks: int = self.patch_size
        sigma: float = float(ks) / math.sqrt(2.0)
        self.gk = get_gaussian_kernel2d((ks, ks), (sigma, sigma), True)

        (self.bin_ksize, self.bin_stride, self.pad) = get_sift_bin_ksize_stride_pad(patch_size, num_spatial_bins)

        nw = get_sift_pooling_kernel(ksize=self.bin_ksize).float()
        self.pk = nn.Conv2d(
            1,
            1,
            kernel_size=(nw.size(0), nw.size(1)),
            stride=(self.bin_stride, self.bin_stride),
            padding=(self.pad, self.pad),
            bias=False,
        )
        self.pk.weight.data.copy_(nw.reshape(1, 1, nw.size(0), nw.size(1)))

    def get_pooling_kernel(self) -> Tensor:
        return self.pk.weight.detach()

    def get_weighting_kernel(self) -> Tensor:
        return self.gk.detach()

    def forward(self, input: Tensor) -> Tensor:
        KORNIA_CHECK_SHAPE(input, ["B", "1", f"{self.patch_size}", f"{self.patch_size}"])
        B: int = input.shape[0]
        self.pk = self.pk.to(input.dtype).to(input.device)

        grads = spatial_gradient(input, "diff")
        # unpack the edges
        gx = grads[:, :, 0]
        gy = grads[:, :, 1]

        mag = torch.sqrt(gx * gx + gy * gy + self.eps)
        ori = torch.atan2(gy, gx + self.eps) + 2.0 * pi
        mag = mag * self.gk.expand_as(mag).type_as(mag).to(mag.device)
        o_big = float(self.num_ang_bins) * ori / (2.0 * pi)

        bo0_big_ = torch.floor(o_big)
        wo1_big_ = o_big - bo0_big_
        bo0_big = bo0_big_ % self.num_ang_bins
        bo1_big = (bo0_big + 1) % self.num_ang_bins
        wo0_big = (1.0 - wo1_big_) * mag
        wo1_big = wo1_big_ * mag

        ang_bins = concatenate(
            [
                self.pk((bo0_big == i).to(input.dtype) * wo0_big + (bo1_big == i).to(input.dtype) * wo1_big)
                for i in range(0, self.num_ang_bins)
            ],
            1,
        )
        ang_bins = ang_bins.view(B, -1)
        ang_bins = normalize(ang_bins, p=2)
        ang_bins = torch.clamp(ang_bins, 0.0, float(self.clipval))
        ang_bins = normalize(ang_bins, p=2)
        if self.rootsift:
            ang_bins = torch.sqrt(normalize(ang_bins, p=1) + self.eps)
        return ang_bins


def sift_describe(
    input: Tensor,
    patch_size: int = 41,
    num_ang_bins: int = 8,
    num_spatial_bins: int = 4,
    rootsift: bool = True,
    clipval: float = 0.2,
) -> Tensor:
    r"""Computes the sift descriptor.

    See
    :class: `~kornia.feature.SIFTDescriptor` for details.
    """
    return SIFTDescriptor(patch_size, num_ang_bins, num_spatial_bins, rootsift, clipval)(input)


class DenseSIFTDescriptor(Module):
    """Module, which computes SIFT descriptor densely over the image.

    Args:
        num_ang_bins: Number of angular bins. (8 is default)
        num_spatial_bins: Number of spatial bins per descriptor (4 is default).
    You might want to set odd number and relevant padding to keep feature map size
        spatial_bin_size: Size of a spatial bin in pixels (4 is default)
        clipval: clipping value to reduce single-bin dominance
        rootsift: (bool) if True, RootSIFT (Arandjelović et. al, 2012) is computed
        stride: default 1
        padding: default 0

    Returns:
        Tensor: DenseSIFT descriptor of the image

    Shape:
        - Input: (B, 1, H, W)
        - Output: (B, num_ang_bins * num_spatial_bins ** 2, (H+padding)/stride, (W+padding)/stride)

    Examples::
        >>> input =  torch.rand(2, 1, 200, 300)
        >>> SIFT = DenseSIFTDescriptor()
        >>> descs = SIFT(input) # 2x128x194x294
    """

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_ang_bins={self.num_ang_bins}, "
            f"num_spatial_bins={self.num_spatial_bins}, "
            f"spatial_bin_size={self.spatial_bin_size}, "
            f"rootsift={self.rootsift}, "
            f"stride={self.stride}, "
            f"clipval={self.clipval})"
        )

    def __init__(
        self,
        num_ang_bins: int = 8,
        num_spatial_bins: int = 4,
        spatial_bin_size: int = 4,
        rootsift: bool = True,
        clipval: float = 0.2,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.eps = 1e-10
        self.num_ang_bins = num_ang_bins
        self.num_spatial_bins = num_spatial_bins
        self.spatial_bin_size = spatial_bin_size
        self.clipval = clipval
        self.rootsift = rootsift
        self.stride = stride
        self.pad = padding
        nw = get_sift_pooling_kernel(ksize=self.spatial_bin_size).float()
        self.bin_pooling_kernel = nn.Conv2d(
            1,
            1,
            kernel_size=(nw.size(0), nw.size(1)),
            stride=(1, 1),
            bias=False,
            padding=(nw.size(0) // 2, nw.size(1) // 2),
        )
        self.bin_pooling_kernel.weight.data.copy_(nw.reshape(1, 1, nw.size(0), nw.size(1)))
        self.PoolingConv = nn.Conv2d(
            num_ang_bins,
            num_ang_bins * num_spatial_bins**2,
            kernel_size=(num_spatial_bins, num_spatial_bins),
            stride=(self.stride, self.stride),
            bias=False,
            padding=(self.pad, self.pad),
        )
        self.PoolingConv.weight.data.copy_(
            _get_reshape_kernel(num_ang_bins, num_spatial_bins, num_spatial_bins).float()
        )

    def get_pooling_kernel(self) -> Tensor:
        return self.bin_pooling_kernel.weight.detach()

    def forward(self, input: Tensor) -> Tensor:
        KORNIA_CHECK_SHAPE(input, ["B", "1", "H", "W"])

        B, CH, W, H = input.size()
        self.bin_pooling_kernel = self.bin_pooling_kernel.to(input.dtype).to(input.device)
        self.PoolingConv = self.PoolingConv.to(input.dtype).to(input.device)
        grads = spatial_gradient(input, "diff")
        # unpack the edges
        gx = grads[:, :, 0]
        gy = grads[:, :, 1]
        mag = torch.sqrt(gx * gx + gy * gy + self.eps)
        ori = torch.atan2(gy, gx + self.eps) + 2.0 * pi
        o_big = float(self.num_ang_bins) * ori / (2.0 * pi)

        bo0_big_ = torch.floor(o_big)
        wo1_big_ = o_big - bo0_big_
        bo0_big = bo0_big_ % self.num_ang_bins
        bo1_big = (bo0_big + 1) % self.num_ang_bins
        wo0_big = (1.0 - wo1_big_) * mag
        wo1_big = wo1_big_ * mag
        ang_bins = concatenate(
            [
                self.bin_pooling_kernel(
                    (bo0_big == i).to(input.dtype) * wo0_big + (bo1_big == i).to(input.dtype) * wo1_big
                )
                for i in range(0, self.num_ang_bins)
            ],
            1,
        )

        out_no_norm = self.PoolingConv(ang_bins)
        out = normalize(out_no_norm, dim=1, p=2).clamp_(0, float(self.clipval))
        out = normalize(out, dim=1, p=2)
        if self.rootsift:
            out = torch.sqrt(normalize(out, p=1) + self.eps)
        return out
