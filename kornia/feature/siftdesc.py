from typing import Tuple
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

import kornia


def get_sift_pooling_kernel(ksize: int = 25) -> torch.Tensor:
    """Returns a weighted pooling kernel for SIFT descriptor

    Args:
        ksize: (int): kernel_size

    Returns:
        torch.Tensor: kernel

    Shape:
        Output: :math: `(ksize,ksize)`
    """
    ks_2: float = float(ksize) / 2.0
    xc2: torch.Tensor = ks_2 - (torch.arange(ksize).float() + 0.5 - ks_2).abs()  # type: ignore # noqa
    kernel: torch.Tensor = torch.ger(xc2, xc2) / (ks_2**2)
    return kernel


def get_sift_bin_ksize_stride_pad(patch_size: int,
                                  num_spatial_bins: int) -> Tuple:
    """Returns a tuple with SIFT parameters, given the patch size
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
    out_size: int = (patch_size + 2 * pad - (ksize - 1) - 1) // stride + 1
    if out_size != num_spatial_bins:
        raise ValueError(f"Patch size {patch_size} is incompatible with \
            requested number of spatial bins {num_spatial_bins} \
            for SIFT descriptor. Usually it happens when patch size is too small\
            for num_spatial_bins specified")
    return ksize, stride, pad


class SIFTDescriptor(kornia.nn.SIFTDescriptor):
    """
    Module, which computes SIFT descriptors of given patches

    Args:
        patch_size: (int) Input patch size in pixels (41 is default)
        num_ang_bins: (int) Number of angular bins. (8 is default)
        num_spatial_bins: (int) Number of spatial bins (4 is default)
        clipval: (float) default 0.2
        rootsift: (bool) if True, RootSIFT (Arandjelović et. al, 2012)
        is computed

    Returns:
        torch.Tensor: SIFT descriptor of the patches

    Shape:
        - Input: (B, 1, num_spatial_bins, num_spatial_bins)
        - Output: (B, num_ang_bins * num_spatial_bins ** 2)

    Examples::
        >>> input = torch.rand(23, 1, 32, 32)
        >>> SIFT = kornia.SIFTDescriptor(32, 8, 4)
        >>> descs = SIFT(input) # 23x128
    """

    def __init__(self,
                 patch_size: int = 41,
                 num_ang_bins: int = 8,
                 num_spatial_bins: int = 4,
                 rootsift: bool = True,
                 clipval: float = 0.2,
                 ) -> None:
        super(SIFTDescriptor, self).__init__(
            patch_size, num_ang_bins, num_spatial_bins, rootsift, clipval
        )
        kornia.deprecation_warning("kornia.feature.SIFTDescriptor", "kornia.nn.SIFTDescriptor")


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

    return kornia.nn.SIFTDescriptor(
        patch_size, num_ang_bins, num_spatial_bins, rootsift, clipval)(input)
