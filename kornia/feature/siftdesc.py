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

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.core.utils import _l2_normalize
from kornia.filters import get_gaussian_kernel2d, spatial_gradient
from kornia.geometry.conversions import pi


def _get_reshape_kernel(kd: int, ky: int, kx: int) -> torch.Tensor:
    """Return neigh2channels conv kernel.

    Deliberately not memoised: the result is registered as a buffer, so a shared cache has to clone
    on the way out to keep callers from writing into it, and the clone costs more than rebuilding
    the identity.
    """
    numel: int = kd * ky * kx
    return torch.eye(numel).view(numel, kd, ky, kx)


def _gradient_magnitude_orientation(
    gx: torch.Tensor, gy: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-pixel gradient magnitude and orientation in ``[2pi, 4pi)``.

    ``eps`` keeps ``sqrt`` and ``atan2`` away from their singular point at a zero gradient, where
    both have an undefined backward. A float16 input is lifted to float32 for this step: the
    1e-10 guard is not representable in float16, and a squared float16 gradient underflows long
    before that, so a flat pair of pixels -- ordinary with a 10-bit mantissa -- would send NaN
    into the input gradient. The result is cast back. bfloat16 has float32's exponent range, so
    both the guard and the squares are representable there and it takes the same expression as
    float32 and float64.

    A pixel with an exactly zero gradient has no orientation and contributes nothing: its magnitude
    is zero rather than the guard's ``sqrt(eps)``, and its orientation is ``2pi`` (what the guarded
    ``atan2`` returns) with a zero derivative rather than ``atan2``'s ``1 / eps``. Both were guard
    artefacts: the ``sqrt(eps)`` magnitude made a flat patch's descriptor a unit vector built from
    ``eps`` in float32 and a subnormal one in float16, whose ``1 / norm`` gradient overflowed through
    the float16 cast into a NaN input gradient. A flat patch now has a zero descriptor and a zero
    gradient in every dtype; every other pixel is unchanged.
    """
    dtype = gx.dtype
    if dtype == torch.float16:
        gx = gx.float()
        gy = gy.float()
    sq = gx * gx + gy * gy
    nonzero = sq > 0
    mag = torch.where(nonzero, torch.sqrt(sq + eps), torch.zeros_like(sq))
    ori = torch.where(nonzero, torch.atan2(gy, gx + eps) + 2.0 * pi, torch.full_like(sq, 2.0 * pi))
    return mag.to(dtype), ori.to(dtype)


def get_sift_pooling_kernel(ksize: int = 25) -> torch.Tensor:
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
        num_spatial_bins: the given number of spatial bins.

    Returns:
        ksize, stride, pad.

    """
    ksize: int = 2 * int(patch_size / (num_spatial_bins + 1))
    stride: int = patch_size // num_spatial_bins
    pad: int = ksize // 4
    out_size: int = (patch_size + 2 * pad - (ksize - 1) - 1) // stride + 1
    if out_size != num_spatial_bins:
        raise ValueError(
            f"Patch size {patch_size} is incompatible with the requested number of spatial bins "
            f"{num_spatial_bins} for SIFT descriptor. Usually it happens when patch size is too small "
            "for num_spatial_bins specified"
        )
    return ksize, stride, pad


class SIFTDescriptor(nn.Module):
    r"""nn.Module which computes SIFT descriptors of given patches.

    Args:
        patch_size: Input patch size in pixels.
        num_ang_bins: Number of angular bins.
        num_spatial_bins: Number of spatial bins.
        clipval: clipping value to reduce single-bin dominance
        rootsift: if ``True``, RootSIFT (Arandjelović et. al, 2012) is computed.

    Returns:
        SIFT descriptor of the patches.

    Shape:
        - Input: :math:`(B, 1, \text{patch_size}, \text{patch_size})`
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
        # non-persistent: fully determined by `patch_size`, so it must not enter `state_dict()`
        # (that would break existing checkpoints), but it must still follow `.to()`.
        self.register_buffer("gk", get_gaussian_kernel2d((ks, ks), (sigma, sigma), True), persistent=False)

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

    def get_pooling_kernel(self) -> torch.Tensor:
        """Return the spatial pooling kernel used for histogram accumulation.

        Returns:
            Detached convolution kernel tensor from the pooling layer.
        """
        return self.pk.weight.detach()

    def get_weighting_kernel(self) -> torch.Tensor:
        """Return the Gaussian weighting kernel used before orientation pooling.

        Returns:
            Detached Gaussian kernel tensor.
        """
        return self.gk.detach()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Compute SIFT descriptors for square grayscale patches.

        Args:
            input: Patch tensor shaped
                :math:`(B, 1, \text{patch\_size}, \text{patch\_size})`.

        Returns:
            Descriptor tensor of size
            :math:`(B, \text{num\_ang\_bins} \times \text{num\_spatial\_bins}^2)`.
        """
        KORNIA_CHECK_SHAPE(input, ["B", "1", f"{self.patch_size}", f"{self.patch_size}"])
        B: int = input.shape[0]
        self.pk = self.pk.to(input.dtype).to(input.device)

        grads = spatial_gradient(input, "diff")
        # unpack the edges
        gx = grads[:, :, 0]
        gy = grads[:, :, 1]

        mag, ori = _gradient_magnitude_orientation(gx, gy, self.eps)
        mag = mag * self.gk.expand_as(mag).type_as(mag).to(mag.device)
        o_big = float(self.num_ang_bins) * ori / (2.0 * pi)

        bo0_big_ = torch.floor(o_big)
        wo1_big_ = o_big - bo0_big_
        bo0_big = bo0_big_ % self.num_ang_bins
        bo1_big = (bo0_big + 1) % self.num_ang_bins
        wo0_big = (1.0 - wo1_big_) * mag
        wo1_big = wo1_big_ * mag

        ang_bins = torch.cat(
            [
                self.pk((bo0_big == i).to(input.dtype) * wo0_big + (bo1_big == i).to(input.dtype) * wo1_big)
                for i in range(self.num_ang_bins)
            ],
            1,
        )
        ang_bins = ang_bins.view(B, -1)
        # A constant patch has an all-zero gradient and therefore a zero-norm descriptor; the
        # default `eps` is not representable in float16, where it would come back NaN.
        ang_bins = _l2_normalize(ang_bins, dim=1)
        ang_bins = torch.clamp(ang_bins, 0.0, float(self.clipval))
        ang_bins = _l2_normalize(ang_bins, dim=1)
        if self.rootsift:
            ang_bins = _rootsift(ang_bins, self.eps)
        return ang_bins


def _rootsift(desc: torch.Tensor, eps: float) -> torch.Tensor:
    r"""L1-normalise and take the square root, the RootSIFT step, with a dtype-safe ``eps``.

    ``sqrt`` has an infinite backward at zero, and most bins of a SIFT histogram are zero, so ``eps`` keeps
    the gradient finite. A float16 input cannot carry the 1e-10 guard -- it underflows to zero -- and the
    smallest float16 normal, 6.1e-5, is not neutral: every empty bin would read ``sqrt(6.1e-5) = 0.0078``
    and push the descriptor norm to ~1.004. The step is therefore computed in float32 for a float16 input
    and cast back; float32 and float64 inputs take the same expression as before, unchanged.
    """
    if desc.dtype == torch.float16:
        return torch.sqrt(F.normalize(desc.float(), p=1, eps=1e-12) + eps).to(desc.dtype)
    return torch.sqrt(F.normalize(desc, p=1, eps=1e-12) + eps)


def sift_describe(
    input: torch.Tensor,
    patch_size: int = 41,
    num_ang_bins: int = 8,
    num_spatial_bins: int = 4,
    rootsift: bool = True,
    clipval: float = 0.2,
) -> torch.Tensor:
    r"""Compute the sift descriptor.

    See
    :class: `~kornia.feature.SIFTDescriptor` for details.
    """
    return SIFTDescriptor(patch_size, num_ang_bins, num_spatial_bins, rootsift, clipval)(input)


class DenseSIFTDescriptor(nn.Module):
    """nn.Module, which computes SIFT descriptor densely over the image.

    Args:
        num_ang_bins: Number of angular bins. (8 is default)
        num_spatial_bins: Number of spatial bins per descriptor (4 is default). You might want to set an odd
            number and relevant padding to keep the feature map size.
        spatial_bin_size: Size of a spatial bin in pixels (4 is default)
        clipval: clipping value to reduce single-bin dominance
        rootsift: (bool) if True, RootSIFT (Arandjelović et. al, 2012) is computed
        stride: default 1
        padding: default 0

    Returns:
        torch.Tensor: DenseSIFT descriptor of the image

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

        # Only allocate pooling kernels once during construction
        nw = get_sift_pooling_kernel(ksize=self.spatial_bin_size).float()
        self.register_buffer("_bin_pooling_kernel_weight", nw.reshape(1, 1, nw.size(0), nw.size(1)))
        bin_pooling_kernel = nn.Conv2d(
            1,
            1,
            kernel_size=(nw.size(0), nw.size(1)),
            stride=(1, 1),
            bias=False,
            padding=(nw.size(0) // 2, nw.size(1) // 2),
        )
        bin_pooling_kernel.weight.data.copy_(self._bin_pooling_kernel_weight)
        self.bin_pooling_kernel = bin_pooling_kernel

        Pw = _get_reshape_kernel(num_ang_bins, num_spatial_bins, num_spatial_bins).float()
        self.register_buffer("_poolingconv_weight", Pw)
        PoolingConv = nn.Conv2d(
            num_ang_bins,
            num_ang_bins * num_spatial_bins**2,
            kernel_size=(num_spatial_bins, num_spatial_bins),
            stride=(self.stride, self.stride),
            bias=False,
            padding=(self.pad, self.pad),
        )
        PoolingConv.weight.data.copy_(self._poolingconv_weight)
        self.PoolingConv = PoolingConv

        # Cache pooling kernel torch.Tensor for fast return in get_pooling_kernel
        self._pooling_kernel = self._bin_pooling_kernel_weight.detach()

    def get_pooling_kernel(self) -> torch.Tensor:
        """Return the cached pooling kernel for dense SIFT binning.

        Returns:
            Detached tensor containing pooling weights.
        """
        # Return the cached detached pooling kernel directly for optimal speed
        return self._pooling_kernel

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute dense SIFT descriptors over a full image grid.

        Args:
            input: Grayscale image tensor with shape :math:`(B, 1, H, W)`.

        Returns:
            Dense descriptor map tensor with channel dimension
            ``num_ang_bins * num_spatial_bins**2``.
        """
        KORNIA_CHECK_SHAPE(input, ["B", "1", "H", "W"])

        _B, _CH, _W, _H = input.size()
        self.bin_pooling_kernel = self.bin_pooling_kernel.to(input.dtype).to(input.device)
        self.PoolingConv = self.PoolingConv.to(input.dtype).to(input.device)
        grads = spatial_gradient(input, "diff")
        # unpack the edges
        gx = grads[:, :, 0]
        gy = grads[:, :, 1]
        mag, ori = _gradient_magnitude_orientation(gx, gy, self.eps)
        o_big = float(self.num_ang_bins) * ori / (2.0 * pi)

        bo0_big_ = torch.floor(o_big)
        wo1_big_ = o_big - bo0_big_
        bo0_big = bo0_big_ % self.num_ang_bins
        bo1_big = (bo0_big + 1) % self.num_ang_bins
        wo0_big = (1.0 - wo1_big_) * mag
        wo1_big = wo1_big_ * mag
        ang_bins = torch.cat(
            [
                self.bin_pooling_kernel(
                    (bo0_big == i).to(input.dtype) * wo0_big + (bo1_big == i).to(input.dtype) * wo1_big
                )
                for i in range(self.num_ang_bins)
            ],
            1,
        )

        out_no_norm = self.PoolingConv(ang_bins)
        out = _l2_normalize(out_no_norm, dim=1).clamp_(0, float(self.clipval))
        out = _l2_normalize(out, dim=1)
        if self.rootsift:
            out = _rootsift(out, self.eps)
        return out
