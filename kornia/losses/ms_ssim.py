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

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

# Based on:
# https://github.com/psyrocloud/MS-SSIM_L1_LOSS


class MS_SSIMLoss(nn.Module):
    r"""Creates a criterion that computes MSSIM + L1 loss.

    According to [1], we compute the MS_SSIM + L1 loss as follows:

    .. math::
        \text{loss}(x, y) = \alpha \cdot \mathcal{L_{MSSIM}}(x,y)+(1 - \alpha) \cdot G_\alpha \cdot \mathcal{L_1}(x,y)

    Where:
        - :math:`\alpha` is the weight parameter.
        - :math:`x` and :math:`y` are the reconstructed and true reference images.
        - :math:`\mathcal{L_{MSSIM}}` is the MS-SSIM loss.
        - :math:`G_\alpha` is the sigma values for computing multi-scale SSIM.
        - :math:`\mathcal{L_1}` is the L1 loss.

    Each channel is filtered at every scale. The MS-SSIM of a channel is its luminance at the coarsest scale times its
    contrast-structure at every scale, and :math:`\mathcal{L_{MSSIM}}` is one minus the mean of the per-channel
    MS-SSIM, as in the reference implementation [2] of [1]. The L1 term is filtered at the coarsest scale and averaged
    over the channels.

    Reference:
        [1]: https://research.nvidia.com/sites/default/files/pubs/2017-03_Loss-Functions-for/NN_ImgProc.pdf#page11
        [2]: https://github.com/NVlabs/PL4NN/blob/master/src/loss.py (``MSSSIML1``)

    Args:
        sigmas: gaussian sigma values.
        data_range: the range of the images.
        K: k values.
        alpha : specifies the alpha value
        compensation: specifies the scaling coefficient.
        reduction : Specifies the reduction to apply to the
         output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
         ``'mean'``: the sum of the output will be divided by the number of elements
         in the output, ``'sum'``: the output will be summed.

    Returns:
        The computed loss.

    Shape:
        - Input1: :math:`(N, C, H, W)`.
        - Input2: :math:`(N, C, H, W)`.
        - Output: :math:`(N, H, W)` or scalar if reduction is set to ``'mean'`` or ``'sum'``.

    Note:
        Integer and bool images count as the dtype of the Gaussian masks: float32, unless the module was moved to
        another floating dtype. The images are filtered in the promoted dtype of the two images and the masks, with
        float16 and bfloat16 raised to float32 and autocast disabled on CPU, CUDA and MPS. The loss is returned in the
        promoted dtype of the two images, so two integer images give a loss in the mask dtype. Pixel values are not
        rescaled: pass ``data_range=255.0`` for 8-bit images, which gives the loss of the images divided by 255 at the
        default ``data_range=1.0``.

    Examples:
        >>> input1 = torch.rand(1, 3, 5, 5)
        >>> input2 = torch.rand(1, 3, 5, 5)
        >>> criterion = kornia.losses.MS_SSIMLoss()
        >>> loss = criterion(input1, input2)

    """

    def __init__(
        self,
        sigmas: Sequence[float] = (0.5, 1.0, 2.0, 4.0, 8.0),
        data_range: float = 1.0,
        K: tuple[float, float] = (0.01, 0.03),
        alpha: float = 0.025,
        compensation: float = 200.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.DR: float = data_range
        self.C1: float = (K[0] * data_range) ** 2
        self.C2: float = (K[1] * data_range) ** 2
        self.pad = int(2 * sigmas[-1])
        self.alpha: float = alpha
        self.compensation: float = compensation
        self.reduction: str = reduction

        self.num_scales: int = len(sigmas)

        # One mask per scale; forward repeats them for every channel of the input. The window is odd, 2 * pad + 1
        # wide, so it is centred and the convolutions padded by ``pad`` keep the input's height and width.
        filter_size = 2 * self.pad + 1
        g_masks = torch.stack([self._fspecial_gauss_2d(filter_size, sigma) for sigma in sigmas]).unsqueeze(1)

        # The masks are derived from ``sigmas``, so they are not persisted; see ``_load_from_state_dict``.
        self.register_buffer("_g_masks", g_masks, persistent=False)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Older releases persisted three scale-major copies of every mask, a layout that paired channels with the
        # wrong scales. Accept the key for strict loading, including when nested in another module, and keep the
        # masks built from ``sigmas``.
        state_dict.pop(prefix + "_g_masks", None)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _fspecial_gauss_1d(
        self, size: int, sigma: float, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Create 1-D gauss kernel.

        Args:
            size: the size of gauss kernel.
            sigma: sigma of normal distribution.
            device: device to store the result on.
            dtype: dtype of the result.

        Returns:
            1D kernel (size).

        """
        coords = torch.arange(size, device=device, dtype=dtype)
        coords -= size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(
        self, size: int, sigma: float, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Create 2-D gauss kernel.

        Args:
            size: the size of gauss kernel.
            sigma: sigma of normal distribution.
            device: device to store the result on.
            dtype: dtype of the result.

        Returns:
            2D kernel (size x size).

        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma, device, dtype)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute MS_SSIM loss.

        Args:
            img1: the predicted image with shape :math:`(B, C, H, W)`.
            img2: the target image with a shape of :math:`(B, C, H, W)`.

        Returns:
            Estimated MS-SSIM_L1 loss.

        """
        if not isinstance(img1, torch.Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(img1)}")

        if not isinstance(img2, torch.Tensor):
            raise TypeError(f"Output type is not a torch.Tensor. Got {type(img2)}")

        if not len(img1.shape) == len(img2.shape):
            raise ValueError(f"Input shapes should be same. Got {type(img1)} and {type(img2)}.")

        mask_dtype = self._g_masks.dtype
        if img1.is_complex() or img2.is_complex():
            output_dtype = torch.promote_types(img1.dtype, img2.dtype)
            compute_dtype = output_dtype
            g_masks = self._g_masks
        else:
            # The masks carry fractional weights, so integer and bool images count as the mask dtype.
            dtype1 = img1.dtype if img1.is_floating_point() else mask_dtype
            dtype2 = img2.dtype if img2.is_floating_point() else mask_dtype
            output_dtype = torch.promote_types(dtype1, dtype2)
            # Half-precision moments overflow (local means above about 181 in float16) and cancel, so they are
            # computed in float32.
            compute_dtype = torch.promote_types(output_dtype, mask_dtype)
            if compute_dtype in (torch.float16, torch.bfloat16):
                compute_dtype = torch.float32
            g_masks = self._g_masks.to(compute_dtype)
            if mask_dtype in (torch.float16, torch.bfloat16):
                # Rounded to half precision, the masks no longer sum to one (in bfloat16 only to within 2.4e-3),
                # which shifts every variance by about (1 - sum) * mu^2: renormalise the float32 copy.
                g_masks = g_masks / g_masks.sum(dim=(-2, -1), keepdim=True)

        img1 = img1.to(compute_dtype)
        img2 = img2.to(compute_dtype)

        CH: int = img1.shape[-3]
        # A grouped convolution gives each input channel a contiguous block of output channels, so the masks are
        # repeated channel-major: output ``c * S + s`` is channel ``c`` filtered at scale ``s``.
        g_masks = torch.jit.annotate(torch.Tensor, g_masks).repeat(CH, 1, 1, 1)

        if img1.device.type == "cpu":
            with torch.autocast(device_type="cpu", enabled=False):
                loss = self._compute_loss(img1, img2, g_masks)
        elif img1.device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=False):
                loss = self._compute_loss(img1, img2, g_masks)
        elif img1.device.type == "mps":
            with torch.autocast(device_type="mps", enabled=False):
                loss = self._compute_loss(img1, img2, g_masks)
        else:
            loss = self._compute_loss(img1, img2, g_masks)
        return loss.to(output_dtype)

    def _compute_loss(self, img1: torch.Tensor, img2: torch.Tensor, g_masks: torch.Tensor) -> torch.Tensor:
        CH: int = img1.shape[-3]
        S: int = self.num_scales

        mux = F.conv2d(img1, g_masks, groups=CH, padding=self.pad)
        muy = F.conv2d(img2, g_masks, groups=CH, padding=self.pad)
        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(img1 * img1, g_masks, groups=CH, padding=self.pad) - mux2
        sigmay2 = F.conv2d(img2 * img2, g_masks, groups=CH, padding=self.pad) - muy2
        sigmaxy = F.conv2d(img1 * img2, g_masks, groups=CH, padding=self.pad) - muxy

        lc = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)
        # Per channel: luminance at the coarsest scale times contrast-structure at every scale.
        lM = lc[:, S - 1 :: S]
        PIcs = cs.unflatten(1, [CH, S]).prod(dim=2)

        # Compute MS-SSIM loss, averaged over the channels
        loss_ms_ssim = 1 - (lM * PIcs).mean(dim=1)

        # TODO: pass pointer to function e.g. to make more custom with mse, cosine, etc.
        # Compute L1 loss
        loss_l1 = F.l1_loss(img1, img2, reduction="none")

        # Average over the channels of the l1 loss filtered at the coarsest scale
        gaussian_l1 = F.conv2d(loss_l1, g_masks[S - 1 :: S], groups=CH, padding=self.pad).mean(1)

        # Compute MS-SSIM + L1 loss
        loss = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss = self.compensation * loss

        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            pass
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")
        return loss
