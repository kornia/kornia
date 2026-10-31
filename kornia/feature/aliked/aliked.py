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

# Adapted from the ALIKED implementation in LightGlue
# (https://github.com/cvg/LightGlue/blob/main/lightglue/aliked.py)
# which itself is based on the original ALIKED code by Zhao Xiaoming et al.:
# https://github.com/Shiaoming/ALIKED
#
# BSD 3-Clause License
#
# Copyright (c) 2022, Zhao Xiaoming
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Authors:
# Xiaoming Zhao, Xingming Wu, Weihai Chen, Peter C.Y. Chen, Qingsong Xu, and
# Zhengguo Li

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.utils import _pair

from kornia.color import grayscale_to_rgb
from kornia.core.download import load_state_dict_from_url
from kornia.geometry.subpix import nms2d

from .deform_conv2d import deform_conv2d

# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------


@dataclass
class ALIKEDFeatures:
    r"""Keypoints, descriptors and scores detected by ALIKED for a single image.

    Since ALIKED detects a varying number of keypoints per image,
    ``ALIKEDFeatures`` is not batched.

    Convention:
        The keypoints, descriptors and scores that :class:`~kornia.feature.ALIKED` returns for one image; its
        Convention block gives their frame and how many there are.

    Args:
        keypoints: pixel coordinates ``(N, 2)`` as ``[x, y]``.
        descriptors: L2-normalised descriptors ``(N, D)``.
        keypoint_scores: detection confidence scores ``(N,)``.
    """

    keypoints: torch.Tensor
    descriptors: torch.Tensor
    keypoint_scores: torch.Tensor

    @property
    def n(self) -> int:
        """Number of detected keypoints."""
        return self.keypoints.shape[0]

    def to(self, *args: Any, **kwargs: Any) -> ALIKEDFeatures:
        """Move all tensors to a new device / dtype."""
        return ALIKEDFeatures(
            self.keypoints.to(*args, **kwargs),
            self.descriptors.to(*args, **kwargs),
            self.keypoint_scores.to(*args, **kwargs),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def get_patches(tensor: torch.Tensor, required_corners: torch.Tensor, ps: int) -> torch.Tensor:
    """Extract ps x ps patches centred at required_corners from a CHW tensor."""
    c, h, w = tensor.shape
    corner = (required_corners - ps / 2 + 1).long()
    corner[:, 0] = corner[:, 0].clamp(min=0, max=w - 1 - ps)
    corner[:, 1] = corner[:, 1].clamp(min=0, max=h - 1 - ps)
    offset = torch.arange(0, ps)
    x, y = torch.meshgrid(offset, offset, indexing="ij")
    patches = torch.stack((x, y)).permute(2, 1, 0).unsqueeze(2)
    patches = patches.to(corner) + corner[None, None]
    pts = patches.reshape(-1, 2)
    sampled = tensor.permute(1, 2, 0)[tuple(pts.T)[::-1]]
    sampled = sampled.reshape(ps, ps, -1, c)
    return sampled.permute(2, 3, 0, 1)


def _reads_channels_last(image: torch.Tensor) -> bool:
    """Whether :meth:`ALIKED.forward` asks for the ``channels_last`` feature map; see ``extract_dense_map``."""
    return image.device.type == "cpu" and image.dtype in (torch.float32, torch.float64)


# ---------------------------------------------------------------------------
# LAF helpers
# ---------------------------------------------------------------------------


def _affine_from_cov(cov: torch.Tensor) -> torch.Tensor:
    """Build a 2x2 affine matrix from a batch of 2x2 covariance matrices.

    The affine is ``U @ diag(sqrt(eigenvalues))``, where ``U`` contains the
    orthonormal eigenvectors of ``cov`` as *columns*, in ascending order of
    eigenvalue.  This gives an ellipse whose axes align with the principal
    directions of the covariance (``A @ A.T == cov``).  The sign of the first
    eigenvector is chosen so that ``det(U) = +1``: the affine is a rotation
    times a scaling, never mirrored.

    Args:
        cov: symmetric positive-semi-definite matrices ``(N, 2, 2)``.

    Returns:
        Affine matrices ``(N, 2, 2)``.
    """
    # eigh returns eigenvalues sorted ascending; columns of eigenvectors are evecs.
    # torch.linalg.eigh is not implemented for float16/bfloat16 on CUDA, so promote
    # to float32 and cast the result back.
    orig_dtype = cov.dtype
    if cov.dtype in (torch.float16, torch.bfloat16):
        cov = cov.float()
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    # eigh fixes each eigenvector only up to sign, and the ascending order alone swaps the axes of a covariance that
    # is wider along x, so det(U) is often -1, a mirrored frame. Negate the first eigenvector there.
    det_u = eigenvectors[:, 0, 0] * eigenvectors[:, 1, 1] - eigenvectors[:, 0, 1] * eigenvectors[:, 1, 0]
    sign = 1.0 - 2.0 * (det_u < 0).to(eigenvectors.dtype)  # (N,)
    eigenvectors = eigenvectors * torch.stack([sign, torch.ones_like(sign)], dim=-1)[:, None, :]
    scales = eigenvalues.clamp(min=1e-8).sqrt()  # (N, 2)
    # Each column of eigenvectors multiplied by the corresponding scale.
    return (eigenvectors * scales[:, None, :]).to(orig_dtype)  # (N, 2, 2)


def _laf_from_kpts_and_affine(
    keypoints_px: torch.Tensor,
    affine: torch.Tensor,
) -> torch.Tensor:
    """Assemble a ``(N, 2, 3)`` LAF from pixel keypoints and 2x2 affine matrices.

    Args:
        keypoints_px: ``(N, 2)`` pixel coordinates ``[x, y]``.
        affine: ``(N, 2, 2)`` affine (rotation+scale) part of the LAF.

    Returns:
        LAF tensor ``(N, 2, 3)`` following the kornia convention
        ``[[a, b, cx], [c, d, cy]]``.
    """
    centers = keypoints_px.unsqueeze(-1)  # (N, 2, 1)
    return torch.cat([affine, centers], dim=-1)  # (N, 2, 3)


# ---------------------------------------------------------------------------
# Differentiable Keypoint Detection (DKD)
# ---------------------------------------------------------------------------


class DKD(nn.Module):
    """Differentiable keypoint detection from a score map.

    Args:
        radius: soft detection radius; NMS kernel size is ``2 * radius + 1``. The NMS keeps strict local maxima
            only, unlike the reference's ``simple_nms``; the note of :class:`ALIKED` explains the difference.
        top_k: if ``> 0`` return at most ``top_k`` keypoints per image: its ``top_k`` highest-scoring local maxima,
            or all of them when it has fewer; otherwise threshold mode is used.
        scores_th: score threshold when ``top_k <= 0``.
            If ``> 0`` keep keypoints with ``score > scores_th``, and in an image with none above it, the keypoints
            above that image's mean score; otherwise keep keypoints above the per-image mean score.
        n_limit: maximum number of keypoints returned in threshold mode.
    """

    def __init__(
        self,
        radius: int = 2,
        top_k: int = 0,
        scores_th: float = 0.2,
        n_limit: int = 20000,
    ) -> None:
        super().__init__()
        self.radius = radius
        self.top_k = top_k
        self.scores_th = scores_th
        self.n_limit = n_limit
        self.kernel_size = 2 * self.radius + 1
        self.temperature = 0.1
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, padding=self.radius)
        x = torch.linspace(-self.radius, self.radius, self.kernel_size)
        hw_grid = torch.stack(torch.meshgrid([x, x], indexing="ij")).view(2, -1).t()[:, [1, 0]]
        # derived from ``radius``, so not part of the state dict: the pretrained checkpoints do not carry it
        self.register_buffer("hw_grid", hw_grid, persistent=False)

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
        # A state dict that carries ``hw_grid`` still loads strictly; the grid is rebuilt from ``radius``.
        # ``load_state_dict`` passes a copy, so the caller's dict keeps the key.
        state_dict.pop(prefix + "hw_grid", None)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(
        self,
        scores_map: torch.Tensor,
        sub_pixel: bool = True,
        image_size: Optional[torch.Tensor] = None,
        return_affine: bool = False,
    ) -> tuple[list[torch.Tensor], ...]:
        """Detect keypoints from a score map.

        Args:
            scores_map: ``(B, 1, H, W)`` detection score map.
            sub_pixel: whether to apply soft-argmax sub-pixel refinement.
            image_size: optional ``(B, 2)`` tensor of valid image sizes ``(W, H)``
                for border masking.
            return_affine: if ``True``, also return per-keypoint 2x2 local affine
                matrices estimated from the soft-argmax weight covariance.

        Returns:
            A 3-tuple ``(keypoints, keypoint_scores, score_dispersities)`` — or a
            4-tuple ``(..., local_affines)`` when ``return_affine=True``.
            Each ``keypoints[i]`` is ``(N_i, 2)`` normalised to ``[-1, 1]``.
            Each ``local_affines[i]`` is ``(N_i, 2, 2)`` (only when ``sub_pixel=True``
            and ``return_affine=True``; otherwise identity matrices are returned).
        """
        b, _c, h, w = scores_map.shape
        scores_nograd = scores_map.detach()
        # the candidates: the strict local maxima outside the border (and outside each image's padding)
        is_max = nms2d(scores_nograd, (self.kernel_size, self.kernel_size), mask_only=True)

        # remove border
        is_max[:, :, : self.radius, :] = False
        is_max[:, :, :, : self.radius] = False
        if image_size is not None:
            for i in range(b):
                iw, ih = image_size[i].long()
                is_max[i, :, ih.item() - self.radius :, :] = False
                is_max[i, :, :, iw.item() - self.radius :] = False
        else:
            is_max[:, :, -self.radius :, :] = False
            is_max[:, :, :, -self.radius :] = False

        if self.top_k > 0:
            # at most top_k of the maxima: a non-maximum scores -inf here and is dropped, not used to fill up to top_k
            candidate_scores = scores_nograd.masked_fill(~is_max, float("-inf")).flatten(start_dim=1)
            topk = torch.topk(candidate_scores, min(self.top_k, h * w))
            indices_keypoints = [topk.indices[i][topk.values[i] > float("-inf")] for i in range(b)]
        else:
            # the per-image mean score: the threshold when scores_th <= 0, and the fallback of an image whose maxima
            # are all at or below scores_th, decided for each image alone
            mean = scores_nograd.flatten(start_dim=1).mean(dim=1).reshape(b, 1, 1, 1)
            masks = is_max & (scores_nograd > mean)
            if self.scores_th > 0:
                above = is_max & (scores_nograd > self.scores_th)
                masks = torch.where(above.flatten(start_dim=1).any(dim=1).reshape(b, 1, 1, 1), above, masks)
            masks = masks.flatten(start_dim=1)

            indices_keypoints = []
            scores_view = scores_nograd.flatten(start_dim=1)
            for mask, scores in zip(masks, scores_view):
                indices = mask.nonzero()[:, 0]
                if len(indices) > self.n_limit:
                    kpts_sc = scores[indices]
                    sort_idx = kpts_sc.sort(descending=True)[1]
                    indices = indices[sort_idx[: self.n_limit]]
                indices_keypoints.append(indices)

        # Keypoints are formed in at least float32: a half-precision dtype holds no odd integer above 256 (bfloat16)
        # or 2048 (float16), not even ``w - 1``. The grid of the score lookup is cast to the map's dtype for the call.
        coord_dtype = torch.promote_types(scores_map.dtype, torch.float32)
        wh = torch.tensor([w - 1, h - 1], device=scores_nograd.device, dtype=coord_dtype)

        keypoints = []
        scoredispersitys = []
        kptscores = []
        local_affines: list[torch.Tensor] = []
        if sub_pixel:
            patches = self.unfold(scores_map)  # B x (kernel**2) x (H*W)
            for b_idx in range(b):
                patch = patches[b_idx].t()  # (H*W) x (kernel**2)
                indices_kpt = indices_keypoints[b_idx]
                patch_scores = patch[indices_kpt]  # M x (kernel**2)
                keypoints_xy_nms = torch.stack(
                    [indices_kpt % w, torch.div(indices_kpt, w, rounding_mode="trunc")],
                    dim=1,
                )

                max_v = patch_scores.max(dim=1).values.detach()[:, None]
                x_exp = ((patch_scores - max_v) / self.temperature).exp()
                x_sum = x_exp.sum(dim=1, keepdim=True)

                xy_residual = x_exp @ self.hw_grid / x_sum

                hw_grid_dist2 = (
                    torch.norm(
                        (self.hw_grid[None, :, :] - xy_residual[:, None, :]) / self.radius,
                        dim=-1,
                    )
                    ** 2
                )
                scoredispersity = (x_exp * hw_grid_dist2).sum(dim=1) / x_exp.sum(dim=1)

                keypoints_xy = keypoints_xy_nms.to(coord_dtype) + xy_residual
                keypoints_xy = keypoints_xy / wh * 2 - 1  # -> [-1, 1]

                kptscore = F.grid_sample(
                    scores_map[b_idx].unsqueeze(0),
                    keypoints_xy.view(1, 1, -1, 2).to(scores_map.dtype),
                    mode="bilinear",
                    align_corners=True,
                )[0, 0, 0, :]

                keypoints.append(keypoints_xy)
                scoredispersitys.append(scoredispersity)
                kptscores.append(kptscore)

                if return_affine:
                    # Weighted covariance of hw_grid positions under soft-argmax weights.
                    # w_i = x_exp_i / sum(x_exp), mu = xy_residual (already computed).
                    # cov = sum_i w_i (g_i - mu)(g_i - mu)^T  ->  (N, 2, 2)
                    W = x_exp / x_sum  # (N, K²) normalised weights
                    delta = self.hw_grid[None] - xy_residual[:, None]  # (N, K², 2)
                    cov = torch.einsum("ni,nij,nik->njk", W, delta, delta)  # (N, 2, 2)
                    local_affines.append(_affine_from_cov(cov))
        else:
            for b_idx in range(b):
                indices_kpt = indices_keypoints[b_idx]
                keypoints_xy_nms = torch.stack(
                    [indices_kpt % w, torch.div(indices_kpt, w, rounding_mode="trunc")],
                    dim=1,
                )
                keypoints_xy = keypoints_xy_nms.to(coord_dtype) / wh * 2 - 1
                kptscore = F.grid_sample(
                    scores_map[b_idx].unsqueeze(0),
                    keypoints_xy.view(1, 1, -1, 2).to(scores_map.dtype),
                    mode="bilinear",
                    align_corners=True,
                )[0, 0, 0, :]
                keypoints.append(keypoints_xy)
                scoredispersitys.append(torch.zeros_like(kptscore))
                kptscores.append(kptscore)
                if return_affine:
                    # No soft-argmax weights available; fall back to identity.
                    N_i = keypoints_xy.shape[0]
                    local_affines.append(
                        torch.eye(2, device=scores_map.device, dtype=scores_map.dtype).unsqueeze(0).expand(N_i, -1, -1)
                    )

        if return_affine:
            return keypoints, kptscores, scoredispersitys, local_affines
        return keypoints, kptscores, scoredispersitys


# ---------------------------------------------------------------------------
# Image padding helper
# ---------------------------------------------------------------------------


class InputPadder:
    """Pad an image so that both spatial dimensions are divisible by ``divis_by``."""

    def __init__(self, h: int, w: int, divis_by: int = 8) -> None:
        self.ht = h
        self.wd = w
        # The distance to the next multiple, written without `%`: under dynamic-shape torch.compile a modulo of the
        # image size reached the float offset bound of DeformableConv2d, which inductor (torch 2.14) cannot lower.
        pad_ht = (h + divis_by - 1) // divis_by * divis_by - h
        pad_wd = (w + divis_by - 1) // divis_by * divis_by - w
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        """Pad *x* (BCHW)."""
        return F.pad(x, self._pad, mode="replicate")

    def unpad(self, x: torch.Tensor) -> torch.Tensor:
        """Remove the padding added by :meth:`pad`."""
        ht, wd = x.shape[-2], x.shape[-1]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class DeformableConv2d(nn.Module):
    """Deformable conv2d (DCNv1 or DCNv2) without torchvision dependency.

    Uses the pure-PyTorch :func:`~kornia.feature.aliked.deform_conv2d.deform_conv2d`
    implementation that matches ``torchvision.ops.deform_conv2d``.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: spatial kernel size.
        stride: convolution stride.
        padding: zero-padding size.
        bias: whether to add a learnable bias.
        mask: if ``True`` use DCNv2 modulation mask (3 * k*k offsets instead of 2).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = False,
        mask: bool = False,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask
        self.channel_num = 3 * kernel_size * kernel_size if mask else 2 * kernel_size * kernel_size
        self.offset_conv = nn.Conv2d(
            in_channels,
            self.channel_num,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=dilation,
            bias=True,
        )
        self.regular_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Run the module forward pass.

        Args:
            x: Input feature map with shape :math:`(B, C_{\text{in}}, H, W)`.

        Returns:
            Deformable convolution output with shape :math:`(B, C_{\text{out}}, H', W')`.
        """
        h, w = x.shape[2:]
        max_offset = max(h, w) / 4.0
        out = self.offset_conv(x)
        if self.mask:
            o1, o2, mask_w = torch.chunk(out, 3, dim=1)
            offset = torch.cat((o1, o2), dim=1)
            mask_w = torch.sigmoid(mask_w)
        else:
            offset = out
            mask_w = None
        offset = offset.clamp(-max_offset, max_offset)
        return deform_conv2d(
            input=x,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask_w,
        )


def get_conv(
    inplanes: int,
    planes: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    bias: bool = False,
    conv_type: str = "conv",
    mask: bool = False,
) -> nn.Module:
    """Return a standard or deformable conv2d layer."""
    if conv_type == "conv":
        return nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    if conv_type == "dcn":
        return DeformableConv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=_pair(padding)[0],
            bias=bias,
            mask=mask,
        )
    raise TypeError(f"Unknown conv_type: {conv_type!r}. Expected 'conv' or 'dcn'.")


class ConvBlock(nn.Module):
    """Two-layer conv block with BN and an activation gate."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gate: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        conv_type: str = "conv",
        mask: bool = False,
    ) -> None:
        super().__init__()
        self.gate = nn.ReLU(inplace=True) if gate is None else gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = get_conv(in_channels, out_channels, kernel_size=3, conv_type=conv_type, mask=mask)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = get_conv(out_channels, out_channels, kernel_size=3, conv_type=conv_type, mask=mask)
        self.bn2 = norm_layer(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Run the module forward pass.

        Args:
            x: Input feature map with shape :math:`(B, C_{\text{in}}, H, W)`.

        Returns:
            Feature map with shape :math:`(B, C_{\text{out}}, H, W)` after two gated conv-BN layers.
        """
        x = self.gate(self.bn1(self.conv1(x)))
        return self.gate(self.bn2(self.conv2(x)))


class ResBlock(nn.Module):
    """Residual block (BasicBlock variant) supporting deformable convolutions."""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        gate: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        conv_type: str = "conv",
        mask: bool = False,
    ) -> None:
        super().__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("ResBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ResBlock")
        self.conv1 = get_conv(inplanes, planes, kernel_size=3, conv_type=conv_type, mask=mask)
        self.bn1 = norm_layer(planes)
        self.conv2 = get_conv(planes, planes, kernel_size=3, conv_type=conv_type, mask=mask)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Run the module forward pass.

        Args:
            x: Input feature map with shape :math:`(B, C_{\text{in}}, H, W)`.

        Returns:
            Feature map with shape :math:`(B, C_{\text{out}}, H', W')` after the residual block,
            where ``H'`` and ``W'`` may differ from the input if ``stride > 1``.
        """
        identity = x
        out = self.gate(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.gate(out + identity)


# ---------------------------------------------------------------------------
# Sparse Descriptor with Deformable Heads (SDDH)
# ---------------------------------------------------------------------------


class SDDH(nn.Module):
    """Sparse descriptor head using deformable sampling positions.

    Args:
        dims: input feature dimension.
        kernel_size: patch size used to estimate sampling offsets.
        n_pos: number of deformable sampling positions per keypoint.
        gate: activation function for the offset network.
        conv2d: if ``True`` use a 1x1 conv to aggregate features; otherwise
            use a learnable weighted sum (``agg_weights``).
        mask: if ``True`` use DCNv2-style attention weighting on samples.
    """

    def __init__(
        self,
        dims: int,
        kernel_size: int = 3,
        n_pos: int = 8,
        gate: Optional[nn.Module] = None,
        conv2d: bool = False,
        mask: bool = False,
    ) -> None:
        super().__init__()
        if gate is None:
            gate = nn.ReLU()
        self.kernel_size = kernel_size
        self.n_pos = n_pos
        self.conv2d = conv2d
        self.mask = mask
        self.get_patches_func = get_patches

        self.channel_num = 3 * n_pos if mask else 2 * n_pos
        self.offset_conv = nn.Sequential(
            nn.Conv2d(dims, self.channel_num, kernel_size=kernel_size, stride=1, padding=0, bias=True),
            gate,
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.sf_conv = nn.Conv2d(dims, dims, kernel_size=1, stride=1, padding=0, bias=False)

        if not conv2d:
            agg_weights = nn.Parameter(torch.rand(n_pos, dims, dims))
            self.register_parameter("agg_weights", agg_weights)
        else:
            self.convM = nn.Conv2d(dims * n_pos, dims, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        keypoints: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Compute descriptors at keypoint locations.

        Args:
            x: dense feature map ``(B, C, H, W)``.
            keypoints: list of ``(N_i, 2)`` normalised keypoint coordinates ``[-1, 1]``.

        Returns:
            A pair ``(descriptors, offsets)`` where each element is a list of length B.
        """
        b, c, h, w = x.shape
        # Positions in at least float32, as in DKD; the sampling grid is cast to the feature map's dtype for the call.
        wh = torch.tensor([[w - 1, h - 1]], device=x.device, dtype=torch.promote_types(x.dtype, torch.float32))
        max_offset = max(h, w) / 4.0

        # Each convolution of the head sees one window (offset_conv) or one sample (sf_conv) per keypoint, so it is a
        # matrix product over the flattened channels and window. F.linear with the convolution's weights computes it
        # as one GEMM; conv2d on these (N, C, k, k) inputs made the head about 3x slower on an Apple M1 CPU.
        conv_offset, gate, conv_offset_1x1 = self.offset_conv
        w_offset = conv_offset.weight.flatten(1)
        w_offset_1x1 = conv_offset_1x1.weight.flatten(1)
        w_sf = self.sf_conv.weight.flatten(1)

        offsets_list = []
        descriptors = []

        for ib in range(b):
            xi, kptsi = x[ib], keypoints[ib]
            kptsi_wh = (kptsi / 2 + 0.5) * wh
            N_kpts = len(kptsi)

            if self.kernel_size > 1:
                patch = self.get_patches_func(xi, kptsi_wh.long(), self.kernel_size)
            else:
                kptsi_wh_long = kptsi_wh.long()
                patch = xi[:, kptsi_wh_long[:, 1], kptsi_wh_long[:, 0]].permute(1, 0).reshape(N_kpts, c, 1, 1)

            # the gate gets the (N, C, 1, 1) map the convolution would give it, so a 2-D-only module stays valid
            offset = gate(F.linear(patch.flatten(1), w_offset, conv_offset.bias)[:, :, None, None]).flatten(1)
            offset = F.linear(offset, w_offset_1x1, conv_offset_1x1.bias).clamp(-max_offset, max_offset)

            if self.mask:
                offset = offset.view(N_kpts, 3, self.n_pos).permute(0, 2, 1)
                mask_w = torch.sigmoid(offset[:, :, -1])
                offset = offset[:, :, :-1]
            else:
                offset = offset.view(N_kpts, 2, self.n_pos).permute(0, 2, 1)
                mask_w = None
            offsets_list.append(offset)

            pos = kptsi_wh.unsqueeze(1) + offset  # [N_kpts, n_pos, 2]
            pos = 2.0 * pos / wh[None] - 1
            pos = pos.reshape(1, N_kpts * self.n_pos, 1, 2)

            features = F.grid_sample(xi.unsqueeze(0), pos.to(xi.dtype), mode="bilinear", align_corners=True)
            features = features.reshape(c, N_kpts, self.n_pos).permute(1, 2, 0)  # [N_kpts, n_pos, C]

            if mask_w is not None:
                features = features * mask_w[:, :, None]

            features = torch.selu_(F.linear(features, w_sf, self.sf_conv.bias))  # [N_kpts, n_pos, C]

            if not self.conv2d:
                descs = features.flatten(1) @ self.agg_weights.flatten(0, 1)
            else:
                features = features.transpose(1, 2).flatten(1)  # channel-major, the order of convM's input channels
                descs = F.linear(features, self.convM.weight.flatten(1))

            descs = F.normalize(descs, p=2.0, dim=1)
            descriptors.append(descs)

        return descriptors, offsets_list


# ---------------------------------------------------------------------------
# ALIKED main module
# ---------------------------------------------------------------------------

_ALIKED_CFGS: dict[str, tuple[int, int, int, int, int, int, int]] = {
    # c1, c2, c3, c4, dim, K, M
    "aliked-t16": (8, 16, 32, 64, 64, 3, 16),
    "aliked-n16": (16, 32, 64, 128, 128, 3, 16),
    "aliked-n16rot": (16, 32, 64, 128, 128, 3, 16),
    "aliked-n32": (16, 32, 64, 128, 128, 3, 32),
}

_CHECKPOINT_URLS = [
    "https://huggingface.co/kornia/aliked/resolve/main/{}.pth",
    "https://github.com/Shiaoming/ALIKED/raw/main/models/{}.pth",
]


class ALIKED(nn.Module):
    r"""ALIKED local feature detector and descriptor.

    ALIKED (Adaptive Local Image KEypoint Detection) combines a multi-scale
    ResNet backbone with deformable descriptor sampling (SDDH) and a
    differentiable keypoint detector (DKD).

    See :cite:`zhao2023aliked` for details.

    .. image:: _static/img/ALIKED.png

    Convention:
        - ``ALIKED()`` builds random weights in training mode, where BatchNorm normalises with the batch, updates its
          running statistics on every call, and a single small image raises; call ``.eval()`` for inference.
          :meth:`from_pretrained` returns the trained model in evaluation mode.
        - ``images`` are RGB in :math:`[0, 1]`, the range of the pretrained weights, and the network applies no
          normalisation; a ``(B, 1, H, W)`` image is repeated to three channels. Any ``H`` and ``W`` work: the
          image is replicate-padded around its centre to multiples of 32 and the maps are cropped back, so the padding
          does not move the keypoints.
        - :meth:`forward` returns one :class:`ALIKEDFeatures` per image. Its keypoints are sub-pixel ``(x, y)`` in the
          pixel frame of :func:`~kornia.feature.laf_from_center_scale_ori`
          (see :ref:`Local features <local-features-conventions>`): a strict maximum of the score map in its
          ``2 * nms_radius + 1`` window (see the note), at least ``nms_radius`` pixels from the border, moved by a
          soft-argmax over that window. Its score is the score map read bilinearly there, and the descriptors are
          L2-normalised. The keypoints, and the LAFs of :meth:`forward_laf`, are float32 for a float16 or bfloat16
          model and otherwise in the image's dtype; scores and descriptors keep the model's dtype.
        - With ``detection_threshold > 0`` (threshold mode), the maxima strictly above it are kept, at most
          ``max_num_keypoints`` of them, and at most 20000 for a non-positive ``max_num_keypoints``. An image with no
          maximum above the threshold keeps the maxima above its own mean score instead, whatever the rest of the
          batch. With ``detection_threshold <= 0`` and ``max_num_keypoints > 0`` (top-k mode), the
          ``max_num_keypoints`` strongest maxima are kept, or all of them when there are fewer; with both ``<= 0``,
          the maxima above the image's mean score.
        - ``image_size`` holds the valid ``(W, H)`` of each image of a batch padded at the bottom and right to a
          common size: maxima closer than ``nms_radius`` pixels to that image's last valid row or column are dropped.
        - :meth:`forward_laf` returns padded tensors like :class:`~kornia.feature.LocalFeature`, but ``N`` is the
          largest keypoint count in the batch and the responses are ``(B, N, 1)``. Its ``mask`` multiplies the
          score map, and it takes no ``image_size``.
        - Known defects: on CPU, float16 and bfloat16 score and feature maps go to torch's ``grid_sample`` without
          an upcast: the half-precision kernel can return wrong values, NaN or overflow, and where torch has none the
          call raises (`#5810 <https://github.com/kornia/kornia/issues/5810>`_).

    Args:
        model_name: backbone configuration, one of
            ``'aliked-t16'``, ``'aliked-n16'``, ``'aliked-n16rot'``, ``'aliked-n32'``.
        max_num_keypoints: the keypoint budget, a cap on the number of maxima kept in threshold mode and in top-k
            mode; see the Convention block.
        detection_threshold: a positive value selects threshold mode and is the score a maximum must exceed; zero or
            a negative value selects top-k mode when ``max_num_keypoints`` is positive. See the Convention block.
        nms_radius: NMS radius (kernel size ``= 2 * nms_radius + 1``). The suppression keeps strict local maxima
            only; see the note below.

    Note:
        Keypoints are the strict local maxima of the score map: :func:`~kornia.geometry.subpix.nms2d` keeps a
        position only if its score is greater than every other score in its ``2 * nms_radius + 1`` window. This
        departs on purpose from the reference implementations (Shiaoming/ALIKED and LightGlue's port). Their
        ``simple_nms`` also keeps tied maxima, and in two further rounds it adds the positions that become the
        maximum of their window once the windows around the maxima already kept are zeroed. Those positions lie on
        the flank of a stronger peak and are not local maxima of the score map. In threshold mode, when neither the
        keypoint budget nor the fallback below decides the count, kornia therefore returns a subset of the reference
        keypoints: with the pretrained ``aliked-n16`` and the default threshold it returns up to 10 % fewer keypoints
        than the reference on the Oxford img1 images, and each one left out has a higher score in its window.
        Otherwise the two keypoint sets need not be nested, because the reference ranks those extra positions with
        the maxima: in top-k mode, where it also fills ``top_k`` slots with positions that are not maxima when the
        score map has fewer maxima, and when the budget cuts the count in threshold mode. The reference falls back to
        the mean score only when no image of the batch has a maximum above the threshold; kornia returns maxima only
        and decides the fallback for each image, so an image's keypoints do not depend on the rest of the batch.

    Example:
        >>> aliked = ALIKED.from_pretrained('aliked-n16')  # doctest: +SKIP
        >>> images = torch.rand(1, 3, 256, 256)  # doctest: +SKIP
        >>> features = aliked(images)  # doctest: +SKIP
    """

    n_limit_max: int = 20000

    def __init__(
        self,
        model_name: str = "aliked-n16",
        max_num_keypoints: int = -1,
        detection_threshold: float = 0.2,
        nms_radius: int = 2,
    ) -> None:
        super().__init__()

        if model_name not in _ALIKED_CFGS:
            raise ValueError(f"Unknown model_name {model_name!r}. Choose from {list(_ALIKED_CFGS)}.")

        self.model_name = model_name
        self.max_num_keypoints = max_num_keypoints
        self.detection_threshold = detection_threshold
        self.nms_radius = nms_radius

        c1, c2, c3, c4, dim, K, M = _ALIKED_CFGS[model_name]
        conv_types = ["conv", "conv", "dcn", "dcn"]
        conv2d = False
        mask = False

        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.norm = nn.BatchNorm2d
        self.gate = nn.SELU(inplace=True)

        self.block1 = ConvBlock(3, c1, self.gate, self.norm, conv_type=conv_types[0])
        self.block2 = self._make_resblock(c1, c2, conv_types[1], mask)
        self.block3 = self._make_resblock(c2, c3, conv_types[2], mask)
        self.block4 = self._make_resblock(c3, c4, conv_types[3], mask)

        self.conv1 = _conv1x1(c1, dim // 4)
        self.conv2 = _conv1x1(c2, dim // 4)
        self.conv3 = _conv1x1(c3, dim // 4)
        self.conv4 = _conv1x1(dim, dim // 4)  # note: c4 == dim for all configs

        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        self.upsample32 = nn.Upsample(scale_factor=32, mode="bilinear", align_corners=True)

        self.score_head = nn.Sequential(
            _conv1x1(dim, 8),
            self.gate,
            _conv3x3(8, 4),
            self.gate,
            _conv3x3(4, 4),
            self.gate,
            _conv3x3(4, 1),
        )
        self.desc_head = SDDH(dim, K, M, gate=self.gate, conv2d=conv2d, mask=mask)
        self.dkd = DKD(
            radius=nms_radius,
            top_k=-1 if detection_threshold > 0 else max_num_keypoints,
            scores_th=detection_threshold,
            n_limit=max_num_keypoints if max_num_keypoints > 0 else self.n_limit_max,
        )

    def _make_resblock(self, c_in: int, c_out: int, conv_type: str, mask: bool) -> ResBlock:
        return ResBlock(
            c_in,
            c_out,
            stride=1,
            downsample=nn.Conv2d(c_in, c_out, 1),
            gate=self.gate,
            norm_layer=self.norm,
            conv_type=conv_type,
            mask=mask,
        )

    def extract_dense_map(
        self, image: torch.Tensor, *, channels_last: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the backbone and return ``(feature_map, score_map)``.

        Args:
            image: ``(B, 3, H, W)`` float image. Inputs are internally padded to
                a multiple of 32 and the padding is removed before returning.
            channels_last: if ``True``, ``feature_map`` has the strides of the ``torch.channels_last`` memory
                format (it is a crop of such a tensor when the image was padded): the ``dim`` channels of each pixel
                lie next to each other. Its values equal those of the default
                layout up to the rounding of the normalisation, and ``score_map`` is the same either way. On CPU,
                normalising the map and the sampling of the descriptor head are then several times faster, so
                :meth:`forward` and :meth:`forward_laf` use it for a float32 or float64 CPU image. They keep the
                default elsewhere: on MPS, ``grid_sample`` is about 4x slower with this layout; on CUDA the forward
                is 0.80-0.94x as fast with it (RTX 4090, float32, float16 and bfloat16); and on CPU torch's float16
                and bfloat16 ``grid_sample`` is unreliable in either layout: it returned NaN for this one on an Apple
                M1, and on x86 it crashes the process for a 640 x 480 image in both
                (`#5810 <https://github.com/kornia/kornia/issues/5810>`_). By default each channel is a plane of the
                map, as before. Under ``torch.compile`` on an x86 CPU the layout still costs time. Inductor (torch
                2.14) lowers the descriptor head's ``grid_sample`` to a loop that gathers one channel at a time across
                the sample points, and on this layout each of those reads lands on another cache line. The compiled
                forward of a 416 x 640 image takes 228 ms with this map and 199 ms without, and eager takes 144 ms
                (4 threads).

        Returns:
            ``feature_map``: ``(B, dim, H, W)`` L2-normalised dense features.
            ``score_map``: ``(B, 1, H, W)`` detection score map in ``(0, 1)``.
        """
        div_by = 2**5
        padder = InputPadder(image.shape[-2], image.shape[-1], div_by)
        image = padder.pad(image)

        x1 = self.block1(image)
        x2 = self.block2(self.pool2(x1))
        x3 = self.block3(self.pool4(x2))
        x4 = self.block4(self.pool4(x3))

        x1 = self.gate(self.conv1(x1))
        x2 = self.gate(self.conv2(x2))
        x3 = self.gate(self.conv3(x3))
        x4 = self.gate(self.conv4(x4))

        x2_up = self.upsample2(x2)
        x3_up = self.upsample8(x3)
        x4_up = self.upsample32(x4)
        x1234 = torch.cat([x1, x2_up, x3_up, x4_up], dim=1)

        score_map = torch.sigmoid(self.score_head(x1234))
        if channels_last:
            # Each pixel's channels are normalised in an explicit (B, H, W, C) tensor, whose permutation back has the
            # strides of a channels_last map. Eager computes the same bits as normalising
            # ``x1234.contiguous(memory_format=torch.channels_last)`` over dim 1. Inductor (torch 2.14, x86) differs:
            # for that form it kept the channel-major loop of the map's producer and transposed every 8 x 8 tile into
            # the channels_last store. For this form it writes pixel by pixel, and the compiled forward of a 416 x 640
            # image is 1.11x faster.
            feature_map = F.normalize(x1234.permute(0, 2, 3, 1).contiguous(), p=2, dim=-1).permute(0, 3, 1, 2)
        else:
            feature_map = F.normalize(x1234, p=2, dim=1)

        feature_map = padder.unpad(feature_map)
        score_map = padder.unpad(score_map)

        return feature_map, score_map

    def forward(
        self,
        images: torch.Tensor,
        image_size: Optional[torch.Tensor] = None,
    ) -> list[ALIKEDFeatures]:
        """Detect and describe local features in a batch of images.

        Args:
            images: ``(B, 3, H, W)`` float images (can be grayscale ``(B, 1, H, W)``
                — will be broadcast to 3 channels automatically).
            image_size: optional ``(B, 2)`` tensor of valid ``(W, H)`` for
                border masking when images are padded to a common size.

        Returns:
            A list of :class:`ALIKEDFeatures` of length B, one per image.
            Keypoints are in pixel coordinates ``[x, y]``.
        """
        if images.shape[1] == 1:
            images = grayscale_to_rgb(images)

        feature_map, score_map = self.extract_dense_map(images, channels_last=_reads_channels_last(images))
        keypoints, kptscores, _scoredispersitys = self.dkd(score_map, image_size=image_size)
        descriptors, _offsets = self.desc_head(feature_map, keypoints)

        B = images.shape[0]
        _, _, h, w = images.shape
        wh = torch.tensor([w - 1, h - 1], device=images.device, dtype=torch.promote_types(images.dtype, torch.float32))

        results = []
        for i in range(B):
            # Convert normalised [-1,1] coords back to pixel coordinates
            kps_px = wh * (keypoints[i] + 1) / 2.0
            results.append(ALIKEDFeatures(kps_px, descriptors[i], kptscores[i]))
        return results

    def forward_laf(
        self,
        img: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        compute_affine: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Detect and describe local features, returning results in kornia LAF format.

        With ``compute_affine=True`` the 2x2 block ``A`` of each LAF comes from
        the covariance of the soft-argmax weights in the keypoint's
        ``2 * nms_radius + 1`` detection window, computed inside :class:`DKD`
        without any additional network parameters: ``A @ A.T`` is that
        covariance, so the axes of the frame are its principal axes, and
        ``det A > 0`` (the frame is never mirrored). The frame describes how
        sharply the score map peaks, not an image structure. Its scale
        (:func:`~kornia.feature.get_laf_scale`) is about a pixel or less with
        the pretrained ``aliked-n16``. That is much smaller than the region the descriptor
        samples, so rescale the frames, for example with
        :func:`~kornia.feature.scale_laf`, before using them as a region size.

        All per-image tensors are zero-padded along the keypoint dimension to
        the largest keypoint count in the batch, so that the outputs are proper
        batched tensors.

        Args:
            img: image to extract features with shape ``(B,C,H,W)``.
            mask: optional spatial mask ``(B, 1, H, W)`` with values in
                ``[0, 1]``; the score map is multiplied by this mask before
                keypoint detection so that features are suppressed in masked
                regions.
            compute_affine: if ``True`` (default), estimate the 2x2 affine shape
                of each LAF using ``torch.linalg.eigh`` on the soft-argmax
                covariance.  Set to ``False`` to skip the eigendecomposition and
                return identity affines, which is faster and avoids the linalg
                call entirely (useful when only keypoint positions are needed).

        Returns:
            - Detected local affine frames with shape ``(B,N,2,3)``.
            - Response function values for corresponding LAFs with shape ``(B,N,1)``.
            - Local descriptors of shape ``(B,N,D)``.

        """
        if img.shape[1] == 1:
            img = grayscale_to_rgb(img)

        feature_map, score_map = self.extract_dense_map(img, channels_last=_reads_channels_last(img))

        if mask is not None:
            # Resize mask to score map resolution and apply.
            mask_rs = F.interpolate(
                mask.to(score_map.dtype), size=score_map.shape[-2:], mode="bilinear", align_corners=True
            )
            score_map = score_map * mask_rs

        dkd_out = self.dkd(score_map, return_affine=compute_affine)
        if compute_affine:
            keypoints, kptscores, _scoredispersitys, local_affines = dkd_out
        else:
            keypoints, kptscores, _scoredispersitys = dkd_out
            local_affines = None
        descriptors, _offsets = self.desc_head(feature_map, keypoints)

        B, _, H, W = img.shape
        wh = torch.tensor([W - 1, H - 1], device=img.device, dtype=torch.promote_types(img.dtype, torch.float32))

        if B == 0:
            # torch.stack rejects an empty list. Derive the empty outputs from the dense maps instead of allocating
            # them, so they stay in the autograd graph, and sample the responses as DKD does, so they keep the
            # dtype a non-empty batch gets (float32 under autocast).
            responses = F.grid_sample(score_map, score_map.new_zeros(0, 1, 0, 2), mode="bilinear", align_corners=True)
            return (
                score_map.flatten()[:0].to(wh.dtype).view(0, 0, 2, 3),
                responses.view(0, 0, 1),
                feature_map.flatten()[:0].view(0, 0, feature_map.shape[1]),
            )

        lafs_list: list[torch.Tensor] = []
        for i in range(B):
            kps_px = wh * (keypoints[i] + 1) / 2.0  # (N, 2)
            if local_affines is not None:
                affine_i = local_affines[i]
            else:
                # Identity affine: both axes are unit-scale, no rotation.
                n = kps_px.shape[0]
                affine_i = torch.eye(2, device=img.device, dtype=img.dtype).unsqueeze(0).expand(n, -1, -1)
            laf_i = _laf_from_kpts_and_affine(kps_px, affine_i)  # (N, 2, 3)
            lafs_list.append(laf_i)

        # Pad to the maximum number of keypoints in the batch.
        n_max = max(laf.shape[0] for laf in lafs_list)

        def _pad(t: torch.Tensor, n: int, fill: float = 0.0) -> torch.Tensor:
            pad_n = n - t.shape[0]
            if pad_n == 0:
                return t
            pad_shape = (pad_n,) + t.shape[1:]
            return torch.cat([t, t.new_full(pad_shape, fill)], dim=0)

        lafs = torch.stack([_pad(laf, n_max) for laf in lafs_list])  # (B, N, 2, 3)
        responses = torch.stack([_pad(s, n_max).unsqueeze(-1) for s in kptscores])  # (B, N, 1)
        descs = torch.stack([_pad(d, n_max) for d in descriptors])  # (B, N, D)

        return lafs, responses, descs

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "aliked-n16",
        max_num_keypoints: int = -1,
        detection_threshold: float = 0.2,
        nms_radius: int = 2,
        device: Optional[torch.device] = None,
    ) -> ALIKED:
        """Load a pretrained ALIKED model from the official checkpoint repository.

        Args:
            model_name: one of ``'aliked-t16'``, ``'aliked-n16'``,
                ``'aliked-n16rot'``, ``'aliked-n32'``.
            max_num_keypoints: passed to :class:`ALIKED` constructor.
            detection_threshold: passed to :class:`ALIKED` constructor.
            nms_radius: passed to :class:`ALIKED` constructor.
            device: target device; defaults to CPU.

        Returns:
            Pretrained :class:`ALIKED` in eval mode.
        """
        if device is None:
            device = torch.device("cpu")
        model = cls(
            model_name=model_name,
            max_num_keypoints=max_num_keypoints,
            detection_threshold=detection_threshold,
            nms_radius=nms_radius,
        ).to(device)
        urls = [t.format(model_name) for t in _CHECKPOINT_URLS]
        state_dict = load_state_dict_from_url(urls, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model
