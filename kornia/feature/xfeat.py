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

"""XFeat: Accelerated Features for Lightweight Image Matching.

This module is adapted from the original XFeat implementation:
    https://github.com/verlab/accelerated_features

Original work copyright (c) 2024 Verlab / UFMG, licensed under the Apache License 2.0.
Modifications copyright (c) 2024 Kornia Team, licensed under the Apache License 2.0.

Reference:
    Guilherme Potje, Felipe Cadar, Andre Araujo, Renato Martins, Erickson R. Nascimento.
    "XFeat: Accelerated Features for Lightweight Image Matching", CVPR 2024.
    https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core.check import KORNIA_CHECK
from kornia.geometry.subpix import nms2d


class BasicLayer(nn.Module):
    """Basic convolutional layer: Conv2d -> BatchNorm2d (no affine) -> ReLU.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size. Default: ``3``.
        stride: convolution stride. Default: ``1``.
        padding: convolution padding. Default: ``1``.
        dilation: convolution dilation. Default: ``1``.
        bias: whether to use bias. Default: ``False``.
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
    ) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias
            ),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class XFeatModel(nn.Module):
    """XFeat backbone: CNN feature extractor, keypoint and reliability heads.

    Implements the architecture from
    "XFeat: Accelerated Features for Lightweight Image Matching", CVPR 2024.

    Input: float image tensor :math:`(B, C, H, W)` (grayscale or RGB, any channel count).
    Output:

    - ``feats``: dense descriptors :math:`(B, 64, H/8, W/8)`.
    - ``keypoints``: keypoint logits :math:`(B, 65, H/8, W/8)`.
    - ``heatmap``: reliability map :math:`(B, 1, H/8, W/8)`.

    Note:
        Image normalisation (``InstanceNorm2d``) is wrapped in ``torch.no_grad()``
        following the original design; backpropagating through it is not supported.
    """

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(1)

        self.skip1 = nn.Sequential(
            nn.AvgPool2d(4, stride=4),
            nn.Conv2d(1, 24, 1, stride=1, padding=0),
        )

        self.block1 = nn.Sequential(
            BasicLayer(1, 4, stride=1),
            BasicLayer(4, 8, stride=2),
            BasicLayer(8, 8, stride=1),
            BasicLayer(8, 24, stride=2),
        )

        self.block2 = nn.Sequential(
            BasicLayer(24, 24, stride=1),
            BasicLayer(24, 24, stride=1),
        )

        self.block3 = nn.Sequential(
            BasicLayer(24, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, 1, padding=0),
        )

        self.block4 = nn.Sequential(
            BasicLayer(64, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
        )

        self.block5 = nn.Sequential(
            BasicLayer(64, 128, stride=2),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 64, 1, padding=0),
        )

        self.block_fusion = nn.Sequential(
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
            nn.Conv2d(64, 64, 1, padding=0),
        )

        self.heatmap_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

        self.keypoint_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 65, 1),
        )

        self.fine_matcher = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
        )

    def _unfold2d(self, x: torch.Tensor, ws: int = 2) -> torch.Tensor:
        """Unfold spatial dims by window size ``ws`` and concatenate into channels."""
        B, C, H, W = x.shape
        x = x.unfold(2, ws, ws).unfold(3, ws, ws).reshape(B, C, H // ws, W // ws, ws**2)
        return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H // ws, W // ws)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the XFeat backbone.

        Args:
            x: image tensor of shape :math:`(B, C, H, W)`.

        Returns:
            Tuple of:
            - feats: dense descriptors :math:`(B, 64, H/8, W/8)`.
            - keypoints: keypoint logits :math:`(B, 65, H/8, W/8)`.
            - heatmap: reliability map :math:`(B, 1, H/8, W/8)`.
        """
        with torch.no_grad():
            x = x.mean(dim=1, keepdim=True)
            x = self.norm(x)

        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode="bilinear", align_corners=False)
        x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode="bilinear", align_corners=False)
        feats = self.block_fusion(x3 + x4 + x5)

        heatmap = self.heatmap_head(feats)
        keypoints = self.keypoint_head(self._unfold2d(x, ws=8))
        return feats, keypoints, heatmap


class InterpolateSparse2d(nn.Module):
    """Bilinearly or bicubically sample a dense feature map at sparse 2-D positions.

    Args:
        mode: interpolation mode for :func:`torch.nn.functional.grid_sample`.
            Default: ``'bicubic'``.
        align_corners: passed to ``grid_sample``. Default: ``False``.

    Shape:
        - Input ``x``: :math:`(B, C, H, W)`.
        - Input ``pos``: :math:`(B, N, 2)` integer or float (x, y) coordinates.
        - Output: :math:`(B, N, C)`.
    """

    def __init__(self, mode: str = "bicubic", align_corners: bool = False) -> None:
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def normgrid(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Normalise pixel coordinates to the ``[-1, 1]`` range expected by ``grid_sample``.

        Uses the formula from the original XFeat implementation:
        ``2 * x / [W-1, H-1] - 1``.  Note that ``grid_sample`` is always
        called with ``align_corners=False``; this asymmetry matches the
        convention the pretrained weights were trained with.
        """
        return 2.0 * (x / torch.tensor([W - 1, H - 1], device=x.device, dtype=x.dtype)) - 1.0

    def forward(self, x: torch.Tensor, pos: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Sample ``x`` at positions ``pos``.

        Args:
            x: feature map :math:`(B, C, H, W)`.
            pos: sampling positions :math:`(B, N, 2)` in pixel coordinates.
            H: height used for coordinate normalisation.
            W: width used for coordinate normalisation.

        Returns:
            Sampled features :math:`(B, N, C)`.
        """
        grid = self.normgrid(pos, H, W).unsqueeze(-2).to(x.dtype)
        # align_corners=False is intentional: the pretrained XFeat weights were trained
        # with normgrid using the (W-1) denominator but grid_sample using align_corners=False.
        # Changing either side would break compatibility with the pretrained weights.
        x = F.grid_sample(x, grid, mode=self.mode, align_corners=False)
        return x.permute(0, 2, 3, 1).squeeze(-2)


class XFeat(nn.Module):
    """XFeat sparse and semi-dense local feature extractor and matcher.

    Wraps :class:`XFeatModel` with NMS keypoint detection, descriptor
    interpolation, and mutual nearest-neighbour matching helpers.

    Reference:
        "XFeat: Accelerated Features for Lightweight Image Matching", CVPR 2024.
        https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    .. image:: _static/img/XFeat.png

    Args:
        top_k: maximum number of keypoints to keep per image. Default: ``4096``.
        detection_threshold: minimum keypoint score. Default: ``0.05``.

    Example:
        >>> model = XFeat()
        >>> img = torch.rand(1, 3, 256, 256)
        >>> out = model.detectAndCompute(img)
        >>> out[0]['keypoints'].shape
        torch.Size([..., 2])
    """

    weights_url: str = "https://github.com/verlab/accelerated_features/raw/main/weights/xfeat.pt"

    def __init__(self, top_k: int = 4096, detection_threshold: float = 0.05) -> None:
        super().__init__()
        self.top_k = top_k
        self.detection_threshold = detection_threshold
        self.net = XFeatModel()
        self.interpolator = InterpolateSparse2d("bicubic")
        self._nearest = InterpolateSparse2d("nearest")
        self._bilinear = InterpolateSparse2d("bilinear")

    @classmethod
    def from_pretrained(cls, top_k: int = 4096, detection_threshold: float = 0.05) -> XFeat:
        """Instantiate XFeat with pretrained weights downloaded from the official release.

        Args:
            top_k: maximum number of keypoints to keep. Default: ``4096``.
            detection_threshold: minimum keypoint score. Default: ``0.05``.

        Returns:
            XFeat model with pretrained weights loaded, set to eval mode.
        """
        model = cls(top_k=top_k, detection_threshold=detection_threshold)
        state_dict = torch.hub.load_state_dict_from_url(cls.weights_url, file_name="xfeat.pt")
        model.net.load_state_dict(state_dict)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess_tensor(self, x: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Resize to the largest multiple of 32 not exceeding the input size (min 32); return image and scale ratios."""
        KORNIA_CHECK(x.dim() == 4, "Input must be a 4-D tensor (B, C, H, W)")
        H, W = x.shape[-2:]
        _H, _W = max(32, (H // 32) * 32), max(32, (W // 32) * 32)
        rh, rw = H / _H, W / _W
        x = F.interpolate(x.float(), (_H, _W), mode="bilinear", align_corners=False)
        return x, rh, rw

    @staticmethod
    def _get_kpts_heatmap(kpts: torch.Tensor, softmax_temp: float = 1.0) -> torch.Tensor:
        """Convert 65-channel keypoint logits to a full-resolution heatmap."""
        scores = F.softmax(kpts * softmax_temp, 1)[:, :64]
        B, _, H, W = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        return heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H * 8, W * 8)

    @staticmethod
    def _nms(x: torch.Tensor, threshold: float = 0.05, kernel_size: int = 5) -> torch.Tensor:
        """Non-maximum suppression on a heatmap; returns (B, N, 2) integer keypoint positions."""
        B = x.shape[0]
        pos = nms2d(x, (kernel_size, kernel_size), mask_only=True) & (x > threshold)
        pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos]
        pad_val = max(len(p) for p in pos_batched)
        pos_out = torch.full((B, pad_val, 2), -1, dtype=torch.long, device=x.device)
        for b in range(B):
            pos_out[b, : len(pos_batched[b])] = pos_batched[b]
        return pos_out

    @staticmethod
    def _create_xy(h: int, w: int, device: torch.device) -> torch.Tensor:
        """Create a grid of (x, y) pixel coordinates of shape :math:`(H*W, 2)`."""
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        return torch.cat([x[..., None], y[..., None]], -1).reshape(-1, 2)

    @staticmethod
    def _subpix_softmax2d(heatmaps: torch.Tensor, temp: float = 3.0) -> torch.Tensor:
        """Compute soft-argmax offsets from an (N, H, W) heatmap."""
        N, H, W = heatmaps.shape
        heatmaps = torch.softmax(temp * heatmaps.view(N, H * W), -1).view(N, H, W)
        x, y = torch.meshgrid(
            torch.arange(W, device=heatmaps.device), torch.arange(H, device=heatmaps.device), indexing="xy"
        )
        x = x - (W // 2)
        y = y - (H // 2)
        coords = torch.cat([(x[None] * heatmaps)[..., None], (y[None] * heatmaps)[..., None]], -1)
        return coords.view(N, H * W, 2).sum(1)

    def _match_mnn(
        self, feats1: torch.Tensor, feats2: torch.Tensor, min_cossim: float = 0.82
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mutual nearest-neighbour matching on L2-normalised descriptor pairs."""
        cossim = feats1 @ feats2.t()
        cossim_t = feats2 @ feats1.t()
        _, match12 = cossim.max(dim=1)
        _, match21 = cossim_t.max(dim=1)
        idx0 = torch.arange(len(match12), device=match12.device)
        mutual = match21[match12] == idx0
        if min_cossim > 0:
            cossim_max, _ = cossim.max(dim=1)
            good = cossim_max > min_cossim
            idx0 = idx0[mutual & good]
            idx1 = match12[mutual & good]
        else:
            idx0 = idx0[mutual]
            idx1 = match12[mutual]
        return idx0, idx1

    def _batch_match(
        self, feats1: torch.Tensor, feats2: torch.Tensor, min_cossim: float = -1
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Batched MNN matching; returns a list of ``(idx0, idx1)`` tuples."""
        B = feats1.shape[0]
        cossim = torch.bmm(feats1, feats2.permute(0, 2, 1))
        match12 = torch.argmax(cossim, dim=-1)
        match21 = torch.argmax(cossim.permute(0, 2, 1), dim=-1)
        idx0 = torch.arange(match12.shape[1], device=match12.device)
        batched_matches = []
        for b in range(B):
            mutual = match21[b][match12[b]] == idx0
            if min_cossim > 0:
                cossim_max, _ = cossim[b].max(dim=1)
                good = cossim_max > min_cossim
                idx0_b = idx0[mutual & good]
                idx1_b = match12[b][mutual & good]
            else:
                idx0_b = idx0[mutual]
                idx1_b = match12[b][mutual]
            batched_matches.append((idx0_b, idx1_b))
        return batched_matches

    def _extract_dense(self, x: torch.Tensor, top_k: int = 8000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract coarse descriptors from an FPN feature map, top-k by reliability."""
        if top_k < 1:
            top_k = 100_000_000
        x, rh1, rw1 = self._preprocess_tensor(x)
        M1, _K1, H1 = self.net(x)
        B, C, _H1, _W1 = M1.shape
        xy1 = (self._create_xy(_H1, _W1, M1.device) * 8).expand(B, -1, -1)
        M1 = M1.permute(0, 2, 3, 1).reshape(B, -1, C)
        H1 = H1.permute(0, 2, 3, 1).reshape(B, -1)
        _, top_k_idx = torch.topk(H1, k=min(H1.shape[1], top_k), dim=-1)
        feats = torch.gather(M1, 1, top_k_idx[..., None].expand(-1, -1, C))
        mkpts = torch.gather(xy1, 1, top_k_idx[..., None].expand(-1, -1, 2))
        mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1, -1)
        return mkpts, feats

    def _extract_dualscale(
        self, x: torch.Tensor, top_k: int, s1: float = 0.6, s2: float = 1.3
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract dense features at two scales and merge."""
        x1 = F.interpolate(x, scale_factor=s1, align_corners=False, mode="bilinear")
        x2 = F.interpolate(x, scale_factor=s2, align_corners=False, mode="bilinear")
        mkpts_1, feats_1 = self._extract_dense(x1, int(top_k * 0.20))
        mkpts_2, feats_2 = self._extract_dense(x2, int(top_k * 0.80))
        mkpts = torch.cat([mkpts_1 / s1, mkpts_2 / s2], dim=1)
        sc1 = torch.ones(mkpts_1.shape[:2], device=mkpts_1.device) * (1 / s1)
        sc2 = torch.ones(mkpts_2.shape[:2], device=mkpts_2.device) * (1 / s2)
        return mkpts, torch.cat([sc1, sc2], dim=1), torch.cat([feats_1, feats_2], dim=1)

    def _refine_matches(
        self,
        d0: Dict[str, torch.Tensor],
        d1: Dict[str, torch.Tensor],
        matches: List[Tuple[torch.Tensor, torch.Tensor]],
        batch_idx: int,
        fine_conf: float = 0.25,
    ) -> torch.Tensor:
        """Refine coarse match positions using the fine-matcher MLP with subpixel softmax."""
        idx0, idx1 = matches[batch_idx]
        feats1 = d0["descriptors"][batch_idx][idx0]
        feats2 = d1["descriptors"][batch_idx][idx1]
        mkpts_0 = d0["keypoints"][batch_idx][idx0].clone()
        mkpts_1 = d1["keypoints"][batch_idx][idx1]
        sc0 = d0["scales"][batch_idx][idx0]
        offsets = self.net.fine_matcher(torch.cat([feats1, feats2], dim=-1))
        conf = F.softmax(offsets * 3, dim=-1).max(dim=-1)[0]
        offsets = self._subpix_softmax2d(offsets.view(-1, 8, 8))
        mkpts_0 = mkpts_0 + offsets * sc0[:, None]
        mask_good = conf > fine_conf
        return torch.cat([mkpts_0[mask_good], mkpts_1[mask_good]], dim=-1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def detectAndCompute(
        self,
        x: torch.Tensor,
        top_k: Optional[int] = None,
        detection_threshold: Optional[float] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """Detect sparse keypoints and compute descriptors.

        Args:
            x: image tensor of shape :math:`(B, C, H, W)`.
            top_k: number of keypoints to keep (overrides ``self.top_k``).
            detection_threshold: minimum score (overrides ``self.detection_threshold``).

        Returns:
            List of length ``B``. Each element is a dict with:

            - ``'keypoints'``: :math:`(N, 2)` keypoints in (x, y) pixel coordinates.
            - ``'scores'``: :math:`(N,)` reliability scores.
            - ``'descriptors'``: :math:`(N, 64)` L2-normalised descriptors.
        """
        if top_k is None:
            top_k = self.top_k
        if detection_threshold is None:
            detection_threshold = self.detection_threshold
        x, rh1, rw1 = self._preprocess_tensor(x)
        B, _, _H1, _W1 = x.shape

        M1, K1, H1 = self.net(x)
        M1 = F.normalize(M1, dim=1)

        K1h = self._get_kpts_heatmap(K1)
        mkpts = self._nms(K1h, threshold=detection_threshold, kernel_size=5)

        scores = (self._nearest(K1h, mkpts, _H1, _W1) * self._bilinear(H1, mkpts, _H1, _W1)).squeeze(-1)
        scores[torch.all(mkpts == -1, dim=-1)] = -1

        k = min(top_k, scores.shape[-1])
        scores, idxs = torch.topk(scores, k=k, dim=-1, largest=True, sorted=True)
        mkpts_x = torch.gather(mkpts[..., 0], -1, idxs)
        mkpts_y = torch.gather(mkpts[..., 1], -1, idxs)
        mkpts = torch.cat([mkpts_x[..., None], mkpts_y[..., None]], dim=-1)

        feats = self.interpolator(M1, mkpts, H=_H1, W=_W1)
        feats = F.normalize(feats, dim=-1)

        mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1, 1, -1)

        valid = scores > 0
        return [
            {
                "keypoints": mkpts[b][valid[b]],
                "scores": scores[b][valid[b]],
                "descriptors": feats[b][valid[b]],
            }
            for b in range(B)
        ]

    @torch.inference_mode()
    def detectAndComputeDense(
        self,
        x: torch.Tensor,
        top_k: Optional[int] = None,
        multiscale: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Detect keypoints and compute dense coarse descriptors.

        Args:
            x: image tensor of shape :math:`(B, C, H, W)`.
            top_k: number of features to keep (overrides ``self.top_k``).
            multiscale: use dual-scale (0.6x and 1.3x) extraction. Default: ``True``.

        Returns:
            Dict with:

            - ``'keypoints'``: :math:`(B, K, 2)` coarse keypoints.
            - ``'descriptors'``: :math:`(B, K, 64)` coarse descriptors.
            - ``'scales'``: :math:`(B, K)` extraction scale per keypoint.
        """
        if top_k is None:
            top_k = self.top_k
        if multiscale:
            mkpts, sc, feats = self._extract_dualscale(x, top_k)
        else:
            mkpts, feats = self._extract_dense(x, top_k)
            sc = torch.ones(mkpts.shape[:2], device=mkpts.device)
        return {"keypoints": mkpts, "descriptors": feats, "scales": sc}

    @torch.inference_mode()
    def match_xfeat(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        top_k: Optional[int] = None,
        min_cossim: float = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect, describe and mutually match keypoints from two images.

        Args:
            img1: first image tensor of shape :math:`(1, C, H, W)`.
            img2: second image tensor of shape :math:`(1, C, H, W)`.
            top_k: number of top keypoints to use.
            min_cossim: minimum cosine similarity threshold. Use ``-1`` to disable.

        Returns:
            Tuple ``(mkpts0, mkpts1)`` of matched keypoints, each :math:`(N, 2)`.
        """
        if top_k is None:
            top_k = self.top_k
        out1 = self.detectAndCompute(img1, top_k=top_k)[0]
        out2 = self.detectAndCompute(img2, top_k=top_k)[0]
        idx0, idx1 = self._match_mnn(out1["descriptors"], out2["descriptors"], min_cossim=min_cossim)
        return out1["keypoints"][idx0], out2["keypoints"][idx1]

    @torch.inference_mode()
    def match_xfeat_star(
        self,
        im_set1: torch.Tensor,
        im_set2: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> List[torch.Tensor] | Tuple[torch.Tensor, torch.Tensor]:
        """Extract coarse features, match pairs and refine matches (XFeat*).

        Args:
            im_set1: batch of images :math:`(B, C, H, W)`.
            im_set2: batch of images :math:`(B, C, H, W)`.
            top_k: number of top features to use.

        Returns:
            If ``B > 1``: list of :math:`(N, 4)` tensors with ``(x1, y1, x2, y2)`` matches.
            If ``B == 1``: tuple of :math:`(N, 2)` matched keypoint tensors.
        """
        if top_k is None:
            top_k = self.top_k
        out1 = self.detectAndComputeDense(im_set1, top_k=top_k)
        out2 = self.detectAndComputeDense(im_set2, top_k=top_k)
        idxs_list = self._batch_match(out1["descriptors"], out2["descriptors"])
        B = im_set1.shape[0]
        matches = [self._refine_matches(out1, out2, matches=idxs_list, batch_idx=b) for b in range(B)]
        if B > 1:
            return matches
        return matches[0][:, :2], matches[0][:, 2:]
