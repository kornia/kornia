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
from kornia.core.download import load_state_dict_from_url
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
        r"""Run the module forward pass.

        Args:
            x: Input feature map with shape :math:`(B, C_{\text{in}}, H, W)`.

        Returns:
            Output feature map with shape :math:`(B, C_{\text{out}}, H', W')`.
        """
        return self.layer(x)


class XFeatModel(nn.Module):
    """XFeat backbone: CNN feature extractor, keypoint and reliability heads.

    Implements the architecture from
    "XFeat: Accelerated Features for Lightweight Image Matching", CVPR 2024.

    Input: float image tensor ``(B, C, H, W)`` (grayscale or RGB, any channel count).
    Output:

    - ``feats``: dense descriptors ``(B, 64, H/8, W/8)``.
    - ``keypoints``: keypoint logits ``(B, 65, H/8, W/8)``.
    - ``heatmap``: reliability map ``(B, 1, H/8, W/8)``.

    Convention:
        - The channels are averaged and the result is instance-normalised, so any channel count is accepted and an
          affine change ``a x + b`` of the intensities with ``a > 0`` changes the outputs only through the norm's
          ``eps`` (``1e-5``): negligibly while the variance of the channel mean is much larger than ``eps``, and
          substantially for a low-contrast image, whose variance is close to it (see :class:`XFeat`).
        - ``H`` and ``W`` must be multiples of 8; other sizes raise an error or give maps of mismatched sizes.
          :class:`XFeat` resizes its input to multiples of 32.
        - ``keypoints`` holds 65 logits per 8 x 8 cell: channel ``(y % 8) * 8 + x % 8`` of cell ``(y // 8, x // 8)``
          is pixel ``(x, y)``, and channel 64 is the "no keypoint" dustbin.

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
        return x.permute(0, 1, 4, 2, 3).flatten(start_dim=1, end_dim=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the XFeat backbone.

        Args:
            x: image tensor of shape ``(B, C, H, W)``.

        Returns:
            Tuple of:
            - feats: dense descriptors ``(B, 64, H/8, W/8)``.
            - keypoints: keypoint logits ``(B, 65, H/8, W/8)``.
            - heatmap: reliability map ``(B, 1, H/8, W/8)``.
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

    Convention:
        - ``pos`` holds ``(x, y)`` pixels of an image of height ``H`` and width ``W`` whose feature map has height
          ``h`` and width ``w``. A position ``(px, py)`` is read at ``px * w / (W - 1) - 0.5`` and
          ``py * h / (H - 1) - 0.5`` in the map's pixel frame: ``normgrid``'s ``2 p / (W - 1) - 1`` sampled with
          ``align_corners=False``, the convention the pretrained XFeat weights use. So a map of the image's own size is
          not read at ``p`` itself.
        - Known defects: on CPU, float16 and bfloat16 maps go to torch's ``grid_sample`` without an upcast: the
          half-precision kernel can return wrong values, NaN or overflow, and where torch has none the call raises
          (`#5810 <https://github.com/kornia/kornia/issues/5810>`_).

    Args:
        mode: interpolation mode for :func:`torch.nn.functional.grid_sample`.
            Default: ``'bicubic'``.
        align_corners: stored and not used: ``grid_sample`` is always called with ``align_corners=False``. Default:
            ``False``.

    Shape:
        - Input ``x``: ``(B, C, H, W)``.
        - Input ``pos``: ``(B, N, 2)`` integer or float (x, y) coordinates.
        - Output: ``(B, N, C)``.
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
            x: feature map ``(B, C, H, W)``.
            pos: sampling positions ``(B, N, 2)`` in pixel coordinates.
            H: height used for coordinate normalisation.
            W: width used for coordinate normalisation.

        Returns:
            Sampled features ``(B, N, C)``.
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

    Convention:
        - XFeat defines no ``forward``: call :meth:`detectAndCompute` or :meth:`detectAndComputeDense`.
        - ``XFeat()`` builds random weights in training mode, where BatchNorm normalises with the batch, updates its
          running statistics on every call, and matching a single small image pair raises; call ``.eval()`` for
          inference. :meth:`from_pretrained` returns the trained model in evaluation mode. The inference methods
          run under ``torch.inference_mode``.
        - The image is ``(B, C, H, W)`` with any channel count. Each extraction resizes the image it gets
          bilinearly to :math:`(H', W')`: ``H`` and ``W`` rounded down to multiples of 32, with a minimum of 32.
          :class:`XFeatModel` then averages the channels and instance-normalises the result, so an affine change of the
          intensities with a positive gain changes its maps only through the norm's ``eps``, which weighs more the
          lower the image's contrast. With the pretrained weights and ``top_k=2048``, the :math:`[0, 1]` and
          :math:`[0, 255]` versions of Oxford graf and wall ``img1`` differ in at most 1 keypoint, in 1 to 2 % of them
          with the intensities multiplied by 0.1, and in 35 to 40 % multiplied by 0.01. Every output is float32.
        - :meth:`detectAndCompute` keypoints are integer pixels ``i`` of the resized image, strict maxima of the
          keypoint heatmap in their 5 x 5 window (see the note), reported at ``i * (W / W', H / H')``. That is the
          pixel frame of :func:`~kornia.feature.laf_from_center_scale_ori`
          (see :ref:`Local features <local-features-conventions>`) when ``H`` and ``W`` are multiples of 32;
          otherwise the centre of the resized pixel is at ``(i + 0.5) * (W / W', H / H') - 0.5``.
        - The heatmap value must be strictly above ``detection_threshold``. ``scores`` is that value times the
          reliability map, and the ``top_k`` best keypoints come sorted by it. Each descriptor is 64-D, sampled
          bicubically from the dense map at the keypoint by :class:`InterpolateSparse2d`, and L2-normalised.
        - With ``multiscale=False``, :meth:`detectAndComputeDense` reports cell ``(j, i)`` of the 1/8-resolution map
          at ``8 * (j, i)``, times the same ratio: the top-left pixel of the cell, not its centre. It keeps the
          ``top_k`` cells of highest reliability, or every cell when there are fewer or ``top_k`` is below 1. With
          the default ``multiscale=True``, it rescales the image bilinearly by 0.6 and by 1.3, runs that extraction
          on each with ``int(0.2 * top_k)`` and the remaining ``top_k - int(0.2 * top_k)`` cells, and divides each
          keypoint by its factor; ``scales`` is ``1 / 0.6`` or ``1 / 1.3`` for each keypoint. A share of 0 yields no
          cell from its scale, and a scale with fewer cells than its share returns all of them, so there are never
          more than ``top_k`` keypoints, and exactly ``top_k`` when both scales have enough cells. A ``top_k`` below
          1 keeps every cell of both scales. Its descriptors are not normalised. The reference implementation gives
          the 1.3 scale ``int(0.8 * top_k)`` and reads a share of 0 as "every cell"; kornia does not, because that
          returns more keypoints than ``top_k`` asks for.
        - Known defects: converting the model to float16 or bfloat16 makes every extraction raise, because
          ``_preprocess_tensor`` passes float32 input to the half-precision weights
          (`#5957 <https://github.com/kornia/kornia/issues/5957>`_).

    Args:
        top_k: maximum number of keypoints to keep per image. Default: ``4096``.
        detection_threshold: the keypoint heatmap value a keypoint must exceed, see the Convention block. Default:
            ``0.05``.

    Note:
        Keypoints are the strict local maxima of the keypoint heatmap: :func:`~kornia.geometry.subpix.nms2d` keeps a
        position only if its score is greater than every other score in its full 5x5 window. Tied maxima, and the
        positions within 2 pixels of the border whose window runs off the image, are never keypoints. This departs
        on purpose from the reference implementation (verlab/accelerated_features). Its NMS keeps every position
        equal to the maximum of its window: all positions of a plateau survive, and whether a near-tie survives
        depends on rounding-level differences of the scores. Its window is also clipped at the border, so it judges
        a border position against part of its window. With the pretrained weights on Oxford graf and boat img1,
        every candidate of the reference that kornia leaves out lies in that border strip.

        XFeat* (:meth:`match_xfeat_star`) moves each matched keypoint of the first image by the soft-argmax of the
        fine matcher's 8 x 8 heatmap times the keypoint's scale, measured from bin 3 of each axis. This also departs
        on purpose from the reference implementation, which measures it from bin 4, the window centre: the pretrained
        fine matcher puts a zero displacement at bin 3. Matching Oxford bikes img1 with itself (pretrained weights,
        ``multiscale=False``), the reference's refinement moves the points by (-1.04, -1.25) px on average and leaves
        1 % of them within 1 px of their position, against (-0.04, -0.25) px and 94 % from bin 3. On 18 Oxford affine
        pairs the median homography corner error drops from 2.58 to 2.30 px; IMC 2021 PhotoTourism relative pose
        mAA@10 is the same within noise (0.441 and 0.443 over 813 pairs). Away from a zero displacement the soft-argmax
        still moves a point only about half of the way, as in the reference: on Oxford graf 1-2 to 1-5 its response to
        the true sub-cell offset has a slope of about 0.5, so bin 3 removes the constant part of the error, not that
        shrinkage.

    Example:
        >>> model = XFeat.from_pretrained()
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
            detection_threshold: the keypoint heatmap value a keypoint must exceed. Default: ``0.05``.

        Returns:
            XFeat model with pretrained weights loaded, set to eval mode.
        """
        model = cls(top_k=top_k, detection_threshold=detection_threshold)
        state_dict = load_state_dict_from_url(cls.weights_url, file_name="xfeat.pt")
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
        pad_val = max((len(p) for p in pos_batched), default=0)  # an empty batch pads to no keypoints
        pos_out = torch.full((B, pad_val, 2), -1, dtype=torch.long, device=x.device)
        for b in range(B):
            pos_out[b, : len(pos_batched[b])] = pos_batched[b]
        return pos_out

    @staticmethod
    def _create_xy(h: int, w: int, device: torch.device) -> torch.Tensor:
        """Create a grid of (x, y) pixel coordinates of shape ``(H*W, 2)``."""
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        return torch.cat([x[..., None], y[..., None]], -1).reshape(-1, 2)

    @staticmethod
    def _subpix_softmax2d(heatmaps: torch.Tensor, temp: float = 3.0) -> torch.Tensor:
        """Compute soft-argmax offsets from an (N, H, W) heatmap, measured from bin ``(W // 2 - 1, H // 2 - 1)``.

        That bin, not the window centre the reference implementation uses, is where the pretrained fine matcher puts
        a zero displacement; see the XFeat* note of :class:`XFeat`.
        """
        N, H, W = heatmaps.shape
        heatmaps = torch.softmax(temp * heatmaps.view(N, H * W), -1).view(N, H, W)
        x, y = torch.meshgrid(
            torch.arange(W, device=heatmaps.device), torch.arange(H, device=heatmaps.device), indexing="xy"
        )
        x = x - (W // 2 - 1)
        y = y - (H // 2 - 1)
        coords = torch.cat([(x[None] * heatmaps)[..., None], (y[None] * heatmaps)[..., None]], -1)
        return coords.view(N, H * W, 2).sum(1)

    def _match_mnn(
        self, feats1: torch.Tensor, feats2: torch.Tensor, min_cossim: float = 0.82
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mutual nearest-neighbour matching on L2-normalised descriptor pairs."""
        # If either descriptor set is empty there are no possible matches; return early
        # with empty index tensors.
        if feats1.shape[0] == 0 or feats2.shape[0] == 0:
            empty = torch.empty(0, dtype=torch.long, device=feats1.device)
            return empty, empty.clone()

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
        cossim_max = cossim.amax(dim=-1) if min_cossim > 0 else None
        # Free the forward matrix before the reverse one, so only one (B, N1, N2) matrix is alive at a time.
        del cossim
        # The second product reduces along rows, as _match_mnn does. An argmax down the columns of cossim is a strided
        # reduction, several times slower on CPU than this matrix product, and on CUDA it allocates about six times
        # the matrix in scratch memory.
        match21 = torch.argmax(torch.bmm(feats2, feats1.permute(0, 2, 1)), dim=-1)
        idx0 = torch.arange(match12.shape[1], device=match12.device)
        batched_matches = []
        for b in range(B):
            keep = match21[b][match12[b]] == idx0
            if cossim_max is not None:
                keep = keep & (cossim_max[b] > min_cossim)
            batched_matches.append((idx0[keep], match12[b][keep]))
        return batched_matches

    def _extract_dense(self, x: torch.Tensor, top_k: Optional[int] = 8000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract coarse descriptors from an FPN feature map, top-k by reliability.

        ``top_k=None`` keeps every cell; ``top_k=0`` keeps none.
        """
        x, rh1, rw1 = self._preprocess_tensor(x)
        M1, _K1, H1 = self.net(x)
        B, C, _H1, _W1 = M1.shape
        if top_k is None:
            top_k = _H1 * _W1
        xy1 = (self._create_xy(_H1, _W1, M1.device) * 8).expand(B, -1, -1)
        M1 = M1.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        H1 = H1.permute(0, 2, 3, 1).flatten(start_dim=1)
        if top_k < 1:
            return torch.zeros(B, 0, 2, device=M1.device), M1.new_zeros(B, 0, C)
        _, top_k_idx = torch.topk(H1, k=min(H1.shape[1], top_k), dim=-1)
        feats = torch.gather(M1, 1, top_k_idx[..., None].expand(-1, -1, C))
        mkpts = torch.gather(xy1, 1, top_k_idx[..., None].expand(-1, -1, 2))
        mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1, -1)
        return mkpts, feats

    def _extract_dualscale(
        self, x: torch.Tensor, top_k: Optional[int], s1: float = 0.6, s2: float = 1.3
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract dense features at two scales and merge.

        ``top_k`` is split into ``int(0.2 * top_k)`` for the first scale and the rest for the second;
        ``None`` keeps every cell of both. A zero share yields no cell from its scale.
        """
        if top_k is None:
            k1 = k2 = None
        else:
            k1 = int(top_k * 0.20)
            k2 = int(top_k) - k1
        x2 = F.interpolate(x, scale_factor=s2, align_corners=False, mode="bilinear")
        mkpts_2, feats_2 = self._extract_dense(x2, k2)
        if k1 == 0:
            # no cell is asked of the first scale: skip its resize and backbone pass; the empty outputs take the
            # dtype and device of the computed scale, so the concatenation does not change the output dtype
            mkpts_1 = mkpts_2.new_empty((mkpts_2.shape[0], 0, 2))
            feats_1 = feats_2.new_empty((feats_2.shape[0], 0, feats_2.shape[-1]))
        else:
            x1 = F.interpolate(x, scale_factor=s1, align_corners=False, mode="bilinear")
            mkpts_1, feats_1 = self._extract_dense(x1, k1)
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
            x: image tensor of shape ``(B, C, H, W)``.
            top_k: number of keypoints to keep (overrides ``self.top_k``).
            detection_threshold: the heatmap value to exceed (overrides ``self.detection_threshold``).

        Returns:
            List of length ``B``. Each element is a dict with:

            - ``'keypoints'``: ``(N, 2)`` keypoints in (x, y) pixel coordinates.
            - ``'scores'``: ``(N,)`` keypoint heatmap value times reliability, in descending order.
            - ``'descriptors'``: ``(N, 64)`` L2-normalised descriptors.
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
            x: image tensor of shape ``(B, C, H, W)``.
            top_k: number of features to keep (overrides ``self.top_k``). A value below 1 keeps every cell. With
                ``multiscale=True`` the budget is split between the scales, ``int(0.2 * top_k)`` for the 0.6x
                extraction and the rest for the 1.3x one, so the result never exceeds a positive ``top_k``; a scale
                with a budget of 0 contributes nothing. This departs from the reference implementation, which
                gives the 1.3x scale ``int(0.8 * top_k)`` and reads a budget that rounds down to 0 as "every cell".
            multiscale: use dual-scale (0.6x and 1.3x) extraction. Default: ``True``.

        Returns:
            Dict with:

            - ``'keypoints'``: ``(B, K, 2)`` coarse keypoints.
            - ``'descriptors'``: ``(B, K, 64)`` coarse descriptors.
            - ``'scales'``: ``(B, K)`` extraction scale per keypoint.
        """
        if top_k is None:
            top_k = self.top_k
        # a top_k below 1 means every cell; resolve it once, before the budget is split between the scales
        budget = top_k if top_k >= 1 else None
        if multiscale:
            mkpts, sc, feats = self._extract_dualscale(x, budget)
        else:
            mkpts, feats = self._extract_dense(x, budget)
            sc = torch.ones(mkpts.shape[:2], device=mkpts.device)
        return {"keypoints": mkpts, "descriptors": feats, "scales": sc}

    @torch.inference_mode()
    def match_xfeat(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        top_k: Optional[int] = None,
        min_cossim: float = -1,
    ) -> List[torch.Tensor] | Tuple[torch.Tensor, torch.Tensor]:
        """Detect, describe and mutually match keypoints from two batches of images.

        Pair ``b`` is image ``b`` of ``img1`` with image ``b`` of ``img2``. Each batch goes through
        :meth:`detectAndCompute` in one call, and then the descriptors of each pair are matched on their own by
        mutual nearest neighbour. A backend may round a batched convolution differently from a single-image one (MPS,
        and CUDA with cuDNN's default TF32 convolutions), so a pair matched in a batch can gain or lose a few
        near-tied matches compared with matching it alone.

        Args:
            img1: first images ``(B, C, H, W)``.
            img2: second images ``(B, C, H, W)``, with the same ``B`` as ``img1``.
            top_k: number of top keypoints to use.
            min_cossim: minimum cosine similarity threshold. Use ``-1`` to disable.

        Returns:
            If ``B != 1``: list of ``(N, 4)`` tensors with ``(x1, y1, x2, y2)`` matches, one per pair; an empty
            batch gives an empty list.
            If ``B == 1``: tuple ``(mkpts0, mkpts1)`` of matched keypoints, each ``(N, 2)``.

        See Also:
            :meth:`match_xfeat_star`: the dense, refined variant, with the same return convention.
        """
        KORNIA_CHECK(img1.shape[0] == img2.shape[0], "img1 and img2 must hold the same number of images")
        if top_k is None:
            top_k = self.top_k
        out1 = self.detectAndCompute(img1, top_k=top_k)
        out2 = self.detectAndCompute(img2, top_k=top_k)
        pairs = []
        for f1, f2 in zip(out1, out2):
            idx0, idx1 = self._match_mnn(f1["descriptors"], f2["descriptors"], min_cossim=min_cossim)
            pairs.append((f1["keypoints"][idx0], f2["keypoints"][idx1]))
        if len(pairs) != 1:
            return [torch.cat(pair, dim=-1) for pair in pairs]
        return pairs[0]

    @torch.inference_mode()
    def match_xfeat_star(
        self,
        im_set1: torch.Tensor,
        im_set2: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> List[torch.Tensor] | Tuple[torch.Tensor, torch.Tensor]:
        """Extract coarse features, match pairs and refine matches (XFeat*).

        Args:
            im_set1: batch of images ``(B, C, H, W)``.
            im_set2: batch of images ``(B, C, H, W)``.
            top_k: number of top features to use.

        Returns:
            If ``B != 1``: list of ``(N, 4)`` tensors with ``(x1, y1, x2, y2)`` matches, one per pair; an empty
            batch gives an empty list.
            If ``B == 1``: tuple of ``(N, 2)`` matched keypoint tensors.
        """
        if top_k is None:
            top_k = self.top_k
        out1 = self.detectAndComputeDense(im_set1, top_k=top_k)
        out2 = self.detectAndComputeDense(im_set2, top_k=top_k)
        idxs_list = self._batch_match(out1["descriptors"], out2["descriptors"])
        B = im_set1.shape[0]
        matches = [self._refine_matches(out1, out2, matches=idxs_list, batch_idx=b) for b in range(B)]
        if B != 1:
            return matches
        return matches[0][:, :2], matches[0][:, 2:]
