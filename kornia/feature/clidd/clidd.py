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

# Adapted from the CLIDD implementation (https://github.com/HITCSC/CLIDD).
#
# MIT License
#
# Copyright (c) 2024 HITCSC-Robotics
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""CLIDD: Cross-Layer Independent Deformable Description for local features.

Reference:
    Haodi Yao, Fenghua He, Ning Hao, Yao Su. "CLIDD: Cross-Layer Independent Deformable Description for Efficient and
    Discriminative Local Feature Representation", arXiv:2601.09230, 2026.
"""

from __future__ import annotations

import importlib.util
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core._compat import torch_version_lt
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE
from kornia.core.download import load_state_dict_from_url
from kornia.core.ops import _copied_literal
from kornia.core.utils import _l2_normalize, _samples_half_in_float32
from kornia.geometry.subpix import nms2d

_CLIDD_CONFIGS: Dict[str, Dict[str, int]] = {
    "A48": {"c1": 4, "c2": 4, "c3": 4, "r1": 1, "r2": 1, "r3": 1, "cdesc": 48, "cdetect": 4, "M": 4},
    "N64": {"c1": 8, "c2": 8, "c3": 8, "r1": 1, "r2": 1, "r3": 1, "cdesc": 64, "cdetect": 8, "M": 8},
    "T64": {"c1": 8, "c2": 16, "c3": 24, "r1": 1, "r2": 1, "r3": 1, "cdesc": 64, "cdetect": 8, "M": 8},
    "S64": {"c1": 8, "c2": 24, "c3": 32, "r1": 1, "r2": 1, "r3": 1, "cdesc": 64, "cdetect": 8, "M": 16},
    "M64": {"c1": 16, "c2": 32, "c3": 48, "r1": 1, "r2": 1, "r3": 1, "cdesc": 64, "cdetect": 8, "M": 16},
    "L64": {"c1": 16, "c2": 48, "c3": 96, "r1": 1, "r2": 1, "r3": 1, "cdesc": 64, "cdetect": 8, "M": 16},
    "G128": {"c1": 16, "c2": 64, "c3": 256, "r1": 1, "r2": 1, "r3": 1, "cdesc": 128, "cdetect": 8, "M": 32},
    "E128": {"c1": 16, "c2": 64, "c3": 256, "r1": 1, "r2": 2, "r3": 2, "cdesc": 128, "cdetect": 8, "M": 32},
    "U128": {"c1": 32, "c2": 128, "c3": 256, "r1": 1, "r2": 2, "r3": 2, "cdesc": 128, "cdetect": 8, "M": 32},
}

_WEIGHTS_URL = "https://github.com/HITCSC/CLIDD/releases/download/v1.0/{}.pth"


def _weights_source(variant: str) -> Tuple[str, str]:
    """Return the cache file name and the URL of the authors' v1.0 checkpoint of ``variant``.

    The release names its files after the variant alone (``L64.pth``), so the cache entry in the shared torch hub
    directory carries a ``clidd_`` prefix.
    """
    return f"clidd_{variant}.pth", _WEIGHTS_URL.format(variant)


# Keypoints closer than this to the border of the resized image are dropped, as in the reference implementation.
_BORDER = 4


# Whether the triton package is installed, found without importing it. A module constant, because Dynamo before
# torch 2.15 refuses to trace importlib.util.find_spec in a compiled call.
_TRITON_INSTALLED = importlib.util.find_spec("triton") is not None


# The variants whose descriptors the automatic mode computes with the Triton kernels: the 128-D ones, whose
# (B, C_1 + C_2 + C_3, N, 32) samples make the PyTorch formulation slower than the kernels.
_TRITON_BY_DEFAULT = ("G128", "E128", "U128")


def _uses_triton(use_triton: Optional[bool], variant: str, x: torch.Tensor) -> bool:
    """Resolve the ``use_triton`` option of :class:`CLIDDModel` of ``variant`` for feature maps ``x``."""
    if use_triton is not None and not use_triton:
        return False
    usable = x.is_cuda and _TRITON_INSTALLED and x.dtype in (torch.float32, torch.float16, torch.bfloat16)
    if use_triton:
        KORNIA_CHECK(
            usable,
            "use_triton=True needs the triton package and float32, float16 or bfloat16 feature maps on a CUDA device.",
        )
        return True
    # Inductor before torch 2.7 miscompiles the kernels (see kornia/feature/clidd/_triton.py), which then stay out of
    # compiled graphs: keep the automatic mode compilable there.
    return usable and variant in _TRITON_BY_DEFAULT and not torch_version_lt(2, 7, 0)


class _BasicLayer(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class _ResNetLayer(nn.Module):
    """Two 3 x 3 Conv2d -> BatchNorm2d with a residual connection, a 1 x 1 convolution when the width changes."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_out),
        )
        self.skip: nn.Module = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.layer(x) + self.skip(x))


class CLIDDModel(nn.Module):
    r"""Compute the CLIDD feature maps and score map, and describe keypoints by deformable cross-layer sampling.

    ``net = CLIDDModel("L64").eval()`` builds the random-weight network of a CLIDD variant (``CLIDD.from_pretrained``
    loads trained ones into ``CLIDD.net``). ``maps, score = net(images)`` takes RGB images ``(B, 3, H, W)`` with
    ``H`` and ``W`` multiples of 32 and returns three feature maps at 1/2, 1/8 and 1/32 of the image size and the
    keypoint logits ``(B, 1, H, W)``. ``descriptors = net.describe(maps, grid)`` then takes keypoints ``(B, N, 2)`` as
    ``(x, y)`` in the normalised :math:`[-1, 1]` frame of the image, ``align_corners=False``, and returns
    L2-normalised descriptors ``(B, N, D)``. It detects no keypoints itself: use :class:`CLIDD` for that. It implements
    "CLIDD: Cross-Layer Independent Deformable Description for Efficient and Discriminative Local Feature
    Representation" (arXiv:2601.09230).

    Each descriptor is built from the three maps independently, without a dense fused descriptor map: bilinear samples
    of each map at the keypoint give ``3 M`` offsets in pixels (``M`` per map, clamped to ``max_offset``), each map is
    sampled again at its ``M`` displaced points, and one ``(1, M)`` convolution projects the samples of all three maps
    to the descriptor.

    Convention:
        - A new ``CLIDDModel`` is in training mode; call ``.eval()`` for inference.
        - ``H`` and ``W`` must be multiples of 32, or the maps do not line up. :class:`CLIDD` resizes its input.
        - The sampling reads zeros outside each map, as :func:`torch.nn.functional.grid_sample` with
          ``padding_mode="zeros"`` and ``align_corners=False``. Offsets are pixels of the image the network saw.
        - ``use_triton`` picks the implementation of the sampling and projection: the fused Triton kernels of the
          authors, or the PyTorch formulation, :func:`torch.nn.functional.grid_sample` followed by
          :func:`torch.nn.functional.conv2d`. ``True`` runs the kernels and raises without the ``triton`` package,
          off CUDA, or for float64 maps, which the kernels would compute in float32; ``False`` always runs PyTorch.
          ``None`` (the default) runs the kernels for float32, float16 and bfloat16 maps on CUDA with Triton installed
          and a 128-D variant (``"G128"``, ``"E128"``, ``"U128"``) from torch 2.7, and PyTorch otherwise.
        - Both implementations are differentiable and agree to float32 rounding with TF32 off; the kernels use TF32
          when ``torch.backends.cudnn.allow_tf32`` is set, as the convolution of the PyTorch formulation does. Both
          compile with ``torch.compile(fullgraph=True)`` from torch 2.7. Inductor in torch 2.5 and 2.6 miscompiles the
          kernels, so there they stay out of the graph: a compiled ``use_triton=True`` breaks the graph around them,
          or raises with ``fullgraph=True``.
        - The PyTorch formulation holds the ``(B, C_1 + C_2 + C_3, N, M)`` samples in memory: 0.43 GB for ``"U128"``
          and 4,096 keypoints in float32, against 28 MB for the kernels. On an RTX 4090 (torch 2.15, 640 x 480,
          4,096 keypoints, eager), :class:`CLIDD` with the kernels was 1.10-1.36x faster for the 128-D variants and
          1.14-1.39x slower for the others, whose six kernel launches cost more than their small PyTorch operations.
        - Float16 and bfloat16 maps are sampled in float32 by the PyTorch formulation, as by the kernels. The sampling
          positions are float32, or float64 for float64 keypoints, in both.

    Args:
        variant: one of ``"A48"``, ``"N64"``, ``"T64"``, ``"S64"``, ``"M64"``, ``"L64"``, ``"G128"``, ``"E128"``,
            ``"U128"``; the number is the descriptor size.
        max_offset: the largest offset, in pixels, of a sampling point from its keypoint along each axis. Default:
            ``128``, as in the pretrained weights.
        use_triton: whether to run the Triton kernels; see the Convention block. Default: ``None`` (automatic).

    Shape:
        - Input: ``(B, 3, H, W)``.
        - Output: the maps ``(B, C_1, H / 2, W / 2)``, ``(B, C_2, H / 8, W / 8)`` and ``(B, C_3, H / 32, W / 32)``, and
          the score logits ``(B, 1, H, W)``.

    See Also:
        - :class:`CLIDD`: runs this network, detects keypoints in its score map and describes them.
        - :class:`XFeatModel`: another lightweight detector-descriptor network, with a dense descriptor map.

    Example:
        >>> net = CLIDDModel("A48").eval()
        >>> maps, score = net(torch.rand(1, 3, 64, 64))
        >>> net.describe(maps, torch.zeros(1, 5, 2)).shape
        torch.Size([1, 5, 48])
    """

    def __init__(self, variant: str = "L64", max_offset: float = 128.0, use_triton: Optional[bool] = None) -> None:
        super().__init__()
        KORNIA_CHECK(
            variant in _CLIDD_CONFIGS, lambda: f"Unknown CLIDD variant {variant!r}: use one of {list(_CLIDD_CONFIGS)}."
        )
        cfg = _CLIDD_CONFIGS[variant]
        c1, c2, c3 = cfg["c1"], cfg["c2"], cfg["c3"]
        cdetect, num_points = cfg["cdetect"], cfg["M"]
        self.variant = variant
        self.max_offset = max_offset
        self.use_triton = use_triton
        self.num_points = num_points

        self.block1 = nn.Sequential(
            _BasicLayer(3, c1, 4, 2, 1),
            _BasicLayer(c1, c1),
            *[_ResNetLayer(c1, c1) for _ in range(cfg["r1"])],
        )
        self.block2 = nn.Sequential(
            nn.AvgPool2d(4, 4),
            *[_ResNetLayer(c1 if i == 0 else c2, c2) for i in range(cfg["r2"])],
        )
        self.block3 = nn.Sequential(
            nn.AvgPool2d(4, 4),
            *[_ResNetLayer(c2 if i == 0 else c3, c3) for i in range(cfg["r3"])],
        )

        self.conv1 = nn.Conv2d(c1, cdetect, 1)
        self.conv2 = nn.Conv2d(c2, cdetect, 1)
        self.conv3 = nn.Conv2d(c3, cdetect, 1)
        self.score_head = nn.Sequential(
            nn.Conv2d(cdetect, cdetect, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cdetect, 4, 3, 1, 1),
            nn.PixelShuffle(2),
        )

        # The offsets (x, y) of M points in each of the 3 maps, and the projection of the samples to the descriptor.
        self.dcn = nn.Conv2d(c1 + c2 + c3, num_points * 2 * 3, 1)
        self.post_conv = nn.Conv2d(c1 + c2 + c3, cfg["cdesc"], (1, num_points))

    def forward(self, x: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Run the backbone and the score head.

        Args:
            x: RGB images ``(B, 3, H, W)`` in :math:`[0, 1]`, ``H`` and ``W`` multiples of 32.

        Returns:
            The feature maps at 1/2, 1/8 and 1/32 resolution and the keypoint logits ``(B, 1, H, W)``.
        """
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        feat = self.conv2(x2) + F.interpolate(self.conv3(x3), scale_factor=4, mode="bilinear", align_corners=False)
        feat = self.conv1(x1) + F.interpolate(feat, scale_factor=4, mode="bilinear", align_corners=False)
        return (x1, x2, x3), self.score_head(feat)

    def _project(
        self,
        maps: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        grids: List[torch.Tensor],
        conv: nn.Conv2d,
        triton: bool,
    ) -> torch.Tensor:
        """Sample each map at its grid ``(B, N, M, 2)`` and project the samples with ``conv``; return ``(B, N, C)``."""
        if triton:
            from ._triton import deformable_sample_project

            weights = conv.weight.split([m.shape[-1] for m in maps], 1)
            out = deformable_sample_project(maps[0], grids[0], weights[0], None, False)
            out = out + deformable_sample_project(maps[1], grids[1], weights[1], None, False)
            return out + deformable_sample_project(maps[2], grids[2], weights[2], conv.bias, False)
        samples = []
        for m, g in zip(maps, grids):
            wide = torch.float32 if _samples_half_in_float32(m) else m.dtype
            s = F.grid_sample(m.to(wide), g.to(wide), mode="bilinear", padding_mode="zeros", align_corners=False)
            samples.append(s.to(m.dtype))
        return conv(torch.cat(samples, 1)).squeeze(-1).transpose(1, 2)

    def describe(self, maps: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], keypoints: torch.Tensor) -> torch.Tensor:
        """Describe keypoints from the feature maps of :meth:`forward`.

        Args:
            maps: the three feature maps of :meth:`forward`.
            keypoints: ``(B, N, 2)`` keypoints as ``(x, y)`` in :math:`[-1, 1]`: pixel ``(i, j)`` of an ``H x W`` image
                is ``((2 i + 1) / W - 1, (2 j + 1) / H - 1)``.

        Returns:
            L2-normalised descriptors ``(B, N, D)``.
        """
        KORNIA_CHECK_SHAPE(keypoints, ["B", "N", "2"])
        if keypoints.shape[1] == 0:
            # conv2d rejects the empty (B, C, 0, M) samples. An empty view of a map, so that a loss summed over no
            # descriptors still backpropagates (to zero gradients) like a non-empty one.
            return maps[0][:, :1, :1, :1].reshape(maps[0].shape[0], 1, 1).expand(-1, 0, self.post_conv.out_channels)
        triton = _uses_triton(self.use_triton, self.variant, maps[0])
        if triton:
            maps = (
                maps[0].permute(0, 2, 3, 1).contiguous(),
                maps[1].permute(0, 2, 3, 1).contiguous(),
                maps[2].permute(0, 2, 3, 1).contiguous(),
            )
            h, w = maps[0].shape[1] * 2, maps[0].shape[2] * 2
        else:
            h, w = maps[0].shape[-2] * 2, maps[0].shape[-1] * 2
        # The positions are at least float32: a half-precision grid is off by a pixel or more on a large map.
        pos_dtype = torch.promote_types(keypoints.dtype, torch.float32)
        grid = keypoints.to(pos_dtype).unsqueeze(2)

        offsets = self._project(maps, [grid, grid, grid], self.dcn, triton)
        offsets = offsets.to(pos_dtype).unflatten(-1, (-1, 2)).clamp(-self.max_offset, self.max_offset)
        offsets = offsets * _copied_literal([2.0 / w, 2.0 / h], device=offsets.device, dtype=offsets.dtype)
        off1, off2, off3 = offsets.chunk(3, -2)

        desc = self._project(maps, [grid + off1, grid + off2, grid + off3], self.post_conv, triton)
        return _l2_normalize(desc, dim=-1)


class CLIDD(nn.Module):
    r"""Detect and describe keypoints with CLIDD, a family of very small and fast learned local features.

    ``clidd = CLIDD.from_pretrained("L64")`` loads trained weights in evaluation mode, and ``features = clidd(images)``
    takes RGB images ``(B, 3, H, W)`` in :math:`[0, 1]`, or grayscale ``(B, 1, H, W)``, and returns one dictionary per
    image with up to ``top_k`` keypoints ``(N, 2)`` as ``(x, y)`` input pixels (integer pixels when ``H`` and ``W`` are
    multiples of 32), their scores ``(N,)`` in descending order and L2-normalised descriptors ``(N, D)``. Match the
    descriptors of two images with :meth:`match`, the dual-softmax mutual nearest neighbours of the authors, or with
    :func:`match_mnn`.

    The variants trade accuracy for speed; the number is the descriptor size ``D``. The authors report, on an NVIDIA
    Jetson Orin-NX: ``"A48"`` 4,252 parameters, 881 FPS; ``"N64"`` 0.019 M, 843 FPS; ``"T64"`` 0.043 M, 804 FPS;
    ``"S64"`` 0.10 M, 746 FPS; ``"M64"`` 0.17 M, 625 FPS; ``"L64"`` 0.35 M, 591 FPS; ``"G128"`` 2.3 M, 475 FPS;
    ``"E128"`` 3.5 M, 431 FPS; ``"U128"`` 4.4 M, 281 FPS.

    Reference:
        Haodi Yao, Fenghua He, Ning Hao, Yao Su. "CLIDD: Cross-Layer Independent Deformable Description for Efficient
        and Discriminative Local Feature Representation", arXiv:2601.09230, 2026. https://github.com/HITCSC/CLIDD

    Convention:
        - ``CLIDD()`` builds random weights in training mode; call ``.eval()`` for inference.
          :meth:`from_pretrained` returns the trained model in evaluation mode.
        - The forward does not change the grad mode: run it under ``torch.no_grad()`` or ``torch.inference_mode()``
          for inference. The descriptors are differentiable with respect to the image and the weights; the keypoints
          and the choice of keypoints are not.
        - The image is resized bilinearly with ``align_corners=False`` to ``(H', W')``: ``H`` and ``W`` rounded down
          to multiples of 32, at least 32. Pixel ``i`` of the resized image is reported at its centre in the input,
          :math:`(i + 0.5) W / W' - 0.5` in :math:`x`, and likewise in :math:`y`, in the pixel convention of
          :func:`~kornia.feature.laf_from_center_scale_ori`, as :class:`XFeat` does. That is exactly ``i`` when ``H``
          and ``W`` are multiples of 32. This departs on purpose from the reference implementation, which resizes
          with ``align_corners=True`` and reports :math:`i W / W'`.
        - A keypoint is a strict local maximum of the score map in its ``2 * nms_radius + 1`` window
          (:func:`~kornia.geometry.subpix.nms2d`, as in :class:`XFeat` and :class:`ALIKED`), at least 4 pixels from
          the border of the resized image, with a score strictly above ``detection_threshold``. Tied maxima are never
          keypoints. This departs on purpose from the reference implementation, which keeps every pixel equal to the
          maximum of its window, so all pixels of a plateau. Scores are the raw logits of the score head. The
          ``top_k`` highest are kept, all of them for ``top_k=None``, in descending order.
        - A ``(B, 1, H, W)`` image is repeated to three channels, and other channel counts raise. The image is cast
          to the dtype of the weights after the resize. Keypoints are float32, or float64 for a float64 model; scores
          and descriptors have the dtype of the weights.

    Args:
        variant: the network configuration; see :class:`CLIDDModel`. Default: ``"L64"``.
        top_k: the number of keypoints kept per image, or ``None`` for all. Default: ``4096``.
        nms_radius: the radius of the suppression window; ``0`` disables the suppression. Default: ``2``.
        detection_threshold: the logit a keypoint must exceed. Default: ``-5.0``, as in the reference.
        use_triton: whether to run the fused Triton kernels; see :class:`CLIDDModel`. Default: ``None``, which runs
            them for the 128-D variants on CUDA from torch 2.7 when the ``triton`` package is installed.

    See Also:
        - :class:`CLIDDModel`: the network, with the feature maps and the deformable descriptor sampling.
        - :class:`XFeat`: lightweight learned keypoints with 64-D descriptors and a refined XFeat* matching.
        - :class:`ALIKED`: learned sub-pixel keypoints with deformable descriptors, also matched by LightGlue.
        - :func:`match_mnn`: mutual nearest-neighbour matching of the descriptors, without the dual softmax.

    Example:
        >>> clidd = CLIDD("A48").eval()
        >>> with torch.no_grad():
        ...     out = clidd(torch.rand(1, 3, 64, 64))
        >>> out[0]["descriptors"].shape[-1]
        48
    """

    def __init__(
        self,
        variant: str = "L64",
        top_k: Optional[int] = 4096,
        nms_radius: int = 2,
        detection_threshold: float = -5.0,
        use_triton: Optional[bool] = None,
    ) -> None:
        super().__init__()
        KORNIA_CHECK(top_k is None or top_k > 0, "top_k must be positive or None.")
        self.top_k = top_k
        self.nms_radius = nms_radius
        self.detection_threshold = detection_threshold
        self.net = CLIDDModel(variant, use_triton=use_triton)

    @classmethod
    def from_pretrained(
        cls,
        variant: str = "L64",
        top_k: Optional[int] = 4096,
        nms_radius: int = 2,
        detection_threshold: float = -5.0,
        use_triton: Optional[bool] = None,
    ) -> CLIDD:
        """Build CLIDD with the authors' pretrained weights of ``variant``, downloaded from their v1.0 release.

        Args:
            variant: the network configuration; see :class:`CLIDDModel`. Default: ``"L64"``.
            top_k: the number of keypoints kept per image, or ``None`` for all. Default: ``4096``.
            nms_radius: the radius of the suppression window. Default: ``2``.
            detection_threshold: the logit a keypoint must exceed. Default: ``-5.0``.
            use_triton: whether to run the fused Triton kernels. Default: ``None`` (automatic).

        Returns:
            The model in evaluation mode.
        """
        model = cls(variant, top_k, nms_radius, detection_threshold, use_triton)
        file_name, url = _weights_source(variant)
        state_dict = load_state_dict_from_url(url, file_name=file_name, map_location="cpu")
        model.net.load_state_dict(state_dict)
        return model.eval()

    def forward(self, images: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """Detect and describe keypoints.

        Args:
            images: RGB images ``(B, 3, H, W)`` in :math:`[0, 1]`, or grayscale ``(B, 1, H, W)``.

        Returns:
            One dictionary per image with ``"keypoints"`` ``(N, 2)``, ``"scores"`` ``(N,)`` and ``"descriptors"``
            ``(N, D)``.
        """
        KORNIA_CHECK_SHAPE(images, ["B", "C", "H", "W"])
        KORNIA_CHECK(images.shape[1] in (1, 3), "CLIDD takes RGB (B, 3, H, W) or grayscale (B, 1, H, W) images.")
        B, _, H, W = images.shape
        if B == 0:
            # nms2d pools windows other than 3, 5 and 7 with max_pool2d, which on MPS returns a (0,) tensor for an empty
            # batch before torch 2.16 (fixed by pytorch#201756).
            return []
        h, w = max(32, H // 32 * 32), max(32, W // 32 * 32)
        dtype = self.net.conv1.weight.dtype
        x = images
        if (h, w) != (H, W):
            wide = torch.float32 if _samples_half_in_float32(x) else x.dtype
            x = F.interpolate(x.to(wide), (h, w), mode="bilinear", align_corners=False)
        x = x.to(dtype)
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)

        maps, score = self.net(x)
        score = score[:, 0]
        detect = score > self.detection_threshold
        if self.nms_radius > 0:
            k = 2 * self.nms_radius + 1
            detect = detect & nms2d(score[:, None], (k, k), mask_only=True)[:, 0]
        inner = torch.zeros_like(detect)
        inner[:, _BORDER:-_BORDER, _BORDER:-_BORDER] = True
        detect = detect & inner

        kpt_dtype = torch.promote_types(dtype, torch.float32)
        size = _copied_literal([w, h], device=x.device, dtype=kpt_dtype)
        ratio = _copied_literal([W / w, H / h], device=x.device, dtype=kpt_dtype)
        out: List[Dict[str, torch.Tensor]] = []
        for b in range(B):
            yx = detect[b].nonzero()
            scores = score[b][yx[:, 0], yx[:, 1]]
            k = scores.shape[0] if self.top_k is None else min(self.top_k, scores.shape[0])
            scores, order = scores.topk(k)
            kpts = yx[order].flip(-1).to(kpt_dtype)
            grid = (kpts + 0.5) / size * 2 - 1
            descs = self.net.describe((maps[0][b : b + 1], maps[1][b : b + 1], maps[2][b : b + 1]), grid[None])[0]
            # Pixel i of the resized image has its centre at (i + 0.5) W / W' - 0.5 of the input.
            out.append({"keypoints": (kpts + 0.5) * ratio - 0.5, "scores": scores, "descriptors": descs})
        return out

    @staticmethod
    def match(
        desc0: torch.Tensor, desc1: torch.Tensor, beta: float = 20.0, min_score: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Match two sets of descriptors by the dual softmax of the authors, keeping mutual nearest neighbours.

        The score of a pair is :math:`P_{ij} = \mathrm{softmax}_j(\beta s_{ij}) \, \mathrm{softmax}_i(\beta s_{ij})` for
        the cosine similarities :math:`s_{ij}`. A pair is kept when each is the other's highest score and the score
        exceeds ``min_score``. The authors use ``beta=30`` for sparse indoor scenes without suppression.

        Args:
            desc0: L2-normalised descriptors ``(N0, D)``.
            desc1: L2-normalised descriptors ``(N1, D)``.
            beta: the inverse temperature of the softmax. Default: ``20``.
            min_score: the score a match must exceed. Default: ``0.01``.

        Returns:
            The scores ``(M, 1)`` and the index pairs ``(M, 2)`` into ``desc0`` and ``desc1``.
        """
        KORNIA_CHECK_SHAPE(desc0, ["N0", "D"])
        KORNIA_CHECK_SHAPE(desc1, ["N1", "D"])
        if desc0.shape[0] == 0 or desc1.shape[0] == 0:
            return desc0.new_zeros(0, 1), torch.zeros(0, 2, dtype=torch.long, device=desc0.device)
        # The sum of two log-softmaxes, in at least float32: the product of the softmaxes underflows to 0 in float16
        # and, for low similarities and a large beta, in float32 as well, which would score a singleton pair 0, not 1.
        wide = torch.promote_types(desc0.dtype, torch.float32)
        logits = (desc0.to(wide) @ desc1.to(wide).t()) * beta
        log_prob = logits.log_softmax(dim=-1) + logits.log_softmax(dim=-2)
        nn01 = log_prob.argmax(dim=1)
        nn10 = log_prob.argmax(dim=0)
        idx0 = torch.arange(log_prob.shape[0], device=log_prob.device)
        mutual = nn10[nn01] == idx0
        idx0, idx1 = idx0[mutual], nn01[mutual]
        scores = log_prob[idx0, idx1].exp()
        keep = scores > min_score
        return scores[keep, None].to(desc0.dtype), torch.stack([idx0[keep], idx1[keep]], dim=-1)
