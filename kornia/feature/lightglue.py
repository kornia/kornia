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
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core._compat import torch_version_lt
from kornia.core.check import KORNIA_CHECK
from kornia.core.download import hf_url, load_state_dict_from_url
from kornia.core.utils import is_exporting
from kornia.feature.laf import laf_to_three_points, scale_laf


def math_clamp(x, min_, max_):
    """Clamp a value to lie within [min, max]."""
    return min(max(x, min_), max_)


def normalize_keypoints(kpts: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    """Normalize torch.Tensor of keypoints."""
    if isinstance(size, torch.Size):
        size = torch.tensor(size)[None]
    # Under an active autocast, cast fp16/bf16 inputs to fp32 so the normalisation
    # arithmetic runs at full precision regardless of the accelerator.
    # Deriving the autocast device from the input tensor itself is correct for any
    # backend (CUDA, NPU, XPU, MPS) and on CPU-only builds where
    # ``torch.accelerator.current_accelerator()`` would be ``None``.
    device_type = kpts.device.type
    if (
        kpts.is_floating_point()
        and kpts.dtype in (torch.float16, torch.bfloat16)
        and torch.is_autocast_enabled(device_type)
    ):
        kpts = kpts.to(torch.float32)
    shift = size.float().to(kpts) / 2
    scale = size.max(1).values.float().to(kpts) / 2
    return (kpts - shift[:, None]) / scale[:, None, None]


def pad_to_length(x: torch.Tensor, length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad torch.Tensor to desired length."""
    if length <= x.shape[-2]:
        return x, torch.ones_like(x[..., :1], dtype=torch.bool)
    pad_tensor = torch.ones(*x.shape[:-2], length - x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype)
    y = torch.cat([x, pad_tensor], dim=-2)
    mask = torch.zeros(*y.shape[:-1], 1, dtype=torch.bool, device=x.device)
    mask[..., : x.shape[-2], :] = True
    return y, mask


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Apply half rotation."""
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding."""
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


class LearnableFourierPositionalEncoding(nn.Module):
    """Implement learnable Fourier features for positional encoding.

    This module encodes spatial coordinates into high-dimensional embeddings
    using a learnable projection and periodic functions.

    Args:
        M: The number of frequencies to use.
        dim: The dimension of the output embedding.
        F_dim: The dimension of the Fourier features. Default: None.
        gamma: The scaling factor for the initialization. Default: 1.0.
    """

    def __init__(self, M: int, dim: int, F_dim: Optional[int] = None, gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode position vector."""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


class TokenConfidence(nn.Module):
    """Predict the confidence score for each token in the sequence.

    Args:
        dim: The dimension of the input feature tokens.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get confidence tokens."""
        dtype = self.token[0].weight.dtype
        orig_dtype = desc0.dtype
        return (
            self.token(desc0.detach().to(dtype)).squeeze(-1).to(orig_dtype),
            self.token(desc1.detach().to(dtype)).squeeze(-1).to(orig_dtype),
        )


class Attention(nn.Module):
    """Implement the optimized multi-head attention mechanism for LightGlue.

    Args:
        allow_flash: Whether to enable FlashAttention for increased performance.
    """

    def __init__(self, allow_flash: bool) -> None:
        super().__init__()
        self.allow_flash = allow_flash
        self._windows_old_torch = sys.platform == "win32" and torch_version_lt(2, 6, 0)
        torch.backends.cuda.enable_flash_sdp(allow_flash)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply multi-head attention to query, key, and value tensors.

        Args:
            q: Query tensor with shape ``(B, H, N_q, D_h)``, where ``B`` is batch size,
                ``H`` is the number of attention heads, ``N_q`` is the number of query tokens,
                and ``D_h`` is the per-head descriptor dimension.
            k: Key tensor with shape ``(B, H, N_k, D_h)``, where ``N_k`` is the number
                of key tokens.
            v: Value tensor with shape ``(B, H, N_k, D_h)``.
            mask: Optional boolean attention mask. ``True`` entries mark token pairs that
                are allowed to attend to each other.

        Returns:
            Attention output with shape ``(B, H, N_q, D_h)``.
        """
        if self.allow_flash and q.device.type == "cuda":
            # use torch 2.0 scaled_dot_product_attention with flash
            args = [x.half().contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args, attn_mask=mask).to(q.dtype)  # type: ignore
            return v if mask is None else v.nan_to_num()
        args = [x.contiguous() for x in [q, k, v]]
        if self._windows_old_torch and q.device.type == "cpu" and q.dtype == torch.float64 and q.ndim == 4:
            # The Windows torch 2.5 CPU flash kernel can return NaNs or corrupt finite values for valid float64
            # inputs. It only accepts 4D inputs: a singleton batch dimension selects math attention without changing
            # process-global backend flags or using a context manager that torch 2.5 cannot compile.
            q_, k_, v_ = (x.unsqueeze(0) for x in args)
            v = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=mask).squeeze(0)
        else:
            v = F.scaled_dot_product_attention(*args, attn_mask=mask)  # type: ignore
        return v if mask is None else v.nan_to_num()


class SelfBlock(nn.Module):
    """Implement a self-attention block to process features within a single image.

    Args:
        embed_dim: The dimension of the feature embeddings.
        num_heads: The number of attention heads.
        flash: Whether to use FlashAttention. Default: False.
        bias: Whether to include bias in the linear layers. Default: True.
    """

    def __init__(self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        KORNIA_CHECK(self.embed_dim % num_heads == 0, "Embed dimension should be dividable by num_heads")
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Update descriptors from one image using self-attention.

        Args:
            x: Descriptor tensor with shape ``(B, N, D)``, where ``B`` is batch size,
                ``N`` is the number of local features, and ``D`` is descriptor dimension.
            encoding: Positional encoding tensor aligned with ``x`` and used for rotary
                position embedding inside attention.
            mask: Optional boolean mask limiting which descriptors may attend to each other.

        Returns:
            Updated descriptor tensor with the same shape as ``x``.
        """
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v, mask=mask)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))


class CrossBlock(nn.Module):
    """Implement a cross-attention block to exchange information between two images.

    Args:
        embed_dim: The dimension of the feature embeddings.
        num_heads: The number of attention heads.
        flash: Whether to use FlashAttention. Default: False.
        bias: Whether to include bias in the linear layers. Default: True.
    """

    def __init__(self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        if flash:
            self.flash = Attention(True)
        else:
            self.flash = None

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the same function to descriptor tensors from both images.

        Args:
            func: Callable applied independently to ``x0`` and ``x1``.
            x0: Descriptor tensor for the first image.
            x1: Descriptor tensor for the second image.

        Returns:
            A tuple containing ``func(x0)`` and ``func(x1)``.
        """
        return func(x0), func(x1)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exchange information between two images using cross-attention.

        Args:
            x0: Descriptor tensor from the first image with shape ``(B, M, D)``,
                where ``M`` is the number of local features in image 0.
            x1: Descriptor tensor from the second image with shape ``(B, N, D)``,
                where ``N`` is the number of local features in image 1.
            mask: Optional boolean mask with shape ``(B, H, M, N)`` or a broadcastable
                equivalent, marking valid descriptor pairs.

        Returns:
            A tuple ``(x0_out, x1_out)`` with updated descriptors for both images.
        """
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = (t.unflatten(-1, (self.heads, -1)).transpose(1, 2) for t in (qk0, qk1, v0, v1))
        if self.flash is not None and qk0.device.type == "cuda":
            m0 = self.flash(qk0, qk1, v1, mask)
            m1 = self.flash(qk1, qk0, v0, mask.transpose(-1, -2) if mask is not None else None)
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)
            if mask is not None:
                sim = sim.masked_fill(~mask, -float("inf"))
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
            m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
            if mask is not None:
                m0, m1 = m0.nan_to_num(), m1.nan_to_num()
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class TransformerLayer(nn.Module):
    """Implement a complete Transformer layer for feature matching.

    This layer typically alternates between self-attention and cross-attention
    to update local feature descriptors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.self_attn = SelfBlock(*args, **kwargs)
        self.cross_attn = CrossBlock(*args, **kwargs)

    def forward(
        self,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
        encoding0: torch.Tensor,
        encoding1: torch.Tensor,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one LightGlue transformer layer over two descriptor sets.

        Args:
            desc0: Descriptor tensor from image 0 with shape ``(B, M, D)``.
            desc1: Descriptor tensor from image 1 with shape ``(B, N, D)``.
            encoding0: Positional encoding for ``desc0``.
            encoding1: Positional encoding for ``desc1``.
            mask0: Optional validity mask for descriptors in image 0.
            mask1: Optional validity mask for descriptors in image 1.

        Returns:
            A tuple containing updated descriptor tensors for image 0 and image 1.
        """
        if mask0 is not None and mask1 is not None:
            return self.masked_forward(desc0, desc1, encoding0, encoding1, mask0, mask1)
        desc0 = self.self_attn(desc0, encoding0)
        desc1 = self.self_attn(desc1, encoding1)
        return self.cross_attn(desc0, desc1)

    # This part is compiled and allows padding inputs
    def masked_forward(
        self,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
        encoding0: torch.Tensor,
        encoding1: torch.Tensor,
        mask0: torch.Tensor,
        mask1: torch.Tensor,
    ) -> torch.Tensor:
        """Run the transformer layer with explicit descriptor validity masks.

        Args:
            desc0: Descriptor tensor from image 0 with shape ``(B, M, D)``.
            desc1: Descriptor tensor from image 1 with shape ``(B, N, D)``.
            encoding0: Positional encoding for image 0 descriptors.
            encoding1: Positional encoding for image 1 descriptors.
            mask0: Boolean mask for valid image 0 descriptors.
            mask1: Boolean mask for valid image 1 descriptors.

        Returns:
            A tuple with masked self-attention and cross-attention updates for both images.
        """
        mask = mask0 & mask1.transpose(-1, -2)
        mask0 = mask0 & mask0.transpose(-1, -2)
        mask1 = mask1 & mask1.transpose(-1, -2)
        desc0 = self.self_attn(desc0, encoding0, mask0)
        desc1 = self.self_attn(desc1, encoding1, mask1)
        return self.cross_attn(desc0, desc1, mask)


def sigmoid_log_double_softmax(sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
    """Create the log assignment matrix from logits and similarity."""
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores


class MatchAssignment(nn.Module):
    """Assign matches between two sets of local features using a dual-softmax approach.

    Args:
        dim: The dimension of the feature descriptors.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build assignment matrix from descriptors."""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def get_matchability(self, desc: torch.Tensor) -> torch.Tensor:
        """Estimate how likely each descriptor is to obtain a valid match.

        Args:
            desc: Descriptor tensor with shape ``(B, N, D)``, where ``N`` is the
                number of local features.

        Returns:
            Matchability probabilities with shape ``(B, N)``.
        """
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)


def filter_matches(scores: torch.Tensor, th: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Obtain matches from a log assignment matrix [Bx M+1 x N+1]."""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1


_KEYPOINT_RANGE_MSG = (
    "Keypoints must lie inside the image: normalized by the image size they have to be in [-1, 1]. The keypoints are "
    "(x, y) pixel coordinates. The image size is (width, height) for the ``image_size`` of LightGlue and "
    "OnnxLightGlue, and (height, width) for the ``hw1`` / ``hw2`` of LightGlueMatcher; the wrong order, or keypoints "
    "detected on a differently sized image, put them out of range."
)


def _check_keypoints_normalized(kpts: torch.Tensor) -> None:
    """Check that keypoints lie in [-1, 1]; skipped under export, where reading the data is not possible."""
    if not is_exporting():
        KORNIA_CHECK(torch.all(kpts >= -1).item() and torch.all(kpts <= 1).item(), _KEYPOINT_RANGE_MSG)


def _select_pair(image: Dict[str, Any], k: int, b: int) -> Dict[str, Any]:
    """Take pair ``k`` of the ``b`` pairs from the inputs of one image, keeping entries shared by the batch."""
    return {
        key: value[k : k + 1] if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == b else value
        for key, value in image.items()
    }


class LightGlue(nn.Module):
    """Implement the LightGlue matcher for sparse local features.

    LightGlue is a deep network that matches local features across image pairs
    using a series of transformer layers and an adaptive pruning mechanism.

    Convention:
        - Complete learned weights return the model in evaluation mode, even if derived buffers are absent from
          the checkpoint. Random or partially loaded learned weights leave it in training mode.
        - The ``keypoints`` of each image are ``(x, y)`` pixels and ``image_size`` is ``(W, H)``; an ``image``
          tensor gives the size from its last two dimensions instead. ``forward`` normalises the keypoints to
          ``(k - size / 2) / (max(size) / 2)``, as the reference implementation does, and raises for a keypoint
          outside :math:`[-1, 1]` after that. The keypoints can be wider than the model, like the float32 keypoints
          of kornia's detectors with a float16 or bfloat16 model; the positional encoding runs in the model's dtype.
        - The ``"sift"``, ``"doghardnet"`` and ``"dog_affnet_hardnet"`` configurations, which set ``add_scale_ori``,
          also read ``scales`` and ``oris`` (radians) of each image; the last two multiply ``scales`` by their
          ``scale_coef`` of 1/6 first, so they take the LAF scale. ``"keynet_affnet_hardnet"``, which sets ``add_laf``,
          reads ``lafs``. :class:`~kornia.feature.LightGlueMatcher` builds them from LAFs.
        - The matching scores, ``matching_scores0``, ``matching_scores1`` and ``scores``, are in the dtype of the
          descriptors with ``mp=False``. With ``mp=True``, autocast sets it, for every configuration and also when
          point pruning removes every point. On CPU, autocast runs in float16 for a float16 model and in bfloat16,
          CPU autocast's default, for any other: a float16 model gives float16 scores for float16 and float32
          descriptors, a float32 or bfloat16 model gives bfloat16 scores for float32 and bfloat16 descriptors, and a
          float64 model gives float64 scores for float64 descriptors. Float16 descriptors with any other model, and
          bfloat16 descriptors with a float16 model, are not supported there
          (`#5929 <https://github.com/kornia/kornia/issues/5929>`_). On MPS, float32, float16 and bfloat16
          descriptors give float32 scores; on CUDA, float32 descriptors give float32 scores. An image without
          descriptors runs no layer, and its zero scores keep the descriptor dtype, also with ``mp=True``.
        - Each pair of a batch gives the matches and scores it gives on its own when point pruning can run, that is
          with ``width_confidence > 0``, no padding to the static lengths of :meth:`compile`, and an image with more
          keypoints than :meth:`pruning_min_kpts` of the device. Pruning keeps a different number of points in each
          pair, so then the pairs run one at a time: ``matches0``, ``matches1``, ``matching_scores0``,
          ``matching_scores1``, ``prune0`` and ``prune1`` stack the rows of the pairs, ``matches`` and ``scores`` list
          them, ``stop`` is the largest number of layers that a pair ran, and there is no ``log_assignment``.
          Otherwise a batch runs as one, and early stopping (``depth_confidence``) ends it at a layer only when every
          pair meets its own criterion. A pair can therefore run more layers in a batch than on its own, and its
          matches can differ.
        - Known defects: on the static-length path of :meth:`compile`, early stopping also counts the unconfident
          padded positions, against the number of real keypoints, so a padded pair can stop later than without
          padding (`#5930 <https://github.com/kornia/kornia/issues/5930>`_).

    Args:
        features: The type of local features to match (e.g., 'superpoint', 'disk'), whose configuration and
            pretrained weights are loaded. ``None`` builds the model from ``default_conf`` and the keyword arguments
            and downloads no weights.
        **conf: keyword arguments that override entries of ``default_conf`` and of the configuration of
            ``features``, such as ``filter_threshold`` or ``depth_confidence``. The configuration of ``features`` sets
            the defaults, and an explicit keyword argument always wins, except ``weights``, which must name the
            checkpoint of ``features`` when ``features`` is given.
    """

    default_conf: ClassVar[Dict[str, Any]] = {
        "name": "lightglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "add_laf": False,  # for KeyNetAffNetHardNet
        "scale_coef": 1.0,  # to compensate for the SIFT scale bigger than KeyNet
        "n_layers": 9,
        "num_heads": 4,
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": 0.95,  # early stopping, disable with -1
        "width_confidence": 0.99,  # point pruning, disable with -1
        "filter_threshold": 0.1,  # match threshold
        "weights": None,
    }

    # Point pruning involves an overhead (gather).
    # Therefore, we only activate it if there are enough keypoints.
    pruning_keypoint_thresholds: ClassVar[Dict[str, Any]] = {
        "cpu": -1,
        "mps": -1,
        "cuda": 1024,
        "flash": 1536,
    }
    required_data_keys: ClassVar[List[str]] = ["image0", "image1"]

    version: ClassVar[str] = "v0.1_arxiv"
    url: ClassVar[str] = "https://github.com/cvg/LightGlue/releases/download/{}/{}_lightglue.pth"

    features: ClassVar[Dict[str, Any]] = {
        "superpoint": {
            "weights": "superpoint_lightglue",
            "input_dim": 256,
        },
        "dedodeb": {
            "weights": "dedodeb_lightglue",
            "input_dim": 256,
        },
        "dedodeg": {
            "weights": "dedodeg_lightglue",
            "input_dim": 256,
        },
        "disk": {
            "weights": "disk_lightglue",
            "input_dim": 128,
        },
        "aliked": {
            "weights": "aliked_lightglue",
            "input_dim": 128,
        },
        "raco-aliked": {
            "weights": "raco_aliked_lightglue",
            "input_dim": 128,
        },
        "xfeat": {
            "weights": "xfeat_lighterglue",
            "input_dim": 64,
            "descriptor_dim": 96,
            "n_layers": 6,
            "num_heads": 1,
            "depth_confidence": -1,
            "width_confidence": 0.95,
            "filter_threshold": 0.1,
        },
        "sift": {
            "weights": "sift_lightglue",
            "input_dim": 128,
            "add_scale_ori": True,
        },
        "dog_affnet_hardnet": {
            "weights": "doghardnet",
            "input_dim": 128,
            "add_scale_ori": True,
            "scale_coef": 1.0 / 6.0,
        },
        "doghardnet": {
            "weights": "doghardnet",
            "input_dim": 128,
            "add_scale_ori": True,
            "scale_coef": 1.0 / 6.0,
        },
        "keynet_affnet_hardnet": {
            "weights": "keynet_affnet_hardnet_lightglue",
            "input_dim": 128,
            "width_confidence": -1,
            "depth_confidence": -1,
            "add_laf": True,
        },
    }

    def __init__(self, features: Optional[str] = "superpoint", **conf_) -> None:
        super().__init__()
        if features is not None:
            KORNIA_CHECK(features in list(self.features.keys()), "Features keys are wrong")
            # the download URL follows `features` and the cached file name `weights`: they must name one checkpoint
            KORNIA_CHECK(
                conf_.get("weights", self.features[features]["weights"]) == self.features[features]["weights"],
                "`weights` names the checkpoint of `features`; pass `features=None` to load other weights",
            )
        preset = self.features[features] if features is not None else {}
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **preset, **conf_})
        KORNIA_CHECK(not (self.conf.add_scale_ori and self.conf.add_laf))  # we use either scale ori, or LAF

        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(conf.input_dim, conf.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(
            2 + 2 * conf.add_scale_ori + 4 * conf.add_laf,
            head_dim,
            head_dim,
        )

        h, n, d = conf.num_heads, conf.n_layers, conf.descriptor_dim
        self.transformers = nn.ModuleList([TransformerLayer(d, h, conf.flash) for _ in range(n)])
        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = nn.ModuleList([TokenConfidence(d) for _ in range(n - 1)])
        self.register_buffer(
            "confidence_thresholds",
            torch.Tensor([self.confidence_threshold(i) for i in range(self.conf.n_layers)]),
        )
        state_dict = None
        if features is not None:
            fname = f"{conf.weights}_{self.version}.pth".replace(".", "-")
            if features == "dog_affnet_hardnet":
                features = "doghardnet"  # new dog model is better for affnet as well
            if features in ["keynet_affnet_hardnet"]:
                fname = "keynet_affnet_hardnet_lightglue.pth"
                url = [
                    hf_url("lightglue", "keynet_affnet_hardnet_lightglue.pth"),
                    "https://cmp.felk.cvut.cz/~mishkdmy/models/keynet_affnet_hardnet_lightlue.pth",
                ]
            elif features in ["dedodeb"]:
                fname = "dedodeb_lightglue.pth"
                url = [
                    hf_url("lightglue", "dedodeb_lightglue.pth"),
                    "https://cmp.felk.cvut.cz/~mishkdmy/models/dedodeb_lightglue.pth",
                ]
            elif features in ["dedodeg"]:
                fname = "dedodeg_lightglue.pth"
                url = [
                    hf_url("lightglue", "dedodeg_lightglue.pth"),
                    "https://cmp.felk.cvut.cz/~mishkdmy/models/dedodeg_lightglue.pth",
                ]
            elif features == "xfeat":
                fname = "xfeat-lighterglue.pt"
                url = [
                    hf_url("lightglue", "xfeat-lighterglue.pt"),
                    "https://github.com/verlab/accelerated_features/raw/main/weights/xfeat-lighterglue.pt",
                ]
            else:
                url = [
                    hf_url("lightglue", f"{features.replace('-', '_')}_lightglue.pth"),
                    self.url.format(self.version, features.replace("-", "_")),
                ]
            state_dict = load_state_dict_from_url(url, file_name=fname)
        elif conf.weights is not None:
            path = Path(__file__).parent
            path = path / f"weights/{self.conf.weights}.pth"
            state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
        if state_dict:
            # xfeat-lighterglue weights are nested under a 'matcher.' prefix
            prefix = "matcher."
            state_dict = {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
            # rename old state dict entries
            for i in range(self.conf.n_layers):
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            incompatible = self.load_state_dict(state_dict, strict=False)
            # Missing derived buffers are harmless, but partial learned weights leave a model to train.
            parameters = dict(self.named_parameters())
            if not any(name in parameters for name in incompatible.missing_keys):
                self.eval()
        # static lengths LightGlue is compiled for (only used with torch.compile)
        self.static_lengths = None

    def compile(
        self, mode: str = "reduce-overhead", static_lengths: Sequence[int] = (256, 512, 768, 1024, 1280, 1536)
    ) -> None:
        """Compile masked transformer layers with ``torch.compile``.

        Args:
            mode: Compilation mode forwarded to ``torch.compile``.
            static_lengths: Descriptor sequence lengths prepared for compiled execution.
                These lengths represent possible numbers of local features.
        """
        if self.conf.width_confidence != -1:
            warnings.warn(
                "Point pruning is partially disabled for compiled forward.",
                stacklevel=2,
            )

        if (
            hasattr(torch, "_inductor")
            and hasattr(torch._inductor, "cudagraph_mark_step_begin")
            and torch.cuda.is_available()
        ):
            torch._inductor.cudagraph_mark_step_begin()
        for i in range(self.conf.n_layers):
            self.transformers[i].masked_forward = torch.compile(
                self.transformers[i].masked_forward, mode=mode, fullgraph=True
            )

        self.static_lengths = static_lengths

    def forward(self, data: dict) -> dict:
        """Match keypoints and descriptors between two images.

        Input (dict):
            image0: dict
                keypoints: [B x M x 2]
                descriptors: [B x M x D]
                image: [B x C x H x W] or image_size: [B x 2]
                scales, oris: [B x M], with add_scale_ori
                lafs: [B x M x 2 x 3], with add_laf
            image1: dict
                keypoints: [B x N x 2]
                descriptors: [B x N x D]
                image: [B x C x H x W] or image_size: [B x 2]
                scales, oris: [B x N], with add_scale_ori
                lafs: [B x N x 2 x 3], with add_laf
        Output (dict):
            log_assignment: [B x M+1 x N+1], not for a batch that prunes points (see the Convention block)
            matches0: [B x M]
            matching_scores0: [B x M]
            matches1: [B x N]
            matching_scores1: [B x N]
            matches: List[[Si x 2]], scores: List[[Si]]
            stop, prune0, prune1: diagnostics of early stopping and point pruning
        """
        for key in self.required_data_keys:
            KORNIA_CHECK(key in data, f"Missing key {key} in data")
        device_type = data["image0"]["keypoints"].device.type
        # CPU autocast defaults to bfloat16, whose outputs cannot meet the float16 tensors of a float16 model, so such a
        # model autocasts to float16 there.
        dtype = torch.float16 if device_type == "cpu" and self.posenc.Wr.weight.dtype == torch.float16 else None
        with torch.autocast(enabled=self.conf.mp, device_type=device_type, dtype=dtype):
            return self._forward(data)

    def _forward(self, data: dict) -> dict:
        data0, data1 = data["image0"], data["image1"]
        kpts0, kpts1 = data0["keypoints"], data1["keypoints"]
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape
        device = kpts0.device
        c = max(m, n)
        do_compile = self.static_lengths and c <= max(self.static_lengths)
        do_point_pruning = self.conf.width_confidence > 0 and not do_compile
        if b > 1 and min(m, n) > 0 and do_point_pruning and c > self.pruning_min_kpts(device):
            # Point pruning keeps a different number of points in each pair, so the pairs run one at a time.
            return self._forward_pairs(data0, data1, b)
        size0, size1 = data0.get("image_size"), data1.get("image_size")
        size0 = size0 if size0 is not None else data0["image"].shape[-2:][::-1]
        size1 = size1 if size1 is not None else data1["image"].shape[-2:][::-1]

        kpts0 = normalize_keypoints(kpts0, size0).clone()
        kpts1 = normalize_keypoints(kpts1, size1).clone()
        _check_keypoints_normalized(kpts0)
        _check_keypoints_normalized(kpts1)
        if self.conf.add_scale_ori:
            kpts0 = torch.cat([kpts0] + [data0[k].unsqueeze(-1) for k in ("scales", "oris")], -1)
            if self.conf.scale_coef != 1.0:
                kpts0[..., -2] = kpts0[..., -2] * self.conf.scale_coef
            kpts1 = torch.cat([kpts1] + [data1[k].unsqueeze(-1) for k in ("scales", "oris")], -1)
            if self.conf.scale_coef != 1.0:
                kpts1[..., -2] = kpts1[..., -2] * self.conf.scale_coef
        elif self.conf.add_laf:
            laf0 = scale_laf(data0["lafs"], self.conf.scale_coef)
            laf1 = scale_laf(data1["lafs"], self.conf.scale_coef)
            laf0 = laf_to_three_points(laf0)
            laf1 = laf_to_three_points(laf1)
            kpts0 = torch.cat(
                [
                    kpts0,
                    normalize_keypoints(laf0[..., 0], size0).clone().to(kpts0.dtype),
                    normalize_keypoints(laf0[..., 1], size0).clone().to(kpts0.dtype),
                ],
                -1,
            )
            kpts1 = torch.cat(
                [
                    kpts1,
                    normalize_keypoints(laf1[..., 0], size1).clone().to(kpts1.dtype),
                    normalize_keypoints(laf1[..., 1], size1).clone().to(kpts1.dtype),
                ],
                -1,
            )

        desc0 = data0["descriptors"].detach().contiguous()
        desc1 = data1["descriptors"].detach().contiguous()

        # Guard against empty-descriptors after pruning
        if desc0.shape[1] == 0 or desc1.shape[1] == 0:
            # No layer runs here, so the outputs keep the input dtype, also under mp=True.
            dtype = desc0.dtype
            empty_m0 = torch.full((b, m), -1, device=device, dtype=torch.long)
            empty_m1 = torch.full((b, n), -1, device=device, dtype=torch.long)
            empty_s0 = torch.zeros((b, m), device=device, dtype=dtype)
            empty_s1 = torch.zeros((b, n), device=device, dtype=dtype)

            return {
                "log_assignment": torch.empty((b, 1, 1), device=device, dtype=dtype),
                "matches0": empty_m0,
                "matches1": empty_m1,
                "matching_scores0": empty_s0,
                "matching_scores1": empty_s1,
                "stop": 0,
                "matches": [torch.empty((0, 2), device=device, dtype=torch.long) for _ in range(b)],
                "scores": [torch.empty((0,), device=device, dtype=dtype) for _ in range(b)],
                "prune0": torch.ones_like(empty_s0) * self.conf.n_layers,
                "prune1": torch.ones_like(empty_s1) * self.conf.n_layers,
            }

        KORNIA_CHECK(desc0.shape[-1] == self.conf.input_dim, "Descriptor dimension does not match input dim in config")
        KORNIA_CHECK(desc1.shape[-1] == self.conf.input_dim, "Descriptor dimension does not match input dim in config")

        device_type = desc0.device.type
        if device_type != "cpu" and torch.is_autocast_enabled(device_type):
            desc0 = desc0.half()
            desc1 = desc1.half()

        mask0, mask1 = None, None
        if do_compile:
            kn = min([k for k in self.static_lengths if k >= c])
            desc0, mask0 = pad_to_length(desc0, kn)
            desc1, mask1 = pad_to_length(desc1, kn)
            kpts0, _ = pad_to_length(kpts0, kn)
            kpts1, _ = pad_to_length(kpts1, kn)
        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        # cache positional embeddings, in the encoder's dtype: kornia's detectors return float32 keypoints for a
        # half-precision model, which would otherwise meet half-precision weights here
        pe_dtype = self.posenc.Wr.weight.dtype
        encoding0 = self.posenc(kpts0.to(pe_dtype))
        encoding1 = self.posenc(kpts1.to(pe_dtype))

        # GNN + final_proj + assignment
        do_early_stop = self.conf.depth_confidence > 0
        pruning_th = self.pruning_min_kpts(device)
        if do_point_pruning:
            ind0 = torch.arange(0, m, device=device)[None].expand(b, -1)
            ind1 = torch.arange(0, n, device=device)[None].expand(b, -1)
            # We store the index of the layer at which pruning is detected.
            prune0 = torch.ones_like(ind0)
            prune1 = torch.ones_like(ind1)
        token0, token1 = None, None
        for i in range(self.conf.n_layers):
            desc0, desc1 = self.transformers[i](desc0, desc1, encoding0, encoding1, mask0=mask0, mask1=mask1)
            if i == self.conf.n_layers - 1:
                continue  # no early stopping or adaptive width at last layer

            if do_early_stop:
                token0, token1 = self.token_confidence[i](desc0, desc1)
                if self.check_if_stop(token0, token1, i, m + n):
                    break
            # Pruning runs here for a single pair only: a batch that would prune runs its pairs one at a time.
            if do_point_pruning and desc0.shape[-2] > pruning_th:
                scores0 = self.log_assignment[i].get_matchability(desc0)
                prunemask0 = self.get_pruning_mask(token0, scores0, i)
                keep0 = torch.where(prunemask0)[1]
                ind0 = ind0.index_select(1, keep0)
                desc0 = desc0.index_select(1, keep0)
                encoding0 = encoding0.index_select(-2, keep0)
                prune0[:, ind0] += 1
            if do_point_pruning and desc1.shape[-2] > pruning_th:
                scores1 = self.log_assignment[i].get_matchability(desc1)
                prunemask1 = self.get_pruning_mask(token1, scores1, i)
                keep1 = torch.where(prunemask1)[1]
                ind1 = ind1.index_select(1, keep1)
                desc1 = desc1.index_select(1, keep1)
                encoding1 = encoding1.index_select(-2, keep1)
                prune1[:, ind1] += 1

        desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]

        # If adaptive width pruning removed all keypoints from either side,
        # return empty matches instead of crashing in filter_matches (see #3564).
        if desc0.shape[-2] == 0 or desc1.shape[-2] == 0:
            # A pass that keeps points takes its scores from the exp of the log assignment, and autocast picks the
            # dtype of both ops: the log assignment is bfloat16 on CPU even where the descriptors stay float32
            # (input_dim == descriptor_dim), and the exp is float32 on CUDA and MPS. So take the dtype from the same
            # ops. Autocast picks it by op and input dtype, not by size, so empty slices give it at no cost.
            log_assignment, _ = self.log_assignment[i](desc0[:, :0], desc1[:, :0])
            dtype = log_assignment.new_zeros(()).exp().dtype
            matches = [torch.zeros((0, 2), dtype=torch.long, device=device) for _ in range(b)]
            mscores = [torch.zeros(0, device=device, dtype=dtype) for _ in range(b)]
            m0_ = torch.full((b, m), -1, device=device, dtype=torch.long)
            m1_ = torch.full((b, n), -1, device=device, dtype=torch.long)
            mscores0_ = torch.zeros((b, m), device=device, dtype=dtype)
            mscores1_ = torch.zeros((b, n), device=device, dtype=dtype)
            return {
                "matches": matches,
                "scores": mscores,
                "matching_scores0": mscores0_,
                "matching_scores1": mscores1_,
                "matches0": m0_,
                "matches1": m1_,
                "stop": i + 1,
                "prune0": prune0 if do_point_pruning else torch.ones_like(torch.arange(m, device=device)[None]),
                "prune1": prune1 if do_point_pruning else torch.ones_like(torch.arange(n, device=device)[None]),
            }

        scores, _ = self.log_assignment[i](desc0, desc1)
        m0, m1, mscores0, mscores1 = filter_matches(scores, self.conf.filter_threshold)

        matches, mscores = [], []
        for k in range(b):
            valid = m0[k] > -1
            m_indices_0 = torch.where(valid)[0]
            m_indices_1 = m0[k][valid]
            if do_point_pruning:
                m_indices_0 = ind0[k, m_indices_0]
                m_indices_1 = ind1[k, m_indices_1]
            matches.append(torch.stack([m_indices_0, m_indices_1], -1))
            mscores.append(mscores0[k][valid])

        # TODO: Remove when hloc switches to the compact format.
        if do_point_pruning:
            m0_ = torch.full((b, m), -1, device=m0.device, dtype=m0.dtype)
            m1_ = torch.full((b, n), -1, device=m1.device, dtype=m1.dtype)
            m0_.scatter_(1, ind0, torch.where(m0 == -1, -1, ind1.gather(1, m0.clamp(min=0))))
            m1_.scatter_(1, ind1, torch.where(m1 == -1, -1, ind0.gather(1, m1.clamp(min=0))))
            mscores0_ = torch.zeros((b, m), device=mscores0.device, dtype=mscores0.dtype)
            mscores1_ = torch.zeros((b, n), device=mscores1.device, dtype=mscores1.dtype)
            mscores0_.scatter_(1, ind0, mscores0)
            mscores1_.scatter_(1, ind1, mscores1)
            m0, m1, mscores0, mscores1 = m0_, m1_, mscores0_, mscores1_
        else:
            prune0 = torch.ones_like(mscores0) * self.conf.n_layers
            prune1 = torch.ones_like(mscores1) * self.conf.n_layers

        return {
            "log_assignment": scores,
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "stop": i + 1,
            "matches": matches,
            "scores": mscores,
            "prune0": prune0,
            "prune1": prune1,
        }

    def _forward_pairs(self, data0: dict, data1: dict, b: int) -> dict:
        """Match the ``b`` pairs of a batch one at a time and stack their outputs.

        The pairs have no common ``log_assignment``, since each covers the points its pair kept, and ``stop`` is the
        largest number of layers a pair ran.
        """
        outs = [
            self._forward({"image0": _select_pair(data0, k, b), "image1": _select_pair(data1, k, b)}) for k in range(b)
        ]
        stacked = ("matches0", "matches1", "matching_scores0", "matching_scores1", "prune0", "prune1")
        out: Dict[str, Any] = {key: torch.cat([o[key] for o in outs]) for key in stacked}
        out["matches"] = [o["matches"][0] for o in outs]
        out["scores"] = [o["scores"][0] for o in outs]
        out["stop"] = max(o["stop"] for o in outs)
        return out

    def confidence_threshold(self, layer_index: int) -> float:
        """Scaled confidence threshold."""
        threshold = 0.8 + 0.1 * math.exp(-4.0 * layer_index / self.conf.n_layers)
        return min(max(threshold, 0), 1)

    def get_pruning_mask(self, confidences: torch.Tensor, scores: torch.Tensor, layer_index: int) -> torch.Tensor:
        """Mask points which should be removed."""
        keep = scores > (1 - self.conf.width_confidence)
        if confidences is not None:  # Low-confidence points are never pruned.
            keep |= confidences <= self.confidence_thresholds[layer_index]
        return keep

    def check_if_stop(
        self,
        confidences0: torch.Tensor,
        confidences1: torch.Tensor,
        layer_index: int,
        num_points: int,
    ) -> torch.Tensor:
        """Evaluate the early-stopping condition of every pair in a batch.

        Args:
            confidences0: Token confidences of image 0, shape ``(B, M)``.
            confidences1: Token confidences of image 1, shape ``(B, N)``.
            layer_index: Index of the layer that computed the confidences.
            num_points: Number of keypoints of one pair that the unconfident points are counted against.

        Returns:
            Whether every pair has a ratio of confident points above ``depth_confidence``: each pair counts its own
            points below the confidence threshold of the layer.
        """
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = 1.0 - (confidences < threshold).float().sum(-1) / num_points
        return (ratio_confident > self.conf.depth_confidence).all()

    def pruning_min_kpts(self, device: torch.device) -> int:
        """Return the minimum keypoint count needed before pruning is enabled.

        Args:
            device: Torch device on which descriptors are processed.

        Returns:
            With point pruning enabled, it runs on an image while the image has more keypoints than this number,
            so ``-1`` means always.
        """
        if self.conf.flash and device.type == "cuda":
            return self.pruning_keypoint_thresholds["flash"]
        return self.pruning_keypoint_thresholds[device.type]
