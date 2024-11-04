import math
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core import (
    Module,
    ModuleList,
    Tensor,
    arange,
    concatenate,
    cos,
    einsum,
    ones,
    ones_like,
    sin,
    stack,
    where,
    zeros,
)
from kornia.core.check import KORNIA_CHECK
from kornia.feature.laf import laf_to_three_points, scale_laf
from kornia.utils._compat import custom_fwd

try:
    from flash_attn.modules.mha import FlashCrossAttention
except ModuleNotFoundError:
    FlashCrossAttention = None

if FlashCrossAttention or hasattr(F, "scaled_dot_product_attention"):
    FLASH_AVAILABLE = True
else:
    FLASH_AVAILABLE = False


def math_clamp(x, min_, max_):  # type: ignore
    return max(min(x, min_), min_)


@custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints(kpts: Tensor, size: Tensor) -> Tensor:
    if isinstance(size, torch.Size):
        size = Tensor(size)[None]
    shift = size.float().to(kpts) / 2
    scale = size.max(1).values.float().to(kpts) / 2
    kpts = (kpts - shift[:, None]) / scale[:, None, None]
    return kpts


def pad_to_length(x: Tensor, length: int) -> Tuple[Tensor, Tensor]:
    if length <= x.shape[-2]:
        return x, ones_like(x[..., :1], dtype=torch.bool)
    pad = ones(*x.shape[:-2], length - x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype)
    y = concatenate([x, pad], dim=-2)
    mask = zeros(*y.shape[:-1], 1, dtype=torch.bool, device=x.device)
    mask[..., : x.shape[-2], :] = True
    return y, mask


def rotate_half(x: Tensor) -> Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(freqs: Tensor, t: Tensor) -> Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


class LearnableFourierPositionalEncoding(Module):
    def __init__(self, M: int, dim: int, F_dim: Optional[int] = None, gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: Tensor) -> Tensor:
        """Encode position vector."""
        projected = self.Wr(x)
        cosines, sines = cos(projected), sin(projected)
        emb = stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


class TokenConfidence(Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, desc0: Tensor, desc1: Tensor) -> Tuple[Tensor, Tensor]:
        """Get confidence tokens."""
        dtype = self.token[0].weight.dtype
        orig_dtype = desc0.dtype
        return (
            self.token(desc0.detach().to(dtype)).squeeze(-1).to(orig_dtype),
            self.token(desc1.detach().to(dtype)).squeeze(-1).to(orig_dtype),
        )


class Attention(Module):
    def __init__(self, allow_flash: bool) -> None:
        super().__init__()
        if allow_flash and not FLASH_AVAILABLE:
            warnings.warn(
                "FlashAttention is not available. For optimal speed, "
                "consider installing torch >= 2.0 or flash-attn.",
                stacklevel=2,
            )
        self.enable_flash = allow_flash and FLASH_AVAILABLE
        self.has_sdp = hasattr(F, "scaled_dot_product_attention")
        if allow_flash and FlashCrossAttention:
            self.flash_ = FlashCrossAttention()
        if self.has_sdp:
            torch.backends.cuda.enable_flash_sdp(allow_flash)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if self.enable_flash and q.device.type == "cuda":
            # use torch 2.0 scaled_dot_product_attention with flash
            if self.has_sdp:
                args = [x.half().contiguous() for x in [q, k, v]]
                v = F.scaled_dot_product_attention(*args, attn_mask=mask).to(q.dtype)  # type: ignore
                return v if mask is None else v.nan_to_num()
            else:
                KORNIA_CHECK(mask is None)
                q, k, v = (x.transpose(-2, -3).contiguous() for x in [q, k, v])
                m = self.flash_(q.half(), stack([k, v], 2).half())
                return m.transpose(-2, -3).to(q.dtype).clone()
        elif self.has_sdp:
            args = [x.contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args, attn_mask=mask)  # type: ignore
            return v if mask is None else v.nan_to_num()
        else:
            s = q.shape[-1] ** -0.5
            sim = einsum("...id,...jd->...ij", q, k) * s
            if mask is not None:
                sim.masked_fill(~mask, -float("inf"))
            attn = F.softmax(sim, -1)
            return einsum("...ij,...jd->...id", attn, v)


class SelfBlock(Module):
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
        x: Tensor,
        encoding: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v, mask=mask)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(concatenate([x, message], -1))


class CrossBlock(Module):
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
        if flash and FLASH_AVAILABLE:
            self.flash = Attention(True)
        else:
            self.flash = None  # type: ignore

    def map_(self, func: Callable, x0: Tensor, x1: Tensor) -> Tuple[Tensor, Tensor]:  # type: ignore
        return func(x0), func(x1)

    def forward(self, x0: Tensor, x1: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = (t.unflatten(-1, (self.heads, -1)).transpose(1, 2) for t in (qk0, qk1, v0, v1))
        if self.flash is not None and qk0.device.type == "cuda":
            m0 = self.flash(qk0, qk1, v1, mask)
            m1 = self.flash(qk1, qk0, v0, mask.transpose(-1, -2) if mask is not None else None)
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = einsum("bhid, bhjd -> bhij", qk0, qk1)
            if mask is not None:
                sim = sim.masked_fill(~mask, -float("inf"))
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            m0 = einsum("bhij, bhjd -> bhid", attn01, v1)
            m1 = einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
            if mask is not None:
                m0, m1 = m0.nan_to_num(), m1.nan_to_num()
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(concatenate([x0, m0], -1))
        x1 = x1 + self.ffn(concatenate([x1, m1], -1))
        return x0, x1


class TransformerLayer(Module):
    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__()
        self.self_attn = SelfBlock(*args, **kwargs)
        self.cross_attn = CrossBlock(*args, **kwargs)

    def forward(
        self,
        desc0: Tensor,
        desc1: Tensor,
        encoding0: Tensor,
        encoding1: Tensor,
        mask0: Optional[Tensor] = None,
        mask1: Optional[Tensor] = None,
    ) -> Tensor:
        if mask0 is not None and mask1 is not None:
            return self.masked_forward(desc0, desc1, encoding0, encoding1, mask0, mask1)
        else:
            desc0 = self.self_attn(desc0, encoding0)
            desc1 = self.self_attn(desc1, encoding1)
            return self.cross_attn(desc0, desc1)

    # This part is compiled and allows padding inputs
    def masked_forward(
        self, desc0: Tensor, desc1: Tensor, encoding0: Tensor, encoding1: Tensor, mask0: Tensor, mask1: Tensor
    ) -> Tensor:
        mask = mask0 & mask1.transpose(-1, -2)
        mask0 = mask0 & mask0.transpose(-1, -2)
        mask1 = mask1 & mask1.transpose(-1, -2)
        desc0 = self.self_attn(desc0, encoding0, mask0)
        desc1 = self.self_attn(desc1, encoding1, mask1)
        return self.cross_attn(desc0, desc1, mask)


def sigmoid_log_double_softmax(sim: Tensor, z0: Tensor, z1: Tensor) -> Tensor:
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


class MatchAssignment(Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: Tensor, desc1: Tensor) -> Tuple[Tensor, Tensor]:
        """Build assignment matrix from descriptors."""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def get_matchability(self, desc: Tensor) -> Tensor:
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)


def filter_matches(scores: Tensor, th: float) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Obtain matches from a log assignment matrix [Bx M+1 x N+1]."""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = arange(m0.shape[1], device=m0.device)[None]
    indices1 = arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = where(mutual0, max0_exp, zero)
    mscores1 = where(mutual1, mscores0.gather(1, m1), zero)
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = where(valid0, m0, -1)
    m1 = where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1


class LightGlue(Module):
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
        "sift": {
            "weights": "sift_lightglue",
            "input_dim": 128,
            "add_scale_ori": True,
        },
        "dog_affnet_hardnet": {
            "weights": "doghardnet",
            "input_dim": 128,
            "width_confidence": -1,
            "depth_confidence": -1,
            "add_scale_ori": True,
            "scale_coef": 1.0 / 6.0,
        },
        "doghardnet": {
            "weights": "doghardnet",
            "input_dim": 128,
            "width_confidence": -1,
            "depth_confidence": -1,
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

    def __init__(self, features: str = "superpoint", **conf_) -> None:  # type: ignore
        super().__init__()
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf_})
        if features is not None:
            KORNIA_CHECK(features in list(self.features.keys()), "Features keys are wrong")
            for k, v in self.features[features].items():
                setattr(conf, k, v)
        KORNIA_CHECK(not (self.conf.add_scale_ori and self.conf.add_laf))  # we use either scale ori, or LAF

        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(conf.input_dim, conf.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()  # type: ignore

        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(
            2 + 2 * conf.add_scale_ori + 4 * conf.add_laf,
            head_dim,
            head_dim,
        )

        h, n, d = conf.num_heads, conf.n_layers, conf.descriptor_dim
        self.transformers = ModuleList([TransformerLayer(d, h, conf.flash) for _ in range(n)])
        self.log_assignment = ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = ModuleList([TokenConfidence(d) for _ in range(n - 1)])
        self.register_buffer(
            "confidence_thresholds",
            Tensor([self.confidence_threshold(i) for i in range(self.conf.n_layers)]),
        )
        state_dict = None
        if features is not None:
            fname = f"{conf.weights}_{self.version}.pth".replace(".", "-")
            if features == "dog_affnet_hardnet":
                features = "doghardnet"  # new dog model is better for affnet as well
            if features in ["keynet_affnet_hardnet"]:
                fname = "keynet_affnet_hardnet_lightlue.pth"
                url = "http://cmp.felk.cvut.cz/~mishkdmy/models/keynet_affnet_hardnet_lightlue.pth"
            elif features in ["dedodeb"]:
                fname = "dedodeb_lightglue.pth"
                url = "http://cmp.felk.cvut.cz/~mishkdmy/models/dedodeb_lightglue.pth"
            elif features in ["dedodeg"]:
                fname = "dedodeg_lightglue.pth"
                url = "http://cmp.felk.cvut.cz/~mishkdmy/models/dedodeg_lightglue.pth"
            else:
                url = self.url.format(self.version, features)
            state_dict = torch.hub.load_state_dict_from_url(url, file_name=fname)
        elif conf.weights is not None:
            path = Path(__file__).parent
            path = path / f"weights/{self.conf.weights}.pth"
            state_dict = torch.load(str(path), map_location="cpu")
        if state_dict:
            # rename old state dict entries
            for i in range(self.conf.n_layers):
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            self.load_state_dict(state_dict, strict=False)
        print("Loaded LightGlue model")
        # static lengths LightGlue is compiled for (only used with torch.compile)
        self.static_lengths = None

    def compile(
        self, mode: str = "reduce-overhead", static_lengths: List[int] = [256, 512, 768, 1024, 1280, 1536]
    ) -> None:
        if self.conf.width_confidence != -1:
            warnings.warn(
                "Point pruning is partially disabled for compiled forward.",
                stacklevel=2,
            )

        for i in range(self.conf.n_layers):
            self.transformers[i].masked_forward = torch.compile(
                self.transformers[i].masked_forward, mode=mode, fullgraph=True
            )

        self.static_lengths = static_lengths  # type: ignore

    def forward(self, data: dict) -> dict:  # type: ignore
        """Match keypoints and descriptors between two images.

        Input (dict):
            image0: dict
                keypoints: [B x M x 2]
                descriptors: [B x M x D]
                image: [B x C x H x W] or image_size: [B x 2]
            image1: dict
                keypoints: [B x N x 2]
                descriptors: [B x N x D]
                image: [B x C x H x W] or image_size: [B x 2]
        Output (dict):
            log_assignment: [B x M+1 x N+1]
            matches0: [B x M]
            matching_scores0: [B x M]
            matches1: [B x N]
            matching_scores1: [B x N]
            matches: List[[Si x 2]], scores: List[[Si]]
        """
        with torch.autocast(enabled=self.conf.mp, device_type="cuda"):
            return self._forward(data)

    def _forward(self, data: dict) -> dict:  # type: ignore
        for key in self.required_data_keys:
            KORNIA_CHECK(key in data, f"Missing key {key} in data")
        data0, data1 = data["image0"], data["image1"]
        kpts0, kpts1 = data0["keypoints"], data1["keypoints"]
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape
        device = kpts0.device
        size0, size1 = data0.get("image_size"), data1.get("image_size")
        size0 = size0 if size0 is not None else data0["image"].shape[-2:][::-1]
        size1 = size1 if size1 is not None else data1["image"].shape[-2:][::-1]

        kpts0 = normalize_keypoints(kpts0, size0).clone()
        kpts1 = normalize_keypoints(kpts1, size1).clone()
        KORNIA_CHECK(torch.all(kpts0 >= -1).item() and torch.all(kpts0 <= 1).item(), "")  # type: ignore
        KORNIA_CHECK(torch.all(kpts1 >= -1).item() and torch.all(kpts1 <= 1).item(), "")  # type: ignore
        if self.conf.add_scale_ori:
            kpts0 = concatenate([kpts0] + [data0[k].unsqueeze(-1) for k in ("scales", "oris")], -1)
            if self.conf.scale_coef != 1.0:
                kpts0[..., -2] = kpts0[..., -2] * self.conf.scale_coef
            kpts1 = concatenate([kpts1] + [data1[k].unsqueeze(-1) for k in ("scales", "oris")], -1)
            if self.conf.scale_coef != 1.0:
                kpts1[..., -2] = kpts1[..., -2] * self.conf.scale_coef
        elif self.conf.add_laf:
            laf0 = scale_laf(data0["lafs"], self.conf.scale_coef)
            laf1 = scale_laf(data1["lafs"], self.conf.scale_coef)
            laf0 = laf_to_three_points(laf0)
            laf1 = laf_to_three_points(laf1)
            kpts0 = concatenate(
                [
                    kpts0,
                    normalize_keypoints(laf0[..., 0], size0).clone().to(kpts0.dtype),
                    normalize_keypoints(laf0[..., 1], size0).clone().to(kpts0.dtype),
                ],
                -1,
            )
            kpts1 = concatenate(
                [
                    kpts1,
                    normalize_keypoints(laf1[..., 0], size1).clone().to(kpts1.dtype),
                    normalize_keypoints(laf1[..., 1], size1).clone().to(kpts1.dtype),
                ],
                -1,
            )

        desc0 = data0["descriptors"].detach().contiguous()
        desc1 = data1["descriptors"].detach().contiguous()

        KORNIA_CHECK(desc0.shape[-1] == self.conf.input_dim, "Descriptor dimension does not match input dim in config")
        KORNIA_CHECK(desc1.shape[-1] == self.conf.input_dim, "Descriptor dimension does not match input dim in config")

        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()

        mask0, mask1 = None, None
        c = max(m, n)
        do_compile = self.static_lengths and c <= max(self.static_lengths)
        if do_compile:
            kn = min([k for k in self.static_lengths if k >= c])
            desc0, mask0 = pad_to_length(desc0, kn)
            desc1, mask1 = pad_to_length(desc1, kn)
            kpts0, _ = pad_to_length(kpts0, kn)
            kpts1, _ = pad_to_length(kpts1, kn)
        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        # cache positional embeddings
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        # GNN + final_proj + assignment
        do_early_stop = self.conf.depth_confidence > 0
        do_point_pruning = self.conf.width_confidence > 0 and not do_compile
        pruning_th = self.pruning_min_kpts(device)
        if do_point_pruning:
            ind0 = arange(0, m, device=device)[None]
            ind1 = arange(0, n, device=device)[None]
            # We store the index of the layer at which pruning is detected.
            prune0 = ones_like(ind0)
            prune1 = ones_like(ind1)
        token0, token1 = None, None
        for i in range(self.conf.n_layers):
            desc0, desc1 = self.transformers[i](desc0, desc1, encoding0, encoding1, mask0=mask0, mask1=mask1)
            if i == self.conf.n_layers - 1:
                continue  # no early stopping or adaptive width at last layer

            if do_early_stop:
                token0, token1 = self.token_confidence[i](desc0, desc1)
                if self.check_if_stop(token0[..., :m, :], token1[..., :n, :], i, m + n):
                    break
            if do_point_pruning and desc0.shape[-2] > pruning_th:
                scores0 = self.log_assignment[i].get_matchability(desc0)
                prunemask0 = self.get_pruning_mask(token0, scores0, i)  # type: ignore
                keep0 = where(prunemask0)[1]
                ind0 = ind0.index_select(1, keep0)
                desc0 = desc0.index_select(1, keep0)
                encoding0 = encoding0.index_select(-2, keep0)
                prune0[:, ind0] += 1
            if do_point_pruning and desc1.shape[-2] > pruning_th:
                scores1 = self.log_assignment[i].get_matchability(desc1)
                prunemask1 = self.get_pruning_mask(token1, scores1, i)  # type: ignore
                keep1 = where(prunemask1)[1]
                ind1 = ind1.index_select(1, keep1)
                desc1 = desc1.index_select(1, keep1)
                encoding1 = encoding1.index_select(-2, keep1)
                prune1[:, ind1] += 1

        desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]
        scores, _ = self.log_assignment[i](desc0, desc1)
        m0, m1, mscores0, mscores1 = filter_matches(scores, self.conf.filter_threshold)
        matches, mscores = [], []
        for k in range(b):
            valid = m0[k] > -1
            m_indices_0 = where(valid)[0]
            m_indices_1 = m0[k][valid]
            if do_point_pruning:
                m_indices_0 = ind0[k, m_indices_0]
                m_indices_1 = ind1[k, m_indices_1]
            matches.append(stack([m_indices_0, m_indices_1], -1))
            mscores.append(mscores0[k][valid])

        # TODO: Remove when hloc switches to the compact format.
        if do_point_pruning:
            m0_ = torch.full((b, m), -1, device=m0.device, dtype=m0.dtype)
            m1_ = torch.full((b, n), -1, device=m1.device, dtype=m1.dtype)
            m0_[:, ind0] = where(m0 == -1, -1, ind1.gather(1, m0.clamp(min=0)))
            m1_[:, ind1] = where(m1 == -1, -1, ind0.gather(1, m1.clamp(min=0)))
            mscores0_ = zeros((b, m), device=mscores0.device)
            mscores1_ = zeros((b, n), device=mscores1.device)
            mscores0_[:, ind0] = mscores0
            mscores1_[:, ind1] = mscores1
            m0, m1, mscores0, mscores1 = m0_, m1_, mscores0_, mscores1_
        else:
            prune0 = ones_like(mscores0) * self.conf.n_layers
            prune1 = ones_like(mscores1) * self.conf.n_layers

        pred = {
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

        return pred

    def confidence_threshold(self, layer_index: int) -> float:
        """Scaled confidence threshold."""
        threshold = 0.8 + 0.1 * math.exp(-4.0 * layer_index / self.conf.n_layers)
        return min(max(threshold, 0), 1)

    def get_pruning_mask(self, confidences: Tensor, scores: Tensor, layer_index: int) -> Tensor:
        """Mask points which should be removed."""
        keep = scores > (1 - self.conf.width_confidence)
        if confidences is not None:  # Low-confidence points are never pruned.
            keep |= confidences <= self.confidence_thresholds[layer_index]
        return keep

    def check_if_stop(
        self,
        confidences0: Tensor,
        confidences1: Tensor,
        layer_index: int,
        num_points: int,
    ) -> Tensor:
        """Evaluate stopping condition."""
        confidences = concatenate([confidences0, confidences1], -1)
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = 1.0 - (confidences < threshold).float().sum() / num_points
        return ratio_confident > self.conf.depth_confidence

    def pruning_min_kpts(self, device: torch.device) -> int:
        if self.conf.flash and FLASH_AVAILABLE and device.type == "cuda":
            return self.pruning_keypoint_thresholds["flash"]
        else:
            return self.pruning_keypoint_thresholds[device.type]
