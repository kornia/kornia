import math
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core import Module, ModuleList, Tensor, arange, concatenate, einsum, ones_like, softmax, stack, where, zeros
from kornia.core.check import KORNIA_CHECK
from kornia.utils._compat import torch_meshgrid

try:
    from flash_attn.modules.mha import FlashCrossAttention
except ModuleNotFoundError:
    FlashCrossAttention = None

if FlashCrossAttention or hasattr(F, 'scaled_dot_product_attention'):
    FLASH_AVAILABLE = True
else:
    FLASH_AVAILABLE = False


def math_clamp(x, min_, max_):  # type: ignore
    return max(min(x, min_), min_)


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints(kpts: Tensor, size: Tensor) -> Tensor:
    if isinstance(size, torch.Size):
        size = Tensor(size)[None]
    shift = size.float().to(kpts) / 2
    scale = size.max(1).values.float().to(kpts) / 2
    kpts = (kpts - shift[:, None]) / scale[:, None, None]
    return kpts


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
        cosines, sines = torch.cos(projected), torch.sin(projected)
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
                'FlashAttention is not available. For optimal speed, '
                'consider installing torch >= 2.0 or flash-attn.',
                stacklevel=2,
            )
        self.enable_flash = allow_flash and FLASH_AVAILABLE
        if allow_flash and FlashCrossAttention:
            self.flash_ = FlashCrossAttention()

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if self.enable_flash and q.device.type == 'cuda':
            if FlashCrossAttention:
                q, k, v = (x.transpose(-2, -3) for x in [q, k, v])
                m = self.flash_(q.half(), stack([k, v], 2).half())
                return m.transpose(-2, -3).to(q.dtype)
            else:  # use torch 2.0 scaled_dot_product_attention with flash
                args = [x.half().contiguous() for x in [q, k, v]]
                with torch.backends.cuda.sdp_kernel(enable_flash=True):
                    return F.scaled_dot_product_attention(*args).to(q.dtype)
        elif hasattr(F, 'scaled_dot_product_attention'):
            args = [x.contiguous() for x in [q, k, v]]
            return F.scaled_dot_product_attention(*args).to(q.dtype)
        else:
            s = q.shape[-1] ** -0.5
            attn = softmax(einsum('...id,...jd->...ij', q, k) * s, -1)
            return einsum('...ij,...jd->...id', attn, v)


class Transformer(Module):
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

    def _forward(self, x: Tensor, encoding: Optional[Tensor] = None) -> Tensor:
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        if encoding is not None:
            q = apply_cached_rotary_emb(encoding, q)
            k = apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(concatenate([x, message], -1))

    def forward(
        self, x0: Tensor, x1: Tensor, encoding0: Optional[Tensor] = None, encoding1: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        return self._forward(x0, encoding0), self._forward(x1, encoding1)


class CrossTransformer(Module):
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

    def forward(self, x0: Tensor, x1: Tensor) -> Tuple[Tensor, Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = (t.unflatten(-1, (self.heads, -1)).transpose(1, 2) for t in (qk0, qk1, v0, v1))
        if self.flash is not None:
            m0 = self.flash(qk0, qk1, v1)
            m1 = self.flash(qk1, qk0, v0)
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = einsum('b h i d, b h j d -> b h i j', qk0, qk1)
            attn01 = softmax(sim, dim=-1)
            attn10 = softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            m0 = einsum('bhij, bhjd -> bhid', attn01, v1)
            m1 = einsum('bhji, bhjd -> bhid', attn10.transpose(-2, -1), v0)
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(concatenate([x0, m0], -1))
        x1 = x1 + self.ffn(concatenate([x1, m1], -1))
        return x0, x1


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
        sim = einsum('bmd,bnd->bmn', mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def scores(self, desc0: Tensor, desc1: Tensor) -> Tuple[Tensor, Tensor]:
        m0 = torch.sigmoid(self.matchability(desc0)).squeeze(-1)
        m1 = torch.sigmoid(self.matchability(desc1)).squeeze(-1)
        return m0, m1


def filter_matches(scores: Tensor, th: float) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    mutual0 = arange(m0.shape[1]).to(m0)[None] == m1.gather(1, m0)
    mutual1 = arange(m1.shape[1]).to(m1)[None] == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = where(mutual0, max0_exp, zero)
    mscores1 = where(mutual1, mscores0.gather(1, m1), zero)
    if th is not None:
        valid0 = mutual0 & (mscores0 > th)
    else:
        valid0 = mutual0
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = where(valid0, m0, m0.new_tensor(-1))
    m1 = where(valid1, m1, m1.new_tensor(-1))
    return m0, m1, mscores0, mscores1


class LightGlue(Module):
    default_conf: ClassVar[Dict[str, Any]] = {
        'name': 'lightglue',  # just for interfacing
        'input_dim': 256,  # input descriptor dimension (autoselected from weights)
        'descriptor_dim': 256,
        'n_layers': 9,
        'num_heads': 4,
        'flash': True,  # enable FlashAttention if available.
        'mp': False,  # enable mixed precision
        'depth_confidence': 0.95,  # early stopping, disable with -1
        'width_confidence': 0.99,  # point pruning, disable with -1
        'filter_threshold': 0.1,  # match threshold
        'weights': None,
    }

    required_data_keys: ClassVar[List[str]] = ['image0', 'image1']

    version: ClassVar[str] = "v0.1_arxiv"
    url: ClassVar[str] = "https://github.com/cvg/LightGlue/releases/download/{}/{}_lightglue.pth"

    features: ClassVar[Dict[str, Any]] = {'superpoint': ('superpoint_lightglue', 256), 'disk': ('disk_lightglue', 128)}

    def __init__(self, features: str = 'superpoint', **conf) -> None:  # type: ignore
        super().__init__()
        temp_conf = {**self.default_conf, **conf}
        if features is not None:
            KORNIA_CHECK(features in list(self.features.keys()), "Features keys are wrong")
            temp_conf['weights'], temp_conf['input_dim'] = self.features[features]
        self.conf = conf_ = SimpleNamespace(**temp_conf)

        if conf_.input_dim != conf_.descriptor_dim:
            self.input_proj = nn.Linear(conf_.input_dim, conf_.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()  # type: ignore

        head_dim = conf_.descriptor_dim // conf_.num_heads
        self.posenc = LearnableFourierPositionalEncoding(2, head_dim, head_dim)

        h, n, d = conf_.num_heads, conf_.n_layers, conf_.descriptor_dim
        self.self_attn = ModuleList([Transformer(d, h, conf_.flash) for _ in range(n)])
        self.cross_attn = ModuleList([CrossTransformer(d, h, conf_.flash) for _ in range(n)])
        self.log_assignment = ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = ModuleList([TokenConfidence(d) for _ in range(n - 1)])

        if features is not None:
            fname = f'{conf_.weights}_{self.version}.pth'.replace('.', '-')
            state_dict = torch.hub.load_state_dict_from_url(self.url.format(self.version, features), file_name=fname)
            self.load_state_dict(state_dict, strict=False)
        elif conf_.weights is not None:
            path = Path(__file__).parent
            path = path / f'weights/{self.conf.weights}.pth'
            state_dict = torch.load(str(path), map_location='cpu')
            self.load_state_dict(state_dict, strict=False)

        print('Loaded LightGlue model')

    def forward(self, data: Dict) -> Dict:  # type: ignore
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
        with torch.cuda.amp.autocast(enabled=self.conf.mp):
            return self._forward(data)

    def _forward(self, data: Dict[str, Dict[str, Tensor]]) -> Dict[str, Any]:
        for key in self.required_data_keys:
            KORNIA_CHECK(key in data, f'Missing key {key} in data')
        data0, data1 = data['image0'], data['image1']
        kpts0_, kpts1_ = data0['keypoints'], data1['keypoints']

        # torch 1.9.0 compatibility
        if hasattr(torch, 'inf'):
            inf = torch.inf
        else:
            inf = math.inf

        b, m, _ = kpts0_.shape
        b, n, _ = kpts1_.shape
        size0, size1 = data0.get('image_size'), data1.get('image_size')
        size0 = size0 if size0 is not None else data0['image'].shape[-2:][::-1]  # type: ignore
        size1 = size1 if size1 is not None else data1['image'].shape[-2:][::-1]  # type: ignore
        kpts0 = normalize_keypoints(kpts0_, size=size0)
        kpts1 = normalize_keypoints(kpts1_, size=size1)

        KORNIA_CHECK(torch.all(kpts0 >= -1).item() and torch.all(kpts0 <= 1).item(), "")  # type: ignore
        KORNIA_CHECK(torch.all(kpts1 >= -1).item() and torch.all(kpts1 <= 1).item(), "")  # type: ignore

        desc0 = data0['descriptors'].detach()
        desc1 = data1['descriptors'].detach()
        KORNIA_CHECK(desc0.shape[-1] == self.conf.input_dim, "Descriptor dimension does not match input dim in config")
        KORNIA_CHECK(desc1.shape[-1] == self.conf.input_dim, "Descriptor dimension does not match input dim in config")

        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()

        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)

        # cache positional embeddings
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        # GNN + final_proj + assignment
        ind0 = arange(0, m).to(device=kpts0.device)[None]
        ind1 = arange(0, n).to(device=kpts0.device)[None]
        prune0 = ones_like(ind0)  # store layer where pruning is detected
        prune1 = ones_like(ind1)
        dec, wic = self.conf.depth_confidence, self.conf.width_confidence
        token0, token1 = None, None
        for i in range(self.conf.n_layers):
            # self+cross attention
            desc0, desc1 = self.self_attn[i](desc0, desc1, encoding0, encoding1)
            desc0, desc1 = self.cross_attn[i](desc0, desc1)
            if i == self.conf.n_layers - 1:
                continue  # no early stopping or adaptive width at last layer
            if dec > 0:  # early stopping
                token0, token1 = self.token_confidence[i](desc0, desc1)
                if self.stop(token0, token1, self.conf_th(i), dec, m + n):
                    break
            if wic > 0:  # point pruning
                match0, match1 = self.log_assignment[i].scores(desc0, desc1)
                mask0 = self.get_mask(token0, match0, self.conf_th(i), 1 - wic)  # type: ignore
                mask1 = self.get_mask(token1, match1, self.conf_th(i), 1 - wic)  # type: ignore
                ind0, ind1 = ind0[mask0][None], ind1[mask1][None]
                desc0, desc1 = desc0[mask0][None], desc1[mask1][None]
                if desc0.shape[-2] == 0 or desc1.shape[-2] == 0:
                    break
                encoding0 = encoding0[:, :, mask0][:, None]
                encoding1 = encoding1[:, :, mask1][:, None]
            prune0[:, ind0] += 1
            prune1[:, ind1] += 1

        if wic > 0:  # scatter with indices after pruning
            scores_, _ = self.log_assignment[i](desc0, desc1)
            dt, dev = scores_.dtype, scores_.device
            scores = zeros(b, m + 1, n + 1, dtype=dt, device=dev)
            scores[:, :-1, :-1] = -inf
            scores[:, ind0[0], -1] = scores_[:, :-1, -1]
            scores[:, -1, ind1[0]] = scores_[:, -1, :-1]
            x, y = torch_meshgrid(ind0[0], ind1[0], indexing='ij')
            scores[:, x, y] = scores_[:, :-1, :-1]
        else:
            scores, _ = self.log_assignment[i](desc0, desc1)

        m0, m1, mscores0, mscores1 = filter_matches(scores, self.conf.filter_threshold)

        matches, mscores = [], []
        for k in range(b):
            valid = m0[k] > -1
            matches.append(stack([where(valid)[0], m0[k][valid]], -1))
            mscores.append(mscores0[k][valid])

        return {
            'log_assignment': scores,
            'matches0': m0,
            'matches1': m1,
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
            'stop': i + 1,
            'prune0': prune0,
            'prune1': prune1,
            'matches': matches,
            'scores': mscores,
        }

    def conf_th(self, i: int) -> float:
        """Scaled confidence threshold."""
        return math_clamp(0.8 + 0.1 * math.exp(-4.0 * float(i) / self.conf.n_layers), 0.0, 1.0)

    def get_mask(self, confidence: Tensor, match: Tensor, conf_th: float, match_th: float) -> Tensor:
        """Mask points which should be removed."""
        if conf_th and confidence is not None:
            mask = where(confidence > conf_th, match, match.new_tensor(1.0)) > match_th
        else:
            mask = match > match_th
        return mask

    def stop(self, token0: Tensor, token1: Tensor, conf_th: float, inl_th: float, seql: int) -> Tensor:
        """Evaluate stopping condition."""
        tokens = concatenate([token0, token1], -1)
        if conf_th:
            pos = 1.0 - (tokens < conf_th).float().sum() / seql
            return pos > inl_th
        else:
            return tokens.mean() > inl_th
