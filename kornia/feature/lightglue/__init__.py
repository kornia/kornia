from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

try:
    from flash_attn.modules.mha import FlashCrossAttention

    FOUND_OFFICIAL_FLASH = True
except ModuleNotFoundError:
    FOUND_OFFICIAL_FLASH = False


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints(
    kpts: torch.Tensor, size: Optional[List[int]] = None, shape: Optional[List[int]] = None
) -> torch.Tensor:
    if size is None:
        assert shape is not None
        _, _, h, w = shape
        one = kpts.new_tensor(1)
        size = torch.stack([one * w, one * h])[None]
    shift = size.float().to(kpts) / 2
    scale = size.max(1).values.float().to(kpts) / 2
    kpts = (kpts - shift[:, None]) / scale[:, None, None]
    return kpts


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None, gamma: float = 1.0) -> None:
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
        return repeat(emb, '... n -> ... (n r)', r=2)


class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """Get confidence tokens."""
        return (self.token(desc0.detach().float()).squeeze(-1), self.token(desc1.detach().float()).squeeze(-1))


class FastAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.s = dim**-0.5

    def forward(self, q, k, v) -> torch.Tensor:
        if hasattr(F, 'scaled_dot_product_attention'):
            q, k, v = (x.contiguous() for x in [q, k, v])
            return F.scaled_dot_product_attention(q, k, v)
        else:
            s = self.s
            attn = F.softmax(torch.einsum('...id,...jd->...ij', q, k) * s, -1)
            return torch.einsum('...ij,...jd->...id', attn, v)


class FlashAttention(nn.Module):
    def __init__(self, *args) -> None:
        super().__init__()
        if FOUND_OFFICIAL_FLASH:
            self.flash = FlashCrossAttention()

    def forward(self, q, k, v) -> torch.Tensor:
        if FOUND_OFFICIAL_FLASH:
            q, k, v = (x.transpose(-2, -3) for x in [q, k, v])
            m = self.flash(q.half(), torch.stack([k, v], 2).half())
            return m.transpose(-2, -3).to(q.dtype)
        else:
            args = [x.half().contiguous() for x in [q, k, v]]
            return F.scaled_dot_product_attention(*args).to(q.dtype)


class Transformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        attn = FlashAttention if flash else FastAttention
        self.inner_attn = attn(self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def _forward(self, x: torch.Tensor, encoding: Optional[torch.Tensor] = None):
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, 'b n (h d three) -> b h n d three', three=3, h=self.num_heads)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        if encoding is not None:
            q = apply_cached_rotary_emb(encoding, q)
            k = apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v)
        message = self.out_proj(rearrange(context, 'b h n d -> b n (h d)'))
        return x + self.ffn(torch.cat([x, message], -1))

    def forward(self, x0, x1, encoding0=None, encoding1=None):
        return self._forward(x0, encoding0), self._forward(x1, encoding1)


class CrossTransformer(nn.Module):
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
            self.flash = FastAttention(dim_head)
        else:
            self.flash = None

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> List[torch.Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (qk0, qk1, v0, v1))
        if self.flash is not None:
            m0 = self.flash(qk0, qk1, v1)
            m1 = self.flash(qk1, qk0, v0)
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = torch.einsum('b h i d, b h j d -> b h i j', qk0, qk1)
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            m0 = torch.einsum('bhij, bhjd -> bhid', attn01, v1)
            m1 = torch.einsum('bhji, bhjd -> bhid', attn10.transpose(-2, -1), v0)
        m0, m1 = self.map_(lambda t: rearrange(t, 'b h n d -> b n (h d)'), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


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
    def __init__(self, dim: float) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """Build assignment matrix from descriptors."""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum('bmd,bnd->bmn', mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def scores(self, desc0: torch.Tensor, desc1: torch.Tensor):
        m0 = torch.sigmoid(self.matchability(desc0)).squeeze(-1)
        m1 = torch.sigmoid(self.matchability(desc1)).squeeze(-1)
        return m0, m1


def filter_matches(scores: torch.Tensor, th: float):
    """Obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    mutual0 = torch.arange(m0.shape[1]).to(m0)[None] == m1.gather(1, m0)
    mutual1 = torch.arange(m1.shape[1]).to(m1)[None] == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    if th is not None:
        valid0 = mutual0 & (mscores0 > th)
    else:
        valid0 = mutual0
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, m0.new_tensor(-1))
    m1 = torch.where(valid1, m1, m1.new_tensor(-1))
    return m0, m1, mscores0, mscores1


class LightGlue(nn.Module):
    default_conf = {
        'name': 'lightglue',  # just for interfacing
        'input_dim': 256,  # input descriptor dimension (autoselected from weights)
        'descriptor_dim': 256,
        'n_layers': 9,
        'num_heads': 4,
        'flash': False,  # enable FlashAttention
        'mp': False,  # enable mixed precision
        'filter_threshold': 0.1,  # match threshold
        'depth_confidence': -1,  # -1 is no early stopping, recommend: 0.95
        'width_confidence': -1,  # -1 is no point pruning, recommend: 0.99
        'weights': None,
    }

    required_data_keys = ['keypoints0', 'keypoints1', 'descriptors0', 'descriptors1']

    version = "v0.1_arxiv"
    url = "https://github.com/cvg/LightGlue/releases/download/{}/{}_lightglue.pth"

    pretrained = {'superpoint': ('superpoint_lightglue', 256), 'disk': ('disk_lightglue', 128)}

    def __init__(self, pretrained='superpoint', **conf) -> None:
        super().__init__()
        self.conf = {**self.default_conf, **conf}
        if pretrained is not None:
            assert pretrained in list(self.pretrained.keys())
            self.conf['weights'], self.conf['input_dim'] = self.pretrained[pretrained]
        self.conf = conf = SimpleNamespace(**self.conf)

        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(conf.input_dim, conf.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(2, head_dim, head_dim)

        h, n, d = conf.num_heads, conf.n_layers, conf.descriptor_dim
        self.self_attn = nn.ModuleList([Transformer(d, h, conf.flash) for _ in range(n)])
        self.cross_attn = nn.ModuleList([CrossTransformer(d, h, conf.flash) for _ in range(n)])
        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = nn.ModuleList([TokenConfidence(d) for _ in range(n - 1)])

        if pretrained is not None:
            fname = f'{conf.weights}_{self.version}.pth'.replace('.', '-')
            state_dict = torch.hub.load_state_dict_from_url(self.url.format(self.version, pretrained), file_name=fname)
            self.load_state_dict(state_dict, strict=False)
        elif conf.weights is not None:
            path = Path(__file__).parent
            path = path / f'weights/{self.conf.weights}.pth'
            state_dict = torch.load(str(path), map_location='cpu')
            self.load_state_dict(state_dict, strict=False)

        print('Loaded LightGlue model')

    def forward(self, data: dict) -> dict:
        """Match keypoints and descriptors between two images.

        Input (dict):
            keypoints0: [B x M x 2], descriptors0: [B x M x D]
            keypoints1: [B x N x 2], descriptors1: [B x N x D]

        Output (dict):
            matches0: [B x M], matching_scores0: [B x M]
            matches1: [B x N], matching_scores1: [B x N]
            log_assignment: [B x M+1 x N+1]
        """
        with torch.autocast(enabled=self.conf.mp, device_type='cuda'):
            return self._forward(data)

    def _forward(self, data: dict) -> dict:
        for key in self.required_data_keys:
            assert key in data, f'Missing key {key} in data'
        kpts0_, kpts1_ = data['keypoints0'], data['keypoints1']
        b, m, _ = kpts0_.shape
        b, n, _ = kpts1_.shape

        kpts0 = normalize_keypoints(kpts0_, size=data.get('image_size0'), shape=data['image0'].shape)
        kpts1 = normalize_keypoints(kpts1_, size=data.get('image_size1'), shape=data['image1'].shape)

        assert torch.all(kpts0 >= -1) and torch.all(kpts0 <= 1)
        assert torch.all(kpts1 >= -1) and torch.all(kpts1 <= 1)

        desc0 = data['descriptors0'].detach()
        desc1 = data['descriptors1'].detach()

        assert desc0.shape[-1] == self.conf.input_dim
        assert desc1.shape[-1] == self.conf.input_dim

        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()

        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)

        # cache positional embeddings
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        # GNN + final_proj + assignment
        ind0 = torch.arange(0, m).to(device=kpts0.device)[None]
        ind1 = torch.arange(0, n).to(device=kpts0.device)[None]
        prune0 = torch.ones_like(ind0)  # store layer where pruning is detected
        prune1 = torch.ones_like(ind1)
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
                mask0 = self.get_mask(token0, match0, self.conf_th(i), 1 - wic)
                mask1 = self.get_mask(token1, match1, self.conf_th(i), 1 - wic)
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
            scores = torch.zeros(b, m + 1, n + 1, dtype=dt, device=dev)
            scores[:, :-1, :-1] = -torch.inf
            scores[:, ind0[0], -1] = scores_[:, :-1, -1]
            scores[:, -1, ind1[0]] = scores_[:, -1, :-1]
            x, y = torch.meshgrid(ind0[0], ind1[0], indexing='ij')
            scores[:, x, y] = scores_[:, :-1, :-1]
        else:
            scores, _ = self.log_assignment[i](desc0, desc1)

        m0, m1, mscores0, mscores1 = filter_matches(scores, self.conf.filter_threshold)

        return {
            'log_assignment': scores,
            'matches0': m0,
            'matches1': m1,
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
            'stop': i + 1,
            'prune0': prune0,
            'prune1': prune1,
        }

    def conf_th(self, i: int) -> float:
        """Scaled confidence threshold."""
        return np.clip(0.8 + 0.1 * np.exp(-4.0 * i / self.conf.n_layers), 0, 1)

    def get_mask(self, confidence: torch.Tensor, match: torch.Tensor, conf_th: float, match_th: float) -> torch.Tensor:
        """Mask points which should be removed."""
        if conf_th and confidence is not None:
            mask = torch.where(confidence > conf_th, match, match.new_tensor(1.0)) > match_th
        else:
            mask = match > match_th
        return mask

    def stop(
        self, token0: torch.Tensor, token1: torch.Tensor, conf_th: float, inl_th: float, seql: int
    ) -> torch.Tensor:
        """Evaluate stopping condition."""
        tokens = torch.cat([token0, token1], -1)
        if conf_th:
            pos = 1.0 - (tokens < conf_th).float().sum() / seql
            return pos > inl_th
        else:
            return tokens.mean() > inl_th
