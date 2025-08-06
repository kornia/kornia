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

import pytest
import torch
import torch.nn.functional as F


# For demonstration, re-define them inline:
def batched_dot_product(x: torch.Tensor, y: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    return (x * y).sum(-1, keepdim)


def normalized_orig(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return v / batched_dot_product(v, v, keepdim=True).add(eps).sqrt()


def normalized_fnorm(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(v, p=2, dim=-1, eps=eps)


# -- Reference normalization using explicit loops (earlier implementation) --
def normalized_ref(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # clone input and iterate over flattened vectors to preserve batch dims
    out = v.clone()
    flat = out.view(-1, v.size(-1))
    for i, vec in enumerate(flat):
        norm = torch.sqrt((vec * vec).sum() + eps)
        flat[i] = vec / norm
    return out


@pytest.fixture(params=[(2, 3), (4, 5, 6), (10, 128)])
def random_batch(request):
    shape = request.param
    torch.manual_seed(0)
    return torch.randn(*shape, dtype=torch.float32)


def test_batched_dot_product_shape(random_batch):
    x = random_batch
    y = random_batch.clone()
    out1 = batched_dot_product(x, y)
    out2 = batched_dot_product(x, y, keepdim=True)
    assert out1.shape == x.shape[:-1]
    assert out2.shape == x.shape[:-1] + (1,)


@pytest.mark.parametrize("fn", [normalized_orig, normalized_fnorm])
def test_unit_norm(random_batch, fn):
    v = random_batch
    out = fn(v)
    assert out.shape == v.shape
    norms = torch.linalg.norm(out.flatten(0, -2), dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


@pytest.mark.parametrize("fn", [normalized_orig, normalized_fnorm])
def test_normalized_against_ref(fn, random_batch):
    v = random_batch
    out = fn(v)
    ref = normalized_ref(v)
    assert torch.allclose(out, ref, atol=1e-6), (
        f"{fn.__name__} differs from reference by {(out - ref).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("fn", [normalized_fnorm])
def test_optimized_matches_orig(fn, random_batch):
    v = random_batch
    out_opt = fn(v)
    out_orig = normalized_orig(v)
    assert torch.allclose(out_opt, out_orig, atol=1e-6), (
        f"{fn.__name__} differs from normalized_orig by {(out_opt - out_orig).abs().max().item():.3e}"
    )


def test_orig_matches_fnormalize(random_batch):
    v = random_batch
    out_orig = normalized_orig(v)
    out_fnorm = F.normalize(v, p=2, dim=-1, eps=1e-6)
    assert torch.allclose(out_orig, out_fnorm, atol=1e-6), (
        f"normalized_orig differs from F.normalize by {(out_orig - out_fnorm).abs().max().item():.3e}"
    )
