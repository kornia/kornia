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

from kornia.feature.lightglue import (
    LightGlue,
    LearnableFourierPositionalEncoding,
    TokenConfidence,
    normalize_keypoints,
    pad_to_length,
    rotate_half,
    apply_cached_rotary_emb,
)

from testing.base import BaseTester


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


def test_normalize_keypoints_smoke():
    kpts = torch.zeros(1, 5, 2)
    size = torch.tensor([[100, 200]])
    out = normalize_keypoints(kpts, size)
    assert out.shape == kpts.shape


def test_normalize_keypoints_center():
    size = torch.tensor([[100, 100]])
    kpts = size.float().unsqueeze(0) / 2  # center pixel
    out = normalize_keypoints(kpts, size)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_normalize_keypoints_range():
    size = torch.tensor([[128, 128]])
    kpts = torch.rand(1, 20, 2) * 128
    out = normalize_keypoints(kpts, size)
    assert out.min() >= -1.0 - 1e-3
    assert out.max() <= 1.0 + 1e-3


def test_pad_to_length_no_pad():
    x = torch.rand(1, 10, 64)
    y, mask = pad_to_length(x, 5)
    assert y is x
    assert mask.shape[-2] == 10


def test_pad_to_length_pads():
    x = torch.rand(1, 5, 64)
    y, mask = pad_to_length(x, 10)
    assert y.shape[-2] == 10
    assert mask.shape[-2] == 10
    assert mask[..., :5, :].all()
    assert not mask[..., 5:, :].any()


def test_rotate_half_shape():
    x = torch.rand(2, 8, 4)
    out = rotate_half(x)
    assert out.shape == x.shape


def test_rotate_half_double_gives_negation():
    x = torch.rand(1, 4, 8)
    assert torch.allclose(rotate_half(rotate_half(x)), -x, atol=1e-6)


def test_apply_cached_rotary_emb_shape():
    B, N, D = 1, 10, 16
    freqs = torch.rand(2, B, 1, N, D)
    t = torch.rand(B, 1, N, D)
    out = apply_cached_rotary_emb(freqs, t)
    assert out.shape == t.shape


# ---------------------------------------------------------------------------
# Module tests
# ---------------------------------------------------------------------------


class TestLearnableFourierPositionalEncoding(BaseTester):
    def test_smoke(self, device, dtype):
        enc = LearnableFourierPositionalEncoding(2, 64).to(device, dtype)
        x = torch.rand(1, 1, 10, 2, device=device, dtype=dtype)
        out = enc(x)
        assert out.shape[0] == 2  # cosines/sines stack

    def test_cardinality(self, device, dtype):
        M, dim = 2, 32
        enc = LearnableFourierPositionalEncoding(M, dim).to(device, dtype)
        B, N = 2, 15
        x = torch.rand(B, 1, N, M, device=device, dtype=dtype)
        out = enc(x)
        assert out.shape[-1] == dim

    def test_gradcheck(self, device):
        enc = LearnableFourierPositionalEncoding(2, 32).to(device, torch.float64)
        x = torch.rand(1, 1, 5, 2, device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(enc, (x,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        enc = LearnableFourierPositionalEncoding(2, 32).to(device, dtype)
        x = torch.rand(1, 1, 5, 2, device=device, dtype=dtype)
        op = torch_optimizer(enc)
        self.assert_close(op(x), enc(x))

    def test_exception(self, device, dtype):
        pass

    def test_module(self, device, dtype):
        enc = LearnableFourierPositionalEncoding(2, 32).to(device, dtype)
        x = torch.rand(1, 1, 5, 2, device=device, dtype=dtype)
        assert enc(x).shape[0] == 2


class TestTokenConfidence(BaseTester):
    def test_smoke(self, device, dtype):
        tc = TokenConfidence(64).to(device, dtype)
        d0 = torch.rand(1, 10, 64, device=device, dtype=dtype)
        d1 = torch.rand(1, 8, 64, device=device, dtype=dtype)
        s0, s1 = tc(d0, d1)
        assert s0.shape == (1, 10)
        assert s1.shape == (1, 8)

    def test_cardinality(self, device, dtype):
        tc = TokenConfidence(32).to(device, dtype)
        B, M, N = 2, 12, 15
        d0 = torch.rand(B, M, 32, device=device, dtype=dtype)
        d1 = torch.rand(B, N, 32, device=device, dtype=dtype)
        s0, s1 = tc(d0, d1)
        assert s0.shape == (B, M)
        assert s1.shape == (B, N)

    def test_scores_in_range(self, device, dtype):
        tc = TokenConfidence(32).to(device, dtype)
        d0 = torch.rand(1, 20, 32, device=device, dtype=dtype)
        d1 = torch.rand(1, 20, 32, device=device, dtype=dtype)
        s0, s1 = tc(d0, d1)
        assert (s0 >= 0).all() and (s0 <= 1).all()
        assert (s1 >= 0).all() and (s1 <= 1).all()

    def test_gradcheck(self, device):
        pass  # TokenConfidence uses detach() on inputs; not differentiable w.r.t. inputs

    def test_dynamo(self, device, dtype, torch_optimizer):
        tc = TokenConfidence(32).to(device, dtype)
        d0 = torch.rand(1, 10, 32, device=device, dtype=dtype)
        d1 = torch.rand(1, 8, 32, device=device, dtype=dtype)
        op = torch_optimizer(tc)
        s0_jit, s1_jit = op(d0, d1)
        s0, s1 = tc(d0, d1)
        self.assert_close(s0_jit, s0)
        self.assert_close(s1_jit, s1)

    def test_exception(self, device, dtype):
        pass

    def test_module(self, device, dtype):
        tc = TokenConfidence(32).to(device, dtype)
        d0 = torch.rand(1, 5, 32, device=device, dtype=dtype)
        d1 = torch.rand(1, 5, 32, device=device, dtype=dtype)
        s0, _s1 = tc(d0, d1)
        assert s0.shape == (1, 5)


# ---------------------------------------------------------------------------
# LightGlue (no pretrained weights) tests
# ---------------------------------------------------------------------------

def _make_lightglue(device, dtype, input_dim=64, n_layers=2):
    """Instantiate a small LightGlue with random weights (features=None)."""
    return LightGlue(
        features=None,
        input_dim=input_dim,
        descriptor_dim=64,
        n_layers=n_layers,
        num_heads=4,
        depth_confidence=-1,
        width_confidence=-1,
        flash=False,
    ).to(device, dtype).eval()


def _make_data(device, dtype, B=1, M=20, N=15, H=64, W=64, D=64):
    """Build a minimal data dict for LightGlue forward."""
    return {
        "image0": {
            "keypoints": torch.rand(B, M, 2, device=device, dtype=dtype) * torch.tensor([W, H], device=device, dtype=dtype),
            "descriptors": torch.rand(B, M, D, device=device, dtype=dtype),
            "image_size": torch.tensor([[W, H]], device=device, dtype=dtype).expand(B, -1),
        },
        "image1": {
            "keypoints": torch.rand(B, N, 2, device=device, dtype=dtype) * torch.tensor([W, H], device=device, dtype=dtype),
            "descriptors": torch.rand(B, N, D, device=device, dtype=dtype),
            "image_size": torch.tensor([[W, H]], device=device, dtype=dtype).expand(B, -1),
        },
    }


class TestLightGlue(BaseTester):
    def test_smoke(self, device, dtype):
        if dtype == torch.float16:
            pytest.skip("LightGlue requires float32 or float64")
        lg = _make_lightglue(device, dtype)
        data = _make_data(device, dtype)
        with torch.no_grad():
            out = lg(data)
        assert "matches0" in out
        assert "matching_scores0" in out

    def test_cardinality(self, device, dtype):
        if dtype == torch.float16:
            pytest.skip("LightGlue requires float32 or float64")
        B, M, N = 1, 20, 15
        lg = _make_lightglue(device, dtype)
        data = _make_data(device, dtype, B=B, M=M, N=N)
        with torch.no_grad():
            out = lg(data)
        assert out["matches0"].shape == (B, M)
        assert out["matches1"].shape == (B, N)
        assert out["matching_scores0"].shape == (B, M)

    def test_image_tensor_input(self, device, dtype):
        """Accept image tensor instead of image_size."""
        if dtype == torch.float16:
            pytest.skip("LightGlue requires float32 or float64")
        B, M, N, H, W, D = 1, 10, 8, 32, 32, 64
        lg = _make_lightglue(device, dtype)
        data = {
            "image0": {
                "keypoints": torch.rand(B, M, 2, device=device, dtype=dtype) * W,
                "descriptors": torch.rand(B, M, D, device=device, dtype=dtype),
                "image": torch.rand(B, 3, H, W, device=device, dtype=dtype),
            },
            "image1": {
                "keypoints": torch.rand(B, N, 2, device=device, dtype=dtype) * W,
                "descriptors": torch.rand(B, N, D, device=device, dtype=dtype),
                "image": torch.rand(B, 3, H, W, device=device, dtype=dtype),
            },
        }
        with torch.no_grad():
            out = lg(data)
        assert "matches0" in out

    def test_matches_are_valid_indices(self, device, dtype):
        if dtype == torch.float16:
            pytest.skip("LightGlue requires float32 or float64")
        B, M, N = 1, 30, 25
        lg = _make_lightglue(device, dtype)
        data = _make_data(device, dtype, B=B, M=M, N=N)
        with torch.no_grad():
            out = lg(data)
        m0 = out["matches0"]  # (B, M), values in [-1, N-1]
        assert (m0 >= -1).all()
        assert (m0 < N).all()

    def test_scores_in_range(self, device, dtype):
        if dtype == torch.float16:
            pytest.skip("LightGlue requires float32 or float64")
        lg = _make_lightglue(device, dtype)
        data = _make_data(device, dtype)
        with torch.no_grad():
            out = lg(data)
        scores = out["matching_scores0"]
        assert (scores >= 0).all()
        assert (scores <= 1).all()

    def test_exception_missing_key(self, device, dtype):
        if dtype == torch.float16:
            pytest.skip("LightGlue requires float32 or float64")
        lg = _make_lightglue(device, dtype)
        with pytest.raises(Exception):
            lg({"image0": {}})

    def test_exception_wrong_features(self, device, dtype):
        with pytest.raises(Exception):
            LightGlue(features="nonexistent_feature_type")

    def test_gradcheck(self, device):
        pass  # LightGlue uses detach() on descriptors; not differentiable end-to-end

    def test_dynamo(self, device, dtype, torch_optimizer):
        pass  # LightGlue uses dynamic control flow; not torch.compile compatible

    def test_module(self, device, dtype):
        if dtype == torch.float16:
            pytest.skip("LightGlue requires float32 or float64")
        lg = _make_lightglue(device, dtype)
        data = _make_data(device, dtype)
        with torch.no_grad():
            out = lg(data)
        assert isinstance(out, dict)

    @pytest.mark.slow
    def test_pretrained_smoke(self, device):
        """Instantiating with real feature type downloads and loads weights."""
        lg = LightGlue(features="disk", depth_confidence=-1, width_confidence=-1).to(device).eval()
        B, M, N, D = 1, 50, 50, 128
        data = _make_data(device, torch.float32, B=B, M=M, N=N, D=D)
        with torch.no_grad():
            out = lg(data)
        assert "matches0" in out
