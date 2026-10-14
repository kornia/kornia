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

import unittest.mock

import pytest
import torch

from kornia.feature.integrated import LightGlueMatcher
from kornia.feature.laf import laf_from_center_scale_ori
from kornia.feature.lightglue import (
    Attention,
    LearnableFourierPositionalEncoding,
    LightGlue,
    TokenConfidence,
    _check_keypoints_normalized,
    apply_cached_rotary_emb,
    normalize_keypoints,
    pad_to_length,
    rotate_half,
)

from testing.base import BaseTester, assert_close

# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


def test_lightglue_empty_after_pruning():
    model = LightGlue(features="superpoint", width_confidence=0.99)
    model.eval()

    data = {
        "image0": {
            "keypoints": torch.empty(1, 0, 2),
            "descriptors": torch.empty(1, 0, 256),
            "image_size": torch.tensor([[640, 480]]),
        },
        "image1": {
            "keypoints": torch.empty(1, 0, 2),
            "descriptors": torch.empty(1, 0, 256),
            "image_size": torch.tensor([[640, 480]]),
        },
    }

    with torch.no_grad():
        out = model(data)

    assert out["matches0"].shape == (1, 0)
    assert out["matches1"].shape == (1, 0)
    assert out["matching_scores0"].shape == (1, 0)
    assert out["matching_scores1"].shape == (1, 0)


def test_lightglue_pruning_removes_all():
    model = LightGlue(features="superpoint", width_confidence=0.0)
    model.eval()

    B, M, D = 1, 8, 256

    data = {
        "image0": {
            "keypoints": torch.rand(B, M, 2),
            "descriptors": torch.rand(B, M, D),
            "image_size": torch.tensor([[640, 480]]),
        },
        "image1": {
            "keypoints": torch.rand(B, M, 2),
            "descriptors": torch.rand(B, M, D),
            "image_size": torch.tensor([[640, 480]]),
        },
    }

    with torch.no_grad():
        out = model(data)

    assert "matches0" in out
    assert "matches1" in out
    assert out["matches0"].shape == (B, M)
    assert out["matches1"].shape == (B, M)


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


def test_normalize_keypoints_autocast_promotes_to_f32():
    """Under CPU autocast, bf16 keypoints are promoted to fp32 for normalization.

    On the main branch (before #4624) the decorator-based ``@AMP_CUSTOM_FWD_F32``
    with ``device_type="cuda"`` did not fire under CPU autocast, so bf16 inputs
    remained bf16.  This head's per-tensor inline autocast guard correctly promotes.
    """
    kpts = torch.randn(1, 5, 2, dtype=torch.bfloat16)
    size = torch.tensor([[100, 200]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = normalize_keypoints(kpts, size)
    assert out.dtype == torch.float32
    assert out.shape == kpts.shape


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


def test_attention_allow_flash_matches_on_cpu():
    """On CPU, ``allow_flash`` never selects the CUDA-only branch, so both settings take the
    same ``scaled_dot_product_attention`` path and must produce identical outputs, with and
    without a boolean mask (exercising the ``nan_to_num`` handling of fully-masked rows).
    """
    torch.manual_seed(0)
    B, H, N, Dh = 1, 2, 5, 8
    q = torch.rand(B, H, N, Dh)
    k = torch.rand(B, H, N, Dh)
    v = torch.rand(B, H, N, Dh)

    # ``Attention.__init__`` flips the process-global ``torch.backends.cuda`` flash-SDPA switch;
    # put back whatever this test found so later tests in a CUDA run do not depend on its order.
    flash_sdp_was_enabled = torch.backends.cuda.flash_sdp_enabled()
    try:
        attn_no_flash = Attention(allow_flash=False)
        attn_flash = Attention(allow_flash=True)

        out_no_flash = attn_no_flash(q, k, v)
        out_flash = attn_flash(q, k, v)
        assert_close(out_no_flash, out_flash)

        # First query row cannot attend to anything: softmax produces NaN there, which
        # ``Attention.forward`` replaces via ``nan_to_num``.
        mask = torch.ones(B, H, N, N, dtype=torch.bool)
        mask[:, :, 0, :] = False

        out_no_flash_masked = attn_no_flash(q, k, v, mask=mask)
        out_flash_masked = attn_flash(q, k, v, mask=mask)
        assert not torch.isnan(out_no_flash_masked).any()
        assert not torch.isnan(out_flash_masked).any()
        assert_close(out_no_flash_masked, out_flash_masked)
    finally:
        torch.backends.cuda.enable_flash_sdp(flash_sdp_was_enabled)


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
        assert (s0 >= 0).all()
        assert (s0 <= 1).all()
        assert (s1 >= 0).all()
        assert (s1 <= 1).all()

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
        s0, _ = tc(d0, d1)
        assert s0.shape == (1, 5)


# ---------------------------------------------------------------------------
# LightGlue (no pretrained weights) tests
# ---------------------------------------------------------------------------


def _make_lightglue(device, dtype, input_dim=64, n_layers=2, mp=False):
    """Instantiate a small LightGlue with random weights (features=None)."""
    return (
        LightGlue(
            features=None,
            input_dim=input_dim,
            descriptor_dim=64,
            n_layers=n_layers,
            num_heads=4,
            depth_confidence=-1,
            width_confidence=-1,
            flash=False,
            mp=mp,
        )
        .to(device, dtype)
        .eval()
    )


def _make_data(device, dtype, B=1, M=20, N=15, H=64, W=64, D=64):
    """Build a minimal data dict for LightGlue forward."""
    return {
        "image0": {
            "keypoints": torch.rand(B, M, 2, device=device, dtype=dtype)
            * torch.tensor([W, H], device=device, dtype=dtype),
            "descriptors": torch.rand(B, M, D, device=device, dtype=dtype),
            "image_size": torch.tensor([[W, H]], device=device, dtype=dtype).expand(B, -1),
        },
        "image1": {
            "keypoints": torch.rand(B, N, 2, device=device, dtype=dtype)
            * torch.tensor([W, H], device=device, dtype=dtype),
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

    def test_mp_forward_uses_input_device_type(self, device, dtype):
        """With ``mp=True``, ``forward`` derives the autocast ``device_type`` from the
        keypoints device (the computation device), not from a hardcoded ``\"cuda\"`` nor
        from the first tensor encountered while scanning the input dict (regression for
        #4624).
        """
        if dtype == torch.float16:
            pytest.skip("LightGlue requires float32 or float64")
        lg = _make_lightglue(device, dtype, mp=True)
        data = _make_data(device, dtype)
        # Put a non-keypoint tensor (image_size) first so a naive "first tensor wins"
        # resolver would select it instead of the keypoints device.
        data["image0"] = {
            "image_size": data["image0"].pop("image_size"),
            **data["image0"],
        }
        seen: dict[str, str] = {}
        real_ac = torch.autocast

        def _spy(*args: object, **kwargs: object):
            seen["device_type"] = kwargs.get("device_type", "")
            return real_ac(*args, **kwargs)

        with unittest.mock.patch("kornia.feature.lightglue.torch.autocast", side_effect=_spy), torch.no_grad():
            out = lg(data)

        assert seen["device_type"] == data["image0"]["keypoints"].device.type
        assert "matches0" in out

    @pytest.mark.slow
    def test_pretrained_smoke(self, device):
        """Instantiating with real feature type downloads and loads weights."""
        lg = LightGlue(features="disk", depth_confidence=-1, width_confidence=-1).to(device).eval()
        B, M, N, D = 1, 50, 50, 128
        data = _make_data(device, torch.float32, B=B, M=M, N=N, D=D)
        with torch.no_grad():
            out = lg(data)
        assert "matches0" in out


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
@pytest.mark.parametrize(
    ("model_device", "autocast_device", "autocast_dtype"),
    [("cpu", "mps", torch.float16), ("mps", "cpu", torch.bfloat16)],
    # Ids must not start with "mps": conftest skips every "[mps" node whose name mentions autocast.
    ids=["model_cpu-autocast_mps", "model_mps-autocast_cpu"],
)
def test_autocast_of_another_device_type_leaves_descriptors_alone(model_device, autocast_device, autocast_dtype):
    """An autocast region of another device type must not cast the descriptors to half (#5198).

    The no-argument ``torch.is_autocast_enabled()`` reports MPS autocast on torch 2.14, so the half cast hit CPU
    descriptors and the CPU float32 projection raised a dtype mismatch.
    """
    lg = _make_lightglue(model_device, torch.float32)
    data = _make_data(model_device, torch.float32)
    with torch.no_grad(), torch.autocast(autocast_device, dtype=autocast_dtype):
        out = lg(data)
    assert out["log_assignment"].dtype == torch.float32


class TestKeypointRangeMessage:
    def test_check_names_the_range_and_the_size_order_5724(self):
        with pytest.raises(Exception, match=r"\[-1, 1\].*\(width, height\)"):
            _check_keypoints_normalized(torch.tensor([[[0.0, 2.0]]]))

    def test_check_accepts_normalized_keypoints_5724(self):
        _check_keypoints_normalized(torch.tensor([[[-1.0, 1.0], [0.0, 0.5]]]))

    def test_forward_with_a_too_small_image_size_5724(self, device):
        lg = _make_lightglue(device, torch.float32)
        data = _make_data(device, torch.float32)
        data["image0"]["image_size"] = torch.tensor([[8.0, 8.0]], device=device)  # keypoints reach 64
        with pytest.raises(Exception, match="Keypoints must lie inside the image"):
            lg(data)


class TestLightGluePruningDtype(BaseTester):
    """``LightGlue`` prunes points in the input dtype (#5837).

    ``LightGlue("disk")`` keeps the preset's pruning (``width_confidence=0.99``, ``depth_confidence=0.95``), with random
    weights in place of the download, two layers, and pruning at any number of keypoints on every device. Layer 0 is
    wired so that each pruning decision has a wide margin in every dtype: the descriptors enter unchanged, the
    matchability reads dimension 0 and the token confidence dimension 1, both +-20. A point of matchability -20 is
    pruned unless its confidence is -20 too, which keeps it as a low-confidence point. Image 1 is image 0 permuted, and
    the last layer's matchability is a constant 10, so a point kept in both images is a mutual match with its copy and
    scores ``sigmoid(10) ** 2``.
    """

    _M = 16
    # Sign of descriptor dimension 0 (matchability) and dimension 1 (token confidence) of each point.
    _MATCHABLE = (1, -1) * 8
    _CONFIDENT = (1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1)

    @staticmethod
    def _model(device, dtype, **conf):
        torch.manual_seed(0)
        with unittest.mock.patch("kornia.feature.lightglue.load_state_dict_from_url", return_value={}):
            lg = LightGlue("disk", n_layers=2, **conf)
        with torch.no_grad():
            lg.input_proj.weight.zero_()
            lg.input_proj.bias.zero_()
            lg.input_proj.weight[:128].copy_(torch.eye(128))
            for linear in (lg.log_assignment[0].matchability, lg.token_confidence[0].token[0]):
                linear.weight.zero_()
                linear.bias.zero_()
            lg.log_assignment[0].matchability.weight[0, 0] = 1.0
            lg.token_confidence[0].token[0].weight[0, 1] = 1.0
            lg.log_assignment[1].matchability.weight.zero_()
            lg.log_assignment[1].matchability.bias.fill_(10.0)
        return lg.to(device, dtype).eval()

    def _scene(self, device, dtype, matchable=_MATCHABLE):
        g = torch.Generator().manual_seed(0)
        desc = 3 * torch.randn(1, self._M, 128, generator=g)
        desc[0, :, 0] = 20 * torch.tensor(matchable, dtype=torch.float32)
        desc[0, :, 1] = 20 * torch.tensor(self._CONFIDENT, dtype=torch.float32)
        kpts = torch.rand(1, self._M, 2, generator=g) * torch.tensor([320.0, 240.0])
        perm = torch.randperm(self._M, generator=g)
        size = torch.tensor([[320, 240]], device=device)
        data = {
            "image0": {"keypoints": kpts.to(device, dtype), "descriptors": desc.to(device, dtype), "image_size": size},
            "image1": {
                "keypoints": kpts[:, perm].to(device, dtype),
                "descriptors": desc[:, perm].to(device, dtype),
                "image_size": size,
            },
        }
        return data, perm

    @staticmethod
    def _forward(model, data):
        thresholds = dict.fromkeys(LightGlue.pruning_keypoint_thresholds, -1)  # prune at any number of keypoints
        with torch.no_grad(), unittest.mock.patch.dict(LightGlue.pruning_keypoint_thresholds, thresholds):
            return model(data)

    @staticmethod
    def _floating_dtypes(out):
        tensors = [v for v in out.values() if torch.is_tensor(v)]
        tensors += [t for v in out.values() if isinstance(v, list) for t in v]
        return {t.dtype for t in tensors if t.is_floating_point()}

    def test_pruning_runs_in_the_input_dtype_5837(self, device, dtype):
        data, perm = self._scene(device, dtype)
        out = self._forward(self._model(device, dtype), data)
        ref = self._forward(self._model(device, torch.float32), self._scene(device, torch.float32)[0])

        pruned = (torch.tensor(self._MATCHABLE) < 0) & (torch.tensor(self._CONFIDENT) > 0)
        inv = torch.argsort(perm)
        kept_score = torch.tensor(10.0, dtype=torch.float64).sigmoid().item() ** 2
        self.assert_close(out["prune0"], torch.where(pruned, 1, 2)[None].to(device))
        self.assert_close(out["prune1"], torch.where(pruned[perm], 1, 2)[None].to(device))
        self.assert_close(out["matches0"], torch.where(pruned, -1, inv)[None].to(device))
        self.assert_close(out["matches1"], torch.where(pruned[perm], -1, perm)[None].to(device))
        expected_scores0 = torch.where(pruned, 0.0, kept_score)[None].to(device, dtype)
        self.assert_close(out["matching_scores0"], expected_scores0)
        self.assert_close(out["matching_scores1"], expected_scores0[:, perm])
        self.assert_close(out["scores"][0], expected_scores0[0, ~pruned.to(device)])
        assert self._floating_dtypes(out) == {dtype}
        # The same model and scene in float32, within the tolerance of the dtype.
        for key in ("matches0", "matches1", "prune0", "prune1"):
            self.assert_close(out[key], ref[key])
        for key in ("matching_scores0", "matching_scores1"):
            self.assert_close(out[key], ref[key].to(dtype))

    def test_no_descriptors_returns_the_input_dtype_5837(self, device, dtype):
        empty = {
            "keypoints": torch.empty(1, 0, 2, device=device, dtype=dtype),
            "descriptors": torch.empty(1, 0, 128, device=device, dtype=dtype),
            "image_size": torch.tensor([[320, 240]], device=device),
        }
        out = self._forward(self._model(device, dtype), {"image0": empty, "image1": empty})
        assert out["matching_scores0"].shape == (1, 0)
        assert self._floating_dtypes(out) == {dtype}

    @staticmethod
    def _skip_without_empty_attention(device, dtype):
        # The layers after the one that pruned every point still run, on empty sequences (#3564).
        qkv = torch.empty(1, 1, 0, 8, device=device, dtype=dtype)
        try:
            torch.nn.functional.scaled_dot_product_attention(qkv, qkv, qkv)
        except RuntimeError:
            pytest.skip("torch rejects attention over empty sequences on this device (MPS on torch 2.5.1)")

    def test_all_points_pruned_returns_the_input_dtype_5837(self, device, dtype):
        self._skip_without_empty_attention(device, dtype)
        # No low-confidence rescue without depth confidence, and every matchability -20.
        data, _ = self._scene(device, dtype, matchable=(-1,) * self._M)
        out = self._forward(self._model(device, dtype, depth_confidence=-1), data)
        self.assert_close(out["prune0"], torch.ones(1, self._M, device=device, dtype=torch.long))
        self.assert_close(out["matches0"], torch.full((1, self._M), -1, device=device))
        self.assert_close(out["matching_scores0"], torch.zeros(1, self._M, device=device, dtype=dtype))
        assert out["scores"][0].shape == (0,)
        assert self._floating_dtypes(out) == {dtype}

    def _mixed_precision_score_dtype(self, device, dtype):
        # With mp=True, autocast sets the dtype the scores are computed in: CPU computes the log assignment and its exp
        # in bfloat16, MPS runs that exp in float32. The model without pruning gives the reference.
        if dtype != torch.float32:
            pytest.skip("mp=True autocasts a float32 model")
        plain = self._forward(self._model(device, dtype, mp=True, width_confidence=-1), self._scene(device, dtype)[0])
        score_dtype = plain["matching_scores0"].dtype
        if device.type in ("cpu", "mps"):
            assert score_dtype == (torch.bfloat16 if device.type == "cpu" else torch.float32)
        return score_dtype

    def test_mixed_precision_prunes_in_the_score_dtype_5837(self, device, dtype):
        score_dtype = self._mixed_precision_score_dtype(device, dtype)
        data = self._scene(device, dtype)[0]
        out = self._forward(self._model(device, dtype, mp=True), data)
        ref = self._forward(self._model(device, dtype), data)
        for key in ("matches0", "matches1", "prune0", "prune1"):
            self.assert_close(out[key], ref[key])
        assert (out["prune0"] == 1).any()
        for key in ("matching_scores0", "matching_scores1"):
            assert out[key].dtype == score_dtype
            self.assert_close(out[key].to(dtype), ref[key], rtol=0.0, atol=1e-2)
        assert out["scores"][0].dtype == score_dtype
        # The early return for an image without descriptors runs no layer, and returns the input dtype.
        empty = {
            "keypoints": torch.empty(1, 0, 2, device=device, dtype=dtype),
            "descriptors": torch.empty(1, 0, 128, device=device, dtype=dtype),
            "image_size": torch.tensor([[320, 240]], device=device),
        }
        out = self._forward(self._model(device, dtype, mp=True), {"image0": empty, "image1": empty})
        assert self._floating_dtypes(out) == {dtype}

    def test_mixed_precision_all_points_pruned_returns_the_score_dtype_5837(self, device, dtype):
        score_dtype = self._mixed_precision_score_dtype(device, dtype)
        self._skip_without_empty_attention(device, dtype)
        data, _ = self._scene(device, dtype, matchable=(-1,) * self._M)
        out = self._forward(self._model(device, dtype, mp=True, depth_confidence=-1), data)
        self.assert_close(out["prune0"], torch.ones(1, self._M, device=device, dtype=torch.long))
        self.assert_close(out["matches0"], torch.full((1, self._M), -1, device=device))
        for key in ("matching_scores0", "matching_scores1"):
            assert out[key].dtype == score_dtype
            assert (out[key] == 0).all()
        assert out["scores"][0].dtype == score_dtype
        assert out["scores"][0].shape == (0,)


class _PositionalEncodingInputs(Exception):
    """Ends a forward pass once both images' keypoint encodings have been recorded."""


class TestConventionsLightGlue(BaseTester):
    # The positional-encoding inputs of image 0 and image 1 for the frames of `_matcher_lafs`, generated in float64
    # from x' = (x - W / 2) / (max(H, W) / 2), y' = (y - H / 2) / (max(H, W) / 2), scale = sqrt(|det A|),
    # orientation = atan2(a01, a00) (radians), and the normalised points c + A[:, 0] and c + A[:, 1] for keynet.
    _XY = {
        (200, 320): [[-0.625, -0.475], [0.5625, 0.4375], [-0.36875, -0.41875]],
        (240, 300): [[-0.6, -0.64], [0.666667, 0.333333], [-0.326667, -0.58]],
    }
    _SIXTH_OF_SCALE = [2.0, 1.25, 1.683251]
    _NEGATED_ORI = [-0.785398, 2.094395, -0.380506]  # in (-pi, pi]
    _ORI = [0.785398, 4.188790, 0.380506]  # in [0, 2 pi)
    _AXIS_POINTS = {
        (200, 320): [
            [-0.571967, -0.528033, -0.571967, -0.421967],
            [0.539062, 0.478095, 0.521905, 0.414062],
            [-0.30625, -0.4375, -0.34375, -0.3625],
        ],
        (240, 300): [
            [-0.543431, -0.696569, -0.543431, -0.583431],
            [0.641667, 0.376635, 0.623365, 0.308333],
            [-0.26, -0.6, -0.3, -0.52],
        ],
    }

    @staticmethod
    def _record_positional_encoding_inputs(lightglue):
        seen = []

        def hook(module, args):
            seen.append(args[0].detach().clone())
            if len(seen) == 2:
                raise _PositionalEncodingInputs

        lightglue.posenc.register_forward_pre_hook(hook)
        return seen

    @staticmethod
    def _matcher_lafs(device, dtype):
        # two rotated frames (scale 12 at 45 degrees, 7.5 at -120 degrees) and a sheared one (a01 != a10), centres
        # off the diagonal
        rotated = laf_from_center_scale_ori(
            torch.tensor([[[60.0, 24.0], [250.0, 170.0]]]),
            torch.tensor([12.0, 7.5]).view(1, 2, 1, 1),
            torch.tensor([45.0, -120.0]).view(1, 2, 1),
        )
        sheared = torch.tensor([[[[10.0, 4.0, 101.0], [-3.0, 9.0, 33.0]]]])
        return torch.cat([rotated, sheared], 1).to(device, dtype)

    def _expected_encoding_input(self, feature_name, hw, device, dtype):
        columns = [torch.tensor(self._XY[hw])]
        if feature_name == "sift":
            columns += [torch.tensor(self._SIXTH_OF_SCALE)[:, None], torch.tensor(self._NEGATED_ORI)[:, None]]
        elif feature_name in ("doghardnet", "dog_affnet_hardnet"):
            columns += [torch.tensor(self._SIXTH_OF_SCALE)[:, None], torch.tensor(self._ORI)[:, None]]
        elif feature_name == "keynet_affnet_hardnet":
            columns += [torch.tensor(self._AXIS_POINTS[hw])]
        return torch.cat(columns, 1)[None].to(device, dtype)

    def test_convention_lightglue_image_size_is_width_height(self, device, dtype):
        # Without point pruning, the batch of two pairs runs as one, and the hook sees both items of each image.
        lg = LightGlue(None, input_dim=8, descriptor_dim=8, n_layers=1, num_heads=1, flash=False, width_confidence=-1)
        lg = lg.to(device, dtype).eval()
        seen = self._record_positional_encoding_inputs(lg)
        kpts = torch.tensor([[[300.0, 30.0], [20.0, 210.0], [160.0, 120.0]]], device=device, dtype=dtype)
        desc = torch.rand(2, 3, 8, device=device, dtype=dtype)
        # image 0: batch item 1 is item 0 transposed, keypoints and (W, H) size alike, so each item needs its own size
        size = torch.tensor([[320, 240], [240, 320]], device=device)
        # image 1: an image tensor gives the size (W, H) = (320, 240) from its last two dimensions
        image = torch.zeros(2, 1, 240, 320, device=device, dtype=dtype)
        with pytest.raises(_PositionalEncodingInputs):
            lg(
                {
                    "image0": {"keypoints": torch.cat([kpts, kpts.flip(-1)]), "descriptors": desc, "image_size": size},
                    "image1": {"keypoints": torch.cat([kpts, kpts]), "descriptors": desc, "image": image},
                }
            )
        # (k - size / 2) / (max(size) / 2): x = (300 - 160) / 160, y = (30 - 120) / 160, and so on; the transposed item
        # gives the swapped coordinates
        expected = torch.tensor([[[0.875, -0.5625], [-0.875, 0.5625], [0.0, 0.0]]], device=device, dtype=dtype)
        self.assert_close(seen[0], torch.cat([expected, expected.flip(-1)]))
        self.assert_close(seen[1], torch.cat([expected, expected]))

    @pytest.mark.parametrize("feature_name", LightGlueMatcher.known_modes)
    def test_convention_lightglue_matcher_gives_each_feature_type_its_keypoint_convention(
        self, device, dtype, feature_name, monkeypatch
    ):
        # A random-weight LightGlue with the feature type's configuration: the weights are not downloaded.
        monkeypatch.setattr("kornia.feature.lightglue.load_state_dict_from_url", lambda *args, **kwargs: {})
        matcher = LightGlueMatcher(feature_name, {"n_layers": 1}).to(device, dtype).eval()
        seen = self._record_positional_encoding_inputs(matcher.matcher)
        lafs = self._matcher_lafs(device, dtype)
        perm = [2, 0, 1]
        desc = torch.rand(3, matcher.matcher.conf.input_dim, device=device, dtype=dtype)
        # hw1 and hw2 are (H, W) and apply to their own image; image 1 holds the frames in another order
        with pytest.raises(_PositionalEncodingInputs):
            matcher(desc, desc[perm], lafs, lafs[:, perm], hw1=(200, 320), hw2=(240, 300))
        # Every type gets the normalised centres. sift adds sigma of a 6-sigma frame (LAF scale / 6) and the negated
        # LAF orientation in (-pi, pi]; doghardnet and dog_affnet_hardnet add LAF scale / 6 and the LAF orientation
        # in [0, 2 pi); keynet_affnet_hardnet adds the normalised ends of the LAF axes.
        self.assert_close(seen[0], self._expected_encoding_input(feature_name, (200, 320), device, dtype))
        self.assert_close(seen[1], self._expected_encoding_input(feature_name, (240, 300), device, dtype)[:, perm])

    def test_convention_lightglue_matcher_without_hw_takes_the_largest_keypoint_coordinates(
        self, device, dtype, monkeypatch
    ):
        monkeypatch.setattr("kornia.feature.lightglue.load_state_dict_from_url", lambda *args, **kwargs: {})
        matcher = LightGlueMatcher("disk", {"n_layers": 1}).to(device, dtype).eval()
        seen = self._record_positional_encoding_inputs(matcher.matcher)
        lafs = self._matcher_lafs(device, dtype)
        desc = torch.rand(3, 128, device=device, dtype=dtype)
        with pytest.raises(_PositionalEncodingInputs):
            matcher(desc, desc[[0, 2]], lafs, lafs[:, [0, 2]])
        # The image size is (max x, max y) of each image's keypoints: (250, 170), then (101, 33).
        expected0 = torch.tensor([[[-0.52, -0.488], [1.0, 0.68], [-0.192, -0.416]]], device=device, dtype=dtype)
        expected1 = torch.tensor([[[0.188119, 0.148515], [1.0, 0.326733]]], device=device, dtype=dtype)
        self.assert_close(seen[0], expected0)
        self.assert_close(seen[1], expected1)

    def test_convention_lightglue_matcher_returns_the_matching_score_and_index_pairs(self, device, dtype, monkeypatch):
        calls = []

        class _FixedLightGlue(torch.nn.Module):
            # Stands in for the network: a fixed assignment from image 0 to image 1 and its matching scores.
            def __init__(self, *args, **kwargs):
                super().__init__()

            def forward(self, data):
                calls.append(data)
                # 0 -> 3, 2 -> 0, 3 -> 2: no pair is its own inverse, so swapped columns give other pairs
                matches0 = torch.tensor([[3, -1, 0, 2]], device=device)
                scores0 = torch.tensor([[0.9, 0.0, 0.35, 0.6]], device=device, dtype=dtype)
                return {"matches0": matches0, "matching_scores0": scores0}

        monkeypatch.setattr("kornia.feature.integrated.LightGlue", _FixedLightGlue)
        matcher = LightGlueMatcher("DISK")
        assert matcher.feature_name == "disk"  # modes are case-insensitive
        lafs = laf_from_center_scale_ori(
            torch.tensor([[[60.0, 24.0], [250.0, 170.0], [101.0, 33.0], [7.0, 140.0]]], device=device, dtype=dtype)
        )
        lafs1 = lafs.clone()
        lafs1[:, 1] = 0.0  # a padding slot
        desc = torch.rand(4, 128, device=device, dtype=dtype)
        scores, idxs = matcher(desc, desc, lafs, lafs1, hw1=(200, 320), hw2=(200, 320))
        # The first value is LightGlue's matching score (higher is better), not a distance; rows are
        # (index in desc1, index in desc2), and unmatched keypoints (-1) are left out.
        assert idxs.tolist() == [[0, 3], [2, 0], [3, 2]]
        self.assert_close(scores, torch.tensor([[0.9], [0.35], [0.6]], device=device, dtype=dtype))
        # Padding is not filtered: the zero LAF reaches the network as a keypoint at (0, 0).
        assert calls[0]["image1"]["keypoints"][0, 1].tolist() == [0.0, 0.0]
        # Fewer than two descriptors on a side give no match without running the network.
        scores, idxs = matcher(desc[:1], desc, lafs[:, :1], lafs1)
        assert scores.shape == (0, 1)
        assert idxs.shape == (0, 2)
        assert len(calls) == 1
        # A name outside known_modes is rejected.
        with pytest.raises(NotImplementedError, match="orb is not supported"):
            LightGlueMatcher("orb")

    def test_convention_lightglue_features_configuration_wins_over_keyword_arguments(self, device, dtype, monkeypatch):
        # "disk" sets input_dim, so the keyword argument loses; filter_threshold is not in its configuration, so the
        # keyword argument stays
        monkeypatch.setattr("kornia.feature.lightglue.load_state_dict_from_url", lambda *args, **kwargs: {})
        lg = LightGlue("disk", input_dim=64, filter_threshold=0.3, n_layers=1)
        assert lg.conf.input_dim == 128
        assert lg.conf.filter_threshold == 0.3


class TestLightGlueAllPrunedScoreDtype(BaseTester):
    """Every point pruned gives zero scores in the dtype of the scores of a pass that keeps points (#5877).

    ``LightGlue(None)`` with random weights, two layers, pruning at any number of keypoints and no early stop. The
    matchability of each layer reads descriptor dimension 0, which is +30 for every point, so layer 0 keeps them all,
    or -30, so it prunes them all: the same model takes both paths. ``input_dim == descriptor_dim``, as in
    ``"superpoint"``, ``"dedodeb"`` and ``"dedodeg"``, makes ``input_proj`` an identity; ``input_dim !=
    descriptor_dim``, as in ``"disk"``, ``"aliked"`` and ``"sift"``, makes it a linear layer, here wired to copy the
    descriptors.
    """

    _DIM = 16

    def _model(self, device, dtype, input_dim, mp):
        torch.manual_seed(0)
        lg = LightGlue(
            None, input_dim=input_dim, descriptor_dim=self._DIM, n_layers=2, num_heads=1, mp=mp, depth_confidence=-1
        )
        with torch.no_grad():
            if input_dim != self._DIM:
                lg.input_proj.weight.zero_()
                lg.input_proj.bias.zero_()
                lg.input_proj.weight[:input_dim].copy_(torch.eye(input_dim))
            for layer in lg.log_assignment:
                layer.matchability.weight.zero_()
                layer.matchability.bias.zero_()
                layer.matchability.weight[0, 0] = 1.0
        return lg.to(device, dtype).eval()

    @staticmethod
    def _forward(model, keep, device, dtype):
        g = torch.Generator().manual_seed(1)
        desc = torch.randn(1, 6, model.conf.input_dim, generator=g)
        desc[..., 0] = 30.0 if keep else -30.0
        image = {
            "keypoints": (torch.rand(1, 6, 2, generator=g) * 100).to(device, dtype),
            "descriptors": desc.to(device, dtype),
            "image_size": torch.tensor([[100, 100]], device=device),
        }
        out = TestLightGluePruningDtype._forward(model, {"image0": image, "image1": dict(image)})
        # Layer 0 keeps a point (it counts 2) or prunes it (1); the early return for every point pruned has no log
        # assignment.
        for key in ("prune0", "prune1"):
            assert (out[key] == (2 if keep else 1)).all()
        assert ("log_assignment" in out) == keep
        return out

    @pytest.mark.parametrize("input_dim", [16, 8], ids=["input_dim_eq_descriptor_dim", "input_dim_ne_descriptor_dim"])
    @pytest.mark.parametrize("mp", [False, True], ids=["mp_off", "mp_on"])
    def test_all_points_pruned_gives_the_score_dtype_of_kept_points_5877(self, device, dtype, input_dim, mp):
        TestLightGluePruningDtype._skip_without_empty_attention(device, dtype)
        model = self._model(device, dtype, input_dim, mp)
        kept = self._forward(model, True, device, dtype)
        pruned = self._forward(model, False, device, dtype)
        score_dtype = kept["matching_scores0"].dtype
        # The input dtype; with mp=True float32 gives the dtype autocast computes the scores in, bfloat16 on CPU and
        # float32 on MPS, and on MPS float16 and bfloat16 give float32 too.
        if not mp or (device.type == "cpu" and dtype != torch.float32):
            assert score_dtype == dtype
        elif device.type == "cpu":
            assert score_dtype == torch.bfloat16
        elif device.type == "mps":
            assert score_dtype == torch.float32
        for out in (kept, pruned):
            for key in ("matching_scores0", "matching_scores1"):
                assert out[key].dtype == score_dtype
            assert out["scores"][0].dtype == score_dtype
        self.assert_close(pruned["matches0"], torch.full((1, 6), -1, device=device))
        self.assert_close(pruned["matching_scores0"], torch.zeros(1, 6, device=device, dtype=score_dtype))
        self.assert_close(pruned["matching_scores1"], torch.zeros(1, 6, device=device, dtype=score_dtype))
        assert pruned["scores"][0].shape == (0,)

    @pytest.mark.parametrize("mp", [False, True], ids=["mp_off", "mp_on"])
    def test_no_descriptors_keeps_the_input_dtype_5877(self, device, dtype, mp):
        # No layer runs, so autocast does not reach these zero scores: on MPS, a float16 or bfloat16 model with mp=True
        # keeps its dtype here, while the scores of kept points are float32.
        empty = {
            "keypoints": torch.empty(1, 0, 2, device=device, dtype=dtype),
            "descriptors": torch.empty(1, 0, self._DIM, device=device, dtype=dtype),
            "image_size": torch.tensor([[100, 100]], device=device),
        }
        model = self._model(device, dtype, self._DIM, mp)
        out = TestLightGluePruningDtype._forward(model, {"image0": empty, "image1": dict(empty)})
        assert TestLightGluePruningDtype._floating_dtypes(out) == {dtype}

    @staticmethod
    def _record_autocast(model, device_type):
        # The autocast state the first layer runs under.
        seen = []

        def hook(module, args):
            seen.append((torch.is_autocast_enabled(device_type), torch.get_autocast_dtype(device_type)))

        model.transformers[0].register_forward_pre_hook(hook)
        return seen

    @pytest.mark.parametrize("input_dim", [16, 8], ids=["input_dim_eq_descriptor_dim", "input_dim_ne_descriptor_dim"])
    def test_convention_lightglue_mixed_precision_cpu_float16_model_autocasts_to_float16_5897(self, device, input_dim):
        if device.type != "cpu":
            pytest.skip("the rule is CPU's: MPS and CUDA autocast to float16 for every model")
        # With mp=True on CPU, a float16 model autocasts to float16, and any other model to bfloat16, CPU autocast's
        # default. Autocast's bfloat16 outputs meeting the float16 tensors of a float16 model raised in the first layer.
        model = self._model(device, torch.float16, input_dim, mp=True)
        seen = self._record_autocast(model, device.type)
        out = self._forward(model, True, device, torch.float16)
        assert seen == [(True, torch.float16)]
        ref = self._forward(self._model(device, torch.float16, input_dim, mp=False), True, device, torch.float16)
        assert TestLightGluePruningDtype._floating_dtypes(out) == {torch.float16}
        assert (ref["matches0"] > -1).any()
        for key in ("matches0", "matches1"):
            self.assert_close(out[key], ref[key])
        # Autocast normalises the keypoints in float32 and promotes concatenations with float32 tensors, so the scores
        # round differently from the plain float16 pass, within two float16 steps at 1.
        for key in ("matching_scores0", "matching_scores1"):
            assert out[key].isfinite().all()
            self.assert_close(out[key], ref[key], rtol=0.0, atol=2e-3)
        float32 = self._model(device, torch.float32, input_dim, mp=True)
        seen = self._record_autocast(float32, device.type)
        self._forward(float32, True, device, torch.float32)
        assert seen == [(True, torch.bfloat16)]

    def test_convention_lightglue_mixed_precision_cpu_float16_model_reads_lafs_5897(self, device):
        if device.type != "cpu":
            pytest.skip("the rule is CPU's: MPS and CUDA autocast to float16 for every model")
        # "keynet_affnet_hardnet" reads the LAFs before the layers, where bfloat16 autocast raised for a float16 model,
        # also for an image without descriptors.

        def keynet(mp):
            torch.manual_seed(0)
            with unittest.mock.patch("kornia.feature.lightglue.load_state_dict_from_url", return_value={}):
                return LightGlue("keynet_affnet_hardnet", n_layers=2, mp=mp).to(device, torch.float16).eval()

        g = torch.Generator().manual_seed(1)
        kpts = torch.rand(1, 6, 2, generator=g) * 80 + 10
        lafs = laf_from_center_scale_ori(
            kpts, torch.rand(1, 6, 1, 1, generator=g) * 5 + 1, torch.rand(1, 6, 1, generator=g) * 360
        )
        image = {
            "keypoints": kpts.to(device, torch.float16),
            "descriptors": torch.randn(1, 6, 128, generator=g).to(device, torch.float16),
            "image_size": torch.tensor([[100, 100]], device=device),
            "lafs": lafs.to(device, torch.float16),
        }
        model = keynet(True)
        seen = self._record_autocast(model, device.type)
        out = TestLightGluePruningDtype._forward(model, {"image0": image, "image1": dict(image)})
        assert seen == [(True, torch.float16)]
        ref = TestLightGluePruningDtype._forward(keynet(False), {"image0": image, "image1": dict(image)})
        assert TestLightGluePruningDtype._floating_dtypes(out) == {torch.float16}
        assert (ref["matches0"] > -1).any()
        for key in ("matches0", "matches1"):
            self.assert_close(out[key], ref[key])
        for key in ("matching_scores0", "matching_scores1"):  # tolerance as for the other configurations
            assert out[key].isfinite().all()
            self.assert_close(out[key], ref[key], rtol=0.0, atol=2e-3)
        empty = {
            "keypoints": torch.empty(1, 0, 2, device=device, dtype=torch.float16),
            "descriptors": torch.empty(1, 0, 128, device=device, dtype=torch.float16),
            "image_size": torch.tensor([[100, 100]], device=device),
            "lafs": torch.empty(1, 0, 2, 3, device=device, dtype=torch.float16),
        }
        out = TestLightGluePruningDtype._forward(keynet(True), {"image0": empty, "image1": dict(empty)})
        assert out["matching_scores0"].shape == (1, 0)
        assert TestLightGluePruningDtype._floating_dtypes(out) == {torch.float16}


class TestLightGlueBatchedPairs(BaseTester):
    """Each pair of a batch gives the matches it gives on its own (#5898).

    The model is the one of :class:`TestLightGluePruningDtype`: two layers, and layer 0 prunes a point of matchability
    -20 (descriptor dimension 0) unless its token confidence (dimension 1) is -20 too, which also counts against early
    stopping. Image 1 of a pair is image 0 permuted, and a point kept in both images matches its copy. Each pair has its
    own descriptors, keypoints, permutation, and matchable and confident points.
    """

    _M = 16

    def _pair(self, device, dtype, seed, matchable, confident, confident1=None, size=(320, 240)):
        # confident1, in the order of image 0, overrides the confidences that image 1 copies from image 0. The
        # keypoints lie in the (W, H) = (320, 240) top-left corner of an image of the given size.
        g = torch.Generator().manual_seed(seed)
        desc = 3 * torch.randn(1, self._M, 128, generator=g)
        desc[0, :, 0] = 20 * torch.tensor(matchable, dtype=torch.float32)
        desc[0, :, 1] = 20 * torch.tensor(confident, dtype=torch.float32)
        kpts = torch.rand(1, self._M, 2, generator=g) * torch.tensor([320.0, 240.0])
        perm = torch.randperm(self._M, generator=g)
        desc1 = desc.clone()
        if confident1 is not None:
            desc1[0, :, 1] = 20 * torch.tensor(confident1, dtype=torch.float32)
        size = torch.tensor([size], device=device)
        data = {
            "image0": {"keypoints": kpts.to(device, dtype), "descriptors": desc.to(device, dtype), "image_size": size},
            "image1": {
                "keypoints": kpts[:, perm].to(device, dtype),
                "descriptors": desc1[:, perm].to(device, dtype),
                "image_size": size,
            },
        }
        return data, perm

    @staticmethod
    def _stack(pairs):
        return {
            image: {key: torch.cat([p[image][key] for p in pairs]) for key in pairs[0][image]} for image in pairs[0]
        }

    @staticmethod
    def _forward(model, data, pruning_threshold=-1):
        # -1 prunes at any number of keypoints; a larger threshold, such as CUDA's, turns pruning off below it.
        thresholds = dict.fromkeys(LightGlue.pruning_keypoint_thresholds, pruning_threshold)
        with torch.no_grad(), unittest.mock.patch.dict(LightGlue.pruning_keypoint_thresholds, thresholds):
            return model(data)

    def _assert_pair(self, out, k, single, exact):
        # Pair k of a batch output against the output of that pair on its own.
        tol = {"rtol": 0.0, "atol": 0.0} if exact else {}
        for key in ("matches0", "matches1", "prune0", "prune1"):
            self.assert_close(out[key][k : k + 1], single[key], rtol=0.0, atol=0.0)
        self.assert_close(out["matches"][k], single["matches"][0], rtol=0.0, atol=0.0)
        for key in ("matching_scores0", "matching_scores1"):
            self.assert_close(out[key][k : k + 1], single[key], **tol)
        self.assert_close(out["scores"][k], single["scores"][0], **tol)

    def test_convention_lightglue_batch_with_point_pruning_matches_each_pair_5898(self, device, dtype):
        model = TestLightGluePruningDtype._model(device, dtype)
        matchable = [(1, -1) * 8, (1, 1, 1, -1) * 4, (1, -1) * 8]
        confident = [TestLightGluePruningDtype._CONFIDENT, (1,) * 7 + (-1,) + (1,) * 8, (1,) * 16]
        # Pair 0 prunes 6 points of each image and pair 1 prunes 3; pair 2 has no unconfident point, so early stopping
        # ends it after layer 0, before any pruning. Pair 1 has its own image size.
        sizes = [(320, 240), (400, 300), (320, 240)]
        pairs = [self._pair(device, dtype, k, matchable[k], confident[k], size=sizes[k]) for k in range(3)]
        singles = [self._forward(model, data) for data, _ in pairs]
        assert [single["stop"] for single in singles] == [2, 2, 1]
        # A batch can give one image size for all its pairs.
        shared_size = self._stack([pairs[0][0], pairs[2][0]])
        for image in shared_size.values():
            image["image_size"] = image["image_size"][:1]
        for batch, members in ((shared_size, [0, 2]), (self._stack([data for data, _ in pairs]), [0, 1, 2])):
            out = self._forward(model, batch)
            # The pairs keep different numbers of points, so they have no common log assignment, and stop is the
            # largest number of layers a pair ran.
            assert "log_assignment" not in out
            assert out["stop"] == 2
            for row, k in enumerate(members):
                self._assert_pair(out, row, singles[k], exact=True)
            # The pairs that ran both layers match each point they kept to its copy, as an index into their own input.
            for row, k in enumerate(members):
                if k == 2:
                    continue
                perm = pairs[k][1]
                pruned = (torch.tensor(matchable[k]) < 0) & (torch.tensor(confident[k]) > 0)
                self.assert_close(out["prune0"][row], torch.where(pruned, 1, 2).to(device))
                self.assert_close(out["matches0"][row], torch.where(pruned, -1, torch.argsort(perm)).to(device))
                self.assert_close(out["matches1"][row], torch.where(pruned[perm], -1, perm).to(device))
                kept = torch.where(~pruned)[0]
                self.assert_close(out["matches"][row], torch.stack([kept, torch.argsort(perm)[kept]], -1).to(device))

    def test_convention_lightglue_batch_with_a_pair_whose_points_are_all_pruned_5898(self, device, dtype):
        TestLightGluePruningDtype._skip_without_empty_attention(device, dtype)
        # Without early stopping, no low-confidence point is kept, and layer 0 prunes every point of the second pair.
        model = TestLightGluePruningDtype._model(device, dtype, depth_confidence=-1)
        pairs = [
            self._pair(device, dtype, k, matchable, TestLightGluePruningDtype._CONFIDENT)
            for k, matchable in enumerate([(1, -1) * 8, (-1,) * 16])
        ]
        singles = [self._forward(model, data) for data, _ in pairs]
        assert [("log_assignment" in single) for single in singles] == [True, False]
        out = self._forward(model, self._stack([data for data, _ in pairs]))
        assert "log_assignment" not in out
        for k in range(2):
            self._assert_pair(out, k, singles[k], exact=True)
        self.assert_close(out["matches0"][1], torch.full((self._M,), -1, device=device))
        assert out["scores"][1].shape == (0,)

    def test_convention_lightglue_batch_below_the_pruning_threshold_runs_together_5898(self, device, dtype):
        # Point pruning runs only in an image with more keypoints than the device threshold (on CUDA, 1536 by default
        # and 1024 with flash=False): below it, a batch runs as one, with its log assignment, and gives the prune0 and
        # prune1 of each pair on its own.
        model = TestLightGluePruningDtype._model(device, dtype)
        matchable = [(1, -1) * 8, (1, 1, 1, -1) * 4]
        pairs = [self._pair(device, dtype, k, matchable[k], TestLightGluePruningDtype._CONFIDENT) for k in range(2)]
        singles = [self._forward(model, data, pruning_threshold=1000) for data, _ in pairs]
        out = self._forward(model, self._stack([data for data, _ in pairs]), pruning_threshold=1000)
        assert out["log_assignment"].shape == (2, self._M + 1, self._M + 1)
        assert out["stop"] == 2
        for k in range(2):
            self._assert_pair(out, k, singles[k], exact=False)
            self.assert_close(out["log_assignment"][k : k + 1], singles[k]["log_assignment"])
        self.assert_close(out["prune0"], torch.ones(2, self._M, device=device, dtype=torch.long))

    def test_convention_lightglue_batch_stops_early_when_every_pair_meets_its_criterion_5898(self, device, dtype):
        # Without pruning, a batch runs as one. Early stopping after layer 0 needs more than 95 % of the 32 points of a
        # pair confident: one unconfident point passes, two do not. The batch stops once every pair passes.
        model = TestLightGluePruningDtype._model(device, dtype, width_confidence=-1)
        everywhere = (1,) * self._M
        pairs = [
            self._pair(device, dtype, k, everywhere, unconfident, everywhere)
            for k, unconfident in enumerate(
                [(1,) * 3 + (-1,) + (1,) * 12, (1,) * 5 + (-1,) + (1,) * 10, (-1, -1) + (1,) * 14]
            )
        ]
        singles = [self._forward(model, data) for data, _ in pairs]
        assert [single["stop"] for single in singles] == [1, 1, 2]
        out = self._forward(model, self._stack([pairs[0][0], pairs[1][0]]))
        assert out["stop"] == 1
        for k in range(2):
            self._assert_pair(out, k, singles[k], exact=False)
        # A pair runs as many layers as its batch: pair 0 stops after layer 0 on its own, and after layer 1 next to a
        # pair that does not stop.
        out = self._forward(model, self._stack([pairs[0][0], pairs[2][0]]))
        assert out["stop"] == 2
        self._assert_pair(out, 1, singles[2], exact=False)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_convention_lightglue_input_without_keypoints_reads_no_pruning_threshold_5898(
        self, device, dtype, batch_size
    ):
        # The pruning threshold of the device is read only once there are keypoints, so an input without any returns
        # empty matches also on a device type without a threshold.
        empty = {
            "keypoints": torch.empty(batch_size, 0, 2, device=device, dtype=dtype),
            "descriptors": torch.empty(batch_size, 0, 128, device=device, dtype=dtype),
            "image_size": torch.tensor([[320, 240]] * batch_size, device=device),
        }
        model = TestLightGluePruningDtype._model(device, dtype)
        with torch.no_grad(), unittest.mock.patch.dict(LightGlue.pruning_keypoint_thresholds, clear=True):
            out = model({"image0": empty, "image1": dict(empty)})
        assert out["matches0"].shape == (batch_size, 0)
        assert [tuple(m.shape) for m in out["matches"]] == [(0, 2)] * batch_size
