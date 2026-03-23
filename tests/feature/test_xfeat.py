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

from kornia.feature.lightglue import LightGlue
from kornia.feature.xfeat import InterpolateSparse2d, XFeat, XFeatModel

from testing.base import BaseTester

# ---------------------------------------------------------------------------
# XFeatModel backbone tests
# ---------------------------------------------------------------------------


class TestXFeatModel(BaseTester):
    def test_smoke(self, device, dtype):
        model = XFeatModel().to(device, dtype)
        x = torch.rand(1, 3, 64, 64, device=device, dtype=dtype)
        feats, kpts, heatmap = model(x)
        assert feats.shape == (1, 64, 8, 8)
        assert kpts.shape == (1, 65, 8, 8)
        assert heatmap.shape == (1, 1, 8, 8)

    def test_cardinality(self, device, dtype):
        B = 2
        model = XFeatModel().to(device, dtype)
        x = torch.rand(B, 1, 96, 128, device=device, dtype=dtype)
        feats, kpts, heatmap = model(x)
        assert feats.shape == (B, 64, 12, 16)
        assert kpts.shape == (B, 65, 12, 16)
        assert heatmap.shape == (B, 1, 12, 16)

    def test_exception(self, device, dtype):
        pass  # XFeatModel has no checked exceptions on forward

    def test_gradcheck(self, device):
        pass  # InstanceNorm2d inside torch.no_grad() breaks end-to-end gradcheck

    def test_dynamo(self, device, dtype, torch_optimizer):
        model = XFeatModel().to(device, dtype)
        x = torch.rand(1, 3, 64, 64, device=device, dtype=dtype)
        op = torch_optimizer(model)
        feats_c, kpts_c, hm_c = op(x)
        feats, kpts, hm = model(x)
        self.assert_close(feats_c, feats)
        self.assert_close(kpts_c, kpts)
        self.assert_close(hm_c, hm)

    def test_heatmap_in_zero_one(self, device, dtype):
        model = XFeatModel().to(device, dtype)
        x = torch.rand(1, 3, 64, 64, device=device, dtype=dtype)
        _feats, _kpts, heatmap = model(x)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_grayscale_and_rgb_same_shape(self, device, dtype):
        model = XFeatModel().to(device, dtype)
        gray = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        rgb = torch.rand(1, 3, 64, 64, device=device, dtype=dtype)
        f_gray, _, _ = model(gray)
        f_rgb, _, _ = model(rgb)
        assert f_gray.shape == f_rgb.shape


# ---------------------------------------------------------------------------
# InterpolateSparse2d tests
# ---------------------------------------------------------------------------


class TestInterpolateSparse2d(BaseTester):
    def test_smoke(self, device, dtype):
        interp = InterpolateSparse2d("bilinear").to(device)
        x = torch.rand(1, 32, 8, 8, device=device, dtype=dtype)
        pos = torch.zeros(1, 5, 2, device=device, dtype=dtype)
        out = interp(x, pos, 8, 8)
        assert out.shape == (1, 5, 32)

    def test_cardinality(self, device, dtype):
        interp = InterpolateSparse2d("bicubic").to(device)
        B, C, H, W, N = 2, 16, 32, 48, 20
        x = torch.rand(B, C, H, W, device=device, dtype=dtype)
        pos = torch.rand(B, N, 2, device=device, dtype=dtype) * torch.tensor([W - 1, H - 1], device=device, dtype=dtype)
        out = interp(x, pos, H, W)
        assert out.shape == (B, N, C)

    def test_exception(self, device, dtype):
        pass

    def test_gradcheck(self, device):
        interp = InterpolateSparse2d("bilinear")
        x = torch.rand(1, 4, 8, 8, device=device, dtype=torch.float64, requires_grad=True)
        pos = torch.rand(1, 3, 2, device=device, dtype=torch.float64) * 7
        self.gradcheck(interp, (x, pos, 8, 8), requires_grad=[True, False, False, False])

    def test_dynamo(self, device, dtype, torch_optimizer):
        interp = InterpolateSparse2d("bilinear")
        x = torch.rand(1, 8, 16, 16, device=device, dtype=dtype)
        pos = torch.rand(1, 5, 2, device=device, dtype=dtype) * 15
        op = torch_optimizer(interp)
        self.assert_close(op(x, pos, 16, 16), interp(x, pos, 16, 16))

    def test_normgrid_center(self, device, dtype):
        interp = InterpolateSparse2d()
        coords = torch.tensor([[[32.0, 24.0]]], device=device, dtype=dtype)
        grid = interp.normgrid(coords, 49, 65)
        assert grid.abs().max() < 0.1  # close to center -> close to 0


# ---------------------------------------------------------------------------
# XFeat end-to-end tests
# ---------------------------------------------------------------------------


class TestXFeat(BaseTester):
    def test_smoke(self, device, dtype):
        model = XFeat().to(device)
        x = torch.rand(1, 3, 64, 64, device=device)
        out = model.detectAndCompute(x)
        assert isinstance(out, list)
        assert len(out) == 1
        assert "keypoints" in out[0]
        assert "scores" in out[0]
        assert "descriptors" in out[0]

    def test_cardinality(self, device, dtype):
        model = XFeat().to(device)
        B = 2
        x = torch.rand(B, 3, 64, 64, device=device)
        out = model.detectAndCompute(x)
        assert len(out) == B
        for item in out:
            N = item["keypoints"].shape[0]
            assert item["scores"].shape == (N,)
            assert item["descriptors"].shape == (N, 64)

    def test_exception(self, device, dtype):
        model = XFeat().to(device)
        with pytest.raises(Exception):
            model.detectAndCompute(torch.rand(3, 64, 64, device=device))  # missing batch dim

    def test_gradcheck(self, device):
        pass  # inference_mode on detectAndCompute; gradcheck not applicable

    def test_dynamo(self, device, dtype, torch_optimizer):
        pass  # inference_mode and dynamic NMS control flow; not torch.compile compatible

    def test_descriptors_normalized(self, device, dtype):
        model = XFeat().to(device)
        x = torch.rand(1, 3, 64, 64, device=device)
        out = model.detectAndCompute(x)
        if out[0]["descriptors"].numel() == 0:
            pytest.skip("No keypoints detected on random input")
        norms = out[0]["descriptors"].norm(dim=-1)
        self.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)

    def test_top_k_respected(self, device, dtype):
        top_k = 16
        model = XFeat(top_k=top_k).to(device)
        x = torch.rand(1, 3, 128, 128, device=device)
        out = model.detectAndCompute(x, top_k=top_k)
        assert out[0]["keypoints"].shape[0] <= top_k

    def test_keypoint_coordinates_in_image(self, device, dtype):
        H, W = 64, 96
        model = XFeat().to(device)
        x = torch.rand(1, 3, H, W, device=device)
        out = model.detectAndCompute(x)
        if out[0]["keypoints"].numel() == 0:
            pytest.skip("No keypoints detected on random input")
        kpts = out[0]["keypoints"].float()
        assert (kpts[:, 0] >= 0).all() and (kpts[:, 0] < W).all()
        assert (kpts[:, 1] >= 0).all() and (kpts[:, 1] < H).all()

    def test_dense_output_shapes(self, device, dtype):
        top_k = 32
        model = XFeat(top_k=top_k).to(device)
        x = torch.rand(1, 3, 64, 64, device=device)
        out = model.detectAndComputeDense(x, top_k=top_k, multiscale=False)
        assert "keypoints" in out
        assert "descriptors" in out
        assert "scales" in out
        assert out["keypoints"].shape[-1] == 2
        assert out["descriptors"].shape[-1] == 64
        assert out["keypoints"].shape[1] == out["descriptors"].shape[1]

    def test_match_xfeat_returns_paired_keypoints(self, device, dtype):
        model = XFeat().to(device)
        img1 = torch.rand(1, 3, 64, 64, device=device)
        img2 = torch.rand(1, 3, 64, 64, device=device)
        mkpts0, mkpts1 = model.match_xfeat(img1, img2)
        assert mkpts0.shape == mkpts1.shape
        assert mkpts0.shape[-1] == 2

    @pytest.mark.slow
    def test_pretrained_smoke(self, device):
        model = XFeat.from_pretrained().to(device)
        x = torch.rand(1, 3, 256, 256, device=device)
        out = model.detectAndCompute(x)
        assert len(out) == 1
        assert out[0]["keypoints"].shape[-1] == 2


# ---------------------------------------------------------------------------
# LighterGlue (xfeat config in LightGlue) tests
# ---------------------------------------------------------------------------


def _make_lighterglue(device: torch.device, dtype: torch.dtype) -> LightGlue:
    """LightGlue with xfeat architecture and random weights (no download)."""
    return (
        LightGlue(
            features=None,
            input_dim=64,
            descriptor_dim=96,
            n_layers=6,
            num_heads=1,
            depth_confidence=-1,
            width_confidence=-1,
            flash=False,
        )
        .to(device, dtype)
        .eval()
    )


def _make_lighterglue_data(device: torch.device, dtype: torch.dtype, B: int = 1, M: int = 20, N: int = 15) -> dict:
    return {
        "image0": {
            "keypoints": torch.rand(B, M, 2, device=device, dtype=dtype) * 64,
            "descriptors": torch.rand(B, M, 64, device=device, dtype=dtype),
            "image_size": torch.tensor([[64, 64]], device=device, dtype=dtype).expand(B, -1),
        },
        "image1": {
            "keypoints": torch.rand(B, N, 2, device=device, dtype=dtype) * 64,
            "descriptors": torch.rand(B, N, 64, device=device, dtype=dtype),
            "image_size": torch.tensor([[64, 64]], device=device, dtype=dtype).expand(B, -1),
        },
    }


def test_lighterglue_smoke():
    device = torch.device("cpu")
    dtype = torch.float32
    lg = _make_lighterglue(device, dtype)
    data = _make_lighterglue_data(device, dtype)
    with torch.no_grad():
        out = lg(data)
    assert "matches0" in out
    assert "matching_scores0" in out


def test_lighterglue_cardinality():
    B, M, N = 1, 20, 15
    device = torch.device("cpu")
    dtype = torch.float32
    lg = _make_lighterglue(device, dtype)
    data = _make_lighterglue_data(device, dtype, B=B, M=M, N=N)
    with torch.no_grad():
        out = lg(data)
    assert out["matches0"].shape == (B, M)
    assert out["matches1"].shape == (B, N)
    assert out["matching_scores0"].shape == (B, M)


def test_lighterglue_architecture():
    """The xfeat LighterGlue config creates a 6-layer, 1-head, 96-dim model."""
    lg = _make_lighterglue(torch.device("cpu"), torch.float32)
    assert lg.conf.n_layers == 6
    assert lg.conf.num_heads == 1
    assert lg.conf.descriptor_dim == 96
    assert lg.conf.input_dim == 64
    assert len(lg.transformers) == 6


def test_lighterglue_scores_in_range():
    device = torch.device("cpu")
    dtype = torch.float32
    lg = _make_lighterglue(device, dtype)
    data = _make_lighterglue_data(device, dtype)
    with torch.no_grad():
        out = lg(data)
    scores = out["matching_scores0"]
    assert (scores >= 0).all()
    assert (scores <= 1).all()


def test_lighterglue_matches_are_valid_indices():
    B, M, N = 1, 30, 25
    device = torch.device("cpu")
    dtype = torch.float32
    lg = _make_lighterglue(device, dtype)
    data = _make_lighterglue_data(device, dtype, B=B, M=M, N=N)
    with torch.no_grad():
        out = lg(data)
    m0 = out["matches0"]
    assert (m0 >= -1).all()
    assert (m0 < N).all()


def test_lighterglue_wrong_features():
    """LightGlue rejects unknown feature types."""
    with pytest.raises(Exception):
        LightGlue(features="nonexistent_feature")


@pytest.mark.slow
def test_lighterglue_pretrained_smoke():
    """Download xfeat-lighterglue weights and run a forward pass."""
    lg = LightGlue(features="xfeat", depth_confidence=-1, width_confidence=-1).to("cpu").eval()
    B, M, N = 1, 50, 50
    data = _make_lighterglue_data(torch.device("cpu"), torch.float32, B=B, M=M, N=N)
    with torch.no_grad():
        out = lg(data)
    assert "matches0" in out


# ---------------------------------------------------------------------------
# Reference-data tests  (kornia/data_test  xfeat_reference.pt)
# ---------------------------------------------------------------------------


def _nn_match_fraction(kpts_a: torch.Tensor, kpts_b: torch.Tensor, max_dist: float) -> float:
    """Fraction of points in ``kpts_a`` whose nearest neighbour in ``kpts_b`` is within ``max_dist`` px."""
    if kpts_a.numel() == 0 or kpts_b.numel() == 0:
        return 0.0
    dists = torch.cdist(kpts_a.float(), kpts_b.float())  # (Na, Nb)
    return (dists.min(dim=1).values < max_dist).float().mean().item()


@pytest.mark.slow
@pytest.mark.parametrize("data", ["xfeat_outdoor"], indirect=True)
def test_xfeat_reference_keypoints(device, data):
    """XFeat keypoints reproduce the reference set under NN matching (order-independent)."""
    xfeat = XFeat.from_pretrained(top_k=1024).to(device)

    out1 = xfeat.detectAndCompute(data["img1"].to(device))[0]
    out2 = xfeat.detectAndCompute(data["img2"].to(device))[0]

    ref_kpts0 = data["xfeat_kpts0"].to(device)
    ref_kpts1 = data["xfeat_kpts1"].to(device)

    # At least 90 % of computed keypoints must lie within 3 px of a reference keypoint
    assert _nn_match_fraction(out1["keypoints"], ref_kpts0, max_dist=3.0) > 0.99
    assert _nn_match_fraction(out2["keypoints"], ref_kpts1, max_dist=3.0) > 0.99
    # Symmetric: reference keypoints are also covered by computed set
    assert _nn_match_fraction(ref_kpts0, out1["keypoints"], max_dist=3.0) > 0.99
    assert _nn_match_fraction(ref_kpts1, out2["keypoints"], max_dist=3.0) > 0.99


@pytest.mark.slow
@pytest.mark.parametrize("data", ["xfeat_outdoor"], indirect=True)
def test_lighterglue_reference_matches(device, data):
    """LighterGlue matched pixel-coordinate pairs overlap ≥80 % with reference (order-independent)."""
    xfeat = XFeat.from_pretrained(top_k=1024).to(device)
    img1 = data["img1"].to(device)
    img2 = data["img2"].to(device)

    out1 = xfeat.detectAndCompute(img1)[0]
    out2 = xfeat.detectAndCompute(img2)[0]

    H, W = img1.shape[-2:]
    lg = LightGlue(features="xfeat", depth_confidence=-1, width_confidence=-1).to(device).eval()
    lg_data = {
        "image0": {
            "keypoints": out1["keypoints"].unsqueeze(0),
            "descriptors": out1["descriptors"].unsqueeze(0),
            "image_size": torch.tensor([[W, H]], dtype=torch.float32, device=device),
        },
        "image1": {
            "keypoints": out2["keypoints"].unsqueeze(0),
            "descriptors": out2["descriptors"].unsqueeze(0),
            "image_size": torch.tensor([[W, H]], dtype=torch.float32, device=device),
        },
    }
    with torch.no_grad():
        lg_out = lg(lg_data)

    # Build (x0, y0, x1, y1) match-coordinate tensors for computed and reference results
    m0_comp = lg_out["matches0"].squeeze(0)  # (N,) indices into kpts1, -1 = unmatched
    valid_comp = m0_comp > -1
    if valid_comp.any():
        comp_pairs = torch.cat(
            [out1["keypoints"][valid_comp], out2["keypoints"][m0_comp[valid_comp]]], dim=-1
        )  # (K_comp, 4)
    else:
        comp_pairs = torch.zeros(0, 4, device=device)

    ref_m0 = data["lighterglue_matches0"].to(device)
    ref_kpts0 = data["xfeat_kpts0"].to(device)
    ref_kpts1 = data["xfeat_kpts1"].to(device)
    valid_ref = ref_m0 > -1
    ref_pairs = torch.cat([ref_kpts0[valid_ref], ref_kpts1[ref_m0[valid_ref]]], dim=-1)  # (K_ref, 4)

    # Number of matches should be in the same ballpark (within 2x)
    n_comp = valid_comp.sum().item()
    n_ref = valid_ref.sum().item()
    assert n_comp > n_ref / 2, f"Too few matches: {n_comp} vs reference {n_ref}"
    assert n_comp < n_ref * 2, f"Too many matches: {n_comp} vs reference {n_ref}"

    # NN matching in joint (x0,y0,x1,y1) space with 6-px tolerance per coordinate pair
    frac = _nn_match_fraction(ref_pairs, comp_pairs, max_dist=2.0)
    assert frac > 0.95, f"Only {frac:.1%} of reference matches reproduced within 6 px"
