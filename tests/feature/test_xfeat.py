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

import pytest
import torch

from kornia.feature.lightglue import LightGlue
from kornia.feature.xfeat import InterpolateSparse2d, XFeat, XFeatModel

from testing.base import (
    BaseTester,
    supports_bicubic_2d_grid_sample,
    supports_bilinear_2d_grid_sample,
    supports_nearest_2d_grid_sample,
    supports_replicate_padding,
)

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
        if dtype in (torch.float32, torch.float64):
            self.assert_close(feats_c, feats, rtol=1e-3, atol=1e-4)
            self.assert_close(kpts_c, kpts, rtol=1e-3, atol=1e-4)
            self.assert_close(hm_c, hm, rtol=1e-3, atol=1e-4)
        else:
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
        if not supports_bilinear_2d_grid_sample(device, dtype):
            pytest.skip("This device does not support bilinear interpolation for 2D grid_sample")
        interp = InterpolateSparse2d("bilinear").to(device)
        x = torch.rand(1, 32, 8, 8, device=device, dtype=dtype)
        pos = torch.zeros(1, 5, 2, device=device, dtype=dtype)
        out = interp(x, pos, 8, 8)
        assert out.shape == (1, 5, 32)

    @pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
    def test_cardinality(self, mode, device, dtype):
        if mode == "bilinear" and not supports_bilinear_2d_grid_sample(device, dtype):
            pytest.skip("This device does not support bilinear interpolation for 2D grid_sample")
        if mode == "bicubic" and not supports_bicubic_2d_grid_sample(device, dtype):
            pytest.skip("This device does not support bicubic interpolation for 2D grid_sample")
        interp = InterpolateSparse2d(mode).to(device)
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
        assert (kpts[:, 0] >= 0).all()
        assert (kpts[:, 0] < W).all()
        assert (kpts[:, 1] >= 0).all()
        assert (kpts[:, 1] < H).all()

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

    def test_pretrained_uses_kornia_downloader(self, monkeypatch):
        # Going through kornia.core.download keeps torch.hub's progress line off
        # stdout, which a bare torch.hub call would leak into doctests (#4005).
        calls = []

        def fake_download(url, **kwargs):
            calls.append((url, kwargs))
            return XFeat().net.state_dict()

        monkeypatch.setattr("kornia.feature.xfeat.load_state_dict_from_url", fake_download)
        model = XFeat.from_pretrained()

        assert calls == [(XFeat.weights_url, {"file_name": "xfeat.pt"})]
        assert not model.net.training


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
# Reference-data tests  (kornia/data_test  xfeat_reference.safetensors)
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

    # At least 99 % of computed keypoints must lie within 3 px of a reference keypoint
    assert _nn_match_fraction(out1["keypoints"], ref_kpts0, max_dist=3.0) > 0.99
    assert _nn_match_fraction(out2["keypoints"], ref_kpts1, max_dist=3.0) > 0.99
    # Symmetric: reference keypoints are also covered by computed set
    assert _nn_match_fraction(ref_kpts0, out1["keypoints"], max_dist=3.0) > 0.99
    assert _nn_match_fraction(ref_kpts1, out2["keypoints"], max_dist=3.0) > 0.99


@pytest.mark.slow
@pytest.mark.parametrize("data", ["xfeat_outdoor"], indirect=True)
def test_lighterglue_reference_matches(device, data):
    """LighterGlue matched pixel-coordinate pairs overlap ≥95 % with reference (order-independent)."""
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

    # NN matching in joint (x0,y0,x1,y1) space with max L2 distance 2.0
    frac = _nn_match_fraction(ref_pairs, comp_pairs, max_dist=2.0)
    assert frac > 0.95, f"Only {frac:.1%} of reference matches reproduced within joint distance 2.0 px"


class TestConventionsXFeat(BaseTester):
    @staticmethod
    def _net_outputs(peaks, h, w, reliability=0.5):
        # What XFeatModel returns for an h x w (resized) image, replaced through a forward hook: keypoint logits with
        # the dustbin (channel 64) at 10 and the given peaks (x, y, logit) in channel (y % 8) * 8 + x % 8 of cell
        # (y // 8, x // 8), a constant reliability map, and dense features (column, row, 1, 0, ...) per cell.
        logits = torch.zeros(1, 65, h // 8, w // 8)
        logits[0, 64] = 10.0
        for x, y, logit in peaks:
            logits[0, (y % 8) * 8 + x % 8, y // 8, x // 8] = logit
        yy, xx = torch.meshgrid(torch.arange(h // 8.0), torch.arange(w // 8.0), indexing="ij")
        feats = torch.zeros(1, 64, h // 8, w // 8)
        feats[0, 0], feats[0, 1], feats[0, 2] = xx, yy, torch.ones_like(xx)
        return feats, logits, torch.full((1, 1, h // 8, w // 8), reliability)

    def _detect(self, image, outputs, **kwargs):
        # detectAndCompute samples the descriptors bicubically, in float32 whatever the input dtype
        if not supports_bicubic_2d_grid_sample(image.device, torch.float32):
            pytest.skip("This device does not support bicubic interpolation for 2D grid_sample")
        xfeat = XFeat(**kwargs).to(image.device).eval()
        seen = []

        def hook(module, inputs, output):
            seen.append(inputs[0])
            return tuple(t.to(image.device) for t in outputs)

        handle = xfeat.net.register_forward_hook(hook)
        try:
            (out,) = xfeat.detectAndCompute(image)
        finally:
            handle.remove()
        return out, seen[0]

    def test_convention_xfeat_keypoints_are_pixels_of_the_resized_grid_times_the_resize_ratio(self, device, dtype):
        # detectAndCompute resizes to (32 floor(H / 32), 32 floor(W / 32)), takes integer heatmap pixels i there and
        # reports i * (W / W', H / H') (the formula of verlab/accelerated_features e92685f): kornia's pixel frame when H
        # and W are multiples of 32. The score is the keypoint heatmap value times the reliability, keypoints come
        # sorted by it, and outputs are float32 whatever the input dtype. Heatmap value of a peak with logit v:
        # exp(v) / (exp(v) + exp(10) + 63), the softmax over the 65 channels.
        peaks = [(22, 13, 11.0), (70, 41, 12.0)]  # in-cell offsets (6, 5) and (6, 1): x % 8 != y % 8
        heat = [math.exp(v) / (math.exp(v) + math.exp(10.0) + 63.0) for _, _, v in peaks]
        expected_scores = torch.tensor([0.5 * heat[1], 0.5 * heat[0]], device=device)  # the stronger, later in raster
        for h, w, peaks_i, expected in (
            (64, 96, peaks, [[70.0, 41.0], [22.0, 13.0]]),
            (96, 64, [(y, x, v) for x, y, v in peaks], [[41.0, 70.0], [13.0, 22.0]]),  # relabelling: transpose
        ):
            image = torch.rand(1, 3, h, w, device=device, dtype=dtype)
            out, seen = self._detect(image, self._net_outputs(peaks_i, h, w))
            assert seen.shape == (1, 3, h, w)
            assert out["keypoints"].dtype == out["scores"].dtype == out["descriptors"].dtype == torch.float32
            assert out["keypoints"].tolist() == expected
            self.assert_close(out["scores"], expected_scores)
            self.assert_close(out["descriptors"].norm(dim=-1), torch.ones(2, device=device))
        # detection_threshold bounds the heatmap value (0.8805 and 0.7305 here), not the score (half of it)
        image = torch.rand(1, 3, 64, 96, device=device, dtype=dtype)
        for threshold, expected in ((0.5, [[70.0, 41.0], [22.0, 13.0]]), (0.8, [[70.0, 41.0]])):
            out, _ = self._detect(image, self._net_outputs(peaks, 64, 96), detection_threshold=threshold)
            assert out["keypoints"].tolist() == expected
        # 100 x 150 is resized to 96 x 128: resized pixel (22, 13) is reported at (22 * 150 / 128, 13 * 100 / 96); its
        # centre in the input's pixel frame would be (22.5 * 150 / 128 - 0.5, 13.5 * 100 / 96 - 0.5).
        image = torch.rand(1, 3, 100, 150, device=device, dtype=dtype)
        out, seen = self._detect(image, self._net_outputs([(22, 13, 12.0)], 96, 128))
        assert seen.shape == (1, 3, 96, 128)
        expected = torch.tensor([[22 * 150 / 128, 13 * 100 / 96]], device=device)
        self.assert_close(out["keypoints"], expected)

    def test_convention_xfeat_dense_keypoints_are_the_top_left_pixels_of_their_8_pixel_cells(self, device, dtype):
        # detectAndComputeDense keeps the cells of highest reliability and reports cell (j, i) at (8 j, 8 i) times the
        # resize ratio: the cell's top-left pixel, not its centre (8 j + 3.5, 8 i + 3.5); the formula is create_xy * 8
        # of verlab/accelerated_features e92685f. Cells (2, 1) and (8, 5) of a 64 x 96 image.
        xfeat = XFeat().to(device).eval()
        for h, w, cells, expected in (
            (64, 96, [(2, 1), (8, 5)], [[16.0, 8.0], [64.0, 40.0]]),
            (96, 64, [(1, 2), (5, 8)], [[8.0, 16.0], [40.0, 64.0]]),  # relabelling: transpose
        ):
            feats, logits, reliability = self._net_outputs([], h, w, reliability=0.1)
            (j0, i0), (j1, i1) = cells
            reliability[0, 0, i0, j0], reliability[0, 0, i1, j1] = 0.9, 0.8
            outputs = tuple(t.to(device) for t in (feats, logits, reliability))
            handle = xfeat.net.register_forward_hook(lambda m, i, o, outputs=outputs: outputs)
            try:
                out = xfeat.detectAndComputeDense(torch.rand(1, 3, h, w, device=device, dtype=dtype), 2, False)
            finally:
                handle.remove()
            assert out["keypoints"].tolist() == [expected]
            assert out["descriptors"].shape == (1, 2, 64)
            # the descriptors are the dense features of those cells, (j, i, 1, 0, ...), not normalised
            assert out["descriptors"][0, :, :3].tolist() == [[float(j0), float(i0), 1.0], [float(j1), float(i1), 1.0]]

    def test_convention_xfeat_model_reads_the_channel_mean_through_an_instance_norm(self, device, dtype):
        # XFeatModel averages the channels and instance-normalises the result (model.py of verlab/accelerated_features
        # e92685f), so any channel count is accepted and an affine change a x + b of the intensities with a > 0 leaves
        # the outputs unchanged, up to the norm's eps.
        tol = {torch.float16: 2e-3, torch.bfloat16: 6e-3}.get(dtype, 1e-4)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = XFeatModel().eval()
            rgb = torch.rand(1, 3, 72, 104)
        model = model.to(device, dtype)
        rgb = rgb.to(device, dtype)
        gray = rgb.mean(dim=1, keepdim=True)
        with torch.no_grad():
            reference = model(rgb)
            for variant in (gray, gray.expand(1, 5, 72, 104), rgb.flip(1)):
                for out, ref in zip(model(variant), reference):
                    self.assert_close(out, ref, rtol=0.0, atol=1e-6 if tol == 1e-4 else tol)
            # torch 2.5.1's CPU float16 instance norm overflows on a [0, 255] image, so float16 scales by 16 instead
            scale = 16.0 if dtype == torch.float16 else 255.0
            for variant in (2.0 * rgb + 1.0, scale * rgb):
                for out, ref in zip(model(variant), reference):
                    self.assert_close(out, ref, rtol=0.0, atol=tol)
            # not every change is invisible: squaring the image moves the keypoint logits
            assert (model(rgb**2)[1] - reference[1]).abs().max() > 0.02

    @pytest.mark.parametrize("mode", ["bilinear", "nearest"])
    def test_convention_interpolate_sparse2d_reads_position_p_at_map_coordinate_p_times_map_size_over_size_minus_one(
        self, device, dtype, mode
    ):
        # pos (x, y) on an H x W image reads the h x w map at (x * w / (W - 1) - 0.5, y * h / (H - 1) - 0.5) in the
        # map's pixel frame: normgrid's 2 p / (W - 1) - 1 sampled with align_corners=False (the formula of
        # verlab/accelerated_features e92685f, InterpolateSparse2d). align_corners is stored and not used. A ramp map
        # reads back that coordinate bilinearly, and its nearest pixel with mode="nearest"; the default bicubic kernel
        # does not reproduce a ramp exactly, but reads at the same coordinate. Half precision is checked to one unit in
        # the last place at the largest value read (below 8).
        if mode == "bilinear" and not supports_bilinear_2d_grid_sample(device, dtype):
            pytest.skip("This device does not support bilinear interpolation for 2D grid_sample")
        if mode == "nearest" and not supports_nearest_2d_grid_sample(device, dtype):
            pytest.skip("This device does not support nearest interpolation for 2D grid_sample")
        h, w, H, W = 8, 12, 64, 96
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij"
        )
        ramp = torch.stack([xx, yy])[None]  # channel 0: column, channel 1: row
        pos = torch.tensor([[[22.0, 13.0], [60.0, 41.0], [33.25, 20.75]]], dtype=torch.float64)
        read = pos * torch.tensor([w / (W - 1), h / (H - 1)], dtype=torch.float64) - 0.5
        # read[0] = [[2.278947, 1.150794], [7.078947, 4.706349], [3.7, 2.134921]]
        expected = read if mode == "bilinear" else read.round()
        tol = {torch.float16: 2.0**-8, torch.bfloat16: 2.0**-5}.get(dtype, 1e-4)
        for align_corners in (False, True):
            interp = InterpolateSparse2d(mode, align_corners=align_corners)
            out = interp(ramp.to(device, dtype), pos.to(device, dtype), H, W)
            self.assert_close(out.cpu().double(), expected, rtol=0.0, atol=tol)
            # relabelling: the transposed map and swapped positions read the swapped coordinates
            out_t = interp(ramp.flip(1).transpose(-1, -2).to(device, dtype), pos.flip(-1).to(device, dtype), W, H)
            self.assert_close(out_t.cpu().double(), expected.flip(-1), rtol=0.0, atol=tol)

    @pytest.mark.parametrize("sampler", ["InterpolateSparse2d", "ALIKED", "SOLD2"])
    def test_wart_samplers_send_cpu_half_maps_to_the_half_grid_sample_5810(self, device, dtype, sampler, monkeypatch):
        # torch's CPU float16 / bfloat16 grid_sample returns wrong values, NaN and overflow on sparse grids into larger
        # maps (#5810), and torch 2.5.1 has no such kernel. XFeat's InterpolateSparse2d, ALIKED (DKD scores and SDDH
        # descriptors) and SOLD2.match call it with the half map and grid directly;
        # kornia.core.utils._torch_grid_sample, which the warps use, samples CPU half in float32 instead. The recorder
        # below computes in float32 so that the pipeline runs on every torch.
        if device.type != "cpu" or dtype not in (torch.float16, torch.bfloat16):
            pytest.skip("the defect is in the CPU float16 / bfloat16 path")
        if sampler == "ALIKED" and not supports_replicate_padding(device, dtype):
            pytest.skip("ALIKED pads with replication, absent for this dtype")
        from kornia.feature import ALIKED, SOLD2

        grid_sample = torch.nn.functional.grid_sample
        calls = []

        def recorder(input, grid, *args, **kwargs):
            calls.append((input.dtype, grid.dtype))
            return grid_sample(input.float(), grid.float(), *args, **kwargs).to(input.dtype)

        monkeypatch.setattr(torch.nn.functional, "grid_sample", recorder)
        with torch.no_grad():
            if sampler == "InterpolateSparse2d":
                InterpolateSparse2d()(
                    torch.rand(1, 8, 8, 12, dtype=dtype), torch.tensor([[[22.0, 13.0]]], dtype=dtype), 64, 96
                )
            elif sampler == "ALIKED":
                ALIKED(model_name="aliked-t16").to(dtype=dtype).eval()(torch.rand(1, 3, 64, 96, dtype=dtype))
            else:
                segments = torch.tensor([[[10.0, 10.0], [50.0, 90.0]], [[20.0, 70.0], [55.0, 20.0]]])
                maps = torch.rand(2, 1, 128, 24, 32, dtype=dtype)
                SOLD2(pretrained=False).match(segments, segments + 1.0, maps[0], maps[1])
        assert calls
        assert set(calls) == {(dtype, dtype)}
