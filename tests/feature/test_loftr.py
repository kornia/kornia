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

import copy
import math
import sys

import pytest
import torch
import torch.nn.functional as F

from kornia.feature import LoFTR
from kornia.feature.loftr import loftr as loftr_module
from kornia.feature.loftr.backbone import ResNetFPN_8_2
from kornia.feature.loftr.loftr import default_cfg
from kornia.feature.loftr.loftr_module import FinePreprocess
from kornia.feature.loftr.loftr_module.linear_attention import LinearAttention
from kornia.feature.loftr.utils.coarse_matching import CoarseMatching
from kornia.feature.loftr.utils.fine_matching import FineMatching
from kornia.geometry import resize, transform_points, warp_perspective

from testing.base import BaseTester
from testing.casts import dict_to


def _loftr_matching_every_cell(device, dtype):
    """Randomly initialised LoFTR with no confidence threshold and no border removal: every mutual nearest cell
    is a match, so the keypoints show the coarse grid of the whole image."""
    cfg = copy.deepcopy(default_cfg)
    cfg["match_coarse"]["thr"] = 0.0
    cfg["match_coarse"]["border_rm"] = 0
    torch.manual_seed(0)
    return LoFTR(pretrained=None, config=cfg).to(device, dtype)


class TestLoFTR(BaseTester):
    @pytest.mark.slow
    def test_pretrained_outdoor_smoke(self, device, dtype):
        loftr = LoFTR("outdoor").to(device, dtype)
        assert loftr is not None

    @pytest.mark.slow
    def test_pretrained_indoor_smoke(self, device, dtype):
        loftr = LoFTR("indoor").to(device, dtype)
        assert loftr is not None

    @pytest.mark.slow
    @pytest.mark.skipif(sys.platform == "win32", reason="this test takes so much memory in the CI with Windows")
    @pytest.mark.parametrize("data", ["loftr_fund"], indirect=True)
    def test_pretrained_indoor(self, device, dtype, data):
        if device.type == "cuda":
            # The stored expectations were generated on CPU. The CUDA matcher
            # selects a different set of correspondences (~37% of coordinates
            # differ, and the match count itself changes), so this comparison
            # cannot hold on CUDA until the expectations are regenerated
            # per-device. See https://github.com/kornia/kornia/issues/4092.
            pytest.skip("stored keypoint expectations are CPU-specific")
        loftr = LoFTR("indoor").to(device, dtype)
        data_dev = dict_to(data, device, dtype)
        with torch.no_grad():
            out = loftr(data_dev)
        self.assert_close(out["keypoints0"], data_dev["loftr_indoor_tentatives0"])
        self.assert_close(out["keypoints1"], data_dev["loftr_indoor_tentatives1"])

    @pytest.mark.slow
    @pytest.mark.skipif(sys.platform == "win32", reason="this test takes so much memory in the CI with Windows")
    @pytest.mark.parametrize("data", ["loftr_homo"], indirect=True)
    def test_pretrained_outdoor(self, device, dtype, data):
        if device.type == "cuda":
            # The stored expectations were generated on CPU. The CUDA matcher
            # selects a different set of correspondences (~37% of coordinates
            # differ, and the match count itself changes), so this comparison
            # cannot hold on CUDA until the expectations are regenerated
            # per-device. See https://github.com/kornia/kornia/issues/4092.
            pytest.skip("stored keypoint expectations are CPU-specific")
        loftr = LoFTR("outdoor").to(device, dtype)
        data_dev = dict_to(data, device, dtype)
        with torch.no_grad():
            out = loftr(data_dev)
        # image0 is 450 x 600; 450 is not a multiple of 8, and its 57 coarse rows sit 8 px apart. The stored
        # tentatives place coarse cell (x, y) at 450 / 57 * (x, y), so they give the matched cells, not the pixels.
        cells0 = torch.round(data_dev["loftr_outdoor_tentatives0"] * 57 / 450)
        self.assert_close(out["keypoints0"], 8 * cells0, rtol=0.0, atol=0.0)
        # keypoints1 follow the ground-truth homography of the pair (pixel-centre frame: subtract 0.5)
        kpts0, kpts1 = out["keypoints0"].cpu().double() - 0.5, out["keypoints1"].cpu().double() - 0.5
        err = (transform_points(data["H_gt"].cpu().double()[None], kpts0[None])[0] - kpts1).norm(dim=-1)
        assert (err < 3).double().mean() > 0.85, err
        assert err.median() < 1.5, err

    @pytest.mark.slow
    def test_mask(self, device):
        patches = torch.rand(1, 1, 32, 32, device=device)
        mask = torch.rand(1, 32, 32, device=device)
        loftr = LoFTR().to(patches.device, patches.dtype)
        sample = {"image0": patches, "image1": patches, "mask0": mask, "mask1": mask}
        with torch.no_grad():
            out = loftr(sample)
        assert out is not None

    def test_masked_cells_do_not_match_5740(self, device):
        """The whole-module view of #5740: with the same coarse cells masked on both sides,
        no returned keypoint may sit in the masked region, for any ``thr``."""
        cfg = copy.deepcopy(default_cfg)
        cfg["match_coarse"]["thr"] = 0.0
        cfg["match_coarse"]["border_rm"] = 0
        gen = torch.Generator(device=device).manual_seed(3)
        patches = torch.rand(1, 1, 64, 64, device=device, generator=gen)
        loftr = LoFTR(pretrained=None, config=cfg).to(device)
        mask = torch.ones(1, 64, 64, device=device)
        mask[..., :16] = 0  # the left quarter of both images is padding
        with torch.no_grad():
            out = loftr({"image0": patches, "image1": patches, "mask0": mask, "mask1": mask})
        kp0, kp1 = out["keypoints0"], out["keypoints1"]
        assert len(kp0) > 0
        # The masked cells are the coarse columns 0 and 1 (pixel x < 16, centers 0 and 8). Before
        # the fix, masked-cell pairs put keypoints at those centers, and the fine level refined
        # them to x in [0, 4] or below 0. A fine refinement of a valid coarse cell (column 2,
        # center 16) can legitimately move a few pixels into the masked band, so the assertion
        # covers the masked cell centres with the fine half-window to spare.
        assert (kp0[:, 0] < 12).sum().item() == 0
        assert (kp1[:, 0] < 12).sum().item() == 0

    @pytest.mark.slow
    @pytest.mark.skipif(sys.platform == "win32", reason="this test takes so much memory in the CI with Windows")
    @pytest.mark.parametrize("data", ["loftr_homo"], indirect=True)
    def test_pretrained_outdoor_images_of_different_sizes_5728(self, device, dtype, data):
        # image0 is a 325 x 403 crop of the pair's first image, image1 that crop warped by a known homography (8 degree
        # rotation and 0.9 scale about (200, 160)) into 339 x 401. No side is a multiple of 8 and the two sizes
        # differ, so each image's FPN crops to its own skip sizes.
        loftr = LoFTR("outdoor").to(device, dtype)
        image0 = data["image0"][..., 60:385, 100:503].to(device, dtype)
        c, s = 0.9 * math.cos(math.radians(8.0)), 0.9 * math.sin(math.radians(8.0))
        shift = torch.tensor([[1.0, 0.0, -200.0], [0.0, 1.0, -160.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
        rot = torch.tensor([[c, -s, 2.0], [s, c, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
        homography = torch.linalg.inv(shift) @ rot @ shift  # pixel-centre frame
        image1 = warp_perspective(image0, homography[None].to(device, dtype), (339, 401), align_corners=True)
        with torch.no_grad():
            out = loftr({"image0": image0, "image1": image1})
        kpts0, kpts1 = out["keypoints0"].cpu().double() - 0.5, out["keypoints1"].cpu().double() - 0.5
        err = (transform_points(homography[None], kpts0[None])[0] - kpts1).norm(dim=-1)
        assert kpts0.shape[0] > 1000
        assert (err < 1).double().mean() > 0.99, err
        assert err.median() < 0.3, err

    @pytest.mark.parametrize("hw1", [(100, 67), (93, 86)])
    def test_keypoints0_on_the_8px_grid_for_any_size_5728(self, device, dtype, hw1):
        # Coarse cell (x, y) sits on pixel (8x, 8y) whatever the image size: the backbone's stride-2 stages have
        # centred kernels and the map sizes are rounded up. H0 = 100 has 13 coarse rows, so 100 / 13 = 7.69 is not
        # the stride. hw1 = (100, 67) runs both images through the backbone together, (93, 86) one by one.
        model = _loftr_matching_every_cell(device, dtype)
        h0, w0 = 100, 67
        image0 = torch.rand(1, 1, h0, w0).to(device, dtype)
        image1 = torch.rand(1, 1, *hw1).to(device, dtype)
        with torch.no_grad():
            kpts0 = model({"image0": image0, "image1": image1})["keypoints0"]
        assert kpts0.shape[0] > 10
        self.assert_close(kpts0, 8 * torch.round(kpts0 / 8), rtol=0.0, atol=0.0)
        # the last coarse column and row are cells 8 and 12, on pixels 64 and 96; the matches reach far into both
        assert kpts0.min() >= 0
        assert kpts0[:, 0].max() <= 8 * 8
        assert kpts0[:, 1].max() <= 8 * 12
        assert kpts0[:, 0].max() >= 8 * 5
        assert kpts0[:, 1].max() >= 8 * 9

    def test_mask_is_read_at_the_coarse_cell_pixels_5728(self, device, dtype):
        # A coarse cell takes the mask value of the pixel it sits on, (8x, 8y). The masks are 1 only on those pixels
        # (image0: on the ones with y < 64 and x < 40, so coarse rows 0..7 and columns 0..4), so every such cell is
        # valid, while sampling the mask at (x, y) * size / coarse size would hit a 0 everywhere but at (0, 0).
        model = _loftr_matching_every_cell(device, dtype)
        h0, w0 = 100, 67
        image0 = torch.rand(1, 1, h0, w0).to(device, dtype)
        image1 = torch.rand(1, 1, 93, 86).to(device, dtype)
        mask0 = torch.zeros(1, h0, w0, device=device, dtype=dtype)
        mask0[:, :64:8, :40:8] = 1
        mask1 = torch.zeros(1, 93, 86, device=device, dtype=dtype)
        mask1[:, ::8, ::8] = 1
        with torch.no_grad():
            kpts0 = model({"image0": image0, "image1": image1, "mask0": mask0, "mask1": mask1})["keypoints0"]
        assert kpts0.shape[0] > 5
        self.assert_close(kpts0, 8 * torch.round(kpts0 / 8), rtol=0.0, atol=0.0)
        assert kpts0[:, 0].max() <= 32
        assert kpts0[:, 1].max() <= 56
        assert kpts0[:, 1].max() >= 8 * 4

    @pytest.mark.slow
    def test_gradcheck(self, device):
        patches = torch.rand(1, 1, 32, 32, device=device, dtype=torch.float64)
        patches05 = resize(patches, (48, 48))
        loftr = LoFTR().to(patches.device, patches.dtype)

        def proxy_forward(x, y):
            return loftr.forward({"image0": x, "image1": y})["keypoints0"]

        self.gradcheck(proxy_forward, (patches, patches05), eps=1e-4, atol=1e-4)

    @pytest.mark.skip("does not like transformer.py:L99, zip iteration")
    def test_jit(self, device, dtype):
        B, C, H, W = 1, 1, 32, 32
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        patches2x = resize(patches, (48, 48))
        sample = {"image0": patches, "image1": patches2x}
        model = LoFTR().to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(model)
        out = model(sample)
        out_jit = model_jit(sample)
        for k, v in out.items():
            self.assert_close(v, out_jit[k])


class TestLoFTRConfigIsPerInstance(BaseTester):
    """Building a LoFTR must not write into ``default_cfg`` or into a caller-passed config."""

    @pytest.fixture
    def offline_pretrained(self, monkeypatch):
        """Let ``LoFTR(pretrained=name)`` run its config selection without downloading a checkpoint."""
        state = {"state_dict": LoFTR(None).state_dict()}
        monkeypatch.setattr(loftr_module, "load_state_dict_from_url", lambda *args, **kwargs: state)

    def test_indoor_new_leaves_default_cfg_untouched_5719(self, offline_pretrained):
        before = copy.deepcopy(default_cfg)
        LoFTR("indoor_new")
        assert default_cfg == before
        assert default_cfg["coarse"]["temp_bug_fix"] is False

    def test_indoor_new_uses_the_fixed_positional_encoding_5719(self, offline_pretrained):
        indoor_new = LoFTR("indoor_new")
        assert indoor_new.config["coarse"]["temp_bug_fix"] is True
        assert indoor_new.pos_encoding.temp_bug_fix is True
        assert not torch.allclose(indoor_new.pos_encoding.pe, LoFTR(None).pos_encoding.pe)

    def test_later_default_model_keeps_the_outdoor_positional_encoding_5719(self, offline_pretrained):
        expected_pe = LoFTR(None).pos_encoding.pe.clone()
        LoFTR("indoor_new")
        later = LoFTR(None)
        assert later.config["coarse"]["temp_bug_fix"] is False
        assert later.pos_encoding.temp_bug_fix is False
        self.assert_close(later.pos_encoding.pe, expected_pe)

    def test_caller_config_is_not_mutated_5719(self, offline_pretrained):
        config = copy.deepcopy(default_cfg)
        before = copy.deepcopy(config)
        model = LoFTR("indoor_new", config=config)
        assert config == before
        assert model.config is not config
        assert model.config["coarse"]["temp_bug_fix"] is True

    def test_instances_do_not_share_config_5719(self):
        a, b = LoFTR(None), LoFTR(None)
        assert a.config is not b.config
        assert a.config["coarse"] is not b.config["coarse"]
        assert a.config is not default_cfg


class TestCoarseMatching(BaseTester):
    """CoarseMatching.get_coarse_match's training-mode gt-padding branch
    (self.training=True) previously crashed unconditionally: zip() was
    called with 4 separate 2-element lists instead of 2 separate
    4-element lists, so `for x, y in zip(...)` tried to unpack each
    resulting 4-tuple into 2 variables and raised
    `ValueError: too many values to unpack (expected 2)` on every call.
    No existing test exercised this training-mode path at all.

    The grid is deliberately asymmetric (L=16, S=12) and the predictions
    deterministic (no RNG) so the pairing/ordering/count the fix changes
    -- not just "does it crash" or "is the dtype right" -- are pinned.
    Verified against 8 single-token mutations of the production fix
    (reverting the zip() shape, dropping the dtype fix, swapping the gt
    id lists, swapping the concat order, swapping which side is padding,
    flipping the zero-fill to ones, and disabling the branch entirely):
    all 8 are killed by this test, where the previous dtype-only
    assertions caught only 2 of the 8."""

    CFG = {
        "thr": 0.01,
        "border_rm": 0,
        "train_coarse_percent": 0.5,
        "train_pad_num_gt_min": 2,
        "match_type": "dual_softmax",
        "dsmax_temperature": 0.1,
    }
    H0C, W0C, H1C, W1C = 4, 4, 3, 4  # L=16, S=12 -- i/j ranges are distinguishable
    N_PRED, N_GT = 5, 20

    def _data(self, device):
        L, S = self.H0C * self.W0C, self.H1C * self.W1C
        return {
            "hw0_c": (self.H0C, self.W0C),
            "hw1_c": (self.H1C, self.W1C),
            "hw0_i": (self.H0C * 8, self.W0C * 8),
            "hw1_i": (self.H1C * 8, self.W1C * 8),
            "stride_c": 8,
            "spv_b_ids": torch.zeros(self.N_GT, dtype=torch.long, device=device),
            "spv_i_ids": torch.full((self.N_GT,), L - 1, dtype=torch.long, device=device),
            "spv_j_ids": torch.full((self.N_GT,), S - 1, dtype=torch.long, device=device),
        }

    def _conf(self, device, dtype):
        L, S = self.H0C * self.W0C, self.H1C * self.W1C
        conf = torch.zeros(1, L, S, device=device, dtype=dtype)
        for k in range(self.N_PRED):  # exactly N_PRED mutual-NN matches, no RNG
            conf[0, k, k] = 0.5 + 0.01 * k
        return conf

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_training_gt_padding(self, device, dtype):
        cm = CoarseMatching(self.CFG).to(device)
        cm.train()
        out = cm.get_coarse_match(self._conf(device, dtype), self._data(device))

        L, S = self.H0C * self.W0C, self.H1C * self.W1C
        n_train = int(max(L, S) * self.CFG["train_coarse_percent"])  # 8
        n_pad = max(n_train - self.N_PRED, self.CFG["train_pad_num_gt_min"])  # 3
        assert out["b_ids"].shape[0] == self.N_PRED + n_pad
        assert (out["b_ids"] == 0).all()
        assert (out["i_ids"] < L).all()
        assert (out["j_ids"] < S).all()
        assert out["i_ids"][: self.N_PRED].tolist() == list(range(self.N_PRED))
        assert out["j_ids"][: self.N_PRED].tolist() == list(range(self.N_PRED))
        assert (out["i_ids"][self.N_PRED :] == L - 1).all()
        assert (out["j_ids"][self.N_PRED :] == S - 1).all()
        assert not out["gt_mask"][: self.N_PRED].any()
        assert out["gt_mask"][self.N_PRED :].all()
        assert out["mconf"].numel() == self.N_PRED
        assert out["mconf"].dtype == dtype
        assert out["mkpts0_c"].shape == (self.N_PRED, 2)

    @pytest.mark.parametrize("thr", [0.0, -1.0])
    @pytest.mark.parametrize("mask0_on", [False, True])
    @pytest.mark.parametrize("mask1_on", [False, True])
    def test_masked_cell_pair_is_never_a_match_5740(self, device, mask0_on, mask1_on, thr):
        """A pair of masked coarse cells has the confidence 1 / (L * S): the row and the
        column of a fully masked cell are uniform under the two softmaxes, and all other
        entries are exactly 0, so the pair is also a mutual nearest neighbour. With ``thr``
        below 1 / (L * S) it was returned as a match, with keypoints in the padding of both
        images. One-sided masking is covered too: the confidence of any pair that involves a
        masked cell is 0. A negative ``thr`` must not admit those zero confidences either:
        ``get_coarse_match`` drops ``mconf == 0`` from the keypoints but not from ``b_ids``."""
        cfg = {
            "thr": thr,
            "border_rm": 0,
            "train_coarse_percent": 0.5,
            "train_pad_num_gt_min": 2,
            "match_type": "dual_softmax",
            "dsmax_temperature": 0.1,
        }
        gen = torch.Generator(device=device).manual_seed(0)
        feat0 = torch.randn(1, 12, 8, device=device, generator=gen)
        feat1 = torch.randn(1, 9, 8, device=device, generator=gen)
        mask0, mask1 = torch.ones(1, 12, device=device), torch.ones(1, 9, device=device)
        if mask0_on:
            mask0[0, 3] = 0
        if mask1_on:
            mask1[0, 4] = 0
        data = {
            "hw0_c": (3, 4),
            "hw1_c": (3, 3),
            "hw0_i": (24, 32),
            "hw1_i": (24, 24),
            "stride_c": 8,
        }
        cm = CoarseMatching(cfg).to(device)
        cm.eval()
        cm(feat0, feat1, data, mask_c0=mask0, mask_c1=mask1)
        pairs = set(zip(data["i_ids"].tolist(), data["j_ids"].tolist()))
        assert len(data["b_ids"]) == len(data["mconf"]) == len(data["mkpts0_c"])
        if mask0_on or mask1_on:
            # any pair that involves a masked cell has the confidence 0: one-sided masking
            # already achieved that through the -1e9 similarity fill (the softmax over the
            # masked row or column gives it no weight), both-sided masking is what needed the
            # explicit zero fill after the softmaxes
            assert (3, 4) not in pairs
            assert data["conf_matrix"][0, 3, 4].item() == 0.0
        else:
            # control: without any mask the pair keeps its normal confidence, the fix must not
            # suppress it (whether it is a mutual nearest neighbour depends on the RNG, so only
            # the confidence is asserted)
            assert data["conf_matrix"][0, 3, 4].item() > 0.0

    @pytest.mark.parametrize("masked_side", [0, 1])
    def test_independently_optional_coarse_masks_5768(self, device, masked_side):
        cfg = dict(self.CFG)
        cfg["thr"] = 0.0
        generator = torch.Generator(device=device).manual_seed(0)
        feat0 = torch.randn(1, 12, 8, device=device, generator=generator)
        feat1 = torch.randn(1, 9, 8, device=device, generator=generator)
        masked = torch.ones(1, 12 if masked_side == 0 else 9, device=device)
        masked[0, [0, 5, 7, 8, 11] if masked_side == 0 else [0, 3, 5]] = 0
        all_valid = torch.ones(1, 9 if masked_side == 0 else 12, device=device)

        def run(include_other_mask):
            masks = (
                (masked, all_valid if include_other_mask else None)
                if masked_side == 0
                else (all_valid if include_other_mask else None, masked)
            )
            data = {
                "hw0_c": (3, 4),
                "hw1_c": (3, 3),
                "hw0_i": (24, 32),
                "hw1_i": (24, 24),
                "stride_c": 8,
            }
            CoarseMatching(cfg).to(device).eval()(feat0, feat1, data, mask_c0=masks[0], mask_c1=masks[1])
            return data

        one_sided = run(False)
        explicit_valid = run(True)
        assert torch.equal(one_sided["conf_matrix"], explicit_valid["conf_matrix"])
        assert torch.equal(one_sided["i_ids"], explicit_valid["i_ids"])
        assert torch.equal(one_sided["j_ids"], explicit_valid["j_ids"])
        selected = one_sided["i_ids"] if masked_side == 0 else one_sided["j_ids"]
        assert selected.numel() > 0
        assert bool(masked[0, selected].all())

    @pytest.mark.parametrize("masked_side", [0, 1])
    def test_sinkhorn_applies_independently_optional_coarse_masks_5768(self, device, masked_side):
        cfg = dict(self.CFG)
        cfg["thr"] = 0.0
        cfg["sparse_spvs"] = False
        cm = CoarseMatching(cfg).to(device).eval()
        cm.match_type = "sinkhorn"
        cm.skh_prefilter = False
        cm.skh_iters = 1
        cm.bin_score = torch.nn.Parameter(torch.tensor(1.0, device=device))
        transport_inputs = []

        def fake_transport(scores, *_):
            transport_inputs.append(scores.detach().clone())
            return torch.zeros(scores.shape[0], scores.shape[1] + 1, scores.shape[2] + 1, device=device)

        cm.log_optimal_transport = fake_transport
        generator = torch.Generator(device=device).manual_seed(0)
        feat0 = torch.randn(1, 12, 8, device=device, generator=generator)
        feat1 = torch.randn(1, 9, 8, device=device, generator=generator)
        masked = torch.ones(1, 12 if masked_side == 0 else 9, device=device)
        masked[0, [0, 5]] = 0
        masks = (masked, None) if masked_side == 0 else (None, masked)
        data = {
            "hw0_c": (3, 4),
            "hw1_c": (3, 3),
            "hw0_i": (24, 32),
            "hw1_i": (24, 24),
            "stride_c": 8,
        }

        cm(feat0, feat1, data, mask_c0=masks[0], mask_c1=masks[1])
        invalid = ~masked.bool()[..., None] if masked_side == 0 else ~masked.bool()[:, None]
        invalid = invalid.expand_as(transport_inputs[0])
        assert bool((transport_inputs[0][invalid] == -1e9).all())
        assert bool((data["conf_matrix"][invalid] == 0).all())
        selected = data["i_ids"] if masked_side == 0 else data["j_ids"]
        assert selected.numel() > 0
        assert bool(masked[0, selected].all())

    @pytest.mark.parametrize(("mask0_on", "mask1_on"), [(True, False), (False, True), (True, True)])
    def test_masked_sinkhorn_training_backward_5768(self, device, mask0_on, mask1_on):
        """The zero fill after the transport must not write into the ``exp`` output, which autograd saves: an in-place
        fill made ``backward`` raise for every masked Sinkhorn training step."""
        cfg = dict(self.CFG)
        cfg["thr"] = 0.0
        cfg["sparse_spvs"] = False
        cm = CoarseMatching(cfg).to(device).train()
        cm.match_type = "sinkhorn"
        cm.skh_prefilter = False
        cm.skh_iters = 1
        cm.bin_score = torch.nn.Parameter(torch.tensor(1.0, device=device))
        # A differentiable stand-in for SuperGlue's log_optimal_transport (not shipped), with a dustbin row and column.
        cm.log_optimal_transport = lambda scores, *_: F.log_softmax(F.pad(scores, (0, 1, 0, 1)), dim=-1)
        generator = torch.Generator(device=device).manual_seed(0)
        feat0 = torch.randn(1, 12, 8, device=device, generator=generator, requires_grad=True)
        feat1 = torch.randn(1, 9, 8, device=device, generator=generator, requires_grad=True)
        mask0 = torch.ones(1, 12, device=device)
        mask0[0, [0, 5]] = 0
        mask1 = torch.ones(1, 9, device=device)
        mask1[0, 3] = 0
        data = {
            "hw0_c": (3, 4),
            "hw1_c": (3, 3),
            "hw0_i": (24, 32),
            "hw1_i": (24, 24),
            "stride_c": 8,
            "spv_b_ids": torch.zeros(2, dtype=torch.long, device=device),
            "spv_i_ids": torch.tensor([1, 2], device=device),
            "spv_j_ids": torch.tensor([1, 2], device=device),
        }

        cm(feat0, feat1, data, mask_c0=mask0 if mask0_on else None, mask_c1=mask1 if mask1_on else None)
        data["conf_matrix"].sum().backward()
        assert bool(torch.isfinite(feat0.grad).all())
        assert bool(torch.isfinite(feat1.grad).all())

    @pytest.mark.parametrize("masked_side", [0, 1])
    def test_one_sided_mask_is_applied_per_sample_5768(self, device, masked_side):
        """Each sample of a batch keeps its own mask, and the dual softmax runs over that sample's valid cells only:
        the confidences of the valid cells equal the dual softmax of the unmasked submatrix, all others are 0."""
        cfg = dict(self.CFG)
        cfg["thr"] = 0.0
        generator = torch.Generator(device=device).manual_seed(0)
        feat0 = torch.randn(2, 12, 8, device=device, generator=generator)
        feat1 = torch.randn(2, 9, 8, device=device, generator=generator)
        masked = torch.ones(2, 12 if masked_side == 0 else 9, device=device)
        masked[0, [0, 5, 7]] = 0
        masked[1, [1, 2, 8]] = 0
        data = {
            "hw0_c": (3, 4),
            "hw1_c": (3, 3),
            "hw0_i": (24, 32),
            "hw1_i": (24, 24),
            "stride_c": 8,
        }
        masks = (masked, None) if masked_side == 0 else (None, masked)
        CoarseMatching(cfg).to(device).eval()(feat0, feat1, data, mask_c0=masks[0], mask_c1=masks[1])

        sim = torch.einsum("nlc,nsc->nls", feat0 / 8**0.5, feat1 / 8**0.5) / cfg["dsmax_temperature"]
        for b in range(2):
            valid = masked[b].bool()
            expected = torch.zeros_like(sim[b])
            if masked_side == 0:
                expected[valid] = sim[b][valid].softmax(0) * sim[b][valid].softmax(1)
            else:
                expected[:, valid] = sim[b][:, valid].softmax(0) * sim[b][:, valid].softmax(1)
            self.assert_close(data["conf_matrix"][b], expected)

    def test_eval_path_unaffected(self, device):
        """Eval mode skips the gt-padding branch entirely -- assert that
        directly (gt_mask all False, count == predictions only), not just
        a dtype that would also hold if the branch silently ran."""
        cm = CoarseMatching(self.CFG).to(device)
        cm.eval()
        out = cm.get_coarse_match(self._conf(device, torch.float32), self._data(device))
        assert out["mconf"].dtype == torch.float32
        assert out["gt_mask"].sum() == 0
        assert out["mconf"].numel() == out["b_ids"].shape[0] == self.N_PRED

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_masked_dual_softmax_is_finite_5728(self, device, dtype):
        # Masked pairs get a similarity of -1e9, which float16 cannot hold, so float16 takes its lowest finite value
        # instead; -inf would make an entirely masked row or column nan. Cell 3 of image0 and cell 5 of image1 are
        # masked, so row 3 and column 5 of the similarity matrix are masked entirely.
        gen = torch.Generator().manual_seed(0)
        feat0, feat1 = torch.randn(1, 12, 8, generator=gen), torch.randn(1, 9, 8, generator=gen)
        mask0, mask1 = torch.ones(1, 12), torch.ones(1, 9)
        mask0[0, 3], mask1[0, 5] = 0, 0
        cm = CoarseMatching(self.CFG).to(device).eval()

        def conf_matrix(dt):
            data = {"hw0_c": (3, 4), "hw1_c": (3, 3), "stride_c": 8}
            args = (t.to(device, dt) for t in (feat0, feat1, mask0, mask1))
            f0, f1, m0, m1 = args
            cm(f0, f1, data, mask_c0=m0, mask_c1=m1)
            return data["conf_matrix"]

        conf = conf_matrix(dtype)
        assert conf.dtype == dtype
        assert bool(torch.isfinite(conf).all())
        self.assert_close(conf.float(), conf_matrix(torch.float32), rtol=0.0, atol=2e-2)
        valid1 = torch.arange(9, device=device) != 5
        assert conf[0, 3, valid1].float().max() < 1e-3

    def test_keypoints_use_the_coarse_stride_5728(self, device, dtype):
        # image0 100 x 67 -> 13 x 9 coarse cells, image1 93 x 86 -> 12 x 11: cell (x, y) of either image sits on
        # pixel (8x, 8y). The ratios 100 / 13, 93 / 12, 67 / 9 and 86 / 11 all differ from 8 and from each other.
        h0c, w0c, h1c, w1c = 13, 9, 12, 11
        i_ids = torch.tensor([0, 10, 47, h0c * w0c - 1], device=device)
        j_ids = torch.tensor([3, 40, 77, h1c * w1c - 1], device=device)
        conf = torch.zeros(1, h0c * w0c, h1c * w1c, device=device, dtype=dtype)
        conf[0, i_ids, j_ids] = 0.5
        data = {"hw0_i": (100, 67), "hw1_i": (93, 86), "hw0_c": (h0c, w0c), "hw1_c": (h1c, w1c), "stride_c": 8}
        cm = CoarseMatching(self.CFG).to(device).eval()
        out = cm.get_coarse_match(conf, data)
        assert out["i_ids"].tolist() == i_ids.tolist()
        expected0 = torch.tensor([[0, 0], [8, 8], [16, 40], [64, 96]], device=device, dtype=dtype)
        expected1 = torch.tensor([[24, 0], [56, 24], [0, 56], [80, 88]], device=device, dtype=dtype)
        self.assert_close(out["mkpts0_c"], expected0, rtol=0.0, atol=0.0)
        self.assert_close(out["mkpts1_c"], expected1, rtol=0.0, atol=0.0)


class TestCoarseMatchingSelection(BaseTester):
    """The row max / column max selection and the dual softmax against their reference formulations."""

    CFG = {**TestCoarseMatching.CFG, "train_pad_num_gt_min": 200}

    @staticmethod
    def _reference_selection(cm, conf_matrix, data):
        """The boolean-mask selection of get_coarse_match before the row max / column max rewrite (eval mode)."""
        from kornia.feature.loftr.utils.coarse_matching import mask_border, mask_border_with_padding

        (h0, w0), (h1, w1), n = data["hw0_c"], data["hw1_c"], conf_matrix.shape[0]
        mask = (conf_matrix > max(cm.thr, 0.0)).reshape(n, h0, w0, h1, w1)
        if "mask0" not in data:
            mask_border(mask, cm.border_rm, False)
        else:
            mask_border_with_padding(mask, cm.border_rm, False, data["mask0"], data["mask1"])
        mask = mask.reshape(n, h0 * w0, h1 * w1)
        mask = (
            mask
            * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0])
            * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])
        )
        mask_v, all_j_ids = mask.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        return b_ids, i_ids, j_ids, conf_matrix[b_ids, i_ids, j_ids]

    @staticmethod
    def _features(device, dtype, n, length, size, channels=16, seed=0, scale=4.0):
        # image-1 cells are noisy copies of image-0 cells, so most rows have a clear, untied mutual maximum
        gen = torch.Generator().manual_seed(seed)
        feat_c1 = torch.randn(n, size, channels, generator=gen) * scale
        rows = torch.randint(0, size, (n, length), generator=gen)
        feat_c0 = torch.gather(feat_c1, 1, rows[..., None].expand(-1, -1, channels))
        feat_c0 = feat_c0 + torch.randn(n, length, channels, generator=gen) * (scale / 2)
        return feat_c0.to(device, dtype), feat_c1.to(device, dtype)

    @pytest.mark.parametrize("thr", [0.0, 0.2])
    @pytest.mark.parametrize("border_rm", [0, 1, 2])
    @pytest.mark.parametrize("padded", [False, True])
    def test_selection_matches_the_boolean_mask_reference(self, device, dtype, thr, border_rm, padded):
        cm = CoarseMatching({**self.CFG, "thr": thr, "border_rm": border_rm}).to(device).eval()
        (h0, w0), (h1, w1) = (10, 12), (11, 9)
        feat_c0, feat_c1 = self._features(device, dtype, 2, h0 * w0, h1 * w1)
        data = {"hw0_c": (h0, w0), "hw1_c": (h1, w1), "stride_c": 8}
        if padded:
            # per-sample valid extents, as mask_border_with_padding reads them
            mask0 = torch.zeros(2, h0, w0, device=device)
            mask1 = torch.zeros(2, h1, w1, device=device)
            mask0[0, :9, :10], mask0[1] = 1, 1
            mask1[0], mask1[1, :10, :8] = 1, 1
            data.update(mask0=mask0, mask1=mask1)
        with torch.no_grad():
            cm(feat_c0, feat_c1, data)
        expected = self._reference_selection(cm, data["conf_matrix"], data)
        assert len(expected[0]) >= 5  # border 2 on the padded samples leaves 10
        for name, value in zip(("b_ids", "i_ids", "j_ids", "mconf"), expected):
            self.assert_close(data[name], value, rtol=0.0, atol=0.0)


class TestFineMatching(BaseTester):
    @staticmethod
    def _run(device, dtype, peaked):
        """Run FineMatching on a peaked or a uniform heatmap; return `std` and the input it flows back to."""
        M, W, C = 4, 5, 8
        feat_f0 = torch.zeros(M, W * W, C, device=device, dtype=dtype)
        feat_f1 = torch.zeros(M, W * W, C, device=device, dtype=dtype)
        if peaked:
            # One dominant cell, so the softmax is a spike and the variance of both coordinates is zero.
            feat_f0[:, W * W // 2, 0] = 30.0
            feat_f1[:, W * W // 2, 0] = 30.0
        feat_f1.requires_grad_(True)
        data = {
            "hw0_i": (40, 40),
            "hw0_f": (20, 20),
            "stride_f": 2,
            "mkpts0_c": torch.zeros(M, 2, device=device, dtype=dtype),
            "mkpts1_c": torch.zeros(M, 2, device=device, dtype=dtype),
            "mconf": torch.ones(M, device=device, dtype=dtype),
            "b_ids": torch.zeros(M, dtype=torch.long, device=device),
        }
        FineMatching()(feat_f0, feat_f1, data)
        return data["expec_f"][:, 2], feat_f1

    def test_regression_std_gradient_is_finite_on_a_peaked_heatmap_4229(self, device, dtype):
        # A heatmap peaked on one cell has zero variance. `std` was sqrt(clamp(var, min=1e-10)): in float16 the
        # floor is 0, and torch < 2.14 passes clamp's gradient through at the bound, so sqrt'(0) = inf reached
        # the features as nan. torch 2.14 zeroes clamp's gradient at the bound, so on 2.14 this passes on the
        # old code too; the torch 2.5.1 / 2.9.1 legs are the ones that discriminate.
        std, feat_f1 = self._run(device, dtype, peaked=True)
        std.sum().backward()
        assert bool(torch.isfinite(feat_f1.grad).all()), feat_f1.grad
        # The value is unchanged, and it is `2 * sqrt(floor)`, not `sqrt(floor)`: `std` sums over the two
        # coordinate axes, so both floored terms land in it. Asserted exactly -- the sum of two identical
        # terms is their double in every dtype -- because the default tolerance is wider than the quantity
        # being pinned, and would accept `std == 0`, a dropped `sqrt` and a `1e-12` floor alike.
        self.assert_close(std, torch.full_like(std, 1e-10).sqrt() * 2, rtol=0.0, atol=0.0)

    def test_std_is_the_real_deviation_above_the_floor(self, device, dtype):
        # The floor branch is only half the behaviour: nothing else in the repo reads `expec_f` or `std` at
        # all -- the pretrained tests compare keypoints, which come from `coords_normalized` -- so without
        # this, forcing the floor branch to be taken always, which corrupts `std` for every real heatmap,
        # passes the whole file. A uniform heatmap over the normalized grid `linspace(-1, 1, 5)` has
        # `var = mean(g**2) = 0.5` on each axis, so `std` is `2 * sqrt(0.5)` rather than the floor.
        std, feat_f1 = self._run(device, dtype, peaked=False)
        self.assert_close(std, torch.full_like(std, 0.5).sqrt() * 2)
        std.sum().backward()
        assert bool(torch.isfinite(feat_f1.grad).all()), feat_f1.grad

    def test_offset_uses_the_fine_stride_5728(self, device, dtype):
        # The fine window of a match is 5 x 5 fine cells, 2 px apart, so a heatmap expectation at normalized
        # (u, v) moves keypoints1 by (W // 2) * 2 * (u, v) = 4 (u, v) px, whatever the image size. image0
        # 99 x 67 has 50 x 34 fine cells; 99 / 50 = 1.98 is not the stride.
        M, W, C = 3, 5, 8
        feat_f0 = torch.zeros(M, W * W, C, device=device, dtype=dtype)
        feat_f1 = torch.zeros(M, W * W, C, device=device, dtype=dtype)
        feat_f0[:, W * W // 2, 0] = 30.0
        for m, cell in enumerate((4, 16, 21)):  # (x, y) = (4, 0), (1, 3), (1, 4) in the window
            feat_f1[m, cell, 0] = 30.0
        data = {
            "hw0_i": (99, 67),
            "hw0_f": (50, 34),
            "stride_f": 2,
            "mkpts0_c": torch.tensor([[8.0, 16.0], [24.0, 0.0], [64.0, 96.0]], device=device, dtype=dtype),
            "mkpts1_c": torch.tensor([[40.0, 8.0], [0.0, 32.0], [80.0, 88.0]], device=device, dtype=dtype),
            "mconf": torch.ones(M, device=device, dtype=dtype),
            "b_ids": torch.zeros(M, dtype=torch.long, device=device),
        }
        with torch.no_grad():
            FineMatching()(feat_f0, feat_f1, data)
        offset = data["mkpts1_f"] - data["mkpts1_c"]
        self.assert_close(offset, data["expec_f"][:, :2] * 4)
        # the peaks sit 2 fine cells right / up of the centre, 1 left / 1 down, 1 left / 2 down
        self.assert_close(
            offset,
            torch.tensor([[4.0, -4.0], [-2.0, 2.0], [-2.0, 4.0]], device=device, dtype=dtype),
            rtol=0.0,
            atol=0.05,
        )
        self.assert_close(data["mkpts0_f"], data["mkpts0_c"], rtol=0.0, atol=0.0)


class TestFinePreprocess(BaseTester):
    def test_windows_centred_on_the_coarse_cells_5728(self, device, dtype):
        # Coarse cell (x, y) sits on fine cell (4x, 4y) for any image size. image0 100 x 67 has 13 x 9 coarse and
        # 50 x 34 fine cells, image1 93 x 86 has 12 x 11 and 47 x 43; 50 // 13 = 3 is not the stride. The fine
        # maps hold their own (row, column) indices, so each window shows where it was cut.
        cfg = {
            "fine_concat_coarse_feat": False,
            "fine_window_size": 5,
            "coarse": {"d_model": 4},
            "fine": {"d_model": 2},
        }

        def index_map(h, w):
            rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            return torch.stack([rows, cols])[None].to(device, dtype)

        b_ids = torch.zeros(4, dtype=torch.long, device=device)
        i_ids = torch.tensor([0, 10, 47, 13 * 9 - 1], device=device)
        j_ids = torch.tensor([3, 40, 77, 12 * 11 - 1], device=device)
        data = {
            "hw0_c": (13, 9),
            "hw0_f": (50, 34),
            "stride_c": 8,
            "stride_f": 2,
            "b_ids": b_ids,
            "i_ids": i_ids,
            "j_ids": j_ids,
        }
        feat_c = torch.zeros(1, 1, 4, device=device, dtype=dtype)
        with torch.no_grad():
            win0, win1 = FinePreprocess(cfg).to(device, dtype)(
                index_map(50, 34), index_map(47, 43), feat_c, feat_c, data
            )
        assert win0.shape == win1.shape == (4, 25, 2)
        centre0 = torch.stack([i_ids // 9, i_ids % 9], -1) * 4
        centre1 = torch.stack([j_ids // 11, j_ids % 11], -1) * 4
        self.assert_close(win0[:, 12], centre0.to(dtype), rtol=0.0, atol=0.0)
        self.assert_close(win1[:, 12], centre1.to(dtype), rtol=0.0, atol=0.0)
        # the window spans +-2 fine cells: its last element is two cells down and right of the centre
        self.assert_close(win0[1:3, 24], (centre0[1:3] + 2).to(dtype), rtol=0.0, atol=0.0)


    @pytest.mark.parametrize("fine_hw", [((50, 34), (47, 43)), ((33, 37), (29, 41))])
    def test_gathered_windows_equal_the_unfolded_ones(self, device, dtype, fine_hw):
        # FinePreprocess cuts only the windows of the matched cells; they must be exactly the windows of
        # F.unfold(feat, 5, stride=4, padding=2) indexed by the cell ids, for both images of a batch of two.
        cfg = {
            "fine_concat_coarse_feat": False,
            "fine_window_size": 5,
            "coarse": {"d_model": 4},
            "fine": {"d_model": 3},
        }
        gen = torch.Generator().manual_seed(0)
        (hf0, wf0), (hf1, wf1) = fine_hw
        feat_f0 = torch.randn(2, 3, hf0, wf0, generator=gen).to(device, dtype)
        feat_f1 = torch.randn(2, 3, hf1, wf1, generator=gen).to(device, dtype)

        def unfolded(feat):
            out = F.unfold(feat, kernel_size=(5, 5), stride=4, padding=2)
            return out.reshape(out.shape[0], feat.shape[1], 25, -1).permute(0, 3, 2, 1)

        ref0, ref1 = unfolded(feat_f0), unfolded(feat_f1)
        b_ids = torch.tensor([0, 1, 1, 0, 1], device=device)
        i_ids = torch.tensor([0, 5, ref0.shape[1] - 1, 17, 30], device=device)
        j_ids = torch.tensor([ref1.shape[1] - 1, 0, 9, 41, 2], device=device)
        data = {"stride_c": 8, "stride_f": 2, "b_ids": b_ids, "i_ids": i_ids, "j_ids": j_ids}
        feat_c = torch.zeros(2, 1, 4, device=device, dtype=dtype)
        with torch.no_grad():
            win0, win1 = FinePreprocess(cfg).to(device, dtype)(feat_f0, feat_f1, feat_c, feat_c, data)
        self.assert_close(win0, ref0[b_ids, i_ids], rtol=0.0, atol=0.0)
        self.assert_close(win1, ref1[b_ids, j_ids], rtol=0.0, atol=0.0)


class TestResNetFPN(BaseTester):
    def test_fpn_upsamples_by_2_then_crops_5728(self, device, dtype):
        # Each FPN level adds to its lateral map the coarser level upsampled by 2 (align_corners=True), cropped to
        # the lateral size. A 65 x 74 input (1 and 2 mod 8) crops at both levels: the 1/8 map is 9 x 10 and the 1/4
        # map 17 x 19, not 18 x 20; the 1/4 map is 17 x 19 and the 1/2 map 33 x 37, not 34 x 38. Upsampling to the
        # lateral size instead reads the coarser map at other positions.
        torch.manual_seed(0)
        fpn = ResNetFPN_8_2(default_cfg["resnetfpn"]).to(device, dtype).eval()
        seen = {}

        def keep(name, output=True):
            def hook(module, args, out=None):
                seen[name] = out if output else args[0]

            return hook

        for name, module in (("x1", fpn.layer1), ("x2", fpn.layer2), ("x3_out", fpn.layer3_outconv)):
            module.register_forward_hook(keep(name))
        fpn.layer2_outconv2.register_forward_pre_hook(keep("sum2", output=False))
        fpn.layer2_outconv2.register_forward_hook(keep("x2_out"))
        fpn.layer1_outconv2.register_forward_pre_hook(keep("sum1", output=False))
        with torch.no_grad():
            fpn(torch.rand(1, 1, 65, 74).to(device, dtype))
            assert seen["x3_out"].shape[2:] == (9, 10)
            assert seen["x2"].shape[2:] == (17, 19)
            assert seen["x1"].shape[2:] == (33, 37)
            for coarse, lateral, total in (
                (seen["x3_out"], fpn.layer2_outconv(seen["x2"]), seen["sum2"]),
                (seen["x2_out"], fpn.layer1_outconv(seen["x1"]), seen["sum1"]),
            ):
                h, w = lateral.shape[2:]
                up = F.interpolate(coarse, scale_factor=2.0, mode="bilinear", align_corners=True)[..., :h, :w]
                self.assert_close(total, lateral + up)


class TestLinearAttention(BaseTester):
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_padded_queries_stay_finite_5728(self, device, dtype):
        # A padded query has a zero feature map, so its normaliser 1 / eps = 1e6 would exceed float16's 65504. The
        # padded rows must come out as 0, not as 0 * inf = nan, the gradients must stay finite as well, and the
        # rest must not change.
        gen = torch.Generator().manual_seed(0)
        q, k, v = (torch.randn(1, n, 2, 4, generator=gen) for n in (6, 5, 5))
        q_mask, kv_mask = torch.ones(1, 6), torch.ones(1, 5)
        q_mask[0, [1, 4]], kv_mask[0, 2] = 0, 0

        def attend(dt):
            q_, k_, v_ = (t.to(device, dt).requires_grad_() for t in (q, k, v))
            out = LinearAttention()(q_, k_, v_, q_mask.to(device, dt), kv_mask.to(device, dt))
            out.sum().backward()
            return out.detach(), q_.grad, k_.grad, v_.grad

        out, *grads = attend(dtype)
        out32, *grads32 = attend(torch.float32)
        assert bool(torch.isfinite(out).all())
        self.assert_close(out[0, [1, 4]], torch.zeros_like(out[0, [1, 4]]), rtol=0.0, atol=0.0)
        self.assert_close(out.float(), out32, rtol=0.0, atol=5e-2)
        for grad, grad32 in zip(grads, grads32):
            assert bool(torch.isfinite(grad).all()), grad
            self.assert_close(grad.float(), grad32, rtol=0.0, atol=5e-2)
        self.assert_close(grads[0][0, [1, 4]], torch.zeros_like(grads[0][0, [1, 4]]), rtol=0.0, atol=0.0)


class TestConventionsLoFTR(BaseTester):
    # Per batch item: a coarse cell (row, column) of image 0 (59 x 85 pixels, 8 x 11 coarse cells), its partner in
    # image 1 (64 x 74 pixels, 8 x 10 cells), the feature amplitude, and the peak (x, y) of the fine heatmap in the
    # 5 x 5 window of image 1, whose centre is (2, 2). The cells keep border_rm = 2 cells from the edges.
    _PAIRS = {
        0: [((3, 7), (5, 3), 16.0, (3, 1)), ((5, 2), (2, 6), 16.0, (0, 4)), ((4, 4), (3, 4), 4.0, (2, 2))],
        1: [((2, 3), (4, 7), 16.0, (4, 2))],
    }

    def _hooked_loftr(self, device, dtype, swap):
        """Return a random-weight LoFTR whose coarse and fine matching inputs are replaced by designed features."""
        pairs = {
            b: [((p1, p0, a, peak) if swap else (p0, p1, a, peak)) for p0, p1, a, peak in items]
            for b, items in self._PAIRS.items()
        }

        def coarse_features(module, args):
            # one-hot features: each designed cell is similar to its partner only
            feat0, feat1, data = args
            feat0, feat1 = torch.zeros_like(feat0), torch.zeros_like(feat1)
            w0, w1 = data["hw0_c"][1], data["hw1_c"][1]
            channel = 0
            for b, items in pairs.items():
                for (r0, c0), (r1, c1), amplitude, _ in items:
                    feat0[b, r0 * w0 + c0, channel] = amplitude
                    feat1[b, r1 * w1 + c1, channel] = amplitude
                    channel += 1
            return feat0, feat1, data

        def fine_features(module, args):
            # a fine heatmap peaked on the designed cell of each match's window
            feat0, feat1, data = args
            feat0, feat1 = torch.zeros_like(feat0), torch.zeros_like(feat1)
            feat0[:, feat0.shape[1] // 2, 0] = 30.0
            w0 = data["hw0_c"][1]
            peaks = {(b, r0 * w0 + c0): peak for b, items in pairs.items() for (r0, c0), _, _, peak in items}
            for m, (b, i) in enumerate(zip(data["b_ids"].tolist(), data["i_ids"].tolist())):
                x, y = peaks[(b, i)]
                feat1[m, y * 5 + x, 0] = 30.0
            return feat0, feat1, data

        torch.manual_seed(0)
        model = LoFTR(pretrained=None).to(device, dtype)
        model.coarse_matching.register_forward_pre_hook(coarse_features)
        model.fine_matching.register_forward_pre_hook(fine_features)
        return model

    def test_convention_loftr_keypoints_are_eight_times_the_coarse_cell_plus_the_fine_offset(self, device, dtype):
        image0 = torch.rand(2, 1, 59, 85, device=device, dtype=dtype)
        image1 = torch.rand(2, 1, 64, 74, device=device, dtype=dtype)
        with torch.no_grad():
            out = self._hooked_loftr(device, dtype, swap=False)({"image0": image0, "image1": image1})
        # keypoints0 = 8 * (column, row) of the coarse cell. keypoints1 = 8 * (column, row) of the partner plus the
        # fine offset, 2 px per fine cell: the peaks (3, 1), (0, 4), (4, 2) are (+1, -1), (-2, +2), (+2, 0) cells
        # from the window centre. kornia converts neither: both stay in the reference implementation's frame, in which
        # the pretrained models put pixel centres at +0.5 (subtract 0.5 for kornia's frame).
        expected0 = [[56.0, 24.0], [16.0, 40.0], [24.0, 16.0]]
        expected1 = [[24.0 + 2.0, 40.0 - 2.0], [48.0 - 4.0, 16.0 + 4.0], [56.0 + 4.0, 32.0]]
        self.assert_close(out["keypoints0"], torch.tensor(expected0, device=device, dtype=dtype))
        self.assert_close(out["keypoints1"], torch.tensor(expected1, device=device, dtype=dtype))
        assert out["batch_indexes"].tolist() == [0, 0, 1]
        # confidence is the coarse dual-softmax score: similarity 16 * 16 / 256 / 0.1 = 10 against the zero similarity
        # of the 79 other cells of image 1 and the 87 other cells of image 0. The pair of amplitude 4 (similarity
        # 0.625) scores about 5e-4, e**0.625 / (e**0.625 + 79) * e**0.625 / (e**0.625 + 87), below thr = 0.2, and is
        # dropped.
        confidence = 1.0 / ((1.0 + 79.0 * math.exp(-10.0)) * (1.0 + 87.0 * math.exp(-10.0)))
        self.assert_close(out["confidence"], torch.full((3,), confidence, device=device, dtype=dtype))
        # relabelling: with the images swapped the same cell pairs come out with their roles swapped
        with torch.no_grad():
            swapped = self._hooked_loftr(device, dtype, swap=True)({"image0": image1, "image1": image0})
        expected0 = [[48.0, 16.0], [24.0, 40.0], [56.0, 32.0]]
        expected1 = [[16.0 - 4.0, 40.0 + 4.0], [56.0 + 2.0, 24.0 - 2.0], [24.0 + 4.0, 16.0]]
        self.assert_close(swapped["keypoints0"], torch.tensor(expected0, device=device, dtype=dtype))
        self.assert_close(swapped["keypoints1"], torch.tensor(expected1, device=device, dtype=dtype))
        assert swapped["batch_indexes"].tolist() == [0, 0, 1]

    @pytest.mark.slow
    @pytest.mark.parametrize("data", ["loftr_homo"], indirect=True)
    def test_convention_pretrained_loftr_puts_pixel_centres_at_one_half(self, device, dtype, data):
        # A 2x bilinear upsampling (align_corners=False) maps x to 2x in the frame where pixel centres sit at i + 0.5,
        # and to 2x + 0.5 in kornia's frame; the pretrained matches follow the first.
        loftr = LoFTR("outdoor").to(device, dtype)
        image0 = data["image0"][..., 60:263, 100:417].to(device, dtype)  # 203 x 317, off-centre
        image1 = F.interpolate(image0, scale_factor=2.0, mode="bilinear", align_corners=False)
        with torch.no_grad():
            out = loftr({"image0": image0, "image1": image1})
        residual = out["keypoints1"].cpu().double() - 2 * out["keypoints0"].cpu().double()
        inliers = residual.norm(dim=-1) < 3
        assert inliers.double().mean() > 0.9
        median = residual[inliers].median(0).values
        assert median.abs().max() < 0.25, median
