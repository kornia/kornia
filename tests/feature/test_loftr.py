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

import sys

import pytest
import torch

from kornia.feature import LoFTR
from kornia.feature.loftr.utils.coarse_matching import CoarseMatching
from kornia.geometry import resize

from testing.base import BaseTester
from testing.casts import dict_to


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
        self.assert_close(out["keypoints0"], data_dev["loftr_outdoor_tentatives0"])
        self.assert_close(out["keypoints1"], data_dev["loftr_outdoor_tentatives1"])

    @pytest.mark.slow
    def test_mask(self, device):
        patches = torch.rand(1, 1, 32, 32, device=device)
        mask = torch.rand(1, 32, 32, device=device)
        loftr = LoFTR().to(patches.device, patches.dtype)
        sample = {"image0": patches, "image1": patches, "mask0": mask, "mask1": mask}
        with torch.no_grad():
            out = loftr(sample)
        assert out is not None

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


class TestCoarseMatching:
    """CoarseMatching.get_coarse_match's training-mode gt-padding branch
    (self.training=True) previously crashed unconditionally: zip() was
    called with 4 separate 2-element lists instead of 2 separate
    4-element lists, so `for x, y in zip(...)` tried to unpack each
    resulting 4-tuple into 2 variables and raised
    `ValueError: too many values to unpack (expected 2)` on every call.
    No existing test exercised this training-mode path at all."""

    def _config(self):
        return {
            "thr": 0.01,
            "border_rm": 0,
            "train_coarse_percent": 0.5,
            "train_pad_num_gt_min": 2,
            "match_type": "dual_softmax",
            "dsmax_temperature": 0.1,
        }

    def _data(self, h0c=4, w0c=4, h1c=4, w1c=4, n_gt=20):
        L, S = h0c * w0c, h1c * w1c
        return {
            "hw0_c": (h0c, w0c),
            "hw1_c": (h1c, w1c),
            "hw0_i": (h0c * 8, w0c * 8),
            "hw1_i": (h1c * 8, w1c * 8),
            "spv_b_ids": torch.zeros(n_gt, dtype=torch.long),
            "spv_i_ids": torch.randint(0, L, (n_gt,)),
            "spv_j_ids": torch.randint(0, S, (n_gt,)),
        }

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_training_gt_padding_does_not_crash_and_preserves_dtype(self, dtype):
        torch.manual_seed(0)
        cm = CoarseMatching(self._config())
        cm.train()
        conf_matrix = torch.rand(1, 16, 16, dtype=dtype)

        out = cm.get_coarse_match(conf_matrix, self._data())

        assert out["mconf"].dtype == dtype, (
            f"mconf silently changed dtype ({dtype} -> {out['mconf'].dtype}) through the "
            "gt-padding concatenation -- mconf_gt must be created in mconf's own dtype."
        )
        assert out["b_ids"].shape == out["i_ids"].shape == out["j_ids"].shape

    def test_eval_path_unaffected(self):
        torch.manual_seed(0)
        cm = CoarseMatching(self._config())
        cm.eval()
        conf_matrix = torch.rand(1, 16, 16, dtype=torch.float32)
        out = cm.get_coarse_match(conf_matrix, self._data())
        assert out["mconf"].dtype == torch.float32
