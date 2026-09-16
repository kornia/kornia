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
from kornia.feature.loftr.utils.fine_matching import FineMatching
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


class TestFineMatching(BaseTester):
    def test_convention_std_gradient_is_finite_on_a_peaked_heatmap_4229(self, device, dtype):
        # A heatmap peaked on one cell has zero variance. `std` was sqrt(clamp(var, min=1e-10)): in float16 the
        # floor is 0, and torch < 2.14 passes clamp's gradient through at the bound, so sqrt'(0) = inf reached
        # the features as nan. torch 2.14 zeroes clamp's gradient at the bound, so on 2.14 this passes on the
        # old code too; the torch 2.5.1 / 2.9.1 legs are the ones that discriminate.
        M, W, C = 4, 5, 8
        center = W * W // 2
        feat_f0 = torch.zeros(M, W * W, C, device=device, dtype=dtype)
        feat_f1 = torch.zeros(M, W * W, C, device=device, dtype=dtype)
        feat_f0[:, center, 0] = 30.0
        feat_f1[:, center, 0] = 30.0
        feat_f1.requires_grad_(True)
        data = {
            "hw0_i": (40, 40),
            "hw0_f": (20, 20),
            "mkpts0_c": torch.zeros(M, 2, device=device, dtype=dtype),
            "mkpts1_c": torch.zeros(M, 2, device=device, dtype=dtype),
            "mconf": torch.ones(M, device=device, dtype=dtype),
            "b_ids": torch.zeros(M, dtype=torch.long, device=device),
        }
        FineMatching()(feat_f0, feat_f1, data)
        std = data["expec_f"][:, 2]
        std.sum().backward()
        assert bool(torch.isfinite(feat_f1.grad).all()), feat_f1.grad
        # The value is unchanged: sqrt of the floor as represented in the input dtype.
        self.assert_close(std, torch.full_like(std, 1e-10).sqrt())
