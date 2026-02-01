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

from kornia.core._compat import torch_version_ge
from kornia.feature import LoFTR
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
    @pytest.mark.skipif(torch_version_ge(1, 10), reason="RuntimeError: CUDA out of memory with pytorch>=1.10")
    @pytest.mark.skipif(sys.platform == "win32", reason="this test takes so much memory in the CI with Windows")
    @pytest.mark.parametrize("data", ["loftr_fund"], indirect=True)
    def test_pretrained_indoor(self, device, dtype, data):
        loftr = LoFTR("indoor").to(device, dtype)
        data_dev = dict_to(data, device, dtype)
        with torch.no_grad():
            out = loftr(data_dev)
        self.assert_close(out["keypoints0"], data_dev["loftr_indoor_tentatives0"])
        self.assert_close(out["keypoints1"], data_dev["loftr_indoor_tentatives1"])

    @pytest.mark.slow
    @pytest.mark.skipif(torch_version_ge(1, 10), reason="RuntimeError: CUDA out of memory with pytorch>=1.10")
    @pytest.mark.skipif(sys.platform == "win32", reason="this test takes so much memory in the CI with Windows")
    @pytest.mark.parametrize("data", ["loftr_homo"], indirect=True)
    def test_pretrained_outdoor(self, device, dtype, data):
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
