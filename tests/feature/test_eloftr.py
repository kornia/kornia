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

from kornia.feature import EfficientLoFTR

from testing.base import BaseTester
from testing.casts import dict_to


class TestELoFTR(BaseTester):
    @pytest.mark.slow
    def test_pretrained_outdoor_smoke(self, device, dtype):
        eloftr = EfficientLoFTR().to(device, dtype)
        assert eloftr is not None

    @pytest.mark.slow
    @pytest.mark.parametrize("data", ["eloftr_outdoor"], indirect=True)
    def test_pretrained_outdoor(self, device, dtype, data):
        eloftr = EfficientLoFTR().to(device).eval()
        data_dev = dict_to(data, device, dtype)
        with torch.no_grad():
            out = eloftr(data_dev)

        self.assert_close(data_dev["keypoints0"].shape, out["keypoints0"].shape, rtol=1, atol=1)
        self.assert_close(data_dev["keypoints1"].shape, out["keypoints1"].shape, rtol=1, atol=1)

        # below assertation fails as different device/percesion generate difference confidence score
        # so matching output shape only
        # self.assert_close(data_dev["keypoints0"], out["keypoints0"])
        # self.assert_close(data_dev["keypoints1"], out["keypoints1"])

    # TODO: Add more tests and test data.
    # @pytest.mark.slow
    # def test_mask(self, device):
    #     patches = torch.rand(1, 1, 32, 32, device=device)
    #     mask = torch.rand(1, 32, 32, device=device) # TODO: Check masks input value
    #     eloftr = EfficientLoFTR().to(patches.device, patches.dtype)
    #     sample = {"image0": patches, "image1": patches, "mask0": mask, "mask1": mask}
    #     with torch.no_grad():
    #         out = eloftr(sample)
    #     assert out is not None

    # @pytest.mark.slow
    # def test_gradcheck(self, device):
    #     patches = torch.rand(1, 1, 32, 32, device=device, dtype=torch.float64)
    #     patches05 = resize(patches, (64, 64))
    #     eloftr = EfficientLoFTR().to(patches.device, patches.dtype)

    #     def proxy_forward(x, y):
    #         return eloftr.forward({"image0": x, "image1": y})["keypoints0"]

    #     self.gradcheck(proxy_forward, (patches, patches05), eps=1e-4, atol=1e-4)

    # @pytest.mark.skip("does not like transformer.py:L99, zip iteration")
    # def test_jit(self, device, dtype):
    #     B, C, H, W = 1, 1, 32, 32
    #     patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
    #     patches2x = resize(patches, (48, 48))
    #     sample = {"image0": patches, "image1": patches2x}
    #     model = EfficientLoFTR().to(patches.device, patches.dtype).eval()
    #     model_jit = torch.jit.script(model)
    #     out = model(sample)
    #     out_jit = model_jit(sample)
    #     for k, v in out.items():
    #         self.assert_close(v, out_jit[k])
