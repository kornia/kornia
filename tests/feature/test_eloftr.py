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
        eloftr = EfficientLoFTR().to(device, dtype).eval()
        data_dev = dict_to(data, device, dtype)
        with torch.no_grad():
            out = eloftr(data_dev)

        self.assert_close(data_dev["keypoints0"].shape, out["keypoints0"].shape, rtol=1, atol=1)
        self.assert_close(data_dev["keypoints1"].shape, out["keypoints1"].shape, rtol=1, atol=1)
