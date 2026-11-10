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

import importlib

import pytest
import torch

from kornia.filters.filter import filter2d_separable
from kornia.geometry import ScalePyramid

from testing.base import BaseTester


class TestScalePyramidCpu(BaseTester):
    def test_fast_blur_matches_separable_convolution(self, monkeypatch, device, dtype):
        if device.type != "cpu" or dtype not in (torch.float32, torch.float64):
            pytest.skip("the optimized scale-pyramid blur supports native CPU float32 and float64 only")
        # Exercise the generic ScalePyramid integration even on hosts where
        # oneDNN would normally keep the existing convolution implementation.
        monkeypatch.setattr(importlib.import_module("kornia.filters.gaussian"), "_HAS_MKLDNN", False)
        pyramid = ScalePyramid(n_levels=2).to(device=device, dtype=dtype)
        image = torch.rand(2, 1, 256, 256, device=device, dtype=dtype)
        kernel = pyramid._gk_0

        actual = pyramid._blur_fast(image, kernel)
        expected = filter2d_separable(image, kernel.view(1, -1), kernel.view(1, -1), "reflect")

        tolerance = 5e-7 if dtype == torch.float32 else 1e-12
        self.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
