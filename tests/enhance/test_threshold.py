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

import kornia
from kornia.core import tensor

from testing.base import BaseTester


class TestThreshold(BaseTester):
    thresh_fns = [
        kornia.enhance.thresh_binary,
        kornia.enhance.thresh_binary_inv,
        kornia.enhance.thresh_trunc,
        kornia.enhance.thresh_tozero,
        kornia.enhance.thresh_tozero_inv,
    ]

    thresh_modules = [
        kornia.enhance.ThreshBinary,
        kornia.enhance.ThreshBinaryInv,
        kornia.enhance.ThreshTrunc,
        kornia.enhance.ThreshToZero,
        kornia.enhance.ThreshToZeroInv,
    ]

    def get_binary_input(self, device, dtype, chan_last=False):
        # src = torch.randint(0, 2, (1, 3, 10, 10), device=device, dtype=dtype)
        src = torch.rand((1, 3, 10, 10), device=device, dtype=dtype).round()

        if chan_last:
            src = src.permute(0, 2, 3, 1)

        return src, 0.5, 255.0

    def test_thresh_binary(self, device, dtype):
        src, thresh, maxval = self.get_binary_input(device, dtype)

        out = kornia.enhance.thresh_binary(src, 0.5, maxval, beta=1000.0)
        hard_out = torch.where(src > thresh, maxval, 0).to(device=device, dtype=dtype)

        self.assert_close(out, hard_out, atol=1e-2, rtol=1e-3)

        out_inv = kornia.enhance.thresh_binary_inv(src, thresh, maxval, beta=1000.0)
        self.assert_close(maxval - out_inv, out, atol=1e-2, rtol=1e-3)

    def test_thresh_tozero(self, device, dtype):
        src, thresh, _ = self.get_binary_input(device, dtype)

        out = kornia.enhance.thresh_tozero(src, 0.5, beta=1000.0)
        hard_out = torch.where(src > thresh, src, 0).to(device=device, dtype=dtype)

        self.assert_close(out, hard_out, atol=1e-2, rtol=1e-3)

        out_inv = kornia.enhance.thresh_tozero_inv(src, thresh, beta=1000.0)
        self.assert_close(src - out_inv, out, atol=1e-2, rtol=1e-3)

    def test_thresh_trunc(self, device, dtype):
        src, thresh, _ = self.get_binary_input(device, dtype)

        out = kornia.enhance.thresh_trunc(src, thresh, beta=1000.0)
        hard_out = torch.where(src > thresh, thresh, src).to(device=device, dtype=dtype)

        self.assert_close(out, hard_out, atol=1e-2, rtol=1e-3)

    def test_per_channel_thresh(self, device, dtype):
        src, _, _ = self.get_binary_input(device, dtype, chan_last=True)

        thresh = tensor([0.5, 0, 1], dtype=dtype, device=device)

        out = kornia.enhance.thresh_binary(src, thresh, 1.0, beta=1000.0)
        hard_out = torch.where(src > thresh, 1.0, 0.0).to(device=device, dtype=dtype)

        self.assert_close(out.round(), hard_out.round(), atol=1e-2, rtol=1e-3)

    @pytest.mark.parametrize("fns", list(zip(thresh_modules, thresh_fns)))
    def test_module(self, device, dtype, fns):
        mod, fn = fns
        src, thresh, maxval = self.get_binary_input(device, dtype)

        # maxval may be passed in as beta for 2-arg functions
        # this is just to check that the results are consistent
        mod = mod(thresh, maxval)
        self.assert_close(mod(src), fn(src, thresh, maxval))

    @pytest.mark.parametrize("thresh_fn", thresh_fns)
    def test_gradcheck(self, device, dtype, thresh_fn):
        src, thresh, maxval = self.get_binary_input(device, dtype)

        # maxval may be passed in as beta for 2-arg functions
        # this is just to check that the results are consistent
        self.gradcheck(thresh_fn, (src, thresh, maxval))
