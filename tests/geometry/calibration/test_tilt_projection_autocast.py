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

from kornia.geometry.calibration.distort import tilt_projection

from testing.base import BaseTester, skip_unsupported_autocast


class TestTiltProjectionAutocast(BaseTester):
    @pytest.mark.parametrize("execution", ["eager", "scripted"])
    @pytest.mark.parametrize("amp_dtype", ["float16", "bfloat16"])
    @pytest.mark.parametrize("return_inverse", [False, True])
    def test_convention_float32_matrix_and_gradient(self, execution, amp_dtype, return_inverse, device):
        skip_unsupported_autocast(device, amp_dtype)
        # Offline scalar reference: build OpenCV's Rx and Ry, multiply Pz @ Ry @ Rx,
        # and invert the resulting 3x3 matrix by its adjugate for the inverse case.
        # Weighted derivatives use centered differences of sum((3*i+j+1)*T[i,j]), h=1e-6.
        if return_inverse:
            values = [
                [1.0050209184004555, 0.0, 0.0],
                [0.020338844941192688, 1.0203388449411925, 0.0],
                [-0.20271003550867248, 0.10237536342322193, 1.0254618830224573],
            ]
            derivatives = [10.090730934386727, -3.7987818108931037]
        else:
            values = [
                [0.9950041652780258, 0.0, 0.0],
                [-0.019833838076209875, 0.9800665778412417, 0.0],
                [0.19866933079506122, -0.09784339500725571, 0.975170327201816],
            ]
            derivatives = [-9.572493835818818, 3.8553252128536997]

        angles = torch.tensor([0.1, 0.2], device=device, dtype=torch.float32, requires_grad=True)
        op = torch.jit.script(tilt_projection) if execution == "scripted" else tilt_projection
        with torch.autocast(device.type, dtype=getattr(torch, amp_dtype)):
            actual = op(angles[0], angles[1], return_inverse)
        assert actual.dtype == torch.float32
        self.assert_close(actual, torch.tensor(values, device=device, dtype=torch.float32), atol=2e-7, rtol=0)

        weights = torch.arange(1, 10, device=device, dtype=torch.float32).reshape(3, 3)
        gradient = torch.autograd.grad((actual * weights).sum(), angles)[0]
        self.assert_close(gradient, torch.tensor(derivatives, device=device, dtype=torch.float32), atol=3e-6, rtol=0)
