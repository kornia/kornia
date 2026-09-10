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
from torch._dynamo.testing import CompileCounter

from kornia.geometry.calibration import distort_points, undistort_points

from testing.base import BaseTester


class TestCalibrationCompile(BaseTester):
    @pytest.mark.parametrize("op", [distort_points, undistort_points])
    @pytest.mark.parametrize("num_coefficients", [4, 14])
    def test_dynamo_fullgraph(self, device, dtype, torch_optimizer, op, num_coefficients):
        # #4286: tensor-dependent tilt checks must not break capture or specialize on coefficient values.
        points = torch.tensor([[[1.0, 2.0], [6.0, 5.0]]], device=device, dtype=dtype)
        camera = torch.tensor([[[100.0, 0.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        dist = torch.zeros(1, num_coefficients, device=device, dtype=dtype)
        dist[:, :4] = torch.tensor([0.1, 0.01, 0.001, 0.0001], device=device, dtype=dtype)
        counter = CompileCounter()
        counted = torch.compile(op, backend=counter, fullgraph=True)
        compiled = torch_optimizer(op, fullgraph=True)
        for tilt in [0.0, 0.1]:
            if num_coefficients == 14:
                dist[:, 12] = tilt
                dist[:, 13] = -2 * tilt
            expected = op(points, camera, dist)
            self.assert_close(counted(points, camera, dist), expected)
            self.assert_close(compiled(points, camera, dist), expected)
        assert counter.frame_count == 1
