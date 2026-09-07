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

from kornia.geometry.vector import Vector2, Vector3
from kornia.sensors.camera.projection_model import Z1Projection

from testing.base import BaseTester


class TestProjection(BaseTester):
    @pytest.mark.skip(reason="Unnecessary test")
    def test_smoke(self, device, dtype):
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_gradcheck(self, device):
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_module(self, device, dtype):
        pass

    def test_project(self, device, dtype):
        projection = Z1Projection()
        points = torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [6.0, 6.0, 2.0], [9.0, 9.0, 3.0]],
            device=device,
            dtype=dtype,
        )
        expected = torch.tensor([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0], [3.0, 3.0]], device=device, dtype=dtype)
        self.assert_close(projection.project(Vector3(points)).data, expected)

    def test_unproject(self, device, dtype):
        projection = Z1Projection()
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0], [3.0, 3.0]], device=device, dtype=dtype)
        expected = torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [6.0, 6.0, 2.0], [9.0, 9.0, 3.0]],
            device=device,
            dtype=dtype,
        )
        self.assert_close(
            projection.unproject(
                Vector2(points),
                torch.tensor([1.0, 1.0, 2.0, 3.0], device=device, dtype=dtype),
            ).data,
            expected,
        )

    def test_unproject_scalar_depth(self, device, dtype):
        """Regression test: scalar depth must preserve device and dtype.

        PR #4340: Z1Projection.unproject used to build a CPU float32 tensor
        for python int/float depth, which raised RuntimeError on every
        accelerator and widened float16/bfloat16 results to float32.
        """
        projection = Z1Projection()
        points = Vector2(torch.tensor([[0.25, 0.5]], device=device, dtype=dtype))
        expected = projection.unproject(
            points,
            torch.tensor([4.0], device=device, dtype=dtype),
        ).data
        for depth in (4, 4.0):
            out = projection.unproject(points, depth)
            assert out.data.device.type == device.type, f"expected device {device.type}, got {out.data.device.type}"
            assert out.data.dtype == dtype, f"expected dtype {dtype}, got {out.data.dtype}"
            self.assert_close(out.data, expected, atol=0.0, rtol=0.0)
        # Also verify a vector depth still works unchanged
        out_vec = projection.unproject(points, torch.tensor([4.0], device=device, dtype=dtype))
        self.assert_close(out_vec.data, expected, atol=0.0, rtol=0.0)
