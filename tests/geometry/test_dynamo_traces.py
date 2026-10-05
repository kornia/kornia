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

from kornia.geometry.plane import Hyperplane
from kornia.geometry.vector import Scalar, Vector2, Vector3

from testing.base import BaseTester, dynamo_is_available


@pytest.mark.skipif(not dynamo_is_available(), reason="no Dynamo on this torch/python pair")
class TestGeometryEagerBackendTraces(BaseTester):
    # Dynamo on torch 2.5.1 cannot trace ``isinstance`` against a ``A | B`` union (#5371). The test names keep
    # "compile" and "dynamo" out, so the ordinary jobs, including the torch 2.5.1 legs that have Dynamo, run them.

    def test_eager_backend_traces_vector3_from_coords(self, device, dtype):
        def fn(x):
            return Vector3.from_coords(x, 2 * x, -x).data

        torch._dynamo.reset()
        x = torch.tensor([0.5, -1.0, 2.0, 3.0], device=device, dtype=dtype)
        self.assert_close(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    def test_eager_backend_traces_vector2_from_coords(self, device, dtype):
        def fn(x):
            return Vector2.from_coords(x, 2 * x).data

        torch._dynamo.reset()
        x = torch.tensor([0.5, -1.0, 2.0, 3.0], device=device, dtype=dtype)
        self.assert_close(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    def test_eager_backend_traces_hyperplane_signed_distance(self, device, dtype):
        normal = Vector3(torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype))
        plane = Hyperplane(normal, Scalar(torch.tensor(0.5, device=device, dtype=dtype)))

        def fn(p):
            return plane.signed_distance(p).data

        torch._dynamo.reset()
        p = torch.tensor([[1.0, 2.0, 3.0], [0.0, -1.0, -0.5]], device=device, dtype=dtype)
        self.assert_close(torch.compile(fn, backend="eager", fullgraph=True)(p), fn(p))
